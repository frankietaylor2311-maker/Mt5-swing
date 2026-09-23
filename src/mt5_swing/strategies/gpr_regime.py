"""GPR / VIX regime filter — scale FX exposure; optional safe-haven USD tilt.

Literature hook: Caldara–Iacoviello geopolitical risk; FX vol / uncertainty
compresses carry and momentum (Menkhoff et al.). High VIX / high GPR / high EPU regimes
are associated with USD strength (safe-haven) and carry crashes.

Rules (research priors, lagged only — no HO tuning):
- Compute z-score of lagged GPR and/or VIX vs trailing window.
- If either z >= z_high -> risk_scale = cool (default 0.35) and optional
  usd_tilt toward long-USD expressions on the pair book.
- If both z <= z_low -> full scale (1.0).
- Else linear interpolate between cool and 1.0.

All inputs must already include publication lag from the macro loaders; this
module applies an extra signal_lag on the regime series before scaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.strategies.base import Signal

# Pairs where +1 weight = long USD (safe-haven tilt direction)
USD_LONG_PAIRS = {
    "EURUSD": -1,  # long USD = short EURUSD
    "GBPUSD": -1,
    "AUDUSD": -1,
    "NZDUSD": -1,
    "USDJPY": +1,  # long USDJPY = long USD
    "USDCAD": +1,
    "USDCHF": +1,
}


@dataclass
class GprRegimeConfig:
    z_window: int = 252
    z_high: float = 1.0
    z_low: float = 0.0
    cool: float = 0.35
    signal_lag: int = 1
    use_gpr: bool = True
    use_vix: bool = True
    use_epu: bool = True
    usd_tilt: float = 0.15  # max additive tilt toward USD in high-stress
    min_periods: int = 60


def _zscore(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def align_macro_to_index(macro: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """As-of merge (ffill) macro levels onto a trading calendar."""
    s = macro.copy()
    s.index = pd.DatetimeIndex(s.index)
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")
    idx = index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
    return s.reindex(idx, method="ffill")


def regime_z(
    gpr: pd.Series | None,
    vix: pd.Series | None,
    index: pd.DatetimeIndex,
    *,
    cfg: GprRegimeConfig,
    epu: pd.Series | None = None,
) -> pd.Series:
    """Combined stress z = max(z_gpr, z_vix, z_epu) on index (then signal_lag).

    Baker–Bloom–Davis EPU enters as an additional uncertainty state variable
    with a fixed a-priori threshold (same z_high as GPR/VIX) — not grid-searched.
    """
    zs = []
    if cfg.use_gpr and gpr is not None and len(gpr):
        g = align_macro_to_index(gpr, index)
        zs.append(_zscore(g, cfg.z_window, cfg.min_periods))
    if cfg.use_vix and vix is not None and len(vix):
        v = align_macro_to_index(vix, index)
        zs.append(_zscore(v, cfg.z_window, cfg.min_periods))
    if getattr(cfg, "use_epu", False) and epu is not None and len(epu):
        e = align_macro_to_index(epu, index)
        zs.append(_zscore(e, cfg.z_window, cfg.min_periods))
    if not zs:
        return pd.Series(0.0, index=index, name="regime_z")
    z = zs[0]
    for other in zs[1:]:
        z = pd.concat([z, other], axis=1).max(axis=1)
    z = z.shift(int(cfg.signal_lag))
    z.name = "regime_z"
    return z


def risk_scale_from_z(z: pd.Series, *, cfg: GprRegimeConfig) -> pd.Series:
    """Map regime z -> (cool, 1.0] scale; causal via already-lagged z."""
    cool = float(cfg.cool)
    zh, zl = float(cfg.z_high), float(cfg.z_low)
    scale = pd.Series(1.0, index=z.index, dtype=float)
    scale = scale.where(~(z >= zh), cool)
    mid = (z > zl) & (z < zh)
    if zh > zl:
        frac = ((z - zl) / (zh - zl)).clip(0, 1)
        interp = 1.0 - frac * (1.0 - cool)
        scale = scale.where(~mid, interp)
    scale = scale.fillna(1.0).clip(cool, 1.0)
    scale.name = "risk_scale"
    return scale


def usd_tilt_weights(
    z: pd.Series,
    symbols: list[str],
    *,
    cfg: GprRegimeConfig,
) -> pd.DataFrame:
    """Additive USD-safe-haven tilt weights when z is elevated."""
    zh = float(cfg.z_high)
    intensity = ((z - 0.0) / max(zh, 1e-6)).clip(0, 1) * float(cfg.usd_tilt)
    rows = {}
    for sym in symbols:
        sign = USD_LONG_PAIRS.get(sym.upper(), 0)
        rows[sym] = intensity * sign
    return pd.DataFrame(rows, index=z.index).fillna(0.0)


def apply_regime_to_weights(
    base_weights: pd.DataFrame,
    gpr: pd.Series | None,
    vix: pd.Series | None,
    *,
    cfg: GprRegimeConfig | None = None,
    epu: pd.Series | None = None,
) -> pd.DataFrame:
    """Scale base_weights by risk_scale and add optional USD tilt."""
    cfg = cfg or GprRegimeConfig()
    z = regime_z(gpr, vix, base_weights.index, cfg=cfg, epu=epu)
    scale = risk_scale_from_z(z, cfg=cfg)
    scaled = base_weights.mul(scale, axis=0)
    if cfg.usd_tilt > 0:
        tilt = usd_tilt_weights(z, list(base_weights.columns), cfg=cfg)
        for c in tilt.columns:
            if c in scaled.columns:
                scaled[c] = scaled[c] + tilt[c]
    scaled.attrs["strategy"] = "gpr_regime"
    return scaled


def apply_regime_to_returns(
    port_ret: pd.Series,
    gpr: pd.Series | None,
    vix: pd.Series | None,
    *,
    cfg: GprRegimeConfig | None = None,
    epu: pd.Series | None = None,
) -> pd.Series:
    """Scale a single portfolio return series by lagged regime risk_scale."""
    cfg = cfg or GprRegimeConfig()
    z = regime_z(gpr, vix, port_ret.index, cfg=cfg, epu=epu)
    scale = risk_scale_from_z(z, cfg=cfg)
    out = port_ret * scale.shift(1).fillna(1.0)
    out.name = (port_ret.name or "port") + "_gpr_regime"
    return out


class GprRegimeFilter:
    name = "gpr_regime"

    def __init__(self, **kwargs):
        fields = GprRegimeConfig.__dataclass_fields__
        self.cfg = GprRegimeConfig(**{k: v for k, v in kwargs.items() if k in fields})

    def scale_weights(
        self,
        weights: pd.DataFrame,
        gpr: pd.Series | None = None,
        vix: pd.Series | None = None,
    ) -> pd.DataFrame:
        return apply_regime_to_weights(weights, gpr, vix, cfg=self.cfg)


class GprRegimePairSignal:
    """Flatten single-pair signals when lagged stress z >= z_high."""

    name = "gpr_regime_pair"

    def __init__(self, z_high: float = 1.0, stress_col: str = "stress_z"):
        self.z_high = float(z_high)
        self.stress_col = stress_col

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        base = data.get("base_signal")
        if base is None:
            base = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        if self.stress_col not in data.columns:
            return base.astype(int)
        z = data[self.stress_col]
        out = base.copy().astype(int)
        out = out.mask(z >= self.z_high, int(Signal.FLAT))
        return out.astype(int)
