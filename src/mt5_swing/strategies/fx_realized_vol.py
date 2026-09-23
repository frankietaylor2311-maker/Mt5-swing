"""Menkhoff-style global FX realized-volatility risk factor (not VIX).

Literature
----------
Menkhoff, Sarno, Schmeling & Schrimpf (2012), "Carry Trades and Global Foreign
Exchange Volatility," *Journal of Finance*. Global FX volatility is a priced
state variable: high FX vol predicts carry underperformance / risk-off USD
demand. Their level proxy is the cross-sectional average of absolute daily
currency returns; innovations matter for risk premia.

This module builds a **true FX** realized-vol state from Yahoo USD-major D1
returns already in the repo (G10 subset: EUR/GBP/AUD/NZD/JPY/CAD/CHF) — distinct
from the equity-VIX proxy already studied in ``gpr_regime`` / ``scholarly_combo``.

Construction (PIT, fixed priors — no holdout tuning)
----------------------------------------------------
1. Map USD-pair returns → currency-vs-USD returns (``pair_returns_to_currency_returns``).
2. Daily global FX vol level: equal-weight mean of |r_ccy| across available FX
   (Menkhoff-style absolute-return average). Optional dollar-neutral check:
   demean currency returns cross-sectionally before abs — same level up to scale.
3. Trailing RV: rolling mean of daily level over ``rv_windows`` (default 21 / 63).
4. Causal z-score vs trailing ``z_window`` (default 252), then ``signal_lag``.
5. Tradables:
   - **Standalone USD tilt:** when z(FX-RV) ≥ z_high, long-USD equal-weight book;
     when z ≤ z_low, flat (or mild long-foreign). Continuous intensity in between.
   - **Carry conditioner:** scale scholarly carry weights by risk_scale ∈ [cool, 1]
     from the same FX-RV z (Menkhoff: cool carry in high FX vol).

All thresholds are frozen a-priori (match VIX/GPR combo priors: z_high=1,
cool=0.35). Do **not** claim ~1%/month FTMO consistency from this factor alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mt5_swing.strategies.carry_rank import (
    CarryRankConfig,
    carry_weights_from_rates,
    currency_weights_to_pair_weights,
    expand_weights_to_daily,
    portfolio_returns_from_weights,
)
from mt5_swing.strategies.fx_momentum import pair_returns_to_currency_returns
from mt5_swing.strategies.gpr_regime import USD_LONG_PAIRS, risk_scale_from_z


@dataclass
class FxRealizedVolConfig:
    """Fixed literature priors — do not grid-search on holdout."""

    rv_windows: tuple[int, ...] = (21, 63)
    z_window: int = 252
    min_periods_z: int = 60
    min_periods_rv: int = 15
    z_high: float = 1.0
    z_low: float = 0.0
    cool: float = 0.35
    signal_lag: int = 1
    usd_tilt: float = 0.5  # gross |w| when fully risk-off (standalone)
    dollar_neutral_abs: bool = False  # if True, abs(r - cross-section mean)
    cost_bps_side: float = 1.5
    # Carry conditioner uses same cool / z thresholds
    carry_n_long: int = 2
    carry_n_short: int = 2


def _ensure_utc_index(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    out = obj.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out.index = idx
    return out


def daily_fx_vol_level(
    pair_ret: pd.DataFrame,
    *,
    dollar_neutral_abs: bool = False,
) -> pd.Series:
    """Menkhoff-style daily global FX vol: EW mean of |currency returns|.

    Optional ``dollar_neutral_abs`` uses |r_i − mean_j r_j| (removes common
    dollar factor from the abs average). Default matches the paper's simple
    equal-weight absolute-return average on foreign currencies vs USD.
    """
    pret = _ensure_utc_index(pair_ret)
    ccy = pair_returns_to_currency_returns(pret)
    foreign = ccy.drop(columns=["USD"], errors="ignore")
    if foreign.empty:
        return pd.Series(dtype=float, name="fx_vol_level")
    if dollar_neutral_abs:
        demeaned = foreign.sub(foreign.mean(axis=1), axis=0)
        level = demeaned.abs().mean(axis=1)
    else:
        level = foreign.abs().mean(axis=1)
    level.name = "fx_vol_level"
    return level


def trailing_fx_rv(
    daily_level: pd.Series,
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Causal trailing mean of daily FX-vol level (realized vol proxy)."""
    s = daily_level.astype(float)
    out = s.rolling(int(window), min_periods=int(min_periods)).mean()
    out.name = f"fx_rv_{int(window)}d"
    return out


def fx_rv_zscore(
    rv: pd.Series,
    *,
    z_window: int,
    min_periods: int,
    signal_lag: int,
) -> pd.Series:
    """Trailing z of RV, then ``signal_lag`` (PIT)."""
    mu = rv.rolling(int(z_window), min_periods=int(min_periods)).mean()
    sd = rv.rolling(int(z_window), min_periods=int(min_periods)).std()
    z = (rv - mu) / sd.replace(0.0, np.nan)
    z = z.shift(int(signal_lag))
    z.name = "fx_rv_z"
    return z


def fx_rv_innovation(
    rv: pd.Series,
    *,
    ar_window: int = 63,
    min_periods: int = 40,
    signal_lag: int = 1,
) -> pd.Series:
    """Simple causal innovation: RV − trailing mean (AR(1)-lite unexpected vol).

    Menkhoff et al. emphasize *innovations* in global FX vol. We use RV minus
    its trailing mean (equivalent to residual vs constant local mean) — no
    rolling OLS of RV on lag(RV) to keep the prior transparent and avoid
    extra parameters. Shifted by ``signal_lag``.
    """
    expected = rv.rolling(int(ar_window), min_periods=int(min_periods)).mean()
    innov = (rv - expected).shift(int(signal_lag))
    innov.name = "fx_rv_innov"
    return innov


def usd_tilt_pair_weights_from_z(
    z: pd.Series,
    pair_columns: list[str],
    *,
    cfg: FxRealizedVolConfig,
) -> pd.DataFrame:
    """Map FX-RV z → long-USD weights (high z → long USD).

    Intensity = clip((z - z_low) / (z_high - z_low), 0, 1) * usd_tilt.
    Distributed equally across available USD_LONG_PAIRS symbols.
    """
    zh, zl = float(cfg.z_high), float(cfg.z_low)
    denom = max(zh - zl, 1e-6)
    intensity = ((z - zl) / denom).clip(0.0, 1.0) * float(cfg.usd_tilt)
    cols = [c for c in pair_columns if c.upper() in {p.upper() for p in USD_LONG_PAIRS}]
    if not cols:
        cols = list(pair_columns)
    n = max(len(cols), 1)
    w = pd.DataFrame(0.0, index=z.index, columns=cols)
    for sym in cols:
        sign = USD_LONG_PAIRS.get(sym.upper(), 0)
        # +sign on symbol = long USD; allocate intensity / n
        w[sym] = (intensity / n) * sign
    w = w.fillna(0.0)
    w.attrs["strategy"] = "fxrv_usd_tilt"
    return w


def innov_usd_tilt_weights(
    innov: pd.Series,
    pair_columns: list[str],
    *,
    cfg: FxRealizedVolConfig,
    innov_scale: float | None = None,
) -> pd.DataFrame:
    """Positive FX-RV innovation → long USD (risk-off), scaled by softsign.

    Intensity = softsign(innov / scale) clipped to [0, 1] * usd_tilt when innov>0;
    flat when innov≤0. ``innov_scale`` defaults to trailing MAD of innov (causal
    expanding then rolling) — fixed formula, not HO-tuned threshold grid.
    """
    innov = innov.astype(float)
    if innov_scale is None:
        # Causal scale: rolling MAD of |innov|, min_periods from cfg
        mad = innov.abs().rolling(cfg.z_window, min_periods=cfg.min_periods_z).median()
        scale = mad.replace(0.0, np.nan).fillna(innov.abs().median() or 1e-6)
    else:
        scale = float(innov_scale)
    raw = (innov / scale).clip(-5, 5)
    # Only positive innovations → USD long; negative → flat (not short USD)
    intensity = raw.clip(lower=0.0).clip(upper=1.0) * float(cfg.usd_tilt)
    cols = [c for c in pair_columns if c.upper() in {p.upper() for p in USD_LONG_PAIRS}]
    if not cols:
        cols = list(pair_columns)
    n = max(len(cols), 1)
    w = pd.DataFrame(0.0, index=innov.index, columns=cols)
    for sym in cols:
        sign = USD_LONG_PAIRS.get(sym.upper(), 0)
        w[sym] = (intensity / n) * sign
    w = w.fillna(0.0)
    w.attrs["strategy"] = "fxrv_innov_usd"
    return w


def apply_fxrv_cool_to_weights(
    base_weights: pd.DataFrame,
    z: pd.Series,
    *,
    cfg: FxRealizedVolConfig,
) -> pd.DataFrame:
    """Scale base (carry) weights by FX-RV risk_scale ∈ [cool, 1]."""
    # Reuse gpr_regime mapper with a tiny shim config
    from mt5_swing.strategies.gpr_regime import GprRegimeConfig

    gcfg = GprRegimeConfig(
        z_high=cfg.z_high,
        z_low=cfg.z_low,
        cool=cfg.cool,
        signal_lag=0,  # z already lagged
        usd_tilt=0.0,
    )
    # risk_scale_from_z expects a Series named like regime_z
    scale = risk_scale_from_z(z, cfg=gcfg)
    # Align
    scale = scale.reindex(base_weights.index).ffill().fillna(1.0)
    out = base_weights.mul(scale, axis=0)
    out.attrs["strategy"] = base_weights.attrs.get("strategy", "carry") + "_fxrv_cool"
    return out


def apply_costs(port: pd.Series, weights: pd.DataFrame, *, bps_side: float) -> pd.Series:
    """Simple turnover cost: bps per side on abs weight change."""
    if bps_side <= 0 or weights.empty:
        return port
    dw = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = dw * (float(bps_side) / 10_000.0)
    # Cost paid on the day weights change; port already uses lag(w)*r
    out = port - cost.reindex(port.index).fillna(0.0)
    out.name = port.name
    return out


def build_fxrv_state(
    pair_ret: pd.DataFrame,
    *,
    cfg: FxRealizedVolConfig | None = None,
    window: int = 21,
) -> dict[str, pd.Series]:
    """Return daily level, trailing RV, z, and innovation for one RV window."""
    cfg = cfg or FxRealizedVolConfig()
    pret = _ensure_utc_index(pair_ret)
    level = daily_fx_vol_level(pret, dollar_neutral_abs=cfg.dollar_neutral_abs)
    rv = trailing_fx_rv(level, window=window, min_periods=min(cfg.min_periods_rv, window))
    z = fx_rv_zscore(
        rv,
        z_window=cfg.z_window,
        min_periods=cfg.min_periods_z,
        signal_lag=cfg.signal_lag,
    )
    innov = fx_rv_innovation(
        rv,
        ar_window=max(window, 63),
        min_periods=max(cfg.min_periods_rv, 40),
        signal_lag=cfg.signal_lag,
    )
    return {"level": level, "rv": rv, "z": z, "innov": innov}


def fx_realized_vol_factor_returns(
    pair_ret: pd.DataFrame,
    rates: pd.DataFrame | None = None,
    *,
    cfg: FxRealizedVolConfig | None = None,
) -> dict[str, pd.Series]:
    """Build standalone FX-RV factors + optional carry×FX-RV cool.

    Returns dict of daily portfolio return series (post optional costs).
    """
    cfg = cfg or FxRealizedVolConfig()
    pret = _ensure_utc_index(pair_ret)
    cols = list(pret.columns)
    out: dict[str, pd.Series] = {}

    # Raw carry (for conditioner baseline) if rates provided
    carry_daily: pd.DataFrame | None = None
    if rates is not None and not rates.empty:
        ccfg = CarryRankConfig(
            n_long=cfg.carry_n_long,
            n_short=cfg.carry_n_short,
            signal_lag=cfg.signal_lag,
        )
        ccy_w = carry_weights_from_rates(rates, cfg=ccfg)
        carry_pw = currency_weights_to_pair_weights(ccy_w)
        carry_daily = expand_weights_to_daily(carry_pw, pret.index, signal_lag=cfg.signal_lag)
        # Align columns
        for c in cols:
            if c not in carry_daily.columns:
                carry_daily[c] = 0.0
        carry_daily = carry_daily.reindex(columns=cols).fillna(0.0)
        carry_r = portfolio_returns_from_weights(carry_daily, pret)
        carry_r = apply_costs(carry_r, carry_daily, bps_side=cfg.cost_bps_side)
        carry_r.name = "carry_raw"
        out["carry_raw"] = carry_r

    for window in cfg.rv_windows:
        st = build_fxrv_state(pret, cfg=cfg, window=int(window))
        tag = f"{int(window)}d"

        # --- Standalone: USD tilt on high FX-RV z ---
        w_tilt = usd_tilt_pair_weights_from_z(st["z"], cols, cfg=cfg)
        for c in cols:
            if c not in w_tilt.columns:
                w_tilt[c] = 0.0
        w_tilt = w_tilt.reindex(columns=cols).fillna(0.0)
        r_tilt = portfolio_returns_from_weights(w_tilt, pret)
        r_tilt = apply_costs(r_tilt, w_tilt, bps_side=cfg.cost_bps_side)
        r_tilt.name = f"fxrv_usd_tilt_{tag}"
        out[r_tilt.name] = r_tilt

        # --- Standalone: positive innovation → long USD ---
        w_inn = innov_usd_tilt_weights(st["innov"], cols, cfg=cfg)
        for c in cols:
            if c not in w_inn.columns:
                w_inn[c] = 0.0
        w_inn = w_inn.reindex(columns=cols).fillna(0.0)
        r_inn = portfolio_returns_from_weights(w_inn, pret)
        r_inn = apply_costs(r_inn, w_inn, bps_side=cfg.cost_bps_side)
        r_inn.name = f"fxrv_innov_usd_{tag}"
        out[r_inn.name] = r_inn

        # Binary high-vol long-USD (z >= z_high) vs flat — frozen cutoff
        high = (st["z"] >= cfg.z_high).astype(float)
        w_bin = w_tilt.mul(0.0)
        # Rebuild at full usd_tilt when high
        intensity = high * float(cfg.usd_tilt)
        n = max(len([c for c in cols if c.upper() in USD_LONG_PAIRS]), 1)
        for sym in cols:
            sign = USD_LONG_PAIRS.get(sym.upper(), 0)
            w_bin[sym] = (intensity / n) * sign
        r_bin = portfolio_returns_from_weights(w_bin, pret)
        r_bin = apply_costs(r_bin, w_bin, bps_side=cfg.cost_bps_side)
        r_bin.name = f"fxrv_highvol_usd_{tag}"
        out[r_bin.name] = r_bin

        # --- Carry conditioner ---
        if carry_daily is not None:
            cooled = apply_fxrv_cool_to_weights(carry_daily, st["z"], cfg=cfg)
            r_cool = portfolio_returns_from_weights(cooled, pret)
            r_cool = apply_costs(r_cool, cooled, bps_side=cfg.cost_bps_side)
            r_cool.name = f"carry_fxrv_cool_{tag}"
            out[r_cool.name] = r_cool

            # Low-vol only carry: full carry when z <= z_low, else flat
            low = (st["z"] <= cfg.z_low).astype(float)
            # When z is NaN early, stay flat
            low = low.where(st["z"].notna(), 0.0)
            gated = carry_daily.mul(low, axis=0)
            r_gate = portfolio_returns_from_weights(gated, pret)
            r_gate = apply_costs(r_gate, gated, bps_side=cfg.cost_bps_side)
            r_gate.name = f"carry_fxrv_lowvol_only_{tag}"
            out[r_gate.name] = r_gate

    # Equal-weight blend of primary standalone legs (21d tilt + innov)
    blend_keys = [k for k in ("fxrv_usd_tilt_21d", "fxrv_innov_usd_21d") if k in out]
    if len(blend_keys) >= 2:
        blend = sum(out[k] for k in blend_keys) / len(blend_keys)
        blend.name = "fxrv_ew_21d"
        out["fxrv_ew_21d"] = blend

    return out


class FxRealizedVolBasket:
    """Panel helper exposing FX-RV state + factor returns."""

    name = "fx_realized_vol"

    def __init__(self, **kwargs):
        fields = FxRealizedVolConfig.__dataclass_fields__
        self.cfg = FxRealizedVolConfig(
            **{k: v for k, v in kwargs.items() if k in fields}
        )

    def state(self, pair_ret: pd.DataFrame, window: int = 21) -> dict[str, pd.Series]:
        return build_fxrv_state(pair_ret, cfg=self.cfg, window=window)

    def portfolio_returns(
        self,
        pair_ret: pd.DataFrame,
        rates: pd.DataFrame | None = None,
    ) -> dict[str, pd.Series]:
        return fx_realized_vol_factor_returns(pair_ret, rates, cfg=self.cfg)
