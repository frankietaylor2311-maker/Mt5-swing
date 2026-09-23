"""Bilateral AI-GPR role decompositions → FX (Caldara–Iacoviello).

Literature priors (frozen — not holdout-tuned)
---------------------------------------------
- Caldara & Iacoviello (2022) AER / AI-GPR: elevated geopolitical *threats* and
  *acts*, and oil-region GPR, associate with risk-off / safe-haven USD (and often
  CHF/JPY) and commodity-currency pressure.
- Distinct from: aggregate GPR regime (§8 / ``gpr_regime.py``), country-GPRC_*
  sorts (§9 / ``country_gpr_fx.py``), news-event intensity (§10), CRR/ToT
  commodity FX.

Legs
----
- ``ai_threats_usd`` (primary): high z(THREATS_GPR_AI) → long USD / short risk FX
- ``ai_acts_usd``: high z(ACTS_GPR_AI) → long USD
- ``ai_gpr_usd``: high z(GPR_AI) → long USD
- ``oil_gpr_usd`` / ``oil_threats_usd``: oil-GPR / oil-threats → USD haven
- ``oil_me_vs_non``: z(GPR_OIL_MiddleEast − GPR_NONOIL) → USD tilt
- ``carry_ai_threats_cool``: scholarly carry × threats risk_scale ∈ [cool, 1]
  (evaluate as sleeve factor only — **do not** apply to locked fx4plus)
- ``ai_ew``: EW of main USD-tilt legs

PIT: loader ``pub_lag_days=1``; this module adds ``signal_lag`` trading days on
aligned trailing z before weights. ``portfolio_returns_from_weights`` adds an
extra weight lag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.strategies.carry_rank import (
    CarryRankConfig,
    carry_weights_from_rates,
    currency_weights_to_pair_weights,
    expand_weights_to_daily,
    portfolio_returns_from_weights,
)
from mt5_swing.strategies.gpr_regime import (
    GprRegimeConfig,
    USD_LONG_PAIRS,
    align_macro_to_index,
    risk_scale_from_z,
)
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class AiGprFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # trading days after pub-lagged AI-GPR is known
    z_window: int = 252  # trailing trading-day z (frozen daily-macro prior)
    min_periods: int = 60
    z_high: float = 1.0
    z_low: float = 0.0
    cool: float = 0.35
    usd_tilt: float = 0.5
    cost_bps_side: float = 1.5
    carry_n_long: int = 2
    carry_n_short: int = 2
    binary_usd: bool = True


def trailing_z(s: pd.Series, *, lookback: int, min_periods: int) -> pd.Series:
    mu = s.rolling(lookback, min_periods=min_periods).mean()
    sd = s.rolling(lookback, min_periods=min_periods).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def align_and_z(
    series: pd.Series,
    index: pd.DatetimeIndex,
    *,
    cfg: AiGprFxConfig,
) -> pd.Series:
    """As-of align → trailing z → signal_lag trading days."""
    aligned = align_macro_to_index(series, index)
    z = trailing_z(aligned, lookback=cfg.z_window, min_periods=cfg.min_periods)
    if cfg.signal_lag > 0:
        z = z.shift(int(cfg.signal_lag))
    z.name = f"{series.name or 'macro'}_z"
    return z


def usd_tilt_weights_from_z(
    z: pd.Series,
    pair_columns: list[str],
    *,
    cfg: AiGprFxConfig,
) -> pd.DataFrame:
    """Long-USD weights when AI-GPR role z elevated (safe-haven prior)."""
    if cfg.binary_usd:
        intensity = (z >= cfg.z_high).astype(float) * float(cfg.usd_tilt)
        intensity = intensity.where(z.notna(), 0.0)
    else:
        denom = max(float(cfg.z_high) - float(cfg.z_low), 1e-6)
        intensity = ((z - cfg.z_low) / denom).clip(0.0, 1.0) * float(cfg.usd_tilt)
        intensity = intensity.fillna(0.0)
    cols = [c for c in pair_columns if c.upper() in {p.upper() for p in USD_LONG_PAIRS}]
    if not cols:
        cols = list(pair_columns)
    n = max(len(cols), 1)
    w = pd.DataFrame(0.0, index=z.index, columns=cols)
    for sym in cols:
        sign = USD_LONG_PAIRS.get(sym.upper(), 0)
        w[sym] = (intensity / n) * sign
    return w.fillna(0.0)


def _weights_to_returns(
    w: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: AiGprFxConfig,
    name: str,
) -> pd.Series:
    cols = list(pair_ret.columns)
    for c in cols:
        if c not in w.columns:
            w[c] = 0.0
    w = w.reindex(columns=cols).fillna(0.0)
    r = portfolio_returns_from_weights(w, pair_ret)
    r = apply_costs(r, w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


def ai_gpr_factor_returns(
    pair_ret: pd.DataFrame,
    *,
    threats: pd.Series | None = None,
    acts: pd.Series | None = None,
    gpr_ai: pd.Series | None = None,
    oil_gpr: pd.Series | None = None,
    oil_threats: pd.Series | None = None,
    oil_me_vs_non: pd.Series | None = None,
    rates: pd.DataFrame | None = None,
    cfg: AiGprFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Build AI-GPR role FX factor daily returns (post optional costs)."""
    cfg = cfg or AiGprFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    cols = list(pret.columns)
    out: dict[str, pd.Series] = {}

    role_map = (
        ("ai_threats_usd", threats),
        ("ai_acts_usd", acts),
        ("ai_gpr_usd", gpr_ai),
        ("oil_gpr_usd", oil_gpr),
        ("oil_threats_usd", oil_threats),
        ("oil_me_vs_non", oil_me_vs_non),
    )
    for name, series in role_map:
        if series is None or not len(series):
            continue
        z = align_and_z(series, pret.index, cfg=cfg)
        out[name] = _weights_to_returns(
            usd_tilt_weights_from_z(z, cols, cfg=cfg), pret, cfg=cfg, name=name
        )

    # Scholarly carry cooled by AI threats (NOT for locked sleeve overlay)
    if rates is not None and not rates.empty and threats is not None and len(threats):
        ccfg = CarryRankConfig(
            n_long=cfg.carry_n_long,
            n_short=cfg.carry_n_short,
            signal_lag=cfg.signal_lag,
        )
        ccy_w = carry_weights_from_rates(rates, cfg=ccfg)
        carry_pw = currency_weights_to_pair_weights(ccy_w)
        carry_daily = expand_weights_to_daily(carry_pw, pret.index, signal_lag=cfg.signal_lag)
        for c in cols:
            if c not in carry_daily.columns:
                carry_daily[c] = 0.0
        carry_daily = carry_daily.reindex(columns=cols).fillna(0.0)
        raw = portfolio_returns_from_weights(carry_daily, pret)
        raw = apply_costs(raw, carry_daily, bps_side=cfg.cost_bps_side)
        raw.name = "carry_raw"
        out["carry_raw"] = raw

        gcfg = GprRegimeConfig(
            z_high=cfg.z_high,
            z_low=cfg.z_low,
            cool=cfg.cool,
            signal_lag=0,  # z already lagged
            usd_tilt=0.0,
        )
        z = align_and_z(threats, pret.index, cfg=cfg)
        scale = risk_scale_from_z(z, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
        cooled = carry_daily.mul(scale, axis=0)
        r = portfolio_returns_from_weights(cooled, pret)
        r = apply_costs(r, cooled, bps_side=cfg.cost_bps_side)
        r.name = "carry_ai_threats_cool"
        out[r.name] = r

    blend_keys = [
        k
        for k in ("ai_threats_usd", "ai_acts_usd", "ai_gpr_usd", "oil_gpr_usd")
        if k in out
    ]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "ai_ew"
        out["ai_ew"] = blend

    return out


__all__ = [
    "AiGprFxConfig",
    "ai_gpr_factor_returns",
    "align_and_z",
    "trailing_z",
    "usd_tilt_weights_from_z",
]
