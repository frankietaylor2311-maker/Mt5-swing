"""Du–Schreger CIP-conditioned Lustig–Verdelhan carry FX factors.

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Lustig, Roussanov & Verdelhan (2011): FX carry sorted by interest-rate
  differentials (high-minus-low).
- Du, Tepper & Verdelhan (2018), *JF*: observed CIP / cross-currency basis
  deviations reflect intermediary / dollar-funding stress; carry crashes when
  funding is stressed.
- Du, Im & Schreger (2018) / Du, Keerati & Schreger (2025): government-bond CIP
  ``cip_govt`` (bps) and U.S. Treasury Premium (−cross-sectional mean CIP)
  measure synthetic-dollar sovereign cost / Treasury convenience.

Claim (a priori)
----------------
IR3M / policy-rate carry earns more when aggregate CIP / dollar-funding stress
is **low**. Scale down or gate carry when stress is elevated:
``cip_stress ≡ UST premium = −mean(G10 cip_govt)`` (more-negative mean CIP /
high |UST premium|). Honesty always-on carry and/or inverse gate.

Legs
----
- ``carry_low_cip_stress`` (**PRIMARY**): trade scholarly cash-rate carry only
  when lagged CIP-stress z ≤ 0 (flat in elevated stress).
- ``carry_cip_cool``: carry × risk_scale ∈ [cool, 1] from CIP-stress z
  (cool when z ≥ z_high).
- ``carry_raw``: always-on Lustig–Verdelhan carry (honesty baseline).
- ``carry_high_cip_stress``: inverse gate — carry only when stress z ≥ z_high.
- ``cip_stress_haven_usd``: long USD when lagged CIP-stress z ≥ z_high.
- ``cip_carry_ew``: EW of primary + cool + haven (when present).

PIT
---
CIP panel: loader ``pub_lag_days=1`` → month-end; this module adds
``signal_lag_months`` (default 1) on stress z + **1 trading-day** weight lag.
Rates: existing FRED carry loaders' PIT lags + ``carry_signal_lag`` trading days
via ``CarryRankConfig`` / ``expand_weights_to_daily``.
``portfolio_returns_from_weights`` applies an extra weight lag.

Gate threshold: fixed z (z_high=1.0 / loose ≤0) — mirror funding_liq / ACM;
**do not** grid on holdout.

Distinct from: fwd_carry §19, funding_liq §20, IG OAS §36, ACM TP §44,
CIP XS §67, combo §8, capital-sleeve §53 / soft EW §66.
Explicit: do **not** overlay on the locked FTMO sleeve.
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
from mt5_swing.strategies.cip_basis_fx import trailing_z
from mt5_swing.strategies.gpr_regime import (
    GprRegimeConfig,
    USD_LONG_PAIRS,
    risk_scale_from_z,
)
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class CipConditionedCarryFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag_months: int = 1  # months after pub-lagged month-end CIP known
    carry_signal_lag: int = 1  # trading days on carry weight expand
    z_window: int = 60  # months (~5y), match CIP XS §67
    min_periods: int = 24
    z_high: float = 1.0
    z_low: float = 0.0
    cool: float = 0.35
    usd_tilt: float = 0.5
    cost_bps_side: float = 1.5
    carry_n_long: int = 2
    carry_n_short: int = 2
    weight_lag_days: int = 1  # daily lag after monthly z ffill


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _month_start(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    return pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start"), tz="UTC")


def _month_end(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    return pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="end"), tz="UTC")


def cip_stress_from_panel(cip_panel: pd.DataFrame) -> pd.Series:
    """Aggregate CIP stress = UST premium = −mean(G10 foreign cip_govt).

    High / more-positive stress ↔ more-negative mean CIP ↔ elevated Treasury
    premium / dollar-funding specialness channel used to cool carry.
    """
    foreign = cip_panel[[c for c in cip_panel.columns if str(c).upper() != "USD"]]
    stress = (-foreign.mean(axis=1, skipna=True)).rename("cip_stress")
    return stress


def align_cip_stress_z_daily(
    cip_stress: pd.Series,
    pair_index: pd.DatetimeIndex,
    *,
    cfg: CipConditionedCarryFxConfig,
) -> pd.Series:
    """Month-end CIP stress → trailing z → signal_lag months → daily ffill + 1d lag."""
    s = cip_stress.copy()
    s.index = _month_start(pd.DatetimeIndex(s.index))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    z = trailing_z(s, lookback=cfg.z_window, min_periods=cfg.min_periods)
    if cfg.signal_lag_months > 0:
        z = z.shift(int(cfg.signal_lag_months))
    z_me = z.copy()
    z_me.index = _month_end(pd.DatetimeIndex(z_me.index))
    pair_index = _ensure_utc(pd.DatetimeIndex(pair_index))
    z_daily = z_me.reindex(pair_index, method="ffill")
    if cfg.weight_lag_days > 0:
        z_daily = z_daily.shift(int(cfg.weight_lag_days))
    z_daily.name = "cip_stress_z"
    return z_daily


def _carry_daily_weights(
    rates: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: CipConditionedCarryFxConfig,
) -> pd.DataFrame:
    ccfg = CarryRankConfig(
        n_long=cfg.carry_n_long,
        n_short=cfg.carry_n_short,
        signal_lag=cfg.carry_signal_lag,
    )
    ccy_w = carry_weights_from_rates(rates, cfg=ccfg)
    carry_pw = currency_weights_to_pair_weights(ccy_w)
    carry_daily = expand_weights_to_daily(
        carry_pw, pair_ret.index, signal_lag=cfg.carry_signal_lag
    )
    cols = list(pair_ret.columns)
    for c in cols:
        if c not in carry_daily.columns:
            carry_daily[c] = 0.0
    return carry_daily.reindex(columns=cols).fillna(0.0)


def _weights_to_returns(
    w: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: CipConditionedCarryFxConfig,
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


def usd_tilt_from_cip_stress_z(
    z_daily: pd.Series,
    pair_columns: list[str],
    *,
    cfg: CipConditionedCarryFxConfig,
) -> pd.DataFrame:
    """Long USD when lagged CIP-stress z ≥ z_high (haven / funding-stress tilt)."""
    on = (z_daily >= float(cfg.z_high)).astype(float) * float(cfg.usd_tilt)
    on = on.where(z_daily.notna(), 0.0)
    cols = [c for c in pair_columns if c.upper() in {p.upper() for p in USD_LONG_PAIRS}]
    if not cols:
        cols = list(pair_columns)
    n = max(len(cols), 1)
    w = pd.DataFrame(0.0, index=z_daily.index, columns=list(pair_columns))
    for sym in cols:
        sign = USD_LONG_PAIRS.get(sym.upper(), 0)
        w[sym] = (on / n) * sign
    return w.fillna(0.0)


def cip_conditioned_carry_factor_returns(
    pair_ret: pd.DataFrame,
    cip_panel: pd.DataFrame,
    *,
    rates: pd.DataFrame | None = None,
    cfg: CipConditionedCarryFxConfig | None = None,
    cip_stress_override: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Build CIP-conditioned carry / USD-tilt FX factor daily returns."""
    cfg = cfg or CipConditionedCarryFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    cols = list(pret.columns)
    out: dict[str, pd.Series] = {}

    stress = cip_stress_override
    if stress is None:
        stress = cip_stress_from_panel(cip_panel)
    z_daily = align_cip_stress_z_daily(stress, pret.index, cfg=cfg)

    # USD haven tilt from CIP stress (natural companion; distinct from §67 XS)
    out["cip_stress_haven_usd"] = _weights_to_returns(
        usd_tilt_from_cip_stress_z(z_daily, cols, cfg=cfg),
        pret,
        cfg=cfg,
        name="cip_stress_haven_usd",
    )

    if rates is None or rates.empty:
        return out

    carry_daily = _carry_daily_weights(rates, pret, cfg=cfg)

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
    scale = risk_scale_from_z(z_daily, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
    cooled = carry_daily.mul(scale, axis=0)
    r_cool = portfolio_returns_from_weights(cooled, pret)
    r_cool = apply_costs(r_cool, cooled, bps_side=cfg.cost_bps_side)
    r_cool.name = "carry_cip_cool"
    out[r_cool.name] = r_cool

    # Primary: trade carry only when CIP stress z ≤ 0 (low-stress regime)
    gate_lo = (z_daily <= 0.0).astype(float)
    gate_lo = gate_lo.where(z_daily.notna(), 0.0).reindex(pret.index).fillna(0.0)
    gated_lo = carry_daily.mul(gate_lo, axis=0)
    r_lo = portfolio_returns_from_weights(gated_lo, pret)
    r_lo = apply_costs(r_lo, gated_lo, bps_side=cfg.cost_bps_side)
    r_lo.name = "carry_low_cip_stress"
    out[r_lo.name] = r_lo

    # Honesty inverse: trade carry only when stress z ≥ z_high
    gate_hi = (z_daily >= float(cfg.z_high)).astype(float)
    gate_hi = gate_hi.where(z_daily.notna(), 0.0).reindex(pret.index).fillna(0.0)
    gated_hi = carry_daily.mul(gate_hi, axis=0)
    r_hi = portfolio_returns_from_weights(gated_hi, pret)
    r_hi = apply_costs(r_hi, gated_hi, bps_side=cfg.cost_bps_side)
    r_hi.name = "carry_high_cip_stress"
    out[r_hi.name] = r_hi

    blend_keys = [
        k
        for k in ("carry_low_cip_stress", "carry_cip_cool", "cip_stress_haven_usd")
        if k in out
    ]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "cip_carry_ew"
        out["cip_carry_ew"] = blend

    return out


__all__ = [
    "CipConditionedCarryFxConfig",
    "cip_conditioned_carry_factor_returns",
    "cip_stress_from_panel",
    "align_cip_stress_z_daily",
    "usd_tilt_from_cip_stress_z",
    "trailing_z",
]
