"""Du–Schreger CIP-conditioned Menkhoff FX momentum factors (§82).

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Menkhoff, Sarno, Schmeling & Schrimpf (2012), *JFE*: Currency Momentum Strategies
  — cross-sectional FX momentum (formation ~3m, skip ~1m, long winners / short losers).
- Menkhoff, Sarno, Schmeling & Schrimpf (2012), *JF*: high global FX / equity
  vol → risk-off / crash states for momentum (funding / intermediary stress channel).
- Du, Tepper & Verdelhan (2018), *JF*: observed CIP / cross-currency basis
  deviations reflect intermediary / dollar-funding stress.
- Du, Im & Schreger (2018) / Du, Keerati & Schreger (2025): government-bond CIP
  ``cip_govt`` (bps) and U.S. Treasury Premium (−cross-sectional mean CIP)
  measure synthetic-dollar sovereign cost / Treasury convenience.

Claim (a priori)
----------------
Menkhoff cross-sectional FX momentum earns more when aggregate CIP /
dollar-funding stress is **low**. Scale down or gate momentum when stress is
elevated: ``cip_stress ≡ UST premium = −mean(G10 cip_govt)``. Parallel to
CIP×carry §68 / CIP soft §71 and to VIX/GPR×mom §74 / EPU/TPU×mom §79 —
**not** a cooler overlay on locked fx4plus.

Legs
----
- ``mom_low_cip`` (**PRIMARY**): scholarly FX momentum only when lagged
  CIP-stress z ≤ 0 (flat in elevated stress).
- ``mom_cip_cool``: mom × risk_scale ∈ [cool, 1] from CIP-stress z
  (cool when z ≥ z_high).
- ``mom_raw``: always-on Menkhoff momentum (honesty baseline).
- ``mom_high_cip``: inverse gate — mom only when stress z ≥ z_high.
- ``cip_stress_haven_usd``: long USD when lagged CIP-stress z ≥ z_high
  (same construction as §68/§71).
- ``mom_cip_stack``: EW of mom_cip_cool ⊕ cip_stress_haven_usd
  (continuous cool stacked with haven tilt).
- ``mom_cip_ew``: EW of mom_low_cip ⊕ mom_cip_cool ⊕ cip_stress_haven_usd.
- ``mom_cip_regime``: mom_low_cip + cip_stress_haven_usd (exclusive regimes:
  mom when z≤0, haven when z≥z_high).

PIT
---
CIP panel: loader ``pub_lag_days=1`` → month-end; this module adds
``signal_lag_months`` (default 1) on stress z + **1 trading-day** weight lag
(reuse ``align_cip_stress_z_daily`` from §68 — same z_window=60m as §68/§71).
Momentum: ``FxMomentumConfig`` defaults (formation=63d, skip=21d,
n_long=n_short=2, signal_lag=1) via ``expand_weights_to_daily``.
``portfolio_returns_from_weights`` applies an extra weight lag.

Gate / cool thresholds: fixed z (z_high=1.0 / gate ≤0) — mirror §68/§71.
**Do not** grid on holdout.

Distinct from: combo §8, raw fx_momentum, CIP×carry §68, CIP soft §71,
VIX/GPR mom §74, EPU/TPU mom §79, soft/carry/value/REER stress waves §72–§81,
capital-sleeve §53/§70.
Explicit: do **not** overlay coolers on locked ``fx4plus_gbpcad_d1_voltarget_0025``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mt5_swing.strategies.carry_rank import (
    currency_weights_to_pair_weights,
    expand_weights_to_daily,
    portfolio_returns_from_weights,
)
from mt5_swing.strategies.cip_conditioned_carry_fx import (
    CipConditionedCarryFxConfig,
    align_cip_stress_z_daily,
    cip_stress_from_panel,
    usd_tilt_from_cip_stress_z,
)
from mt5_swing.strategies.fx_momentum import FxMomentumConfig, momentum_weights_from_returns
from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class CipConditionedMomFxConfig:
    """Fixed research priors — do not grid / holdout-tune.

    CIP stress fields mirror §68/§71 (monthly z_window=60). Mom fields mirror
    §74/§79 Menkhoff construction.
    """

    # CIP monthly z (§68/§71 mirror)
    signal_lag_months: int = 1
    weight_lag_days: int = 1  # after month-end ffill of stress z
    z_window: int = 60  # months (~5y)
    min_periods: int = 24
    z_high: float = 1.0
    z_low: float = 0.0
    cool: float = 0.35
    usd_tilt: float = 0.5
    cost_bps_side: float = 1.5
    # Menkhoff FX momentum defaults (fx_momentum.FxMomentumConfig — not HO-tuned)
    formation_days: int = 63
    skip_days: int = 21
    n_long: int = 2
    n_short: int = 2
    mom_signal_lag: int = 1
    mom_min_periods: int = 40


PRIMARY = "mom_low_cip"


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _cip_cfg_from_mom(cfg: CipConditionedMomFxConfig) -> CipConditionedCarryFxConfig:
    """Reuse §68 align_cip_stress_z_daily / usd_tilt config shape."""
    return CipConditionedCarryFxConfig(
        signal_lag_months=cfg.signal_lag_months,
        weight_lag_days=cfg.weight_lag_days,
        z_window=cfg.z_window,
        min_periods=cfg.min_periods,
        z_high=cfg.z_high,
        z_low=cfg.z_low,
        cool=cfg.cool,
        usd_tilt=cfg.usd_tilt,
        cost_bps_side=cfg.cost_bps_side,
    )


def _mom_daily_weights(
    pair_ret: pd.DataFrame,
    *,
    cfg: CipConditionedMomFxConfig,
) -> pd.DataFrame:
    mcfg = FxMomentumConfig(
        formation_days=cfg.formation_days,
        skip_days=cfg.skip_days,
        n_long=cfg.n_long,
        n_short=cfg.n_short,
        signal_lag=cfg.mom_signal_lag,
        min_periods=cfg.mom_min_periods,
    )
    ccy_w = momentum_weights_from_returns(pair_ret, cfg=mcfg)
    mom_pw = currency_weights_to_pair_weights(ccy_w)
    mom_daily = expand_weights_to_daily(
        mom_pw, pair_ret.index, signal_lag=cfg.mom_signal_lag
    )
    cols = list(pair_ret.columns)
    for c in cols:
        if c not in mom_daily.columns:
            mom_daily[c] = 0.0
    return mom_daily.reindex(columns=cols).fillna(0.0)


def _weights_to_returns(
    w: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: CipConditionedMomFxConfig,
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


def cip_conditioned_mom_factor_returns(
    pair_ret: pd.DataFrame,
    cip_panel: pd.DataFrame,
    *,
    cfg: CipConditionedMomFxConfig | None = None,
    cip_stress_override: pd.Series | None = None,
    include_haven: bool = True,
) -> dict[str, pd.Series]:
    """Build CIP-conditioned Menkhoff FX momentum factor daily returns.

    Distinct from §68 (CIP×carry), §71 (CIP×soft), §74 (VIX/GPR×mom),
    §79 (EPU/TPU×mom). Explicit: do **not** overlay coolers on locked fx4plus.
    """
    cfg = cfg or CipConditionedMomFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    out: dict[str, pd.Series] = {}

    if pret.empty or pret.shape[1] < 3:
        return out

    cip_cfg = _cip_cfg_from_mom(cfg)
    stress = cip_stress_override
    if stress is None:
        stress = cip_stress_from_panel(cip_panel)
    z_daily = align_cip_stress_z_daily(stress, pret.index, cfg=cip_cfg)

    mom_daily = _mom_daily_weights(pret, cfg=cfg)

    raw = portfolio_returns_from_weights(mom_daily, pret)
    raw = apply_costs(raw, mom_daily, bps_side=cfg.cost_bps_side)
    raw.name = "mom_raw"
    out["mom_raw"] = raw

    gcfg = GprRegimeConfig(
        z_high=cfg.z_high,
        z_low=cfg.z_low,
        cool=cfg.cool,
        signal_lag=0,  # z already lagged
        usd_tilt=0.0,
    )
    scale = risk_scale_from_z(z_daily, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
    out["mom_cip_cool"] = _weights_to_returns(
        mom_daily.mul(scale, axis=0), pret, cfg=cfg, name="mom_cip_cool"
    )

    # Primary: trade mom only when CIP stress z ≤ 0
    gate_lo = (z_daily <= 0.0).astype(float)
    gate_lo = gate_lo.where(z_daily.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["mom_low_cip"] = _weights_to_returns(
        mom_daily.mul(gate_lo, axis=0), pret, cfg=cfg, name="mom_low_cip"
    )

    # Honesty inverse: mom only when stress z ≥ z_high
    gate_hi = (z_daily >= float(cfg.z_high)).astype(float)
    gate_hi = gate_hi.where(z_daily.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["mom_high_cip"] = _weights_to_returns(
        mom_daily.mul(gate_hi, axis=0), pret, cfg=cfg, name="mom_high_cip"
    )

    if include_haven:
        cols = list(pret.columns)
        haven_w = usd_tilt_from_cip_stress_z(z_daily, cols, cfg=cip_cfg)
        out["cip_stress_haven_usd"] = _weights_to_returns(
            haven_w, pret, cfg=cfg, name="cip_stress_haven_usd"
        )

    # Stack: continuous cool ⊕ haven tilt
    stack_keys = [k for k in ("mom_cip_cool", "cip_stress_haven_usd") if k in out]
    if len(stack_keys) >= 2:
        stack = pd.concat([out[k] for k in stack_keys], axis=1).mean(axis=1)
        stack.name = "mom_cip_stack"
        out["mom_cip_stack"] = stack

    # EW of primary + cool + haven (§68/§71 mirror)
    blend_keys = [
        k
        for k in ("mom_low_cip", "mom_cip_cool", "cip_stress_haven_usd")
        if k in out
    ]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "mom_cip_ew"
        out["mom_cip_ew"] = blend

    # Regime switch: mom when low-stress + haven when high-stress (exclusive)
    if "mom_low_cip" in out and "cip_stress_haven_usd" in out:
        regime = out["mom_low_cip"].fillna(0.0) + out["cip_stress_haven_usd"].fillna(0.0)
        # Preserve NaN where both source legs NaN
        both_nan = out["mom_low_cip"].isna() & out["cip_stress_haven_usd"].isna()
        regime = regime.where(~both_nan)
        regime.name = "mom_cip_regime"
        out["mom_cip_regime"] = regime

    return out


__all__ = [
    "PRIMARY",
    "CipConditionedMomFxConfig",
    "cip_conditioned_mom_factor_returns",
    "cip_stress_from_panel",
    "align_cip_stress_z_daily",
]
