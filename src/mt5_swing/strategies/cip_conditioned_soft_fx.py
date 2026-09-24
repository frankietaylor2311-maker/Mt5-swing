"""CIP-stress-conditioned Dahlquist soft-signal EW FX factors (§71).

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Dahlquist & Hasseltoft (2020), *JFE*: economic-momentum soft signals;
  equal-weight of macro-momentum XS factors (repo §66 ``soft_ew_macro5`` /
  SOFT_LEGS).
- Du, Tepper & Verdelhan (2018), *JF*: CIP / cross-currency basis deviations
  reflect intermediary / dollar-funding stress.
- Du, Keerati & Schreger (2025): government-bond CIP ``cip_govt`` and U.S.
  Treasury Premium (−cross-sectional mean CIP) measure synthetic-dollar
  sovereign cost / Treasury convenience (``cip_dataset_v4``).

Claim (a priori)
----------------
Soft-stack macro momentum (§66) earns more when aggregate CIP / dollar-funding
stress is **low**. Trade soft EW only when stress is subdued; cool or sit out
when intermediary dollar funding stress is elevated.
``cip_stress ≡ UST premium = −mean(G10 cip_govt)`` at tenor **5y**
(DEFAULT_TENOR — same §67/§68 a priori).

Legs
----
- ``soft_low_cip_stress`` (**PRIMARY**): §66 ``soft_ew_macro5`` × binary gate
  (on when lagged CIP-stress z ≤ 0).
- ``soft_cip_cool``: soft_ew_macro5 × continuous cool ∈ [cool, 1] from z.
- ``soft_raw``: ungated soft_ew_macro5 honesty baseline (§66).
- ``soft_high_cip_stress``: honesty inverse — soft only when z ≥ z_high.
- ``cip_stress_haven_usd``: long USD when lagged CIP-stress z ≥ z_high
  (same construction as §68).
- ``soft_cip_ew``: EW of soft_low_cip_stress ⊕ soft_cip_cool ⊕ haven.

PIT
---
CIP panel: loader ``pub_lag_days=1`` → month-end; this module adds
``signal_lag_months`` (default 1) on stress z + **1 trading-day** weight lag
(reuse ``align_cip_stress_z_daily`` from §68). Soft legs keep source-wave PIT
lags from §66 (costs 1.5 bps/side already inside source factor returns).

Gate threshold: fixed z (z_high=1.0 / loose ≤0) — mirror funding_liq / ACM /
§68; **do not** grid on holdout.

Distinct from: soft_ew §66 (ungated), soft-stack CIP enrichment §69
(EW membership includes CIP soft legs), CIP-conditioned carry §68 (gates
Lustig–Verdelhan *carry*, not soft macro EW), CIP XS §67, capital-sleeve
§53 / §70, combo §8.
Explicit: do **not** overlay coolers on locked ``fx4plus_gbpcad_d1_voltarget_0025``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mt5_swing.strategies.cip_conditioned_carry_fx import (
    CipConditionedCarryFxConfig,
    align_cip_stress_z_daily,
    cip_stress_from_panel,
    usd_tilt_from_cip_stress_z,
)
from mt5_swing.strategies.carry_rank import portfolio_returns_from_weights
from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z
from mt5_swing.strategies.yield_curve_fx import apply_costs


# Alias: §71 reuses §68 CIP-stress config fields (same fixed priors).
CipConditionedSoftFxConfig = CipConditionedCarryFxConfig

PRIMARY = "soft_low_cip_stress"


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _scale_series(
    soft: pd.Series,
    scale: pd.Series,
    *,
    name: str,
) -> pd.Series:
    """Element-wise soft × scale (aligned); flat when scale=0 / soft NaN preserved."""
    soft = soft.astype(float).copy()
    soft.index = _ensure_utc(pd.DatetimeIndex(soft.index))
    sc = scale.reindex(soft.index).astype(float)
    out = soft * sc
    # When soft is NaN keep NaN; when scale NaN treat as flat (0)
    out = out.where(soft.notna())
    out = out.where(sc.notna(), 0.0)
    out.name = name
    return out


def _haven_returns(
    z_daily: pd.Series,
    pair_ret: pd.DataFrame,
    *,
    cfg: CipConditionedSoftFxConfig,
) -> pd.Series:
    cols = list(pair_ret.columns)
    w = usd_tilt_from_cip_stress_z(z_daily, cols, cfg=cfg)
    for c in cols:
        if c not in w.columns:
            w[c] = 0.0
    w = w.reindex(columns=cols).fillna(0.0)
    r = portfolio_returns_from_weights(w, pair_ret)
    r = apply_costs(r, w, bps_side=cfg.cost_bps_side)
    r.name = "cip_stress_haven_usd"
    return r


def cip_conditioned_soft_factor_returns(
    soft_ew: pd.Series,
    cip_panel: pd.DataFrame,
    *,
    pair_ret: pd.DataFrame | None = None,
    cfg: CipConditionedSoftFxConfig | None = None,
    cip_stress_override: pd.Series | None = None,
    include_haven: bool = True,
) -> dict[str, pd.Series]:
    """Build CIP-stress-conditioned soft-signal EW FX factor daily returns.

    ``soft_ew`` is the §66 ``soft_ew_macro5`` / ``soft_ew5`` daily return series
    (costs already applied inside source legs). Gating/cooling multiplies that
    series by the CIP-stress gate/cool — distinct from §68 (carry weights) and
    §69 (CIP soft legs inside EW membership).
    """
    cfg = cfg or CipConditionedSoftFxConfig()
    soft = soft_ew.astype(float).copy()
    soft.index = _ensure_utc(pd.DatetimeIndex(soft.index))
    soft.name = soft.name or "soft_ew_macro5"

    stress = cip_stress_override
    if stress is None:
        stress = cip_stress_from_panel(cip_panel)
    z_daily = align_cip_stress_z_daily(stress, soft.index, cfg=cfg)

    out: dict[str, pd.Series] = {}

    raw = soft.copy()
    raw.name = "soft_raw"
    out["soft_raw"] = raw

    gcfg = GprRegimeConfig(
        z_high=cfg.z_high,
        z_low=cfg.z_low,
        cool=cfg.cool,
        signal_lag=0,  # z already lagged
        usd_tilt=0.0,
    )
    scale = risk_scale_from_z(z_daily, cfg=gcfg).reindex(soft.index).ffill().fillna(1.0)
    out["soft_cip_cool"] = _scale_series(soft, scale, name="soft_cip_cool")

    # Primary: trade soft only when CIP stress z ≤ 0
    gate_lo = (z_daily <= 0.0).astype(float)
    gate_lo = gate_lo.where(z_daily.notna(), 0.0).reindex(soft.index).fillna(0.0)
    out["soft_low_cip_stress"] = _scale_series(soft, gate_lo, name="soft_low_cip_stress")

    # Honesty inverse: soft only when stress z ≥ z_high
    gate_hi = (z_daily >= float(cfg.z_high)).astype(float)
    gate_hi = gate_hi.where(z_daily.notna(), 0.0).reindex(soft.index).fillna(0.0)
    out["soft_high_cip_stress"] = _scale_series(
        soft, gate_hi, name="soft_high_cip_stress"
    )

    if include_haven and pair_ret is not None and not pair_ret.empty:
        pret = pair_ret.copy()
        pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
        out["cip_stress_haven_usd"] = _haven_returns(z_daily, pret, cfg=cfg)

    blend_keys = [
        k
        for k in ("soft_low_cip_stress", "soft_cip_cool", "cip_stress_haven_usd")
        if k in out
    ]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "soft_cip_ew"
        out["soft_cip_ew"] = blend

    return out


__all__ = [
    "PRIMARY",
    "CipConditionedSoftFxConfig",
    "cip_conditioned_soft_factor_returns",
    "cip_stress_from_panel",
    "align_cip_stress_z_daily",
]
