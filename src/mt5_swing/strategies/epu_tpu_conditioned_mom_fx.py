"""EPU/TPU-conditioned Menkhoff FX momentum factors (§79).

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Menkhoff, Sarno, Schmeling & Schrimpf (2012), *JFE*: Currency Momentum Strategies
  — cross-sectional FX momentum (formation ~3m, skip ~1m, long winners / short losers).
- Menkhoff, Sarno, Schmeling & Schrimpf (2012), *JF*: high global FX / equity
  vol → risk-off / crash states for momentum.
- Baker, Bloom & Davis (2016), *QJE*: news-based Economic Policy Uncertainty
  (EPU). We use **US EPU** (FRED USEPUINDXM) as primary Stress A; GEPU as
  documented aggregate fall-through if US missing.
- Baker–Bloom–Davis categorical *Trade policy* EPU = US Trade Policy
  Uncertainty (TPU) on policyuncertainty.com — Stress B.

Claim (a priori)
----------------
Menkhoff cross-sectional FX momentum earns more when economic /
trade-policy uncertainty is **low**. Scale down or gate momentum when EPU or
TPU stress is elevated. Parallel to §74 (VIX/GPR×mom) but with free
EPU/TPU **monthly** stress series (same as §76/§77/§78) — **not** a cooler
overlay on locked fx4plus.

Legs
----
- ``mom_low_epu`` (**PRIMARY**): scholarly FX momentum only when
  lagged EPU z ≤ 0 (flat when elevated).
- ``mom_epu_cool``: mom × risk_scale ∈ [cool, 1] from EPU z
  (cool when z ≥ z_high).
- ``mom_low_tpu``: mom only when lagged TPU z ≤ 0.
- ``mom_tpu_cool``: mom × cool from TPU z.
- ``mom_raw``: always-on Menkhoff momentum (honesty).
- ``mom_high_epu``: honesty inverse — mom only when EPU z ≥ z_high.
- ``mom_epu_tpu_stack``: sequential EPU then TPU cool (product of scales).
- ``mom_epu_tpu_ew``: EW of low_epu ⊕ epu_cool ⊕ low_tpu ⊕ tpu_cool.

PIT
---
EPU / TPU loaders apply ``pub_lag_months=1``; this module adds
``signal_lag_months`` (default 1) on **monthly** trailing z (z_window=60m,
CIP §71 / EPU soft §76 / REER §77 / carry §78 mirror) + ``weight_lag_days=1``.
Momentum: ``FxMomentumConfig`` defaults (formation=63d, skip=21d,
n_long=n_short=2, signal_lag=1) via ``expand_weights_to_daily``.
``portfolio_returns_from_weights`` applies an extra weight lag.

Gate / cool thresholds: fixed z (z_high=1.0 / gate ≤0) — mirror §71–§78.
Literature often uses other cutoffs; documented symmetry, **not** HO-tuned.

Distinct from: combo §8, raw fx_momentum, soft VIX/GPR §72, carry VIX/GPR §73,
mom VIX/GPR §74, value VIX/GPR §75, EPU soft §76, EPU×REER §77, EPU×carry §78,
capital-sleeve §53/§70, standalone EPU/TPU §18.
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
from mt5_swing.strategies.epu_tpu_conditioned_soft_fx import (
    EpuTpuConditionedSoftFxConfig,
    align_monthly_stress_z_daily,
    load_epu_series,
    load_tpu_series,
)
from mt5_swing.strategies.fx_momentum import FxMomentumConfig, momentum_weights_from_returns
from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class EpuTpuConditionedMomFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    # EPU/TPU monthly z (CIP §71 / EPU soft §76 / REER §77 / carry §78 mirror)
    signal_lag_months: int = 1
    weight_lag_days: int = 1  # after month-end ffill of stress z
    z_window: int = 60  # months
    min_periods: int = 24
    z_high: float = 1.0  # cool thresh — §71–§78 symmetry
    z_low: float = 0.0
    cool: float = 0.35
    cost_bps_side: float = 1.5
    # Menkhoff FX momentum defaults (fx_momentum.FxMomentumConfig — not HO-tuned)
    formation_days: int = 63
    skip_days: int = 21
    n_long: int = 2
    n_short: int = 2
    mom_signal_lag: int = 1
    mom_min_periods: int = 40
    # Documented: lit often uses other cutoffs; this wave keeps 1.0 for symmetry
    lit_z_high_note: float = 1.5


PRIMARY = "mom_low_epu"

_TPU_AUTO = object()


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _mom_daily_weights(
    pair_ret: pd.DataFrame,
    *,
    cfg: EpuTpuConditionedMomFxConfig,
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
    cfg: EpuTpuConditionedMomFxConfig,
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


def _soft_cfg_from_mom(
    cfg: EpuTpuConditionedMomFxConfig,
) -> EpuTpuConditionedSoftFxConfig:
    """Reuse §76 align_monthly_stress_z_daily config shape."""
    return EpuTpuConditionedSoftFxConfig(
        signal_lag_months=cfg.signal_lag_months,
        weight_lag_days=cfg.weight_lag_days,
        z_window=cfg.z_window,
        min_periods=cfg.min_periods,
        z_high=cfg.z_high,
        z_low=cfg.z_low,
        cool=cfg.cool,
        cost_bps_side=cfg.cost_bps_side,
    )


def epu_tpu_conditioned_mom_factor_returns(
    pair_ret: pd.DataFrame,
    *,
    epu: pd.Series | None = None,
    tpu: pd.Series | None | object = _TPU_AUTO,
    cfg: EpuTpuConditionedMomFxConfig | None = None,
    allow_tpu_missing: bool = True,
) -> dict[str, pd.Series]:
    """Build EPU/TPU-conditioned Menkhoff FX momentum factor daily returns.

    Distinct from §74 (VIX/GPR×mom), §78 (EPU×carry), §76 (soft EPU),
    §77 (REER EPU). Explicit: do **not** overlay coolers on locked fx4plus.

    ``tpu`` semantics: default auto-load; pass ``None`` for explicit EPU-only
    fall-through (TPU companions omitted). If auto-load fails and
    ``allow_tpu_missing``, same EPU-only honesty path.
    """
    cfg = cfg or EpuTpuConditionedMomFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    out: dict[str, pd.Series] = {}

    if pret.empty or pret.shape[1] < 3:
        return out

    if epu is None:
        epu = load_epu_series(series="US", pub_lag_months=1)

    tpu_available = True
    if tpu is _TPU_AUTO:
        try:
            tpu = load_tpu_series(pub_lag_months=1)
        except Exception:
            if not allow_tpu_missing:
                raise
            tpu = None
            tpu_available = False
    elif tpu is None:
        tpu_available = False

    soft_cfg = _soft_cfg_from_mom(cfg)
    z_epu = align_monthly_stress_z_daily(epu, pret.index, cfg=soft_cfg)
    z_tpu = (
        align_monthly_stress_z_daily(tpu, pret.index, cfg=soft_cfg)
        if tpu is not None
        else None
    )

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
    epu_scale = (
        risk_scale_from_z(z_epu, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
    )
    out["mom_epu_cool"] = _weights_to_returns(
        mom_daily.mul(epu_scale, axis=0), pret, cfg=cfg, name="mom_epu_cool"
    )

    # Primary: trade mom only when EPU z ≤ 0
    gate_epu_lo = (z_epu <= 0.0).astype(float)
    gate_epu_lo = gate_epu_lo.where(z_epu.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["mom_low_epu"] = _weights_to_returns(
        mom_daily.mul(gate_epu_lo, axis=0), pret, cfg=cfg, name="mom_low_epu"
    )

    # Honesty inverse: mom only when EPU z ≥ z_high
    gate_epu_hi = (z_epu >= float(cfg.z_high)).astype(float)
    gate_epu_hi = gate_epu_hi.where(z_epu.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["mom_high_epu"] = _weights_to_returns(
        mom_daily.mul(gate_epu_hi, axis=0), pret, cfg=cfg, name="mom_high_epu"
    )

    if tpu_available and z_tpu is not None:
        tpu_scale = (
            risk_scale_from_z(z_tpu, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
        )
        out["mom_tpu_cool"] = _weights_to_returns(
            mom_daily.mul(tpu_scale, axis=0), pret, cfg=cfg, name="mom_tpu_cool"
        )

        gate_tpu_lo = (z_tpu <= 0.0).astype(float)
        gate_tpu_lo = (
            gate_tpu_lo.where(z_tpu.notna(), 0.0).reindex(pret.index).fillna(0.0)
        )
        out["mom_low_tpu"] = _weights_to_returns(
            mom_daily.mul(gate_tpu_lo, axis=0), pret, cfg=cfg, name="mom_low_tpu"
        )

        stack_scale = (epu_scale * tpu_scale).clip(
            lower=float(cfg.cool) ** 2, upper=1.0
        )
        out["mom_epu_tpu_stack"] = _weights_to_returns(
            mom_daily.mul(stack_scale, axis=0),
            pret,
            cfg=cfg,
            name="mom_epu_tpu_stack",
        )

        blend_keys = [
            k
            for k in (
                "mom_low_epu",
                "mom_epu_cool",
                "mom_low_tpu",
                "mom_tpu_cool",
            )
            if k in out
        ]
        if len(blend_keys) >= 2:
            blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
            blend.name = "mom_epu_tpu_ew"
            out["mom_epu_tpu_ew"] = blend
    else:
        out["mom_epu_tpu_stack"] = out["mom_epu_cool"].copy()
        out["mom_epu_tpu_stack"].name = "mom_epu_tpu_stack"
        blend = pd.concat(
            [out["mom_low_epu"], out["mom_epu_cool"]], axis=1
        ).mean(axis=1)
        blend.name = "mom_epu_tpu_ew"
        out["mom_epu_tpu_ew"] = blend

    return out


__all__ = [
    "PRIMARY",
    "EpuTpuConditionedMomFxConfig",
    "epu_tpu_conditioned_mom_factor_returns",
    "load_epu_series",
    "load_tpu_series",
    "align_monthly_stress_z_daily",
]
