"""EPU/TPU-conditioned Lustig–Verdelhan carry FX factors (§78).

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Lustig, Roussanov & Verdelhan (2011): FX carry sorted by interest-rate
  differentials (high-minus-low).
- Baker, Bloom & Davis (2016), *QJE*: news-based Economic Policy Uncertainty
  (EPU). We use **US EPU** (FRED USEPUINDXM) as primary Stress A; GEPU as
  documented aggregate fall-through if US missing.
- Baker–Bloom–Davis categorical *Trade policy* EPU = US Trade Policy
  Uncertainty (TPU) on policyuncertainty.com — Stress B.

Claim (a priori)
----------------
IR3M / policy-rate Lustig–Verdelhan carry earns more when economic /
trade-policy uncertainty is **low**. Scale down or gate carry when EPU or
TPU stress is elevated. Parallel to §73 (VIX/GPR×carry) but with free
EPU/TPU **monthly** stress series (same as §76/§77) — **not** a cooler
overlay on locked fx4plus.

Legs
----
- ``carry_low_epu`` (**PRIMARY**): scholarly cash-rate carry only when
  lagged EPU z ≤ 0 (flat when elevated).
- ``carry_epu_cool``: carry × risk_scale ∈ [cool, 1] from EPU z
  (cool when z ≥ z_high).
- ``carry_low_tpu``: carry only when lagged TPU z ≤ 0.
- ``carry_tpu_cool``: carry × cool from TPU z.
- ``carry_raw``: always-on Lustig–Verdelhan carry (honesty).
- ``carry_high_epu``: honesty inverse — carry only when EPU z ≥ z_high.
- ``carry_epu_tpu_stack``: sequential EPU then TPU cool (product of scales).
- ``carry_epu_tpu_ew``: EW of low_epu ⊕ epu_cool ⊕ low_tpu ⊕ tpu_cool.

PIT
---
EPU / TPU loaders apply ``pub_lag_months=1``; this module adds
``signal_lag_months`` (default 1) on **monthly** trailing z (z_window=60m,
CIP §71 / EPU soft §76 / REER §77 mirror) + ``weight_lag_days=1``.
Rates: existing FRED carry loaders' PIT lags + ``carry_signal_lag`` trading
days via ``CarryRankConfig`` / ``expand_weights_to_daily``.
``portfolio_returns_from_weights`` applies an extra weight lag.

Gate / cool thresholds: fixed z (z_high=1.0 / gate ≤0) — mirror §71–§77.
Literature often uses other cutoffs; documented symmetry, **not** HO-tuned.

Distinct from: fwd_carry §19, funding_liq §20, IG OAS §36, ACM TP §44,
CIP XS §67, CIP×carry §68, soft EW §66, soft CIP §69, capital-sleeve §53/§70,
CIP-conditioned soft §71, VIX/GPR soft §72, VIX/GPR×carry §73, VIX/GPR×mom
§74, VIX/GPR×PPP §75, EPU/TPU soft §76, EPU/TPU×REER §77, combo §8,
standalone EPU/TPU §18.
Explicit: do **not** overlay coolers on locked ``fx4plus_gbpcad_d1_voltarget_0025``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mt5_swing.strategies.carry_rank import (
    CarryRankConfig,
    carry_weights_from_rates,
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
from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class EpuTpuConditionedCarryFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    # EPU/TPU monthly z (CIP §71 / EPU soft §76 / REER §77 mirror — not 252d daily)
    signal_lag_months: int = 1
    weight_lag_days: int = 1  # after month-end ffill of stress z
    z_window: int = 60  # months
    min_periods: int = 24
    z_high: float = 1.0  # cool thresh — §71–§77 symmetry
    z_low: float = 0.0
    cool: float = 0.35
    cost_bps_side: float = 1.5
    carry_n_long: int = 2
    carry_n_short: int = 2
    carry_signal_lag: int = 1
    # Documented: lit often uses other cutoffs; this wave keeps 1.0 for symmetry
    lit_z_high_note: float = 1.5


PRIMARY = "carry_low_epu"

_TPU_AUTO = object()


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _carry_daily_weights(
    rates: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: EpuTpuConditionedCarryFxConfig,
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
    cfg: EpuTpuConditionedCarryFxConfig,
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


def _soft_cfg_from_carry(
    cfg: EpuTpuConditionedCarryFxConfig,
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


def epu_tpu_conditioned_carry_factor_returns(
    pair_ret: pd.DataFrame,
    *,
    rates: pd.DataFrame,
    epu: pd.Series | None = None,
    tpu: pd.Series | None | object = _TPU_AUTO,
    cfg: EpuTpuConditionedCarryFxConfig | None = None,
    allow_tpu_missing: bool = True,
) -> dict[str, pd.Series]:
    """Build EPU/TPU-conditioned Lustig–Verdelhan carry FX factor daily returns.

    Distinct from §73 (VIX/GPR×carry), §68 (CIP×carry), §76 (soft EPU),
    §77 (REER EPU). Explicit: do **not** overlay coolers on locked fx4plus.

    ``tpu`` semantics: default auto-load; pass ``None`` for explicit EPU-only
    fall-through (TPU companions omitted). If auto-load fails and
    ``allow_tpu_missing``, same EPU-only honesty path.
    """
    cfg = cfg or EpuTpuConditionedCarryFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    out: dict[str, pd.Series] = {}

    if rates is None or rates.empty:
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

    soft_cfg = _soft_cfg_from_carry(cfg)
    z_epu = align_monthly_stress_z_daily(epu, pret.index, cfg=soft_cfg)
    z_tpu = (
        align_monthly_stress_z_daily(tpu, pret.index, cfg=soft_cfg)
        if tpu is not None
        else None
    )

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
    epu_scale = (
        risk_scale_from_z(z_epu, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
    )
    out["carry_epu_cool"] = _weights_to_returns(
        carry_daily.mul(epu_scale, axis=0), pret, cfg=cfg, name="carry_epu_cool"
    )

    # Primary: trade carry only when EPU z ≤ 0
    gate_epu_lo = (z_epu <= 0.0).astype(float)
    gate_epu_lo = gate_epu_lo.where(z_epu.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["carry_low_epu"] = _weights_to_returns(
        carry_daily.mul(gate_epu_lo, axis=0), pret, cfg=cfg, name="carry_low_epu"
    )

    # Honesty inverse: carry only when EPU z ≥ z_high
    gate_epu_hi = (z_epu >= float(cfg.z_high)).astype(float)
    gate_epu_hi = gate_epu_hi.where(z_epu.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["carry_high_epu"] = _weights_to_returns(
        carry_daily.mul(gate_epu_hi, axis=0), pret, cfg=cfg, name="carry_high_epu"
    )

    if tpu_available and z_tpu is not None:
        tpu_scale = (
            risk_scale_from_z(z_tpu, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
        )
        out["carry_tpu_cool"] = _weights_to_returns(
            carry_daily.mul(tpu_scale, axis=0), pret, cfg=cfg, name="carry_tpu_cool"
        )

        gate_tpu_lo = (z_tpu <= 0.0).astype(float)
        gate_tpu_lo = (
            gate_tpu_lo.where(z_tpu.notna(), 0.0).reindex(pret.index).fillna(0.0)
        )
        out["carry_low_tpu"] = _weights_to_returns(
            carry_daily.mul(gate_tpu_lo, axis=0), pret, cfg=cfg, name="carry_low_tpu"
        )

        stack_scale = (epu_scale * tpu_scale).clip(
            lower=float(cfg.cool) ** 2, upper=1.0
        )
        out["carry_epu_tpu_stack"] = _weights_to_returns(
            carry_daily.mul(stack_scale, axis=0),
            pret,
            cfg=cfg,
            name="carry_epu_tpu_stack",
        )

        blend_keys = [
            k
            for k in (
                "carry_low_epu",
                "carry_epu_cool",
                "carry_low_tpu",
                "carry_tpu_cool",
            )
            if k in out
        ]
        if len(blend_keys) >= 2:
            blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
            blend.name = "carry_epu_tpu_ew"
            out["carry_epu_tpu_ew"] = blend
    else:
        out["carry_epu_tpu_stack"] = out["carry_epu_cool"].copy()
        out["carry_epu_tpu_stack"].name = "carry_epu_tpu_stack"
        blend = pd.concat(
            [out["carry_low_epu"], out["carry_epu_cool"]], axis=1
        ).mean(axis=1)
        blend.name = "carry_epu_tpu_ew"
        out["carry_epu_tpu_ew"] = blend

    return out


__all__ = [
    "PRIMARY",
    "EpuTpuConditionedCarryFxConfig",
    "epu_tpu_conditioned_carry_factor_returns",
    "load_epu_series",
    "load_tpu_series",
    "align_monthly_stress_z_daily",
]
