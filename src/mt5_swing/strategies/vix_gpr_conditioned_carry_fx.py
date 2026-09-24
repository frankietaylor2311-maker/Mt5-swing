"""VIX/GPR-conditioned Lustig–Verdelhan carry FX factors (§73).

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Lustig, Roussanov & Verdelhan (2011): FX carry sorted by interest-rate
  differentials (high-minus-low).
- Menkhoff, Sarno, Schmeling & Schrimpf (2012), *JF*: high global FX / equity
  vol → carry crashes / risk-off. We proxy global FX vol with **VIX**
  (``data/macro/vix_yahoo.csv`` / FRED VIXCLS).
- Caldara & Iacoviello (2022), *AER* / Liu & Zhang (2024), *JBF*: elevated
  aggregate GPR → risk-off. Free series ``gpr_daily.csv``.

Claim (a priori)
----------------
IR3M / policy-rate Lustig–Verdelhan carry earns more when global FX-vol /
geopolitical stress is **low**. Scale down or gate carry when VIX or GPR
stress is elevated. Parallel to §68 (CIP×carry) but with free VIX/GPR stress
series already in the repo — **not** a cooler overlay on locked fx4plus.

Legs
----
- ``carry_low_vix`` (**PRIMARY**, Menkhoff): scholarly cash-rate carry only
  when lagged VIX z ≤ 0 (flat when elevated).
- ``carry_vix_cool``: carry × risk_scale ∈ [cool, 1] from VIX z
  (cool when z ≥ z_high).
- ``carry_low_gpr``: carry only when lagged GPR z ≤ 0.
- ``carry_gpr_cool``: carry × cool from GPR z.
- ``carry_raw``: always-on Lustig–Verdelhan carry (honesty).
- ``carry_high_vix``: honesty inverse — carry only when VIX z ≥ z_high.
- ``carry_vix_gpr_stack``: sequential VIX then GPR cool (product of scales).
- ``carry_vix_gpr_ew``: EW of low_vix ⊕ vix_cool ⊕ low_gpr ⊕ gpr_cool.

PIT
---
VIX / GPR loaders apply ``bar_lag=1``; this module adds ``signal_lag`` (default
1 trading day) on trailing z — same dual lag as §72 / funding_liq.
Rates: existing FRED carry loaders' PIT lags + ``carry_signal_lag`` trading
days via ``CarryRankConfig`` / ``expand_weights_to_daily``.
``portfolio_returns_from_weights`` applies an extra weight lag.

Gate / cool thresholds: fixed z (z_high=1.0 / gate ≤0) — mirror §71 / §72 /
funding_liq / ACM. Literature often uses **1.5** for pure GPR risk-off; for
**this** wave we keep z≥1 for §71/§72 symmetry — documented, not HO-tuned.

Distinct from: fwd_carry §19, funding_liq §20, IG OAS §36, ACM TP §44,
CIP XS §67, CIP×carry §68, soft EW §66, soft CIP §69, capital-sleeve §53/§70,
CIP-conditioned soft §71, VIX/GPR-conditioned soft §72, combo §8,
quest_fred_carry_gpr_study, gpr_regime standalone, AI-GPR §25, country-GPR §45.
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
from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z
from mt5_swing.strategies.vix_gpr_conditioned_soft_fx import (
    align_stress_z_daily,
    load_gpr_series,
    load_vix_series,
)
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class VixGprConditionedCarryFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # trading days after loader bar_lag=1
    z_window: int = 252  # daily, like §72 — NOT monthly CIP 60
    min_periods: int = 60
    z_high: float = 1.0  # cool thresh — §71/§72 symmetry (lit GPR often 1.5)
    z_low: float = 0.0
    cool: float = 0.35
    cost_bps_side: float = 1.5
    carry_n_long: int = 2
    carry_n_short: int = 2
    carry_signal_lag: int = 1
    # Documented: literature GPR cool often z≥1.5; this wave keeps 1.0 for symmetry
    gpr_lit_z_high_note: float = 1.5


PRIMARY = "carry_low_vix"


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _carry_daily_weights(
    rates: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: VixGprConditionedCarryFxConfig,
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
    cfg: VixGprConditionedCarryFxConfig,
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


def _soft_cfg_from_carry(cfg: VixGprConditionedCarryFxConfig):
    """Reuse §72 align_stress_z_daily config shape."""
    from mt5_swing.strategies.vix_gpr_conditioned_soft_fx import (
        VixGprConditionedSoftFxConfig,
    )

    return VixGprConditionedSoftFxConfig(
        signal_lag=cfg.signal_lag,
        z_window=cfg.z_window,
        min_periods=cfg.min_periods,
        z_high=cfg.z_high,
        z_low=cfg.z_low,
        cool=cfg.cool,
        cost_bps_side=cfg.cost_bps_side,
    )


def vix_gpr_conditioned_carry_factor_returns(
    pair_ret: pd.DataFrame,
    *,
    rates: pd.DataFrame,
    vix: pd.Series | None = None,
    gpr: pd.Series | None = None,
    cfg: VixGprConditionedCarryFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Build VIX/GPR-conditioned Lustig–Verdelhan carry FX factor daily returns.

    Distinct from §68 (CIP stress) and §72 (soft EW × VIX/GPR). Explicit: do
    **not** overlay coolers on locked fx4plus.
    """
    cfg = cfg or VixGprConditionedCarryFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    out: dict[str, pd.Series] = {}

    if rates is None or rates.empty:
        return out

    if vix is None:
        vix = load_vix_series(bar_lag=1)
    if gpr is None:
        gpr = load_gpr_series(bar_lag=1)

    soft_cfg = _soft_cfg_from_carry(cfg)
    z_vix = align_stress_z_daily(vix, pret.index, cfg=soft_cfg)
    z_gpr = align_stress_z_daily(gpr, pret.index, cfg=soft_cfg)

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
    vix_scale = risk_scale_from_z(z_vix, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
    gpr_scale = risk_scale_from_z(z_gpr, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)

    cooled_vix = carry_daily.mul(vix_scale, axis=0)
    out["carry_vix_cool"] = _weights_to_returns(
        cooled_vix, pret, cfg=cfg, name="carry_vix_cool"
    )

    cooled_gpr = carry_daily.mul(gpr_scale, axis=0)
    out["carry_gpr_cool"] = _weights_to_returns(
        cooled_gpr, pret, cfg=cfg, name="carry_gpr_cool"
    )

    # Primary: trade carry only when VIX z ≤ 0 (Menkhoff)
    gate_vix_lo = (z_vix <= 0.0).astype(float)
    gate_vix_lo = gate_vix_lo.where(z_vix.notna(), 0.0).reindex(pret.index).fillna(0.0)
    gated_lo = carry_daily.mul(gate_vix_lo, axis=0)
    out["carry_low_vix"] = _weights_to_returns(
        gated_lo, pret, cfg=cfg, name="carry_low_vix"
    )

    gate_gpr_lo = (z_gpr <= 0.0).astype(float)
    gate_gpr_lo = gate_gpr_lo.where(z_gpr.notna(), 0.0).reindex(pret.index).fillna(0.0)
    gated_gpr = carry_daily.mul(gate_gpr_lo, axis=0)
    out["carry_low_gpr"] = _weights_to_returns(
        gated_gpr, pret, cfg=cfg, name="carry_low_gpr"
    )

    # Honesty inverse: carry only when VIX z ≥ z_high
    gate_vix_hi = (z_vix >= float(cfg.z_high)).astype(float)
    gate_vix_hi = gate_vix_hi.where(z_vix.notna(), 0.0).reindex(pret.index).fillna(0.0)
    gated_hi = carry_daily.mul(gate_vix_hi, axis=0)
    out["carry_high_vix"] = _weights_to_returns(
        gated_hi, pret, cfg=cfg, name="carry_high_vix"
    )

    # Sequential VIX then GPR cool (product of scales)
    stack_scale = (vix_scale * gpr_scale).clip(lower=float(cfg.cool) ** 2, upper=1.0)
    cooled_stack = carry_daily.mul(stack_scale, axis=0)
    out["carry_vix_gpr_stack"] = _weights_to_returns(
        cooled_stack, pret, cfg=cfg, name="carry_vix_gpr_stack"
    )

    blend_keys = [
        k
        for k in ("carry_low_vix", "carry_vix_cool", "carry_low_gpr", "carry_gpr_cool")
        if k in out
    ]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "carry_vix_gpr_ew"
        out["carry_vix_gpr_ew"] = blend

    return out


__all__ = [
    "PRIMARY",
    "VixGprConditionedCarryFxConfig",
    "vix_gpr_conditioned_carry_factor_returns",
    "align_stress_z_daily",
    "load_vix_series",
    "load_gpr_series",
]
