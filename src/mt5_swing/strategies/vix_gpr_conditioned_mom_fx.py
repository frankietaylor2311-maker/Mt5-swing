"""VIX/GPR-conditioned Menkhoff currency momentum FX factors (§74).

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Menkhoff, Sarno, Schmeling & Schrimpf (2012), *JFE*: Currency Momentum Strategies
  — cross-sectional FX momentum (formation ~3m, skip ~1m, long winners / short losers).
- Menkhoff, Sarno, Schmeling & Schrimpf (2012), *JF*: high global FX / equity
  vol → risk-off / crash states. We proxy global FX vol with **VIX**
  (``data/macro/vix_yahoo.csv`` / FRED VIXCLS).
- Caldara & Iacoviello (2022), *AER* / Liu & Zhang (2024), *JBF*: elevated
  aggregate GPR → risk-off. Free series ``gpr_daily.csv``.

Claim (a priori)
----------------
Menkhoff cross-sectional FX momentum earns more when global FX-vol /
geopolitical stress is **low**. Scale down or gate momentum when VIX or GPR
stress is elevated. Parallel to §73 (carry × VIX/GPR) — **not** a cooler
overlay on locked fx4plus. Classic scholarly gap left after carry×VIX/GPR.

Legs
----
- ``mom_low_vix`` (**PRIMARY**, Menkhoff): scholarly FX momentum only
  when lagged VIX z ≤ 0 (flat when elevated).
- ``mom_vix_cool``: mom × risk_scale ∈ [cool, 1] from VIX z
  (cool when z ≥ z_high).
- ``mom_low_gpr``: mom only when lagged GPR z ≤ 0.
- ``mom_gpr_cool``: mom × cool from GPR z.
- ``mom_raw``: always-on Menkhoff momentum (honesty).
- ``mom_high_vix``: honesty inverse — mom only when VIX z ≥ z_high.
- ``mom_vix_gpr_stack``: sequential VIX then GPR cool (product of scales).
- ``mom_vix_gpr_ew``: EW of low_vix ⊕ vix_cool ⊕ low_gpr ⊕ gpr_cool.

PIT
---
VIX / GPR loaders apply ``bar_lag=1``; this module adds ``signal_lag`` (default
1 trading day) on trailing z — same dual lag as §72 / §73 / funding_liq.
Momentum: ``FxMomentumConfig`` defaults (formation=63d, skip=21d,
n_long=n_short=2, signal_lag=1) via ``expand_weights_to_daily``.
``portfolio_returns_from_weights`` applies an extra weight lag.

Gate / cool thresholds: fixed z (z_high=1.0 / gate ≤0) — mirror §71 / §72 /
§73. Literature often uses **1.5** for pure GPR risk-off; for **this** wave we
keep z≥1 for §71/§72/§73 symmetry — documented, not HO-tuned.

Distinct from: combo §8 (carry+mom+dollar × VIX/GPR together), raw
fx_momentum standalone, soft EW §66, soft CIP §69, CIP XS §67, CIP×carry §68,
CIP-conditioned soft §71, VIX/GPR-conditioned soft §72, VIX/GPR-conditioned
carry §73, capital-sleeve §53/§70, funding_liq §20, gpr_regime, AI-GPR §25,
country-GPR §45. Explicit: do **not** overlay coolers on locked
``fx4plus_gbpcad_d1_voltarget_0025``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mt5_swing.strategies.carry_rank import (
    currency_weights_to_pair_weights,
    expand_weights_to_daily,
    portfolio_returns_from_weights,
)
from mt5_swing.strategies.fx_momentum import FxMomentumConfig, momentum_weights_from_returns
from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z
from mt5_swing.strategies.vix_gpr_conditioned_soft_fx import (
    align_stress_z_daily,
    load_gpr_series,
    load_vix_series,
)
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class VixGprConditionedMomFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # trading days after loader bar_lag=1 (VIX/GPR z)
    z_window: int = 252  # daily, like §72/§73 — NOT monthly CIP 60
    min_periods: int = 60
    z_high: float = 1.0  # cool thresh — §71/§72/§73 symmetry (lit GPR often 1.5)
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
    # Documented: literature GPR cool often z≥1.5; this wave keeps 1.0 for symmetry
    gpr_lit_z_high_note: float = 1.5


PRIMARY = "mom_low_vix"


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _mom_daily_weights(
    pair_ret: pd.DataFrame,
    *,
    cfg: VixGprConditionedMomFxConfig,
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
    cfg: VixGprConditionedMomFxConfig,
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


def _soft_cfg_from_mom(cfg: VixGprConditionedMomFxConfig):
    """Reuse §72/§73 align_stress_z_daily config shape."""
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


def vix_gpr_conditioned_mom_factor_returns(
    pair_ret: pd.DataFrame,
    *,
    vix: pd.Series | None = None,
    gpr: pd.Series | None = None,
    cfg: VixGprConditionedMomFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Build VIX/GPR-conditioned Menkhoff FX momentum factor daily returns.

    Distinct from §73 (carry × VIX/GPR), §72 (soft EW × VIX/GPR), and combo §8.
    Explicit: do **not** overlay coolers on locked fx4plus.
    """
    cfg = cfg or VixGprConditionedMomFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    out: dict[str, pd.Series] = {}

    if pret.empty or pret.shape[1] < 3:
        return out

    if vix is None:
        vix = load_vix_series(bar_lag=1)
    if gpr is None:
        gpr = load_gpr_series(bar_lag=1)

    soft_cfg = _soft_cfg_from_mom(cfg)
    z_vix = align_stress_z_daily(vix, pret.index, cfg=soft_cfg)
    z_gpr = align_stress_z_daily(gpr, pret.index, cfg=soft_cfg)

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
    vix_scale = risk_scale_from_z(z_vix, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
    gpr_scale = risk_scale_from_z(z_gpr, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)

    cooled_vix = mom_daily.mul(vix_scale, axis=0)
    out["mom_vix_cool"] = _weights_to_returns(
        cooled_vix, pret, cfg=cfg, name="mom_vix_cool"
    )

    cooled_gpr = mom_daily.mul(gpr_scale, axis=0)
    out["mom_gpr_cool"] = _weights_to_returns(
        cooled_gpr, pret, cfg=cfg, name="mom_gpr_cool"
    )

    # Primary: trade mom only when VIX z ≤ 0 (Menkhoff)
    gate_vix_lo = (z_vix <= 0.0).astype(float)
    gate_vix_lo = gate_vix_lo.where(z_vix.notna(), 0.0).reindex(pret.index).fillna(0.0)
    gated_lo = mom_daily.mul(gate_vix_lo, axis=0)
    out["mom_low_vix"] = _weights_to_returns(
        gated_lo, pret, cfg=cfg, name="mom_low_vix"
    )

    gate_gpr_lo = (z_gpr <= 0.0).astype(float)
    gate_gpr_lo = gate_gpr_lo.where(z_gpr.notna(), 0.0).reindex(pret.index).fillna(0.0)
    gated_gpr = mom_daily.mul(gate_gpr_lo, axis=0)
    out["mom_low_gpr"] = _weights_to_returns(
        gated_gpr, pret, cfg=cfg, name="mom_low_gpr"
    )

    # Honesty inverse: mom only when VIX z ≥ z_high
    gate_vix_hi = (z_vix >= float(cfg.z_high)).astype(float)
    gate_vix_hi = gate_vix_hi.where(z_vix.notna(), 0.0).reindex(pret.index).fillna(0.0)
    gated_hi = mom_daily.mul(gate_vix_hi, axis=0)
    out["mom_high_vix"] = _weights_to_returns(
        gated_hi, pret, cfg=cfg, name="mom_high_vix"
    )

    # Sequential VIX then GPR cool (product of scales)
    stack_scale = (vix_scale * gpr_scale).clip(lower=float(cfg.cool) ** 2, upper=1.0)
    cooled_stack = mom_daily.mul(stack_scale, axis=0)
    out["mom_vix_gpr_stack"] = _weights_to_returns(
        cooled_stack, pret, cfg=cfg, name="mom_vix_gpr_stack"
    )

    blend_keys = [
        k
        for k in ("mom_low_vix", "mom_vix_cool", "mom_low_gpr", "mom_gpr_cool")
        if k in out
    ]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "mom_vix_gpr_ew"
        out["mom_vix_gpr_ew"] = blend

    return out


__all__ = [
    "PRIMARY",
    "VixGprConditionedMomFxConfig",
    "vix_gpr_conditioned_mom_factor_returns",
    "align_stress_z_daily",
    "load_vix_series",
    "load_gpr_series",
]
