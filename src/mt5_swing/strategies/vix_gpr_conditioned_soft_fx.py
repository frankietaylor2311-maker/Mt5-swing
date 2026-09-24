"""VIX/GPR-conditioned Dahlquist soft-signal EW FX factors (§72).

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Dahlquist & Hasseltoft (2020), *JFE*: economic-momentum soft signals;
  equal-weight of macro-momentum XS factors (repo §66 ``soft_ew_macro5`` /
  SOFT_LEGS).
- Menkhoff, Sarno, Schmeling & Schrimpf (2012), *JF*: high global FX / equity
  vol → carry/momentum risk-off. We proxy global FX vol with **VIX**
  (``data/macro/vix_yahoo.csv`` / FRED VIXCLS).
- Caldara & Iacoviello (2022), *AER* / Liu & Zhang (2024), *JBF*: elevated
  aggregate GPR → risk-off. Free series ``gpr_daily.csv``.

Claim (a priori)
----------------
Soft-stack macro momentum (§66) earns more when global FX-vol / geopolitical
stress is **low**. Trade soft EW only when VIX (primary) or GPR stress is
subdued; cool or sit out when stress is elevated.

Legs
----
- ``soft_low_vix`` (**PRIMARY**, Menkhoff): §66 ``soft_ew_macro5`` × binary
  gate (on when lagged VIX z ≤ 0).
- ``soft_vix_cool``: soft_ew_macro5 × continuous cool ∈ [cool, 1] from VIX z.
- ``soft_low_gpr``: soft × binary gate (on when lagged GPR z ≤ 0).
- ``soft_gpr_cool``: soft × continuous cool from GPR z.
- ``soft_raw``: ungated soft_ew_macro5 honesty baseline (§66).
- ``soft_high_vix``: honesty inverse — soft only when VIX z ≥ z_high.
- ``soft_vix_gpr_stack``: sequential VIX then GPR cool (product of scales;
  equivalent of ``apply_vix_gpr_stack`` on soft *returns*).
- ``soft_vix_gpr_ew``: EW of soft_low_vix ⊕ soft_vix_cool ⊕ soft_low_gpr ⊕
  soft_gpr_cool.

PIT
---
VIX / GPR loaders apply ``bar_lag=1``; this module adds ``signal_lag`` (default
1 trading day) on trailing z — same dual lag as ``macro_regimes`` /
funding_liq. Soft legs keep source-wave PIT from §66 (costs 1.5 bps/side
already inside source factor returns).

Gate / cool thresholds: fixed z (z_high=1.0 / gate ≤0) — mirror §71 /
funding_liq / ACM. Literature often uses **1.5** for pure GPR risk-off
(``macro_regimes.apply_gpr_risk_gate`` default); for **this** wave we keep
z≥1 cool for §71 symmetry — documented, not HO-tuned.

Distinct from: soft_ew §66 (ungated), soft CIP §69, CIP×carry §68, CIP XS
§67, CIP-conditioned soft §71 (CIP stress, not VIX/GPR), capital-sleeve
§53 / §70, funding_liq §20, combo §8, gpr_regime standalone, AI-GPR §25,
country-GPR §45.
Explicit: do **not** overlay coolers on locked ``fx4plus_gbpcad_d1_voltarget_0025``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mt5_swing.features.macro_factors import load_gpr_daily, load_vix
from mt5_swing.strategies.funding_liquidity_fx import trailing_z
from mt5_swing.strategies.gpr_regime import (
    GprRegimeConfig,
    align_macro_to_index,
    risk_scale_from_z,
)


@dataclass
class VixGprConditionedSoftFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # trading days after loader bar_lag=1
    z_window: int = 252
    min_periods: int = 60
    z_high: float = 1.0  # cool thresh — §71 symmetry (lit GPR often 1.5; documented)
    z_low: float = 0.0
    cool: float = 0.35
    cost_bps_side: float = 1.5  # informational; costs already in soft legs
    # Documented: literature GPR cool often z≥1.5; this wave keeps 1.0 for §71 symmetry
    gpr_lit_z_high_note: float = 1.5


PRIMARY = "soft_low_vix"


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
    out = out.where(soft.notna())
    out = out.where(sc.notna(), 0.0)
    out.name = name
    return out


def align_stress_z_daily(
    series: pd.Series,
    index: pd.DatetimeIndex,
    *,
    cfg: VixGprConditionedSoftFxConfig | None = None,
) -> pd.Series:
    """As-of align → trailing z → signal_lag trading days (PIT)."""
    cfg = cfg or VixGprConditionedSoftFxConfig()
    idx = _ensure_utc(pd.DatetimeIndex(index))
    aligned = align_macro_to_index(series, idx)
    z = trailing_z(aligned, lookback=cfg.z_window, min_periods=cfg.min_periods)
    if cfg.signal_lag > 0:
        z = z.shift(int(cfg.signal_lag))
    z.name = f"{series.name or 'stress'}_z"
    return z


def load_vix_series(*, bar_lag: int = 1) -> pd.Series:
    """Thin wrapper — Yahoo VIX close with bar lag."""
    return load_vix(bar_lag=bar_lag)


def load_gpr_series(*, bar_lag: int = 1) -> pd.Series:
    """Thin wrapper — Caldara–Iacoviello daily GPR with bar lag."""
    return load_gpr_daily(bar_lag=bar_lag, col="GPR")


def vix_gpr_conditioned_soft_factor_returns(
    soft_ew: pd.Series,
    *,
    vix: pd.Series | None = None,
    gpr: pd.Series | None = None,
    cfg: VixGprConditionedSoftFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Build VIX/GPR-conditioned soft-signal EW FX factor daily returns.

    ``soft_ew`` is the §66 ``soft_ew_macro5`` / ``soft_ew5`` daily return series
    (costs already applied inside source legs). Gating/cooling multiplies that
    series by VIX / GPR stress gate/cool — distinct from §71 (CIP stress) and
    from combo §8 / gpr_regime standalone / funding_liq §20.
    """
    cfg = cfg or VixGprConditionedSoftFxConfig()
    soft = soft_ew.astype(float).copy()
    soft.index = _ensure_utc(pd.DatetimeIndex(soft.index))
    soft.name = soft.name or "soft_ew_macro5"

    if vix is None:
        vix = load_vix_series(bar_lag=1)
    if gpr is None:
        gpr = load_gpr_series(bar_lag=1)

    z_vix = align_stress_z_daily(vix, soft.index, cfg=cfg)
    z_gpr = align_stress_z_daily(gpr, soft.index, cfg=cfg)

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
    vix_scale = risk_scale_from_z(z_vix, cfg=gcfg).reindex(soft.index).ffill().fillna(1.0)
    gpr_scale = risk_scale_from_z(z_gpr, cfg=gcfg).reindex(soft.index).ffill().fillna(1.0)

    out["soft_vix_cool"] = _scale_series(soft, vix_scale, name="soft_vix_cool")
    out["soft_gpr_cool"] = _scale_series(soft, gpr_scale, name="soft_gpr_cool")

    # Primary: trade soft only when VIX z ≤ 0 (Menkhoff)
    gate_vix_lo = (z_vix <= 0.0).astype(float)
    gate_vix_lo = gate_vix_lo.where(z_vix.notna(), 0.0).reindex(soft.index).fillna(0.0)
    out["soft_low_vix"] = _scale_series(soft, gate_vix_lo, name="soft_low_vix")

    gate_gpr_lo = (z_gpr <= 0.0).astype(float)
    gate_gpr_lo = gate_gpr_lo.where(z_gpr.notna(), 0.0).reindex(soft.index).fillna(0.0)
    out["soft_low_gpr"] = _scale_series(soft, gate_gpr_lo, name="soft_low_gpr")

    # Honesty inverse: soft only when VIX z ≥ z_high
    gate_vix_hi = (z_vix >= float(cfg.z_high)).astype(float)
    gate_vix_hi = gate_vix_hi.where(z_vix.notna(), 0.0).reindex(soft.index).fillna(0.0)
    out["soft_high_vix"] = _scale_series(soft, gate_vix_hi, name="soft_high_vix")

    # Sequential VIX then GPR cool (product of scales ≡ apply_vix_gpr_stack on returns)
    stack_scale = (vix_scale * gpr_scale).clip(lower=float(cfg.cool) ** 2, upper=1.0)
    # Allow product down to cool^2 when both hot; clip floor at cool^2 not cool —
    # sequential cools compound. Documented a priori (not HO-tuned).
    out["soft_vix_gpr_stack"] = _scale_series(
        soft, stack_scale, name="soft_vix_gpr_stack"
    )

    blend_keys = [
        k
        for k in ("soft_low_vix", "soft_vix_cool", "soft_low_gpr", "soft_gpr_cool")
        if k in out
    ]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "soft_vix_gpr_ew"
        out["soft_vix_gpr_ew"] = blend

    return out


__all__ = [
    "PRIMARY",
    "VixGprConditionedSoftFxConfig",
    "align_stress_z_daily",
    "load_vix_series",
    "load_gpr_series",
    "vix_gpr_conditioned_soft_factor_returns",
]
