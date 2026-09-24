"""EPU/TPU-conditioned Dahlquist soft-signal EW FX factors (§76).

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Dahlquist & Hasseltoft (2020), *JFE*: economic-momentum soft signals;
  equal-weight of macro-momentum XS factors (repo §66 ``soft_ew_macro5`` /
  SOFT_LEGS).
- Baker, Bloom & Davis (2016), *QJE*: news-based Economic Policy Uncertainty
  (EPU). We use **US EPU** (FRED USEPUINDXM) as primary Stress A; GEPU as
  documented aggregate fall-through if US missing.
- Baker–Bloom–Davis categorical *Trade policy* EPU = US Trade Policy
  Uncertainty (TPU) on policyuncertainty.com — Stress B; distinct from
  aggregate EPU / VIX / GPR / CIP.

Claim (a priori)
----------------
Soft-stack macro momentum (§66) earns more when policy / trade-policy
uncertainty is **low**. Trade soft EW only when lagged EPU (primary) or TPU
stress is subdued; cool or sit out when stress is elevated.

Legs
----
- ``soft_low_epu`` (**PRIMARY**): §66 ``soft_ew_macro5`` × binary gate
  (on when lagged EPU z ≤ 0).
- ``soft_epu_cool``: soft_ew_macro5 × continuous cool ∈ [cool, 1] from EPU z.
- ``soft_low_tpu``: soft × binary gate (on when lagged TPU z ≤ 0).
- ``soft_tpu_cool``: soft × continuous cool from TPU z.
- ``soft_raw``: ungated soft_ew_macro5 honesty baseline (§66).
- ``soft_high_epu``: honesty inverse — soft only when EPU z ≥ z_high.
- ``soft_epu_tpu_stack``: sequential EPU then TPU cool (product of scales).
- ``soft_epu_tpu_ew``: EW of soft_low_epu ⊕ soft_epu_cool ⊕ soft_low_tpu ⊕
  soft_tpu_cool.

PIT
---
EPU / TPU loaders apply ``pub_lag_months=1``; this module adds
``signal_lag_months`` (default 1) on **monthly** trailing z (z_window=60m,
mirroring CIP §71 — not blindly 252d daily) + ``weight_lag_days=1`` after
month-end ffill to the trading calendar. Soft legs keep source-wave PIT from
§66 (costs 1.5 bps/side already inside source factor returns).

Gate / cool thresholds: fixed z (z_high=1.0 / gate ≤0) — mirror §71–§75.
Literature often uses other cutoffs; documented symmetry, **not** HO-tuned.

Distinct from: standalone EPU/TPU wave, soft_ew §66, soft CIP §69,
CIP×carry §68, CIP soft §71, VIX/GPR soft §72, carry/mom/value VIX/GPR
§73–§75, capital-sleeve §53/§70, combo §8.
Explicit: do **not** overlay coolers on locked ``fx4plus_gbpcad_d1_voltarget_0025``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.data.macro_uncertainty import load_epu, load_tpu
from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z


@dataclass
class EpuTpuConditionedSoftFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag_months: int = 1  # months after pub-lagged monthly index known
    weight_lag_days: int = 1  # trading days after month-end ffill
    z_window: int = 60  # months (~5y) — mirror CIP §71, not 252d daily
    min_periods: int = 24
    z_high: float = 1.0  # cool thresh — §71–§75 symmetry
    z_low: float = 0.0
    cool: float = 0.35
    cost_bps_side: float = 1.5  # informational; costs already in soft legs
    # Documented: lit often uses other cutoffs; this wave keeps 1.0 for symmetry
    lit_z_high_note: float = 1.5


PRIMARY = "soft_low_epu"


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


def trailing_z_monthly(
    series: pd.Series, *, lookback: int, min_periods: int
) -> pd.Series:
    """Trailing z on a monthly series (mirror CIP / country-EPU monthly z)."""
    s = series.astype(float)
    mu = s.rolling(lookback, min_periods=min_periods).mean()
    sd = s.rolling(lookback, min_periods=min_periods).std()
    return (s - mu) / sd.replace(0.0, np.nan)


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


def align_monthly_stress_z_daily(
    series: pd.Series,
    index: pd.DatetimeIndex,
    *,
    cfg: EpuTpuConditionedSoftFxConfig | None = None,
) -> pd.Series:
    """Monthly EPU/TPU → trailing monthly z → signal_lag months → daily ffill + 1d.

    Mirrors ``align_cip_stress_z_daily`` (§68/§71): z_window months on the native
    monthly frequency — **not** blindly 252d after daily ffill.
    """
    cfg = cfg or EpuTpuConditionedSoftFxConfig()
    s = series.astype(float).copy()
    s.index = _month_start(pd.DatetimeIndex(s.index))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    z = trailing_z_monthly(s, lookback=cfg.z_window, min_periods=cfg.min_periods)
    if cfg.signal_lag_months > 0:
        z = z.shift(int(cfg.signal_lag_months))
    z_me = z.copy()
    z_me.index = _month_end(pd.DatetimeIndex(z_me.index))
    pair_index = _ensure_utc(pd.DatetimeIndex(index))
    z_daily = z_me.reindex(pair_index, method="ffill")
    if cfg.weight_lag_days > 0:
        z_daily = z_daily.shift(int(cfg.weight_lag_days))
    z_daily.name = f"{series.name or 'stress'}_z"
    return z_daily


def load_epu_series(
    *,
    series: str = "US",
    pub_lag_months: int = 1,
    download: bool = True,
) -> pd.Series:
    """Thin wrapper — Baker–Bloom–Davis US EPU (or GEPU aggregate)."""
    return load_epu(
        series=series, download=download, pub_lag_months=pub_lag_months
    )


def load_tpu_series(
    *,
    pub_lag_months: int = 1,
    download: bool = True,
) -> pd.Series:
    """Thin wrapper — US Trade Policy Uncertainty (categorical Trade policy)."""
    return load_tpu(download=download, pub_lag_months=pub_lag_months)


_TPU_AUTO = object()  # sentinel: try load; None = explicitly unavailable


def epu_tpu_conditioned_soft_factor_returns(
    soft_ew: pd.Series,
    *,
    epu: pd.Series | None = None,
    tpu: pd.Series | None | object = _TPU_AUTO,
    cfg: EpuTpuConditionedSoftFxConfig | None = None,
    allow_tpu_missing: bool = True,
) -> dict[str, pd.Series]:
    """Build EPU/TPU-conditioned soft-signal EW FX factor daily returns.

    ``soft_ew`` is the §66 ``soft_ew_macro5`` / ``soft_ew5`` daily return series
    (costs already applied inside source legs). Gating/cooling multiplies that
    series by EPU / TPU stress gate/cool — distinct from §71 (CIP), §72
    (VIX/GPR), and standalone EPU/TPU wave.

    ``tpu`` semantics: default auto-load; pass ``None`` for explicit EPU-only
    fall-through (TPU companions omitted). If auto-load fails and
    ``allow_tpu_missing``, same EPU-only honesty path.
    """
    cfg = cfg or EpuTpuConditionedSoftFxConfig()
    soft = soft_ew.astype(float).copy()
    soft.index = _ensure_utc(pd.DatetimeIndex(soft.index))
    soft.name = soft.name or "soft_ew_macro5"

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

    z_epu = align_monthly_stress_z_daily(epu, soft.index, cfg=cfg)
    z_tpu = (
        align_monthly_stress_z_daily(tpu, soft.index, cfg=cfg)
        if tpu is not None
        else None
    )

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
    epu_scale = (
        risk_scale_from_z(z_epu, cfg=gcfg).reindex(soft.index).ffill().fillna(1.0)
    )
    out["soft_epu_cool"] = _scale_series(soft, epu_scale, name="soft_epu_cool")

    # Primary: trade soft only when EPU z ≤ 0
    gate_epu_lo = (z_epu <= 0.0).astype(float)
    gate_epu_lo = gate_epu_lo.where(z_epu.notna(), 0.0).reindex(soft.index).fillna(0.0)
    out["soft_low_epu"] = _scale_series(soft, gate_epu_lo, name="soft_low_epu")

    # Honesty inverse: soft only when EPU z ≥ z_high
    gate_epu_hi = (z_epu >= float(cfg.z_high)).astype(float)
    gate_epu_hi = gate_epu_hi.where(z_epu.notna(), 0.0).reindex(soft.index).fillna(0.0)
    out["soft_high_epu"] = _scale_series(soft, gate_epu_hi, name="soft_high_epu")

    if tpu_available and z_tpu is not None:
        tpu_scale = (
            risk_scale_from_z(z_tpu, cfg=gcfg).reindex(soft.index).ffill().fillna(1.0)
        )
        out["soft_tpu_cool"] = _scale_series(soft, tpu_scale, name="soft_tpu_cool")

        gate_tpu_lo = (z_tpu <= 0.0).astype(float)
        gate_tpu_lo = (
            gate_tpu_lo.where(z_tpu.notna(), 0.0).reindex(soft.index).fillna(0.0)
        )
        out["soft_low_tpu"] = _scale_series(soft, gate_tpu_lo, name="soft_low_tpu")

        # Sequential EPU then TPU cool (product of scales)
        stack_scale = (epu_scale * tpu_scale).clip(
            lower=float(cfg.cool) ** 2, upper=1.0
        )
        out["soft_epu_tpu_stack"] = _scale_series(
            soft, stack_scale, name="soft_epu_tpu_stack"
        )

        blend_keys = [
            k
            for k in (
                "soft_low_epu",
                "soft_epu_cool",
                "soft_low_tpu",
                "soft_tpu_cool",
            )
            if k in out
        ]
        if len(blend_keys) >= 2:
            blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
            blend.name = "soft_epu_tpu_ew"
            out["soft_epu_tpu_ew"] = blend
    else:
        # EPU-only fall-through: still board honesty companions without inventing TPU
        out["soft_epu_tpu_stack"] = out["soft_epu_cool"].copy()
        out["soft_epu_tpu_stack"].name = "soft_epu_tpu_stack"
        blend = pd.concat(
            [out["soft_low_epu"], out["soft_epu_cool"]], axis=1
        ).mean(axis=1)
        blend.name = "soft_epu_tpu_ew"
        out["soft_epu_tpu_ew"] = blend

    return out


__all__ = [
    "PRIMARY",
    "EpuTpuConditionedSoftFxConfig",
    "align_monthly_stress_z_daily",
    "trailing_z_monthly",
    "load_epu_series",
    "load_tpu_series",
    "epu_tpu_conditioned_soft_factor_returns",
]
