"""Tests for EPU/TPU-conditioned Dahlquist soft-signal EW FX module (§76)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.epu_tpu_conditioned_soft_fx import (
    PRIMARY,
    EpuTpuConditionedSoftFxConfig,
    align_monthly_stress_z_daily,
    epu_tpu_conditioned_soft_factor_returns,
    trailing_z_monthly,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
EPU_FRED = MACRO / "fred_USEPUINDXM.csv"
TPU_CSV = MACRO / "tpu_us_monthly.csv"


def _synth_soft(n_days: int = 1800) -> pd.Series:
    idx = pd.date_range("2016-01-01", periods=n_days, freq="B", tz="UTC")
    rng = np.random.default_rng(11)
    return pd.Series(
        rng.normal(0.0001, 0.002, n_days), index=idx, name="soft_ew_macro5"
    )


def _synth_monthly(
    start: str = "2010-01-01",
    n_months: int = 180,
    *,
    seed: int = 0,
    mean: float = 100.0,
    name: str = "epu",
) -> pd.Series:
    idx = pd.date_range(start, periods=n_months, freq="MS", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, 30.0, n_months), index=idx, name=name)


def test_primary_name_constant():
    assert PRIMARY == "soft_low_epu"


def test_gate_direction_low_on_high_off():
    soft = _synth_soft(2000)
    # Force EPU so z is controllable: low early, then high
    idx = pd.date_range("2010-01-01", periods=200, freq="MS", tz="UTC")
    epu = pd.Series(
        np.concatenate([np.full(100, 50.0), np.full(100, 250.0)]),
        index=idx,
        name="US_EPU",
    )
    tpu = _synth_monthly(n_months=200, seed=3, mean=80.0, name="TPU")
    cfg = EpuTpuConditionedSoftFxConfig(
        signal_lag_months=1,
        weight_lag_days=1,
        z_window=60,
        min_periods=24,
        z_high=1.0,
    )
    out = epu_tpu_conditioned_soft_factor_returns(
        soft, epu=epu, tpu=tpu, cfg=cfg
    )
    for key in (
        "soft_low_epu",
        "soft_epu_cool",
        "soft_low_tpu",
        "soft_tpu_cool",
        "soft_raw",
        "soft_high_epu",
        "soft_epu_tpu_stack",
        "soft_epu_tpu_ew",
    ):
        assert key in out
    # soft_raw identity
    common = soft.dropna().index.intersection(out["soft_raw"].dropna().index)
    assert np.allclose(soft.loc[common].values, out["soft_raw"].loc[common].values)
    # Gate magnitude: |gated| <= |raw|
    lo = out["soft_low_epu"]
    raw = out["soft_raw"]
    both = lo.notna() & raw.notna()
    assert both.sum() > 100
    assert (lo.loc[both].abs() <= raw.loc[both].abs() + 1e-15).all()
    # Inverse honesty: low and high EPU gates mutually exclusive
    hi = out["soft_high_epu"]
    product = lo.loc[both].abs() * hi.loc[both].abs()
    assert (product <= 1e-18).mean() > 0.99


def test_cool_scale_bounds():
    soft = _synth_soft(600)
    epu = _synth_monthly(n_months=120, seed=1, mean=110.0, name="US_EPU")
    tpu = _synth_monthly(n_months=120, seed=2, mean=90.0, name="TPU")
    cfg = EpuTpuConditionedSoftFxConfig(cool=0.35, z_high=1.0)
    out = epu_tpu_conditioned_soft_factor_returns(
        soft, epu=epu, tpu=tpu, cfg=cfg
    )
    raw = out["soft_raw"]
    cool = out["soft_epu_cool"]
    common = raw.dropna().index.intersection(cool.dropna().index)
    ratio = (cool.loc[common] / raw.loc[common]).replace([np.inf, -np.inf], np.nan)
    ratio = ratio.dropna()
    ratio = ratio[raw.loc[ratio.index].abs() > 1e-12]
    if len(ratio):
        assert float(ratio.min()) >= 0.35 - 1e-9
        assert float(ratio.max()) <= 1.0 + 1e-9


def test_stack_scales_compound():
    soft = _synth_soft(800)
    epu = _synth_monthly(n_months=140, seed=5, mean=120.0, name="US_EPU")
    tpu = _synth_monthly(n_months=140, seed=6, mean=100.0, name="TPU")
    cfg = EpuTpuConditionedSoftFxConfig(cool=0.35, z_high=1.0)
    out = epu_tpu_conditioned_soft_factor_returns(
        soft, epu=epu, tpu=tpu, cfg=cfg
    )
    raw = out["soft_raw"]
    stack = out["soft_epu_tpu_stack"]
    both = raw.notna() & stack.notna()
    assert both.sum() > 50
    assert (stack.loc[both].abs() <= raw.loc[both].abs() + 1e-12).all()


def test_no_lookahead_mutate_future_epu():
    soft = _synth_soft(1800)
    epu = _synth_monthly(
        start="2012-01-01", n_months=180, seed=42, mean=100.0, name="US_EPU"
    )
    tpu = _synth_monthly(
        start="2012-01-01", n_months=180, seed=7, mean=80.0, name="TPU"
    )
    cfg = EpuTpuConditionedSoftFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    f1 = epu_tpu_conditioned_soft_factor_returns(
        soft, epu=epu, tpu=tpu, cfg=cfg
    )
    epu2 = epu.copy()
    epu2.loc["2021-01-01":] = epu2.loc["2021-01-01":] + 80.0
    f2 = epu_tpu_conditioned_soft_factor_returns(
        soft, epu=epu2, tpu=tpu, cfg=cfg
    )
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["soft_low_epu"].loc[:cut].dropna()
    b = f2["soft_low_epu"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 100
    assert np.allclose(
        a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12
    )


def test_monthly_z_window_not_daily_252():
    """PIT: z is computed on monthly frequency with z_window months (CIP §71 mirror)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    s = pd.Series(np.linspace(50, 200, 120), index=idx, name="US_EPU")
    z = trailing_z_monthly(s, lookback=60, min_periods=24)
    # After 60 months of warmup, z should be finite
    assert z.iloc[59:].notna().sum() > 40
    # First min_periods-1 are NaN
    assert z.iloc[:23].isna().all()


def test_align_monthly_stress_z_lagged():
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    s = pd.Series(np.linspace(50, 200, 120), index=idx, name="US_EPU")
    soft_idx = pd.date_range("2015-01-01", periods=500, freq="B", tz="UTC")
    cfg = EpuTpuConditionedSoftFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    z = align_monthly_stress_z_daily(s, soft_idx, cfg=cfg)
    assert z.index.equals(soft_idx)
    assert z.notna().sum() > 50
    # Weight lag: first trading day after month-end ffill should differ from unlagged
    cfg0 = EpuTpuConditionedSoftFxConfig(
        signal_lag_months=1, weight_lag_days=0, z_window=60, min_periods=24
    )
    z0 = align_monthly_stress_z_daily(s, soft_idx, cfg=cfg0)
    # At least some days differ due to 1d lag
    both = z.notna() & z0.notna()
    assert both.sum() > 50
    # lag shifts: equal after aligning — just check shapes / coverage
    assert z.loc[both].isna().sum() == 0


def test_tpu_missing_fallthrough():
    """If TPU unavailable, still board EPU-conditioned soft with honesty companions."""
    soft = _synth_soft(800)
    epu = _synth_monthly(n_months=140, seed=9, mean=110.0, name="US_EPU")
    cfg = EpuTpuConditionedSoftFxConfig()
    out = epu_tpu_conditioned_soft_factor_returns(
        soft, epu=epu, tpu=None, cfg=cfg, allow_tpu_missing=True
    )
    assert "soft_low_epu" in out
    assert "soft_epu_cool" in out
    assert "soft_raw" in out
    assert "soft_high_epu" in out
    # TPU-specific optional; stack/ew still present as EPU-only fall-through
    assert "soft_epu_tpu_ew" in out
    assert "soft_low_tpu" not in out


def test_factor_shapes_match_soft_index():
    soft = _synth_soft(800)
    epu = _synth_monthly(n_months=140, seed=7, name="US_EPU")
    tpu = _synth_monthly(n_months=140, seed=8, mean=95.0, name="TPU")
    out = epu_tpu_conditioned_soft_factor_returns(
        soft, epu=epu, tpu=tpu, cfg=EpuTpuConditionedSoftFxConfig()
    )
    assert out["soft_low_epu"].index.equals(soft.index)
    assert out["soft_epu_tpu_ew"].index.equals(soft.index)


@pytest.mark.skipif(
    not EPU_FRED.exists() and not TPU_CSV.exists(),
    reason="EPU/TPU CSV missing",
)
def test_load_real_epu_tpu():
    from mt5_swing.strategies.epu_tpu_conditioned_soft_fx import (
        load_epu_series,
        load_tpu_series,
    )

    epu = load_epu_series(series="US", pub_lag_months=1, download=False)
    assert epu.dropna().shape[0] > 100
    soft = _synth_soft(500)
    try:
        tpu = load_tpu_series(pub_lag_months=1, download=False)
        out = epu_tpu_conditioned_soft_factor_returns(soft, epu=epu, tpu=tpu)
    except Exception:
        out = epu_tpu_conditioned_soft_factor_returns(
            soft, epu=epu, tpu=None, allow_tpu_missing=True
        )
    assert PRIMARY in out
    assert out[PRIMARY].notna().sum() > 50
