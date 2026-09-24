"""Tests for VIX/GPR-conditioned Dahlquist soft-signal EW FX module (§72)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.vix_gpr_conditioned_soft_fx import (
    PRIMARY,
    VixGprConditionedSoftFxConfig,
    align_stress_z_daily,
    vix_gpr_conditioned_soft_factor_returns,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
VIX_PATH = MACRO / "vix_yahoo.csv"
GPR_PATH = MACRO / "gpr_daily.csv"


def _synth_soft(n_days: int = 1800) -> pd.Series:
    idx = pd.date_range("2016-01-01", periods=n_days, freq="B", tz="UTC")
    rng = np.random.default_rng(11)
    return pd.Series(
        rng.normal(0.0001, 0.002, n_days), index=idx, name="soft_ew_macro5"
    )


def _synth_macro(n_days: int = 2500, *, seed: int = 0, mean: float = 15.0) -> pd.Series:
    idx = pd.date_range("2014-01-01", periods=n_days, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, 5.0, n_days), index=idx, name="macro")


def test_primary_name_constant():
    assert PRIMARY == "soft_low_vix"


def test_gate_direction_low_on_high_off():
    soft = _synth_soft(2000)
    # Force VIX so z is controllable: low early, then high
    idx = pd.date_range("2014-01-01", periods=2500, freq="B", tz="UTC")
    vix = pd.Series(
        np.concatenate([np.full(1250, 10.0), np.full(1250, 40.0)]),
        index=idx,
        name="vix",
    )
    gpr = _synth_macro(2500, seed=3, mean=100.0)
    gpr.name = "gpr"
    cfg = VixGprConditionedSoftFxConfig(
        signal_lag=1, z_window=252, min_periods=60, z_high=1.0
    )
    out = vix_gpr_conditioned_soft_factor_returns(
        soft, vix=vix, gpr=gpr, cfg=cfg
    )
    for key in (
        "soft_low_vix",
        "soft_vix_cool",
        "soft_low_gpr",
        "soft_gpr_cool",
        "soft_raw",
        "soft_high_vix",
        "soft_vix_gpr_stack",
        "soft_vix_gpr_ew",
    ):
        assert key in out
    # soft_raw identity
    common = soft.dropna().index.intersection(out["soft_raw"].dropna().index)
    assert np.allclose(soft.loc[common].values, out["soft_raw"].loc[common].values)
    # Gate magnitude: |gated| <= |raw|
    lo = out["soft_low_vix"]
    raw = out["soft_raw"]
    both = lo.notna() & raw.notna()
    assert both.sum() > 100
    assert (lo.loc[both].abs() <= raw.loc[both].abs() + 1e-15).all()
    # Inverse honesty: low and high VIX gates mutually exclusive
    hi = out["soft_high_vix"]
    product = lo.loc[both].abs() * hi.loc[both].abs()
    assert (product <= 1e-18).mean() > 0.99


def test_cool_scale_bounds():
    soft = _synth_soft(600)
    vix = _synth_macro(800, seed=1, mean=18.0)
    vix.name = "vix"
    gpr = _synth_macro(800, seed=2, mean=90.0)
    gpr.name = "gpr"
    cfg = VixGprConditionedSoftFxConfig(cool=0.35, z_high=1.0)
    out = vix_gpr_conditioned_soft_factor_returns(soft, vix=vix, gpr=gpr, cfg=cfg)
    raw = out["soft_raw"]
    cool = out["soft_vix_cool"]
    common = raw.dropna().index.intersection(cool.dropna().index)
    ratio = (cool.loc[common] / raw.loc[common]).replace([np.inf, -np.inf], np.nan)
    ratio = ratio.dropna()
    ratio = ratio[raw.loc[ratio.index].abs() > 1e-12]
    if len(ratio):
        assert float(ratio.min()) >= 0.35 - 1e-9
        assert float(ratio.max()) <= 1.0 + 1e-9


def test_stack_scales_compound():
    soft = _synth_soft(800)
    vix = _synth_macro(1000, seed=5, mean=20.0)
    vix.name = "vix"
    gpr = _synth_macro(1000, seed=6, mean=110.0)
    gpr.name = "gpr"
    cfg = VixGprConditionedSoftFxConfig(cool=0.35, z_high=1.0)
    out = vix_gpr_conditioned_soft_factor_returns(soft, vix=vix, gpr=gpr, cfg=cfg)
    raw = out["soft_raw"]
    stack = out["soft_vix_gpr_stack"]
    both = raw.notna() & stack.notna()
    assert both.sum() > 50
    # |stack| <= |raw| (product of scales ≤ 1)
    assert (stack.loc[both].abs() <= raw.loc[both].abs() + 1e-12).all()


def test_no_lookahead_mutate_future_vix():
    soft = _synth_soft(1800)
    rng = np.random.default_rng(42)
    idx = pd.date_range("2014-01-01", periods=2200, freq="B", tz="UTC")
    vix = pd.Series(rng.normal(18, 6, 2200), index=idx, name="vix")
    gpr = pd.Series(rng.normal(100, 30, 2200), index=idx, name="gpr")
    cfg = VixGprConditionedSoftFxConfig(signal_lag=1, z_window=252, min_periods=60)
    f1 = vix_gpr_conditioned_soft_factor_returns(soft, vix=vix, gpr=gpr, cfg=cfg)
    vix2 = vix.copy()
    vix2.loc["2021-01-01":] = vix2.loc["2021-01-01":] + 25.0
    f2 = vix_gpr_conditioned_soft_factor_returns(soft, vix=vix2, gpr=gpr, cfg=cfg)
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["soft_low_vix"].loc[:cut].dropna()
    b = f2["soft_low_vix"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 200
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_factor_shapes_match_soft_index():
    soft = _synth_soft(800)
    vix = _synth_macro(1000, seed=7)
    vix.name = "vix"
    gpr = _synth_macro(1000, seed=8, mean=95.0)
    gpr.name = "gpr"
    out = vix_gpr_conditioned_soft_factor_returns(
        soft, vix=vix, gpr=gpr, cfg=VixGprConditionedSoftFxConfig()
    )
    assert out["soft_low_vix"].index.equals(soft.index)
    assert out["soft_vix_gpr_ew"].index.equals(soft.index)


def test_align_stress_z_lagged():
    idx = pd.date_range("2018-01-01", periods=400, freq="B", tz="UTC")
    s = pd.Series(np.linspace(10, 30, 400), index=idx, name="vix")
    soft_idx = pd.date_range("2018-06-01", periods=200, freq="B", tz="UTC")
    cfg = VixGprConditionedSoftFxConfig(signal_lag=1, z_window=60, min_periods=30)
    z = align_stress_z_daily(s, soft_idx, cfg=cfg)
    assert z.index.equals(soft_idx)
    # First signal_lag points after warmup may be NaN from shift — ok
    assert z.notna().sum() > 50


@pytest.mark.skipif(not VIX_PATH.exists() or not GPR_PATH.exists(), reason="VIX/GPR CSV missing")
def test_load_real_vix_gpr():
    from mt5_swing.strategies.vix_gpr_conditioned_soft_fx import (
        load_gpr_series,
        load_vix_series,
    )

    vix = load_vix_series(bar_lag=1)
    gpr = load_gpr_series(bar_lag=1)
    assert vix.dropna().shape[0] > 1000
    assert gpr.dropna().shape[0] > 1000
    soft = _synth_soft(500)
    out = vix_gpr_conditioned_soft_factor_returns(soft, vix=vix, gpr=gpr)
    assert PRIMARY in out
    assert out[PRIMARY].notna().sum() > 50
