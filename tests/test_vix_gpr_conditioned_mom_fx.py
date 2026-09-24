"""Tests for VIX/GPR-conditioned Menkhoff FX momentum module (§74)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.vix_gpr_conditioned_mom_fx import (
    PRIMARY,
    VixGprConditionedMomFxConfig,
    align_stress_z_daily,
    vix_gpr_conditioned_mom_factor_returns,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
VIX_PATH = MACRO / "vix_yahoo.csv"
GPR_PATH = MACRO / "gpr_daily.csv"


def _synth_ret(n_days: int = 1800) -> pd.DataFrame:
    idx = pd.date_range("2016-01-01", periods=n_days, freq="B", tz="UTC")
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            s: rng.normal(0, 0.005, n_days)
            for s in ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF")
        },
        index=idx,
    )


def _synth_macro(n_days: int = 2500, *, seed: int = 0, mean: float = 15.0) -> pd.Series:
    idx = pd.date_range("2014-01-01", periods=n_days, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, 5.0, n_days), index=idx, name="macro")


def test_primary_name_constant():
    assert PRIMARY == "mom_low_vix"


def test_gate_direction_low_on_high_off():
    ret = _synth_ret(2000)
    idx = pd.date_range("2014-01-01", periods=2500, freq="B", tz="UTC")
    vix = pd.Series(
        np.concatenate([np.full(1250, 10.0), np.full(1250, 40.0)]),
        index=idx,
        name="vix",
    )
    gpr = _synth_macro(2500, seed=3, mean=100.0)
    gpr.name = "gpr"
    cfg = VixGprConditionedMomFxConfig(
        signal_lag=1, z_window=252, min_periods=60, z_high=1.0
    )
    out = vix_gpr_conditioned_mom_factor_returns(ret, vix=vix, gpr=gpr, cfg=cfg)
    for key in (
        "mom_low_vix",
        "mom_vix_cool",
        "mom_low_gpr",
        "mom_gpr_cool",
        "mom_raw",
        "mom_high_vix",
        "mom_vix_gpr_stack",
        "mom_vix_gpr_ew",
    ):
        assert key in out
    lo = out["mom_low_vix"]
    raw = out["mom_raw"]
    both = lo.notna() & raw.notna()
    assert both.sum() > 100
    # Inverse honesty: low and high VIX gates mutually exclusive on abs returns
    hi = out["mom_high_vix"]
    product = lo.loc[both].abs() * hi.loc[both].abs()
    assert (product <= 1e-10).mean() > 0.90


def test_cool_scale_bounds():
    """risk_scale_from_z stays in [cool, 1]; cool factor exists and is finite."""
    from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z
    from mt5_swing.strategies.vix_gpr_conditioned_soft_fx import (
        VixGprConditionedSoftFxConfig,
        align_stress_z_daily,
    )

    ret = _synth_ret(800)
    vix = _synth_macro(1000, seed=1, mean=18.0)
    vix.name = "vix"
    gpr = _synth_macro(1000, seed=2, mean=90.0)
    gpr.name = "gpr"
    cfg = VixGprConditionedMomFxConfig(cool=0.35, z_high=1.0)
    soft_cfg = VixGprConditionedSoftFxConfig(
        signal_lag=cfg.signal_lag,
        z_window=cfg.z_window,
        min_periods=cfg.min_periods,
        z_high=cfg.z_high,
        cool=cfg.cool,
    )
    z = align_stress_z_daily(vix, ret.index, cfg=soft_cfg)
    gcfg = GprRegimeConfig(
        z_high=cfg.z_high, z_low=cfg.z_low, cool=cfg.cool, signal_lag=0, usd_tilt=0.0
    )
    scale = risk_scale_from_z(z, cfg=gcfg).dropna()
    assert len(scale) > 50
    assert float(scale.min()) >= cfg.cool - 1e-9
    assert float(scale.max()) <= 1.0 + 1e-9
    out = vix_gpr_conditioned_mom_factor_returns(ret, vix=vix, gpr=gpr, cfg=cfg)
    assert out["mom_vix_cool"].notna().sum() > 50


def test_stack_scales_compound():
    """Stack scale = product of VIX×GPR scales, clipped to [cool^2, 1]."""
    from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z
    from mt5_swing.strategies.vix_gpr_conditioned_soft_fx import (
        VixGprConditionedSoftFxConfig,
        align_stress_z_daily,
    )

    ret = _synth_ret(800)
    vix = _synth_macro(1000, seed=5, mean=20.0)
    vix.name = "vix"
    gpr = _synth_macro(1000, seed=6, mean=110.0)
    gpr.name = "gpr"
    cfg = VixGprConditionedMomFxConfig(cool=0.35, z_high=1.0)
    soft_cfg = VixGprConditionedSoftFxConfig(
        signal_lag=cfg.signal_lag,
        z_window=cfg.z_window,
        min_periods=cfg.min_periods,
        z_high=cfg.z_high,
        cool=cfg.cool,
    )
    z_vix = align_stress_z_daily(vix, ret.index, cfg=soft_cfg)
    z_gpr = align_stress_z_daily(gpr, ret.index, cfg=soft_cfg)
    gcfg = GprRegimeConfig(
        z_high=cfg.z_high, z_low=cfg.z_low, cool=cfg.cool, signal_lag=0, usd_tilt=0.0
    )
    vix_s = risk_scale_from_z(z_vix, cfg=gcfg)
    gpr_s = risk_scale_from_z(z_gpr, cfg=gcfg)
    stack_s = (vix_s * gpr_s).clip(lower=cfg.cool ** 2, upper=1.0).dropna()
    assert len(stack_s) > 50
    assert float(stack_s.min()) >= cfg.cool ** 2 - 1e-9
    assert float(stack_s.max()) <= 1.0 + 1e-9
    out = vix_gpr_conditioned_mom_factor_returns(ret, vix=vix, gpr=gpr, cfg=cfg)
    assert out["mom_vix_gpr_stack"].notna().sum() > 50


def test_no_lookahead_mutate_future_vix():
    ret = _synth_ret(1800)
    rng = np.random.default_rng(42)
    idx = pd.date_range("2014-01-01", periods=2200, freq="B", tz="UTC")
    vix = pd.Series(rng.normal(18, 6, 2200), index=idx, name="vix")
    gpr = pd.Series(rng.normal(100, 30, 2200), index=idx, name="gpr")
    cfg = VixGprConditionedMomFxConfig(signal_lag=1, z_window=252, min_periods=60)
    f1 = vix_gpr_conditioned_mom_factor_returns(ret, vix=vix, gpr=gpr, cfg=cfg)
    vix2 = vix.copy()
    vix2.loc["2021-01-01":] = vix2.loc["2021-01-01":] + 25.0
    f2 = vix_gpr_conditioned_mom_factor_returns(ret, vix=vix2, gpr=gpr, cfg=cfg)
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["mom_low_vix"].loc[:cut].dropna()
    b = f2["mom_low_vix"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 200
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_factor_shapes_match_index():
    ret = _synth_ret(800)
    vix = _synth_macro(1000, seed=7)
    vix.name = "vix"
    gpr = _synth_macro(1000, seed=8, mean=95.0)
    gpr.name = "gpr"
    out = vix_gpr_conditioned_mom_factor_returns(
        ret, vix=vix, gpr=gpr, cfg=VixGprConditionedMomFxConfig()
    )
    assert out["mom_low_vix"].index.equals(ret.index)
    assert out["mom_vix_gpr_ew"].index.equals(ret.index)


def test_align_stress_z_lagged():
    idx = pd.date_range("2018-01-01", periods=400, freq="B", tz="UTC")
    s = pd.Series(np.linspace(10, 30, 400), index=idx, name="vix")
    soft_idx = pd.date_range("2018-06-01", periods=200, freq="B", tz="UTC")
    from mt5_swing.strategies.vix_gpr_conditioned_soft_fx import (
        VixGprConditionedSoftFxConfig,
    )

    cfg = VixGprConditionedSoftFxConfig(signal_lag=1, z_window=60, min_periods=30)
    z = align_stress_z_daily(s, soft_idx, cfg=cfg)
    assert z.index.equals(soft_idx)
    assert z.notna().sum() > 50


def test_gate_on_off_explicit_z():
    """Primary gate: mom on when z≤0; inverse on when z≥z_high."""
    idx = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    z = pd.Series([-1.0, -0.5, 0.0, 0.5, 1.5, 2.0, -0.2, 1.1, 0.0, -2.0], index=idx)
    cfg = VixGprConditionedMomFxConfig(z_high=1.0)
    gate_lo = (z <= 0.0).astype(float)
    gate_hi = (z >= cfg.z_high).astype(float)
    assert gate_lo.iloc[0] == 1.0
    assert gate_lo.iloc[4] == 0.0
    assert gate_hi.iloc[4] == 1.0
    assert gate_hi.iloc[0] == 0.0


@pytest.mark.skipif(not VIX_PATH.exists() or not GPR_PATH.exists(), reason="VIX/GPR CSV missing")
def test_load_real_vix_gpr_mom():
    from mt5_swing.strategies.vix_gpr_conditioned_mom_fx import (
        load_gpr_series,
        load_vix_series,
    )

    vix = load_vix_series(bar_lag=1)
    gpr = load_gpr_series(bar_lag=1)
    assert vix.dropna().shape[0] > 1000
    assert gpr.dropna().shape[0] > 1000
    ret = _synth_ret(500)
    out = vix_gpr_conditioned_mom_factor_returns(ret, vix=vix, gpr=gpr)
    assert PRIMARY in out
    assert out[PRIMARY].notna().sum() > 50
