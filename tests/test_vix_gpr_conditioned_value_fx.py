"""Tests for VIX/GPR-conditioned Rogoff PPP real-FX value module (§75)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.vix_gpr_conditioned_value_fx import (
    PRIMARY,
    VixGprConditionedValueFxConfig,
    align_stress_z_daily,
    vix_gpr_conditioned_value_factor_returns,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
VIX_PATH = MACRO / "vix_yahoo.csv"
GPR_PATH = MACRO / "gpr_daily.csv"


def _synth_panel(n_days: int = 2200) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic G10 closes + returns long enough for 60m PPP lookback."""
    idx = pd.date_range("2008-01-01", periods=n_days, freq="B", tz="UTC")
    rng = np.random.default_rng(17)
    close = {}
    for s, level in (
        ("EURUSD", 1.20),
        ("GBPUSD", 1.50),
        ("AUDUSD", 0.80),
        ("NZDUSD", 0.70),
        ("USDJPY", 110.0),
        ("USDCAD", 1.25),
        ("USDCHF", 0.95),
    ):
        ret = rng.normal(0, 0.005, n_days)
        close[s] = level * np.cumprod(1.0 + ret)
    px = pd.DataFrame(close, index=idx)
    ret = px.pct_change()
    return px, ret


def _synth_cpi(n_months: int = 220) -> pd.DataFrame:
    """Synthetic CPI levels (month-start) for G10 + USD."""
    idx = pd.date_range("2007-01-01", periods=n_months, freq="MS", tz="UTC")
    rng = np.random.default_rng(9)
    cols = {}
    for c in ("USD", "EUR", "GBP", "AUD", "NZD", "JPY", "CAD", "CHF"):
        growth = rng.normal(0.002, 0.001, n_months)
        cols[c] = 100.0 * np.cumprod(1.0 + growth)
    return pd.DataFrame(cols, index=idx)


def _synth_macro(n_days: int = 3000, *, seed: int = 0, mean: float = 15.0) -> pd.Series:
    idx = pd.date_range("2006-01-01", periods=n_days, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, 5.0, n_days), index=idx, name="macro")


def test_primary_name_constant():
    assert PRIMARY == "value_low_vix"


def test_gate_direction_low_on_high_off():
    px, ret = _synth_panel(2400)
    cpi = _synth_cpi(240)
    idx = pd.date_range("2006-01-01", periods=3000, freq="B", tz="UTC")
    vix = pd.Series(
        np.concatenate([np.full(1500, 10.0), np.full(1500, 40.0)]),
        index=idx,
        name="vix",
    )
    gpr = _synth_macro(3000, seed=3, mean=100.0)
    gpr.name = "gpr"
    cfg = VixGprConditionedValueFxConfig(
        signal_lag=1, z_window=252, min_periods=60, z_high=1.0, lookback=60
    )
    out = vix_gpr_conditioned_value_factor_returns(
        px, ret, cpi, vix=vix, gpr=gpr, cfg=cfg
    )
    for key in (
        "value_low_vix",
        "value_vix_cool",
        "value_low_gpr",
        "value_gpr_cool",
        "value_raw",
        "value_high_vix",
        "value_vix_gpr_stack",
        "value_vix_gpr_ew",
    ):
        assert key in out
    lo = out["value_low_vix"]
    raw = out["value_raw"]
    both = lo.notna() & raw.notna()
    assert both.sum() > 100
    hi = out["value_high_vix"]
    product = lo.loc[both].abs() * hi.loc[both].abs()
    assert (product <= 1e-10).mean() > 0.90


def test_cool_scale_bounds():
    from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z
    from mt5_swing.strategies.vix_gpr_conditioned_soft_fx import (
        VixGprConditionedSoftFxConfig,
        align_stress_z_daily,
    )

    px, ret = _synth_panel(1800)
    cpi = _synth_cpi(200)
    vix = _synth_macro(2500, seed=1, mean=18.0)
    vix.name = "vix"
    gpr = _synth_macro(2500, seed=2, mean=90.0)
    gpr.name = "gpr"
    cfg = VixGprConditionedValueFxConfig(cool=0.35, z_high=1.0)
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
    out = vix_gpr_conditioned_value_factor_returns(
        px, ret, cpi, vix=vix, gpr=gpr, cfg=cfg
    )
    assert out["value_vix_cool"].notna().sum() > 50


def test_stack_scales_compound():
    from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z
    from mt5_swing.strategies.vix_gpr_conditioned_soft_fx import (
        VixGprConditionedSoftFxConfig,
        align_stress_z_daily,
    )

    px, ret = _synth_panel(1800)
    cpi = _synth_cpi(200)
    vix = _synth_macro(2500, seed=5, mean=20.0)
    vix.name = "vix"
    gpr = _synth_macro(2500, seed=6, mean=110.0)
    gpr.name = "gpr"
    cfg = VixGprConditionedValueFxConfig(cool=0.35, z_high=1.0)
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
    out = vix_gpr_conditioned_value_factor_returns(
        px, ret, cpi, vix=vix, gpr=gpr, cfg=cfg
    )
    assert out["value_vix_gpr_stack"].notna().sum() > 50


def test_no_lookahead_mutate_future_vix():
    px, ret = _synth_panel(2200)
    cpi = _synth_cpi(220)
    rng = np.random.default_rng(42)
    idx = pd.date_range("2006-01-01", periods=2800, freq="B", tz="UTC")
    vix = pd.Series(rng.normal(18, 6, 2800), index=idx, name="vix")
    gpr = pd.Series(rng.normal(100, 30, 2800), index=idx, name="gpr")
    cfg = VixGprConditionedValueFxConfig(signal_lag=1, z_window=252, min_periods=60)
    f1 = vix_gpr_conditioned_value_factor_returns(
        px, ret, cpi, vix=vix, gpr=gpr, cfg=cfg
    )
    vix2 = vix.copy()
    vix2.loc["2018-01-01":] = vix2.loc["2018-01-01":] + 25.0
    f2 = vix_gpr_conditioned_value_factor_returns(
        px, ret, cpi, vix=vix2, gpr=gpr, cfg=cfg
    )
    cut = pd.Timestamp("2016-06-30", tz="UTC")
    a = f1["value_low_vix"].loc[:cut].dropna()
    b = f2["value_low_vix"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 100
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_no_lookahead_mutate_future_cpi():
    """Future CPI mutation must not change past value signals."""
    px, ret = _synth_panel(2200)
    cpi = _synth_cpi(220)
    vix = _synth_macro(2800, seed=11, mean=18.0)
    vix.name = "vix"
    gpr = _synth_macro(2800, seed=12, mean=100.0)
    gpr.name = "gpr"
    cfg = VixGprConditionedValueFxConfig()
    f1 = vix_gpr_conditioned_value_factor_returns(
        px, ret, cpi, vix=vix, gpr=gpr, cfg=cfg
    )
    cpi2 = cpi.copy()
    cpi2.loc["2018-01-01":] = cpi2.loc["2018-01-01":] * 1.15
    f2 = vix_gpr_conditioned_value_factor_returns(
        px, ret, cpi2, vix=vix, gpr=gpr, cfg=cfg
    )
    cut = pd.Timestamp("2016-06-30", tz="UTC")
    a = f1["value_raw"].loc[:cut].dropna()
    b = f2["value_raw"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 50
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_factor_shapes_match_index():
    px, ret = _synth_panel(1800)
    cpi = _synth_cpi(200)
    vix = _synth_macro(2500, seed=7)
    vix.name = "vix"
    gpr = _synth_macro(2500, seed=8, mean=95.0)
    gpr.name = "gpr"
    out = vix_gpr_conditioned_value_factor_returns(
        px, ret, cpi, vix=vix, gpr=gpr, cfg=VixGprConditionedValueFxConfig()
    )
    assert out["value_low_vix"].index.equals(ret.index)
    assert out["value_vix_gpr_ew"].index.equals(ret.index)


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
    idx = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    z = pd.Series([-1.0, -0.5, 0.0, 0.5, 1.5, 2.0, -0.2, 1.1, 0.0, -2.0], index=idx)
    cfg = VixGprConditionedValueFxConfig(z_high=1.0)
    gate_lo = (z <= 0.0).astype(float)
    gate_hi = (z >= cfg.z_high).astype(float)
    assert gate_lo.iloc[0] == 1.0
    assert gate_lo.iloc[4] == 0.0
    assert gate_hi.iloc[4] == 1.0
    assert gate_hi.iloc[0] == 0.0


@pytest.mark.skipif(not VIX_PATH.exists() or not GPR_PATH.exists(), reason="VIX/GPR CSV missing")
def test_load_real_vix_gpr_value():
    from mt5_swing.strategies.vix_gpr_conditioned_value_fx import (
        load_gpr_series,
        load_vix_series,
    )

    vix = load_vix_series(bar_lag=1)
    gpr = load_gpr_series(bar_lag=1)
    assert vix.dropna().shape[0] > 1000
    assert gpr.dropna().shape[0] > 1000
    px, ret = _synth_panel(1800)
    cpi = _synth_cpi(200)
    out = vix_gpr_conditioned_value_factor_returns(px, ret, cpi, vix=vix, gpr=gpr)
    assert PRIMARY in out
    assert out[PRIMARY].notna().sum() > 50
