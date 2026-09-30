"""Tests for VIX/GPR-conditioned BIS REER HML-FX value module (§81)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.vix_gpr_conditioned_reer_value_fx import (
    PRIMARY,
    VixGprConditionedReerValueFxConfig,
    vix_gpr_conditioned_reer_value_factor_returns,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
VIX_PATH = MACRO / "vix_yahoo.csv"
GPR_PATH = MACRO / "gpr_daily.csv"
REER_USD = MACRO / "fred_RBUSBIS.csv"


def _synth_pairs(n_days: int = 1800) -> pd.DataFrame:
    idx = pd.date_range("2016-01-01", periods=n_days, freq="B", tz="UTC")
    rng = np.random.default_rng(17)
    cols = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
    data = {c: rng.normal(0.0, 0.005, n_days) for c in cols}
    return pd.DataFrame(data, index=idx)


def _synth_reer(
    start: str = "2010-01-01",
    n_months: int = 180,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n_months, freq="MS", tz="UTC")
    rng = np.random.default_rng(seed)
    cols = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
    data = {c: 100.0 + np.cumsum(rng.normal(0.0, 0.8, n_months)) for c in cols}
    return pd.DataFrame(data, index=idx)


def _synth_macro(n_days: int = 3000, *, seed: int = 0, mean: float = 15.0) -> pd.Series:
    idx = pd.date_range("2014-01-01", periods=n_days, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, 5.0, n_days), index=idx, name="macro")


def test_primary_name_constant():
    assert PRIMARY == "reer_low_vix"


def test_gate_direction_low_on_high_off():
    pret = _synth_pairs(2000)
    reer = _synth_reer(n_months=200, seed=1)
    idx = pd.date_range("2014-01-01", periods=3000, freq="B", tz="UTC")
    vix = pd.Series(
        np.concatenate([np.full(1500, 10.0), np.full(1500, 40.0)]),
        index=idx,
        name="vix",
    )
    gpr = _synth_macro(3000, seed=3, mean=100.0)
    gpr.name = "gpr"
    cfg = VixGprConditionedReerValueFxConfig(
        signal_lag=1, z_window=252, min_periods=60, z_high=1.0
    )
    out = vix_gpr_conditioned_reer_value_factor_returns(
        pret, reer, vix=vix, gpr=gpr, cfg=cfg
    )
    for key in (
        "reer_low_vix",
        "reer_vix_cool",
        "reer_low_gpr",
        "reer_gpr_cool",
        "reer_raw",
        "reer_high_vix",
        "reer_vix_gpr_stack",
        "reer_vix_gpr_ew",
    ):
        assert key in out
    lo = out["reer_low_vix"]
    raw = out["reer_raw"]
    both = lo.notna() & raw.notna()
    assert both.sum() > 100
    assert (lo.loc[both].abs() <= raw.loc[both].abs() + 1e-12).mean() > 0.95
    hi = out["reer_high_vix"]
    product = lo.loc[both].abs() * hi.loc[both].abs()
    assert (product <= 1e-12).mean() > 0.95


def test_cool_scale_bounds():
    """Scale bounds on risk_scale_from_z (return ratios deviate when costs reapplied)."""
    from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z
    from mt5_swing.strategies.vix_gpr_conditioned_soft_fx import (
        VixGprConditionedSoftFxConfig,
        align_stress_z_daily,
    )

    pret = _synth_pairs(800)
    reer = _synth_reer(n_months=140, seed=2)
    vix = _synth_macro(2500, seed=1, mean=18.0)
    vix.name = "vix"
    gpr = _synth_macro(2500, seed=2, mean=90.0)
    gpr.name = "gpr"
    cfg = VixGprConditionedReerValueFxConfig(cool=0.35, z_high=1.0)
    soft_cfg = VixGprConditionedSoftFxConfig(
        signal_lag=cfg.signal_lag,
        z_window=cfg.z_window,
        min_periods=cfg.min_periods,
        z_high=cfg.z_high,
        cool=cfg.cool,
    )
    z = align_stress_z_daily(vix, pret.index, cfg=soft_cfg)
    gcfg = GprRegimeConfig(
        z_high=cfg.z_high, z_low=cfg.z_low, cool=cfg.cool, signal_lag=0, usd_tilt=0.0
    )
    scale = risk_scale_from_z(z, cfg=gcfg).dropna()
    assert len(scale) > 50
    assert float(scale.min()) >= cfg.cool - 1e-9
    assert float(scale.max()) <= 1.0 + 1e-9
    out = vix_gpr_conditioned_reer_value_factor_returns(
        pret, reer, vix=vix, gpr=gpr, cfg=cfg
    )
    assert out["reer_vix_cool"].notna().sum() > 50


def test_stack_scales_compound():
    from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z
    from mt5_swing.strategies.vix_gpr_conditioned_soft_fx import (
        VixGprConditionedSoftFxConfig,
        align_stress_z_daily,
    )

    pret = _synth_pairs(800)
    reer = _synth_reer(n_months=140, seed=5)
    vix = _synth_macro(2500, seed=5, mean=20.0)
    vix.name = "vix"
    gpr = _synth_macro(2500, seed=6, mean=110.0)
    gpr.name = "gpr"
    cfg = VixGprConditionedReerValueFxConfig(cool=0.35, z_high=1.0)
    soft_cfg = VixGprConditionedSoftFxConfig(
        signal_lag=cfg.signal_lag,
        z_window=cfg.z_window,
        min_periods=cfg.min_periods,
        z_high=cfg.z_high,
        cool=cfg.cool,
    )
    z_vix = align_stress_z_daily(vix, pret.index, cfg=soft_cfg)
    z_gpr = align_stress_z_daily(gpr, pret.index, cfg=soft_cfg)
    gcfg = GprRegimeConfig(
        z_high=cfg.z_high, z_low=cfg.z_low, cool=cfg.cool, signal_lag=0, usd_tilt=0.0
    )
    vix_s = risk_scale_from_z(z_vix, cfg=gcfg)
    gpr_s = risk_scale_from_z(z_gpr, cfg=gcfg)
    stack_s = (vix_s * gpr_s).clip(lower=cfg.cool ** 2, upper=1.0).dropna()
    assert len(stack_s) > 50
    assert float(stack_s.min()) >= cfg.cool ** 2 - 1e-9
    assert float(stack_s.max()) <= 1.0 + 1e-9
    out = vix_gpr_conditioned_reer_value_factor_returns(
        pret, reer, vix=vix, gpr=gpr, cfg=cfg
    )
    assert out["reer_vix_gpr_stack"].notna().sum() > 50


def test_no_lookahead_mutate_future_vix():
    pret = _synth_pairs(1800)
    reer = _synth_reer(start="2012-01-01", n_months=180, seed=4)
    rng = np.random.default_rng(42)
    idx = pd.date_range("2014-01-01", periods=2800, freq="B", tz="UTC")
    vix = pd.Series(rng.normal(18, 6, 2800), index=idx, name="vix")
    gpr = pd.Series(rng.normal(100, 30, 2800), index=idx, name="gpr")
    cfg = VixGprConditionedReerValueFxConfig(
        signal_lag=1, z_window=252, min_periods=60
    )
    f1 = vix_gpr_conditioned_reer_value_factor_returns(
        pret, reer, vix=vix, gpr=gpr, cfg=cfg
    )
    vix2 = vix.copy()
    vix2.loc["2021-01-01":] = vix2.loc["2021-01-01":] + 25.0
    f2 = vix_gpr_conditioned_reer_value_factor_returns(
        pret, reer, vix=vix2, gpr=gpr, cfg=cfg
    )
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["reer_low_vix"].loc[:cut].dropna()
    b = f2["reer_low_vix"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 50
    assert np.allclose(
        a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12
    )


def test_no_lookahead_mutate_future_reer():
    """Future REER mutation must not change past value signals."""
    pret = _synth_pairs(1800)
    reer = _synth_reer(start="2012-01-01", n_months=180, seed=8)
    vix = _synth_macro(2800, seed=11, mean=18.0)
    vix.name = "vix"
    gpr = _synth_macro(2800, seed=12, mean=100.0)
    gpr.name = "gpr"
    cfg = VixGprConditionedReerValueFxConfig()
    f1 = vix_gpr_conditioned_reer_value_factor_returns(
        pret, reer, vix=vix, gpr=gpr, cfg=cfg
    )
    reer2 = reer.copy()
    reer2.loc["2021-01-01":] = reer2.loc["2021-01-01":] * 1.15
    f2 = vix_gpr_conditioned_reer_value_factor_returns(
        pret, reer2, vix=vix, gpr=gpr, cfg=cfg
    )
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["reer_raw"].loc[:cut].dropna()
    b = f2["reer_raw"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 50
    assert np.allclose(
        a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12
    )


def test_factor_shapes_match_pair_index():
    pret = _synth_pairs(800)
    reer = _synth_reer(n_months=140, seed=7)
    vix = _synth_macro(2500, seed=7)
    vix.name = "vix"
    gpr = _synth_macro(2500, seed=8, mean=95.0)
    gpr.name = "gpr"
    out = vix_gpr_conditioned_reer_value_factor_returns(
        pret, reer, vix=vix, gpr=gpr, cfg=VixGprConditionedReerValueFxConfig()
    )
    assert out["reer_low_vix"].index.equals(pret.index)
    assert out["reer_vix_gpr_ew"].index.equals(pret.index)


def test_gate_on_off_explicit_z():
    idx = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    z = pd.Series([-1.0, -0.5, 0.0, 0.5, 1.5, 2.0, -0.2, 1.1, 0.0, -2.0], index=idx)
    cfg = VixGprConditionedReerValueFxConfig(z_high=1.0)
    gate_lo = (z <= 0.0).astype(float)
    gate_hi = (z >= cfg.z_high).astype(float)
    assert gate_lo.iloc[0] == 1.0
    assert gate_lo.iloc[4] == 0.0
    assert gate_hi.iloc[4] == 1.0
    assert gate_hi.iloc[0] == 0.0


@pytest.mark.skipif(
    not (VIX_PATH.exists() and GPR_PATH.exists() and REER_USD.exists()),
    reason="VIX/GPR/REER CSV missing",
)
def test_load_real_vix_gpr_reer():
    from mt5_swing.data.fred_bis_reer import load_bis_reer_panel
    from mt5_swing.strategies.vix_gpr_conditioned_reer_value_fx import (
        load_gpr_series,
        load_vix_series,
    )

    vix = load_vix_series(bar_lag=1)
    gpr = load_gpr_series(bar_lag=1)
    reer = load_bis_reer_panel(pub_lag_months=2, download=False, force=False)
    pret = _synth_pairs(500)
    assert vix.dropna().shape[0] > 1000
    assert gpr.dropna().shape[0] > 1000
    assert reer.dropna(how="all").shape[0] > 60
    out = vix_gpr_conditioned_reer_value_factor_returns(
        pret, reer, vix=vix, gpr=gpr
    )
    assert PRIMARY in out
    assert out[PRIMARY].notna().sum() > 20
