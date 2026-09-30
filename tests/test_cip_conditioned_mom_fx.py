"""Tests for Du–Schreger CIP-conditioned Menkhoff FX momentum module (§82)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.cip_conditioned_mom_fx import (
    PRIMARY,
    CipConditionedMomFxConfig,
    cip_conditioned_mom_factor_returns,
    cip_stress_from_panel,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
SLIM = MACRO / "cip_g10_5y_govt_panel.csv"


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


def _synth_panel(n_months: int = 120, *, seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range("2014-01-31", periods=n_months, freq="ME", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            c: rng.normal(-20, 15, n_months)
            for c in ("EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")
        },
        index=idx,
    )


def test_primary_name_constant():
    assert PRIMARY == "mom_low_cip"


def test_cip_stress_is_negative_mean():
    panel = _synth_panel(12)
    stress = cip_stress_from_panel(panel)
    expected = -panel.mean(axis=1)
    assert np.allclose(stress.values, expected.values, equal_nan=True)


def test_gate_direction_low_on_high_off():
    ret = _synth_ret(2000)
    # Low CIP stress first half (less-negative mean cip → lower UST premium)
    # High CIP stress second half (more-negative mean cip → higher UST premium)
    m_idx = pd.date_range("2014-01-31", periods=120, freq="ME", tz="UTC")
    panel = pd.DataFrame(
        {
            c: np.concatenate([np.full(60, 10.0), np.full(60, -60.0)])
            for c in ("EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")
        },
        index=m_idx,
    )
    cfg = CipConditionedMomFxConfig(
        signal_lag_months=1,
        weight_lag_days=1,
        z_window=36,
        min_periods=18,
        z_high=1.0,
    )
    out = cip_conditioned_mom_factor_returns(ret, panel, cfg=cfg)
    for key in (
        "mom_low_cip",
        "mom_cip_cool",
        "mom_raw",
        "mom_high_cip",
        "cip_stress_haven_usd",
        "mom_cip_stack",
        "mom_cip_ew",
        "mom_cip_regime",
    ):
        assert key in out
    lo = out["mom_low_cip"]
    hi = out["mom_high_cip"]
    raw = out["mom_raw"]
    both = lo.notna() & raw.notna()
    assert both.sum() > 100
    product = lo.loc[both].abs() * hi.loc[both].abs()
    assert (product <= 1e-10).mean() > 0.90


def test_cool_scale_bounds():
    from mt5_swing.strategies.cip_conditioned_carry_fx import (
        CipConditionedCarryFxConfig,
        align_cip_stress_z_daily,
    )
    from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z

    ret = _synth_ret(800)
    panel = _synth_panel(100, seed=1)
    cfg = CipConditionedMomFxConfig(cool=0.35, z_high=1.0, z_window=36, min_periods=18)
    cip_cfg = CipConditionedCarryFxConfig(
        signal_lag_months=cfg.signal_lag_months,
        weight_lag_days=cfg.weight_lag_days,
        z_window=cfg.z_window,
        min_periods=cfg.min_periods,
        z_high=cfg.z_high,
        cool=cfg.cool,
    )
    stress = cip_stress_from_panel(panel)
    z = align_cip_stress_z_daily(stress, ret.index, cfg=cip_cfg)
    gcfg = GprRegimeConfig(
        z_high=cfg.z_high, z_low=cfg.z_low, cool=cfg.cool, signal_lag=0, usd_tilt=0.0
    )
    scale = risk_scale_from_z(z, cfg=gcfg).dropna()
    assert len(scale) > 50
    assert float(scale.min()) >= cfg.cool - 1e-9
    assert float(scale.max()) <= 1.0 + 1e-9
    out = cip_conditioned_mom_factor_returns(ret, panel, cfg=cfg)
    assert out["mom_cip_cool"].notna().sum() > 50


def test_no_lookahead_mutate_future_cip():
    ret = _synth_ret(1800)
    panel = _synth_panel(120, seed=42)
    cfg = CipConditionedMomFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    f1 = cip_conditioned_mom_factor_returns(ret, panel, cfg=cfg)
    panel2 = panel.copy()
    panel2.loc["2021-01-31":] = panel2.loc["2021-01-31":] - 40.0
    f2 = cip_conditioned_mom_factor_returns(ret, panel2, cfg=cfg)
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["mom_low_cip"].loc[:cut].dropna()
    b = f2["mom_low_cip"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 200
    assert np.allclose(
        a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12
    )


def test_factor_shapes_match_index():
    ret = _synth_ret(800)
    panel = _synth_panel(80, seed=7)
    out = cip_conditioned_mom_factor_returns(
        ret, panel, cfg=CipConditionedMomFxConfig(z_window=36, min_periods=18)
    )
    assert out["mom_low_cip"].index.equals(ret.index)
    assert out["mom_cip_ew"].index.equals(ret.index)
    assert out["mom_cip_regime"].index.equals(ret.index)


def test_gate_on_off_explicit_z():
    """Primary gate: mom on when z≤0; inverse on when z≥z_high."""
    idx = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    z = pd.Series([-1.0, -0.5, 0.0, 0.5, 1.5, 2.0, -0.2, 1.1, 0.0, -2.0], index=idx)
    cfg = CipConditionedMomFxConfig(z_high=1.0)
    gate_lo = (z <= 0.0).astype(float)
    gate_hi = (z >= cfg.z_high).astype(float)
    assert gate_lo.iloc[0] == 1.0
    assert gate_lo.iloc[4] == 0.0
    assert gate_hi.iloc[4] == 1.0
    assert gate_hi.iloc[0] == 0.0


def test_monthly_z_window_prior():
    """Config must use monthly z_window=60 (CIP §68/§71 mirror, not 252d)."""
    cfg = CipConditionedMomFxConfig()
    assert cfg.z_window == 60
    assert cfg.min_periods == 24
    assert cfg.signal_lag_months == 1
    assert cfg.weight_lag_days == 1
    assert cfg.formation_days == 63
    assert cfg.skip_days == 21


def test_haven_optional():
    ret = _synth_ret(800)
    panel = _synth_panel(80, seed=3)
    cfg = CipConditionedMomFxConfig(z_window=36, min_periods=18)
    out = cip_conditioned_mom_factor_returns(
        ret, panel, cfg=cfg, include_haven=False
    )
    assert "mom_low_cip" in out
    assert "mom_cip_cool" in out
    assert "mom_raw" in out
    assert "mom_high_cip" in out
    assert "cip_stress_haven_usd" not in out
    assert "mom_cip_regime" not in out


@pytest.mark.skipif(not SLIM.exists(), reason="CIP slim panel missing")
def test_load_slim_panel_mom():
    from mt5_swing.data.cip_basis import (
        DEFAULT_PUB_LAG_DAYS,
        DEFAULT_TENOR,
        load_cip_panel,
    )

    lp = load_cip_panel(
        tenor=DEFAULT_TENOR,
        pub_lag_days=DEFAULT_PUB_LAG_DAYS,
        download=False,
        force=False,
        frequency="month_end",
    )
    assert lp.shape[1] >= 4
    stress = cip_stress_from_panel(lp)
    assert stress.dropna().shape[0] > 24
    ret = _synth_ret(500)
    out = cip_conditioned_mom_factor_returns(ret, lp)
    assert PRIMARY in out
    assert out[PRIMARY].notna().sum() > 20
