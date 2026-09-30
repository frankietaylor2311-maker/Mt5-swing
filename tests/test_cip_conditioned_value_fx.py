"""Tests for Du–Schreger CIP-conditioned Rogoff PPP / real-FX value module (§83)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.cip_conditioned_value_fx import (
    PRIMARY,
    CipConditionedValueFxConfig,
    cip_conditioned_value_factor_returns,
    cip_stress_from_panel,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
SLIM = MACRO / "cip_g10_5y_govt_panel.csv"


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


def _synth_cip_panel(n_months: int = 120, *, seed: int = 0) -> pd.DataFrame:
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
    assert PRIMARY == "value_low_cip"


def test_cip_stress_is_negative_mean():
    panel = _synth_cip_panel(12)
    stress = cip_stress_from_panel(panel)
    expected = -panel.mean(axis=1)
    assert np.allclose(stress.values, expected.values, equal_nan=True)


def test_gate_direction_low_on_high_off():
    px, ret = _synth_panel(2400)
    cpi = _synth_cpi(240)
    # Low CIP stress first half / high second half
    m_idx = pd.date_range("2006-01-31", periods=240, freq="ME", tz="UTC")
    panel = pd.DataFrame(
        {
            c: np.concatenate([np.full(120, 10.0), np.full(120, -60.0)])
            for c in ("EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")
        },
        index=m_idx,
    )
    cfg = CipConditionedValueFxConfig(
        signal_lag_months=1,
        weight_lag_days=1,
        z_window=60,
        min_periods=24,
        z_high=1.0,
        lookback=60,
    )
    out = cip_conditioned_value_factor_returns(px, ret, cpi, panel, cfg=cfg)
    for key in (
        "value_low_cip",
        "value_cip_cool",
        "value_raw",
        "value_high_cip",
        "cip_stress_haven_usd",
        "value_cip_stack",
        "value_cip_ew",
        "value_cip_regime",
    ):
        assert key in out
    lo = out["value_low_cip"]
    hi = out["value_high_cip"]
    raw = out["value_raw"]
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

    px, ret = _synth_panel(1800)
    cpi = _synth_cpi(200)
    panel = _synth_cip_panel(160, seed=1)
    cfg = CipConditionedValueFxConfig(cool=0.35, z_high=1.0, z_window=36, min_periods=18)
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
    out = cip_conditioned_value_factor_returns(px, ret, cpi, panel, cfg=cfg)
    assert out["value_cip_cool"].notna().sum() > 50


def test_no_lookahead_mutate_future_cip():
    px, ret = _synth_panel(2200)
    cpi = _synth_cpi(220)
    panel = _synth_cip_panel(180, seed=42)
    # Align panel start earlier so stress covers FX window
    panel.index = pd.date_range("2008-01-31", periods=len(panel), freq="ME", tz="UTC")
    cfg = CipConditionedValueFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    f1 = cip_conditioned_value_factor_returns(px, ret, cpi, panel, cfg=cfg)
    panel2 = panel.copy()
    panel2.loc["2018-01-31":] = panel2.loc["2018-01-31":] - 40.0
    f2 = cip_conditioned_value_factor_returns(px, ret, cpi, panel2, cfg=cfg)
    cut = pd.Timestamp("2016-06-30", tz="UTC")
    a = f1["value_low_cip"].loc[:cut].dropna()
    b = f2["value_low_cip"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 100
    assert np.allclose(
        a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12
    )


def test_factor_shapes_match_index():
    px, ret = _synth_panel(1800)
    cpi = _synth_cpi(200)
    panel = _synth_cip_panel(100, seed=7)
    panel.index = pd.date_range("2008-01-31", periods=len(panel), freq="ME", tz="UTC")
    out = cip_conditioned_value_factor_returns(
        px, ret, cpi, panel, cfg=CipConditionedValueFxConfig(z_window=36, min_periods=18)
    )
    assert out["value_low_cip"].index.equals(ret.index)
    assert out["value_cip_ew"].index.equals(ret.index)
    assert out["value_cip_regime"].index.equals(ret.index)


def test_gate_on_off_explicit_z():
    """Primary gate: value on when z≤0; inverse on when z≥z_high."""
    idx = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    z = pd.Series([-1.0, -0.5, 0.0, 0.5, 1.5, 2.0, -0.2, 1.1, 0.0, -2.0], index=idx)
    cfg = CipConditionedValueFxConfig(z_high=1.0)
    gate_lo = (z <= 0.0).astype(float)
    gate_hi = (z >= cfg.z_high).astype(float)
    assert gate_lo.iloc[0] == 1.0
    assert gate_lo.iloc[4] == 0.0
    assert gate_hi.iloc[4] == 1.0
    assert gate_hi.iloc[0] == 0.0


def test_monthly_z_window_prior():
    """Config must use monthly z_window=60 (CIP §68/§71/§82 mirror, not 252d)."""
    cfg = CipConditionedValueFxConfig()
    assert cfg.z_window == 60
    assert cfg.min_periods == 24
    assert cfg.signal_lag_months == 1
    assert cfg.weight_lag_days == 1
    assert cfg.lookback == 60
    assert cfg.n_long == 2
    assert cfg.n_short == 2
    assert cfg.ppp_signal_lag == 1


def test_haven_optional():
    px, ret = _synth_panel(1800)
    cpi = _synth_cpi(200)
    panel = _synth_cip_panel(100, seed=3)
    panel.index = pd.date_range("2008-01-31", periods=len(panel), freq="ME", tz="UTC")
    cfg = CipConditionedValueFxConfig(z_window=36, min_periods=18)
    out = cip_conditioned_value_factor_returns(
        px, ret, cpi, panel, cfg=cfg, include_haven=False
    )
    assert "value_low_cip" in out
    assert "value_cip_cool" in out
    assert "value_raw" in out
    assert "value_high_cip" in out
    assert "cip_stress_haven_usd" not in out
    assert "value_cip_regime" not in out


def test_empty_cpi_returns_empty():
    px, ret = _synth_panel(400)
    panel = _synth_cip_panel(40, seed=1)
    out = cip_conditioned_value_factor_returns(
        px, ret, pd.DataFrame(), panel, cfg=CipConditionedValueFxConfig()
    )
    assert out == {}


@pytest.mark.skipif(not SLIM.exists(), reason="CIP slim panel missing")
def test_load_slim_panel_value():
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
    px, ret = _synth_panel(1800)
    cpi = _synth_cpi(200)
    out = cip_conditioned_value_factor_returns(px, ret, cpi, lp)
    assert PRIMARY in out
    assert out[PRIMARY].notna().sum() > 20
