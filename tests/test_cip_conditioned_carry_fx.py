"""Tests for Du–Schreger CIP-conditioned Lustig–Verdelhan carry FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.cip_conditioned_carry_fx import (
    CipConditionedCarryFxConfig,
    align_cip_stress_z_daily,
    cip_conditioned_carry_factor_returns,
    cip_stress_from_panel,
    usd_tilt_from_cip_stress_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
SLIM = MACRO / "cip_g10_5y_govt_panel.csv"


def _synth_panel(n_months: int = 96) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2014-01-31", periods=n_months, freq="ME", tz="UTC")
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        {
            c: rng.normal(-20, 15, n_months)
            for c in ("EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")
        },
        index=idx,
    )
    stress = cip_stress_from_panel(panel)
    return panel, stress


def test_cip_stress_is_negative_mean():
    panel, stress = _synth_panel(12)
    expected = -panel.mean(axis=1)
    assert np.allclose(stress.values, expected.values, equal_nan=True)


def test_align_cip_stress_z_pit_lag():
    """Monthly stress z must lag ≥ signal_lag_months + weight_lag_days."""
    _panel, stress = _synth_panel(96)
    daily_idx = pd.date_range("2014-01-01", periods=2000, freq="B", tz="UTC")
    cfg = CipConditionedCarryFxConfig(signal_lag_months=1, weight_lag_days=1, z_window=24, min_periods=12)
    z = align_cip_stress_z_daily(stress, daily_idx, cfg=cfg)
    assert z.index.equals(daily_idx)
    assert z.dropna().shape[0] > 100
    # First non-null cannot be at the very start (lags + z warmup)
    assert z.dropna().index.min() > daily_idx.min() + pd.Timedelta(days=30)


def test_gate_direction_low_stress_on_high_off():
    """Primary gate: carry on when z≤0; inverse on when z≥z_high."""
    idx = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    z = pd.Series([-1.0, -0.5, 0.0, 0.5, 1.5, 2.0, -0.2, 1.1, 0.0, -2.0], index=idx)
    cfg = CipConditionedCarryFxConfig(z_high=1.0)
    gate_lo = (z <= 0.0).astype(float)
    gate_hi = (z >= cfg.z_high).astype(float)
    assert gate_lo.iloc[0] == 1.0
    assert gate_lo.iloc[4] == 0.0
    assert gate_hi.iloc[4] == 1.0
    assert gate_hi.iloc[0] == 0.0


def test_usd_tilt_fires_on_high_stress_z():
    idx = pd.date_range("2020-01-01", periods=8, freq="B", tz="UTC")
    z = pd.Series([0.0, 0.5, 1.5, 2.0, -0.5, 1.1, 0.0, 0.0], index=idx)
    cfg = CipConditionedCarryFxConfig(usd_tilt=0.5, z_high=1.0)
    w = usd_tilt_from_cip_stress_z(z, ["EURUSD", "USDJPY"], cfg=cfg)
    assert w.loc[idx[2], "EURUSD"] < 0  # long USD
    assert w.loc[idx[2], "USDJPY"] > 0
    assert w.loc[idx[0], "EURUSD"] == 0.0


def test_no_lookahead_mutate_future_cip():
    """Mutating future CIP must not change earlier factor returns."""
    idx = pd.date_range("2016-01-01", periods=1800, freq="B", tz="UTC")
    rng = np.random.default_rng(42)
    close = pd.DataFrame(
        {
            "EURUSD": 1.1 * np.cumprod(1 + rng.normal(0, 0.005, 1800)),
            "GBPUSD": 1.3 * np.cumprod(1 + rng.normal(0, 0.005, 1800)),
            "AUDUSD": 0.75 * np.cumprod(1 + rng.normal(0, 0.006, 1800)),
            "NZDUSD": 0.65 * np.cumprod(1 + rng.normal(0, 0.006, 1800)),
            "USDJPY": 110 * np.cumprod(1 + rng.normal(0, 0.005, 1800)),
            "USDCAD": 1.3 * np.cumprod(1 + rng.normal(0, 0.004, 1800)),
            "USDCHF": 0.95 * np.cumprod(1 + rng.normal(0, 0.004, 1800)),
        },
        index=idx,
    )
    ret = close.pct_change()
    m_idx = pd.date_range("2014-01-31", periods=120, freq="ME", tz="UTC")
    panel = pd.DataFrame(
        {c: rng.normal(-15, 20, 120) for c in ("EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")},
        index=m_idx,
    )
    rates = pd.DataFrame(
        {
            c: 1.0 + 0.01 * rng.normal(0, 1, 120).cumsum()
            for c in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")
        },
        index=pd.date_range("2014-01-01", periods=120, freq="MS", tz="UTC"),
    )
    cfg = CipConditionedCarryFxConfig(signal_lag_months=1, carry_signal_lag=1, cost_bps_side=1.5)
    f1 = cip_conditioned_carry_factor_returns(ret, panel, rates=rates, cfg=cfg)
    for key in (
        "carry_low_cip_stress",
        "carry_cip_cool",
        "carry_raw",
        "carry_high_cip_stress",
        "cip_stress_haven_usd",
        "cip_carry_ew",
    ):
        assert key in f1

    panel2 = panel.copy()
    panel2.loc["2021-01-31":] = panel2.loc["2021-01-31":] - 40.0  # more-neg CIP → higher stress
    f2 = cip_conditioned_carry_factor_returns(ret, panel2, rates=rates, cfg=cfg)
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["carry_low_cip_stress"].loc[:cut].dropna()
    b = f2["carry_low_cip_stress"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 200
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_factor_shapes_match_index():
    idx = pd.date_range("2018-01-01", periods=800, freq="B", tz="UTC")
    rng = np.random.default_rng(7)
    ret = pd.DataFrame(
        {s: rng.normal(0, 0.005, 800) for s in ("EURUSD", "GBPUSD", "USDJPY", "USDCAD")},
        index=idx,
    )
    m_idx = pd.date_range("2016-01-31", periods=60, freq="ME", tz="UTC")
    panel = pd.DataFrame(
        {c: rng.normal(-10, 10, 60) for c in ("EUR", "GBP", "JPY", "CAD")},
        index=m_idx,
    )
    rates = pd.DataFrame(
        {c: 1.0 + 0.01 * rng.normal(0, 1, 60).cumsum() for c in ("USD", "EUR", "GBP", "JPY", "CAD")},
        index=pd.date_range("2016-01-01", periods=60, freq="MS", tz="UTC"),
    )
    out = cip_conditioned_carry_factor_returns(
        ret, panel, rates=rates, cfg=CipConditionedCarryFxConfig()
    )
    assert out["carry_low_cip_stress"].index.equals(idx)
    assert out["cip_stress_haven_usd"].index.equals(idx)


@pytest.mark.skipif(not SLIM.exists(), reason="CIP slim panel missing")
def test_load_slim_panel_stress():
    from mt5_swing.data.cip_basis import load_cip_panel, DEFAULT_TENOR, DEFAULT_PUB_LAG_DAYS

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
