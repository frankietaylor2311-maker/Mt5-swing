"""Smoke tests for §66 Dahlquist-style soft-signal EW stack."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mt5_swing.strategies.soft_signal_stack_fx import (
    BOARD_SPECS,
    PRIMARY,
    SOFT_LEGS,
    SoftSignalStackConfig,
    equal_weight_daily,
    soft_signal_stack_factor_returns,
)


def test_soft_legs_preregistered():
    assert SOFT_LEGS == [
        "bci_chg_xs",
        "high_ip_xs",
        "high_ppi_xs",
        "low_gdp_xs",
        "low_cars_xs",
    ]
    assert PRIMARY == "soft_ew5"
    assert "soft_ew5" in BOARD_SPECS
    assert set(BOARD_SPECS["soft_ew5"]) == set(SOFT_LEGS)
    # One BCI seed only — no high_bci_z_xs in stack
    assert "high_bci_z_xs" not in SOFT_LEGS
    assert "bci_chg_xs" in SOFT_LEGS


def test_equal_weight_toy_series():
    idx = pd.date_range("2020-01-01", periods=5, freq="B", tz="UTC")
    a = pd.Series([0.01, 0.02, np.nan, 0.04, 0.05], index=idx, name="a")
    b = pd.Series([0.03, 0.00, 0.01, np.nan, 0.01], index=idx, name="b")
    c = pd.Series([np.nan, np.nan, np.nan, np.nan, 0.00], index=idx, name="c")
    legs = {"a": a, "b": b, "c": c}
    ew = equal_weight_daily(legs, ["a", "b", "c"], min_legs=2, name="toy")
    assert ew.name == "toy"
    # day0: a,b → 0.02; day1: a,b → 0.01; day2: only b → NaN (<2);
    # day3: only a → NaN; day4: a,b,c → mean(0.05,0.01,0.00)=0.02
    assert abs(float(ew.iloc[0]) - 0.02) < 1e-12
    assert abs(float(ew.iloc[1]) - 0.01) < 1e-12
    assert np.isnan(ew.iloc[2])
    assert np.isnan(ew.iloc[3])
    assert abs(float(ew.iloc[4]) - 0.02) < 1e-12


def test_nan_handling_and_min_legs():
    idx = pd.date_range("2021-01-01", periods=3, freq="B", tz="UTC")
    legs = {
        "bci_chg_xs": pd.Series([0.1, np.nan, 0.3], index=idx),
        "high_ip_xs": pd.Series([np.nan, np.nan, 0.1], index=idx),
        "high_ppi_xs": pd.Series([0.2, 0.2, np.nan], index=idx),
        "low_gdp_xs": pd.Series([np.nan, np.nan, np.nan], index=idx),
        "low_cars_xs": pd.Series([np.nan, 0.4, 0.5], index=idx),
    }
    board = soft_signal_stack_factor_returns(legs, cfg=SoftSignalStackConfig(min_legs=2))
    assert set(board.keys()) == set(BOARD_SPECS.keys())
    # soft_ew5 day0: bci+ppi → ok; day1: ppi+cars → ok; day2: bci+ip+cars → ok
    assert board["soft_ew5"].notna().sum() == 3
    # soft_ew_honesty needs low_gdp + low_cars; day0 none; day1 only cars; day2 only cars
    # with min_legs=2 → all NaN when low_gdp always missing
    assert board["soft_ew_honesty"].notna().sum() == 0
    # soft_ew_growth day0: bci+ppi; day1: only ppi → NaN; day2: bci+ip
    assert board["soft_ew_growth"].notna().iloc[0]
    assert np.isnan(board["soft_ew_growth"].iloc[1])
    assert board["soft_ew_growth"].notna().iloc[2]


def test_leave_one_out_excludes_named_leg():
    idx = pd.date_range("2022-01-03", periods=2, freq="B", tz="UTC")
    legs = {k: pd.Series([0.01, 0.02], index=idx) for k in SOFT_LEGS}
    board = soft_signal_stack_factor_returns(legs)
    # no_bci should equal EW of the other 4 (all present → mean of 4)
    expected = legs["high_ip_xs"]  # all equal 0.01/0.02 so any subset mean matches
    assert abs(float(board["soft_ew4_no_bci"].iloc[0]) - float(expected.iloc[0])) < 1e-12
    assert abs(float(board["soft_ew5"].iloc[0]) - 0.01) < 1e-12
