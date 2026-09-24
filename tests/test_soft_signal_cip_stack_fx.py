"""Smoke tests for §69 soft-signal CIP-enriched EW stack."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mt5_swing.strategies.cip_basis_fx import CipBasisFxConfig, cip_basis_factor_returns
from mt5_swing.strategies.soft_signal_cip_stack_fx import (
    BOARD_SPECS,
    CIP_SOFT2,
    MACRO_SOFT5,
    PRIMARY,
    SOFT_LEGS_CIP,
    SoftSignalCipStackConfig,
    equal_weight_daily,
    soft_signal_cip_stack_factor_returns,
)
from mt5_swing.strategies.soft_signal_stack_fx import SOFT_LEGS as SOFT_LEGS_66


def test_soft_legs_cip_preregistered():
    assert SOFT_LEGS_CIP == [
        "bci_chg_xs",
        "high_ip_xs",
        "high_ppi_xs",
        "low_gdp_xs",
        "low_cars_xs",
        "ust_premium_haven_usd",
        "cip_ew",
    ]
    assert PRIMARY == "soft_ew7_cip"
    assert "soft_ew7_cip" in BOARD_SPECS
    assert set(BOARD_SPECS["soft_ew7_cip"]) == set(SOFT_LEGS_CIP)
    # One haven seed only — no cip_stress_haven_usd
    assert "cip_stress_haven_usd" not in SOFT_LEGS_CIP
    assert "cip_carry_ew" not in SOFT_LEGS_CIP
    assert CIP_SOFT2 == ["ust_premium_haven_usd", "cip_ew"]
    assert MACRO_SOFT5 == list(SOFT_LEGS_66)


def test_distinct_from_section_66_soft_legs():
    """§69 must not mutate §66 SOFT_LEGS; CIP legs are additive in sibling module."""
    assert SOFT_LEGS_66 == [
        "bci_chg_xs",
        "high_ip_xs",
        "high_ppi_xs",
        "low_gdp_xs",
        "low_cars_xs",
    ]
    assert set(SOFT_LEGS_66).issubset(set(SOFT_LEGS_CIP))
    assert set(SOFT_LEGS_CIP) - set(SOFT_LEGS_66) == {
        "ust_premium_haven_usd",
        "cip_ew",
    }
    assert PRIMARY != "soft_ew5"


def test_equal_weight_nanmean_min_legs():
    idx = pd.date_range("2020-01-01", periods=5, freq="B", tz="UTC")
    a = pd.Series([0.01, 0.02, np.nan, 0.04, 0.05], index=idx, name="a")
    b = pd.Series([0.03, 0.00, 0.01, np.nan, 0.01], index=idx, name="b")
    c = pd.Series([np.nan, np.nan, np.nan, np.nan, 0.00], index=idx, name="c")
    legs = {"a": a, "b": b, "c": c}
    ew = equal_weight_daily(legs, ["a", "b", "c"], min_legs=2, name="toy")
    assert ew.name == "toy"
    assert abs(float(ew.iloc[0]) - 0.02) < 1e-12
    assert abs(float(ew.iloc[1]) - 0.01) < 1e-12
    assert np.isnan(ew.iloc[2])
    assert np.isnan(ew.iloc[3])
    assert abs(float(ew.iloc[4]) - 0.02) < 1e-12


def test_primary_name_and_board_keys():
    idx = pd.date_range("2021-01-01", periods=4, freq="B", tz="UTC")
    legs = {k: pd.Series([0.01, 0.02, 0.03, 0.04], index=idx) for k in SOFT_LEGS_CIP}
    board = soft_signal_cip_stack_factor_returns(
        legs, cfg=SoftSignalCipStackConfig(min_legs=2)
    )
    assert set(board.keys()) == set(BOARD_SPECS.keys())
    assert PRIMARY in board
    assert abs(float(board[PRIMARY].iloc[0]) - 0.01) < 1e-12
    # macro5 baseline equals mean of five macro legs
    macro_mean = float(
        np.mean([legs[k].iloc[0] for k in MACRO_SOFT5])
    )
    assert abs(float(board["soft_ew_macro5"].iloc[0]) - macro_mean) < 1e-12
    # cip2 = mean of two CIP softs
    cip_mean = float(np.mean([legs[k].iloc[0] for k in CIP_SOFT2]))
    assert abs(float(board["soft_ew_cip2"].iloc[0]) - cip_mean) < 1e-12


def test_leave_one_out_excludes_cip_legs():
    idx = pd.date_range("2022-01-03", periods=3, freq="B", tz="UTC")
    # Distinct values so exclusion changes the mean
    vals = {
        "bci_chg_xs": 0.10,
        "high_ip_xs": 0.20,
        "high_ppi_xs": 0.30,
        "low_gdp_xs": 0.40,
        "low_cars_xs": 0.50,
        "ust_premium_haven_usd": 0.60,
        "cip_ew": 0.70,
    }
    legs = {k: pd.Series([v, v, v], index=idx) for k, v in vals.items()}
    board = soft_signal_cip_stack_factor_returns(legs)
    no_haven = [x for x in SOFT_LEGS_CIP if x != "ust_premium_haven_usd"]
    no_cip_ew = [x for x in SOFT_LEGS_CIP if x != "cip_ew"]
    exp_no_haven = float(np.mean([vals[k] for k in no_haven]))
    exp_no_cip = float(np.mean([vals[k] for k in no_cip_ew]))
    assert abs(float(board["soft_ew6_no_haven"].iloc[0]) - exp_no_haven) < 1e-12
    assert abs(float(board["soft_ew6_no_cip_ew"].iloc[0]) - exp_no_cip) < 1e-12
    assert "ust_premium_haven_usd" not in BOARD_SPECS["soft_ew6_no_haven"]
    assert "cip_ew" not in BOARD_SPECS["soft_ew6_no_cip_ew"]
    # growth_cip / honesty_cip membership
    assert set(BOARD_SPECS["soft_ew_growth_cip"]) == set(
        ["bci_chg_xs", "high_ip_xs", "high_ppi_xs", "ust_premium_haven_usd", "cip_ew"]
    )
    assert set(BOARD_SPECS["soft_ew_honesty_cip"]) == set(
        ["low_gdp_xs", "low_cars_xs", "ust_premium_haven_usd", "cip_ew"]
    )


def test_no_lookahead_mutate_future_cip_and_macro():
    """Mutating future CIP panel / macro legs must not change earlier stack returns."""
    idx = pd.date_range("2016-01-01", periods=1800, freq="B", tz="UTC")
    rng = np.random.default_rng(69)
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
    cfg_cip = CipBasisFxConfig(signal_lag=1, cost_bps_side=1.5)
    cip1 = cip_basis_factor_returns(ret, panel, cfg=cfg_cip)
    assert "ust_premium_haven_usd" in cip1 and "cip_ew" in cip1

    # Synthetic macro legs (independent of CIP) — causal daily noise
    macro = {
        k: pd.Series(rng.normal(0, 0.001, len(idx)), index=idx, name=k)
        for k in MACRO_SOFT5
    }
    legs1 = {**macro, "ust_premium_haven_usd": cip1["ust_premium_haven_usd"], "cip_ew": cip1["cip_ew"]}
    board1 = soft_signal_cip_stack_factor_returns(legs1)

    # Mutate future CIP panel
    panel2 = panel.copy()
    panel2.loc["2021-01-31":] = panel2.loc["2021-01-31":] - 40.0
    cip2 = cip_basis_factor_returns(ret, panel2, cfg=cfg_cip)
    # Mutate future macro legs
    macro2 = {k: s.copy() for k, s in macro.items()}
    cut_macro = pd.Timestamp("2020-01-01", tz="UTC")
    for k in macro2:
        macro2[k].loc[cut_macro:] = macro2[k].loc[cut_macro:] + 0.05

    legs2 = {
        **macro2,
        "ust_premium_haven_usd": cip2["ust_premium_haven_usd"],
        "cip_ew": cip2["cip_ew"],
    }
    board2 = soft_signal_cip_stack_factor_returns(legs2)

    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = board1[PRIMARY].loc[:cut].dropna()
    b = board2[PRIMARY].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 200
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)
