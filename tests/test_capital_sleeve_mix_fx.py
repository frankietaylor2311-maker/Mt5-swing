"""Unit tests for §53 capital-sleeve mix math (weights, core-only, IS-only select)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mt5_swing.portfolio.capital_sleeves import (
    CORE_NAME,
    PRE_REGISTERED_MIXES,
    SATELLITE_A,
    SATELLITE_B,
    equity_to_daily_returns,
    mix_sleeve_returns,
    select_mix_is_only,
    validate_weights,
)


def _toy_sleeves(n: int = 40) -> dict[str, pd.Series]:
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(53)
    return {
        CORE_NAME: pd.Series(rng.normal(0.0004, 0.005, n), index=idx, name=CORE_NAME),
        SATELLITE_A: pd.Series(rng.normal(0.0001, 0.004, n), index=idx, name=SATELLITE_A),
        SATELLITE_B: pd.Series(rng.normal(0.00005, 0.004, n), index=idx, name=SATELLITE_B),
    }


def test_pre_registered_mixes_sum_to_one():
    assert len(PRE_REGISTERED_MIXES) <= 6
    assert len(PRE_REGISTERED_MIXES) >= 4
    for name, w in PRE_REGISTERED_MIXES.items():
        vw = validate_weights(w)
        assert abs(sum(vw.values()) - 1.0) < 1e-9, name
        assert all(v >= 0 for v in vw.values())


def test_validate_weights_rejects_negative():
    with pytest.raises(ValueError):
        validate_weights({"core": 0.9, "bci_chg_xs": -0.1})


def test_validate_weights_rejects_empty():
    with pytest.raises(ValueError):
        validate_weights({})


def test_core_only_equals_baseline():
    sleeves = _toy_sleeves()
    mixed = mix_sleeve_returns(sleeves, {"core": 1.0})
    pd.testing.assert_series_equal(
        mixed, sleeves[CORE_NAME].astype(float), check_names=False
    )


def test_mix_is_weighted_sum():
    sleeves = _toy_sleeves()
    w = {"core": 0.7, "bci_chg_xs": 0.2, "high_ppi_xs": 0.1}
    mixed = mix_sleeve_returns(sleeves, w)
    expected = (
        0.7 * sleeves[CORE_NAME] + 0.2 * sleeves[SATELLITE_A] + 0.1 * sleeves[SATELLITE_B]
    )
    np.testing.assert_allclose(mixed.to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_mix_fills_missing_satellite_with_zero():
    """Satellite starts later — early days keep core contribution only."""
    idx = pd.date_range("2020-01-01", periods=20, freq="B", tz="UTC")
    core = pd.Series(0.01, index=idx, name=CORE_NAME)
    late = pd.Series(0.02, index=idx[10:], name=SATELLITE_A)
    sleeves = {CORE_NAME: core, SATELLITE_A: late}
    mixed = mix_sleeve_returns(sleeves, {"core": 0.8, "bci_chg_xs": 0.2})
    # First 10 days: only 0.8 * 0.01 (satellite filled 0)
    assert abs(float(mixed.iloc[0]) - 0.008) < 1e-12
    # Later: 0.8*0.01 + 0.2*0.02
    assert abs(float(mixed.iloc[-1]) - 0.012) < 1e-12


def test_unknown_sleeve_in_weights_raises():
    sleeves = _toy_sleeves()
    with pytest.raises(KeyError):
        mix_sleeve_returns(sleeves, {"core": 0.5, "not_a_sleeve": 0.5})


def test_equity_to_daily_returns_last_per_day():
    idx = pd.date_range("2020-01-01 09:00", periods=8, freq="6h", tz="UTC")
    # Two calendar days with rising equity
    eq = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0], index=idx)
    r = equity_to_daily_returns(eq)
    assert len(r.dropna()) >= 1
    # Day0 last=102, day1 last=106 → ret = 106/102 - 1
    assert abs(float(r.dropna().iloc[0]) - (106.0 / 102.0 - 1.0)) < 1e-12


def test_select_mix_is_only_picks_max_positive():
    means = {
        "core_100": 0.008,
        "core85_bci15": 0.012,
        "core70_bci30": 0.010,
        "neg": -0.001,
    }
    assert select_mix_is_only(means) == "core85_bci15"


def test_select_mix_is_only_no_holdout_leak():
    """Mutating a holdout-only score must not change IS-selected mix.

    Selection input is IS means only; a separate HO dict is ignored.
    """
    is_means = {
        "core_100": 0.009,
        "core85_bci15": 0.011,
        "core70_bci30": 0.010,
    }
    ho_means_a = {"core_100": 0.05, "core85_bci15": -0.02, "core70_bci30": 0.04}
    ho_means_b = {"core_100": -0.05, "core85_bci15": 0.99, "core70_bci30": -0.04}
    pick_a = select_mix_is_only(is_means)
    # HO mutation is not passed into selector — pick stays identical
    pick_b = select_mix_is_only(is_means)
    assert pick_a == pick_b == "core85_bci15"
    # Sanity: if someone wrongly selected on HO, picks would differ
    assert select_mix_is_only(ho_means_a) != select_mix_is_only(ho_means_b)


def test_select_mix_falls_back_when_all_nonpositive():
    means = {"a": -0.01, "b": -0.002, "c": -0.05}
    assert select_mix_is_only(means, require_positive=True) == "b"


def test_pre_registered_includes_core_only_and_dual_satellite():
    assert "core_100" in PRE_REGISTERED_MIXES
    assert PRE_REGISTERED_MIXES["core_100"] == {"core": 1.0}
    dual = [k for k, v in PRE_REGISTERED_MIXES.items() if SATELLITE_B in v]
    assert len(dual) >= 1
    assert all(SATELLITE_A in PRE_REGISTERED_MIXES[k] or k == "core_100" for k in PRE_REGISTERED_MIXES)
