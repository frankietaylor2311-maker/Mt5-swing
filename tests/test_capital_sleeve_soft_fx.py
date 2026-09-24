"""Unit tests for §70 capital-sleeve soft-stack mix (weights, IS-only, §53 untouched)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mt5_swing.portfolio import capital_sleeves as cs53
from mt5_swing.portfolio.capital_sleeves_soft import (
    CORE_NAME,
    PRE_REGISTERED_MIXES_SOFT,
    SATELLITE_A,
    SATELLITE_B,
    mix_sleeve_returns,
    select_mix_is_only,
    validate_weights,
)


def _toy_sleeves(n: int = 40) -> dict[str, pd.Series]:
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(70)
    return {
        CORE_NAME: pd.Series(rng.normal(0.0004, 0.005, n), index=idx, name=CORE_NAME),
        SATELLITE_A: pd.Series(
            rng.normal(0.00012, 0.003, n), index=idx, name=SATELLITE_A
        ),
        SATELLITE_B: pd.Series(
            rng.normal(0.00010, 0.003, n), index=idx, name=SATELLITE_B
        ),
    }


def test_soft_mixes_sum_to_one_and_count_le_6():
    assert 4 <= len(PRE_REGISTERED_MIXES_SOFT) <= 6
    for name, w in PRE_REGISTERED_MIXES_SOFT.items():
        vw = validate_weights(w)
        assert abs(sum(vw.values()) - 1.0) < 1e-9, name
        assert all(v >= 0 for v in vw.values())


def test_soft_mixes_do_not_mutate_section_53():
    """§70 module must not alter §53 PRE_REGISTERED_MIXES keys/weights."""
    assert "core85_bci15" in cs53.PRE_REGISTERED_MIXES
    assert "core85_soft15" not in cs53.PRE_REGISTERED_MIXES
    assert cs53.SATELLITE_A == "bci_chg_xs"
    assert cs53.SATELLITE_B == "high_ppi_xs"
    # Soft mixes use soft stack names only
    for name, w in PRE_REGISTERED_MIXES_SOFT.items():
        for k in w:
            assert k in (CORE_NAME, SATELLITE_A, SATELLITE_B), (name, k)


def test_soft_satellites_are_soft_stack_names():
    assert SATELLITE_A == "soft_ew_macro5"
    assert SATELLITE_B == "soft_ew7_cip"
    assert "core_100" in PRE_REGISTERED_MIXES_SOFT
    assert PRE_REGISTERED_MIXES_SOFT["core_100"] == {CORE_NAME: 1.0}
    dual = [k for k, v in PRE_REGISTERED_MIXES_SOFT.items() if SATELLITE_B in v]
    assert len(dual) >= 1
    assert "core80_soft10_cip10" in PRE_REGISTERED_MIXES_SOFT
    assert "core70_soft20_cip10" in PRE_REGISTERED_MIXES_SOFT


def test_core_only_equals_baseline():
    sleeves = _toy_sleeves()
    mixed = mix_sleeve_returns(sleeves, {CORE_NAME: 1.0})
    pd.testing.assert_series_equal(
        mixed, sleeves[CORE_NAME].astype(float), check_names=False
    )


def test_mix_is_weighted_sum():
    sleeves = _toy_sleeves()
    w = {CORE_NAME: 0.7, SATELLITE_A: 0.2, SATELLITE_B: 0.1}
    mixed = mix_sleeve_returns(sleeves, w)
    expected = (
        0.7 * sleeves[CORE_NAME]
        + 0.2 * sleeves[SATELLITE_A]
        + 0.1 * sleeves[SATELLITE_B]
    )
    np.testing.assert_allclose(mixed.to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_mix_fills_missing_satellite_with_zero():
    idx = pd.date_range("2020-01-01", periods=20, freq="B", tz="UTC")
    core = pd.Series(0.01, index=idx, name=CORE_NAME)
    late = pd.Series(0.02, index=idx[10:], name=SATELLITE_A)
    sleeves = {CORE_NAME: core, SATELLITE_A: late}
    mixed = mix_sleeve_returns(sleeves, {CORE_NAME: 0.8, SATELLITE_A: 0.2})
    assert abs(float(mixed.iloc[0]) - 0.008) < 1e-12
    assert abs(float(mixed.iloc[-1]) - 0.012) < 1e-12


def test_select_mix_is_only_picks_max_positive():
    means = {
        "core_100": 0.008,
        "core85_soft15": 0.012,
        "core70_soft30": 0.010,
        "neg": -0.001,
    }
    assert select_mix_is_only(means) == "core85_soft15"


def test_select_mix_is_only_no_holdout_leak():
    is_means = {
        "core_100": 0.009,
        "core85_soft15": 0.011,
        "core70_soft30": 0.010,
    }
    pick_a = select_mix_is_only(is_means)
    pick_b = select_mix_is_only(is_means)
    assert pick_a == pick_b == "core85_soft15"
    ho_a = {"core_100": 0.05, "core85_soft15": -0.02, "core70_soft30": 0.04}
    ho_b = {"core_100": -0.05, "core85_soft15": 0.99, "core70_soft30": -0.04}
    assert select_mix_is_only(ho_a) != select_mix_is_only(ho_b)


def test_validate_weights_rejects_negative():
    with pytest.raises(ValueError):
        validate_weights({CORE_NAME: 0.9, SATELLITE_A: -0.1})
