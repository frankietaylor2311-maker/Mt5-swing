"""Tests for swap-/forward-aware carry (CIP / IR3M proxies)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_forward_carry import (
    cip_implied_forward_discount_vs_usd,
    forward_carry_coverage,
    load_forward_carry_panels,
    rate_diff_vs_usd,
)
from mt5_swing.strategies.forward_carry_fx import (
    ForwardCarryFxConfig,
    ew_signed_weights,
    forward_carry_factor_returns,
    prepare_forward_carry_scores,
    rank_sort_weights,
    score_rank_agreement,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_cip_fd_same_sign_as_rate_diff():
    idx = pd.date_range("2018-01-01", periods=24, freq="MS", tz="UTC")
    rates = pd.DataFrame(
        {
            "USD": 2.0,
            "AUD": np.linspace(3.0, 5.0, 24),
            "JPY": np.linspace(-0.1, 0.0, 24),
            "EUR": 1.0,
            "GBP": 2.5,
            "CAD": 1.5,
        },
        index=idx,
    )
    fd = cip_implied_forward_discount_vs_usd(rates, tenor_months=3)
    diff = rate_diff_vs_usd(rates)
    # High-AUD / low-JPY ordering preserved
    assert float(fd.iloc[-1]["AUD"]) > float(fd.iloc[-1]["EUR"])
    assert float(fd.iloc[-1]["EUR"]) > float(fd.iloc[-1]["JPY"])
    # Rank agreement with linear diff should be perfect on this panel
    corr = score_rank_agreement(fd, diff)
    assert corr > 0.99


def test_cip_fd_monotonic_in_foreign_rate():
    idx = pd.date_range("2020-01-01", periods=3, freq="MS", tz="UTC")
    low = pd.DataFrame({"USD": [2.0, 2.0, 2.0], "EUR": [1.0, 1.0, 1.0]}, index=idx)
    high = pd.DataFrame({"USD": [2.0, 2.0, 2.0], "EUR": [4.0, 4.0, 4.0]}, index=idx)
    fd_lo = cip_implied_forward_discount_vs_usd(low)
    fd_hi = cip_implied_forward_discount_vs_usd(high)
    assert float(fd_hi["EUR"].iloc[-1]) > float(fd_lo["EUR"].iloc[-1])


def test_rank_sort_long_high_short_low():
    idx = pd.date_range("2015-01-01", periods=12, freq="MS", tz="UTC")
    score = pd.DataFrame(
        {
            "AUD": np.linspace(1, 3, 12),
            "JPY": np.linspace(-2, -1, 12),
            "EUR": 0.0,
            "GBP": 0.5,
        },
        index=idx,
    )
    w = rank_sort_weights(score, n_long=1, n_short=1)
    last = w.iloc[-1]
    assert last["AUD"] > 0
    assert last["JPY"] < 0


def test_ew_signed_requires_both_sides():
    idx = pd.date_range("2016-01-01", periods=6, freq="MS", tz="UTC")
    all_pos = pd.DataFrame({"EUR": 1.0, "GBP": 0.5, "AUD": 0.2}, index=idx)
    assert ew_signed_weights(all_pos).empty
    mixed = pd.DataFrame(
        {"EUR": [1.0] * 6, "GBP": [-0.5] * 6, "AUD": [0.2] * 6, "JPY": [-1.0] * 6},
        index=idx,
    )
    w = ew_signed_weights(mixed)
    assert len(w) == 6
    last = w.iloc[-1]
    assert last[last > 0].sum() == pytest.approx(0.5)
    assert last[last < 0].sum() == pytest.approx(-0.5)


def test_signal_lag_no_lookahead_on_scores():
    idx = pd.date_range("2012-01-01", periods=36, freq="MS", tz="UTC")
    rng = np.random.default_rng(1)
    ir3m = pd.DataFrame(
        {c: rng.normal(0, 0.5, 36) for c in ("EUR", "GBP", "AUD", "CAD")},
        index=idx,
    )
    cfg = ForwardCarryFxConfig(signal_lag=1)
    s1 = prepare_forward_carry_scores(ir3m, ir3m, ir3m, cfg=cfg)["carry_ir3m_xs"]
    ir3m2 = ir3m.copy()
    ir3m2.iloc[-4:] += 5.0
    s2 = prepare_forward_carry_scores(ir3m2, ir3m2, ir3m2, cfg=cfg)["carry_ir3m_xs"]
    cut = -8
    a = s1.iloc[:cut].dropna(how="all")
    b = s2.iloc[:cut].dropna(how="all")
    common = a.index.intersection(b.index)
    assert len(common) > 10
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True)


def test_factor_returns_smoke_causal_and_tc_haircut():
    idx = pd.date_range("2016-01-01", periods=1800, freq="B", tz="UTC")
    rng = np.random.default_rng(11)
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
    m_idx = pd.date_range("2015-01-01", periods=120, freq="MS", tz="UTC")
    ir3m = pd.DataFrame(
        {c: rng.normal(0, 0.4, 120) for c in ("EUR", "GBP", "AUD", "NZD", "JPY", "CAD", "CHF")},
        index=m_idx,
    )
    irstci = ir3m + rng.normal(0, 0.1, ir3m.shape)
    cip = ir3m * 0.25  # scaled FD-like
    cfg = ForwardCarryFxConfig(signal_lag=1, cost_bps_side=1.5, tc_haircut_bps=5.0)
    factors = forward_carry_factor_returns(
        ret, ir3m_diff=ir3m, irstci_diff=irstci, cip_fd=cip, cfg=cfg
    )
    assert "carry_ir3m_xs" in factors
    assert "carry_irstci_xs" in factors
    assert "carry_cip_fd_xs" in factors
    assert "carry_ir3m_ew" in factors
    assert "carry_ir3m_tc5" in factors
    assert "carry_blend_xs" in factors
    # TC haircut must reduce (or equal) cumulative return vs 1.5bps when turnover > 0
    r15 = factors["carry_ir3m_xs"].fillna(0.0)
    r5 = factors["carry_ir3m_tc5"].fillna(0.0)
    assert float((1 + r5).prod()) <= float((1 + r15).prod()) + 1e-12
    # Causal: mutate late IR3M scores → early returns unchanged
    r1 = factors["carry_ir3m_xs"].copy()
    ir3m2 = ir3m.copy()
    ir3m2.iloc[-3:] += 10.0
    r2 = forward_carry_factor_returns(
        ret, ir3m_diff=ir3m2, irstci_diff=irstci, cip_fd=cip, cfg=cfg
    )["carry_ir3m_xs"]
    cut = r1.index[-60]
    a = r1.loc[:cut].dropna()
    b = r2.loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 200
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


@pytest.mark.skipif(
    not (MACRO / "fred_IR3TIB01USM156N.csv").exists(),
    reason="FRED IR3M CSV not cached",
)
def test_load_forward_carry_panels_from_cache():
    panels = load_forward_carry_panels(download=True, pub_lag_months=1)
    assert "USD" in panels["ir3m"].columns
    assert "EUR" in panels["ir3m_diff"].columns
    assert panels["ir3m"].attrs.get("pub_lag_months") == 1
    assert len(panels["ir3m_diff"].dropna(how="all")) > 100
    # CIP FD ranks align with linear IR3M diffs
    corr = score_rank_agreement(panels["cip_fd_ir3m"], panels["ir3m_diff"])
    assert corr > 0.95
    cov = forward_carry_coverage(download=False)
    assert "AUD" in set(cov["currency"])
    assert (cov["approx_level"] == "money_market_ir3m").any()
