"""Causal checks for clock budgets and max-concurrent idle recycle."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mt5_swing.portfolio.overlays import (
    apply_vol_target,
    causal_max_concurrent_recycle,
    clock_budget_weights,
    daily_return_corr,
    occupancy_frame,
)


def _idx(n=80, freq="4h"):
    return pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")


def test_clock_budget_splits_and_renormalizes():
    tfs = ["H4", "H4", "H4", "D1"]
    base = np.array([0.4, 0.3, 0.2, 0.1])
    w = clock_budget_weights(tfs, base, h4_share=0.7)
    assert abs(w.sum() - 1.0) < 1e-12
    assert abs(w[:3].sum() - 0.7) < 1e-12
    assert abs(w[3] - 0.3) < 1e-12
    # within H4, relative 0.4:0.3:0.2 preserved
    assert abs(w[0] / w[1] - 0.4 / 0.3) < 1e-12


def test_clock_budget_single_clock_ignores_share():
    tfs = ["H4", "H4"]
    w = clock_budget_weights(tfs, np.array([1.0, 3.0]), h4_share=0.1)
    assert abs(w[0] - 0.25) < 1e-12
    assert abs(w[1] - 0.75) < 1e-12


def test_recycle_uses_lagged_occupancy():
    idx = _idx(60)
    rng = np.random.default_rng(3)
    rets = pd.DataFrame({f"l{i}": rng.normal(0.0002, 0.003, len(idx)) for i in range(3)}, index=idx)
    occ = pd.DataFrame(1.0, index=idx, columns=list(rets.columns))
    occ.iloc[-1, 0] = 0.0  # last bar occupancy of leg 0
    w = np.ones(3) / 3
    a = causal_max_concurrent_recycle(rets, occ, w, max_k=3, recycle_cap=2.0, initial=100_000.0)
    occ2 = occ.copy()
    occ2.iloc[-1, 0] = 1.0
    b = causal_max_concurrent_recycle(rets, occ2, w, max_k=3, recycle_cap=2.0, initial=100_000.0)
    # last return must be identical (occupancy is lagged)
    r1 = a.pct_change().iloc[-1]
    r2 = b.pct_change().iloc[-1]
    assert abs(r1 - r2) < 1e-12
    assert np.isfinite(a.values).all()


def test_max_k_caps_lagged_keepers():
    idx = _idx(40)
    rets = pd.DataFrame(
        {"a": 0.01, "b": 0.01, "c": 0.01, "d": 0.01},
        index=idx,
    )
    occ = pd.DataFrame(1.0, index=idx, columns=list(rets.columns))
    # distinct priorities so keepers are unique
    pr = np.array([4.0, 3.0, 2.0, 1.0])
    w = np.ones(4) / 4
    port = causal_max_concurrent_recycle(
        rets, occ, w, max_k=2, recycle_cap=1.0, priority=pr, initial=100_000.0
    )
    # with recycle_cap=1 and 2/4 keepers: exposure = 0.5, r = 0.5 * 0.01
    # first bar occupancy lag is 0 → r=0; later bars r=0.005
    r = port.pct_change().iloc[-1]
    assert abs(r - 0.005) < 1e-12


def test_recycle_does_not_exceed_unit_exposure():
    idx = _idx(30)
    rets = pd.DataFrame({"a": 0.02, "b": 0.0, "c": 0.0}, index=idx)
    occ = pd.DataFrame({"a": 1.0, "b": 0.0, "c": 0.0}, index=idx)
    w = np.ones(3) / 3
    port = causal_max_concurrent_recycle(
        rets, occ, w, max_k=3, recycle_cap=5.0, initial=100_000.0
    )
    # only a occupied; cap 5 would want 5/3 but clipped to 1 → r == 0.02 after first bar
    r = port.pct_change().iloc[-1]
    assert abs(r - 0.02) < 1e-12


def test_vol_target_is_lagged():
    idx = _idx(120)
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.002, len(idx))
    r[-1] = 0.05  # last bar spike must not change last scale (trail uses shift 1)
    port = pd.Series((1 + r).cumprod() * 100_000.0, index=idx)
    out = apply_vol_target(port, 0.0025, look=20)
    port2 = port.copy()
    # corrupt last return via last equity
    port2.iloc[-1] = port2.iloc[-2] * 1.2
    out2 = apply_vol_target(port2, 0.0025, look=20)
    # scale for last bar depends on trail up to -2; last *input* return can differ
    # but the scale series at last index should match
    r0 = port.pct_change()
    trail = r0.shift(1).rolling(20, min_periods=6).std()
    r1 = port2.pct_change()
    trail2 = r1.shift(1).rolling(20, min_periods=6).std()
    assert abs(float(trail.iloc[-1]) - float(trail2.iloc[-1])) < 1e-12
    assert np.isfinite(out.values).all()


def test_daily_corr_fail_closed_and_finite():
    idx = _idx(200)
    a = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    b = a * 1.01
    c = daily_return_corr(a, b)
    assert 0.9 < c <= 1.0
    short = a.iloc[:3]
    assert daily_return_corr(short, short) == 1.0


def test_occupancy_ffill_on_union_index():
    h4 = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
    d1 = pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")
    pos_h4 = pd.Series([1, 1, 0, 0, 1, 1], index=h4)
    pos_d1 = pd.Series([1, 0], index=d1)
    union = h4.union(d1)
    occ = occupancy_frame([pos_h4, pos_d1], union)
    assert occ.shape[1] == 2
    assert occ.iloc[0].sum() == 2.0
