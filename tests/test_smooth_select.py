"""Unit tests for IS-only smooth-return selection helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mt5_swing.portfolio.smooth_select import (
    WindowStats,
    expected_monthly_from_daily_vol,
    greedy_decorrelated_pick,
    min_mean_mo_score,
    passes_smooth_constraints,
)


def test_passes_smooth_constraints_hard():
    ok = WindowStats(mean_mo=0.012, pct_pos=0.75, top3=0.50, gates=True, p2t=0.04)
    assert passes_smooth_constraints(ok)
    bad_pos = WindowStats(mean_mo=0.012, pct_pos=0.60, top3=0.50, gates=True)
    assert not passes_smooth_constraints(bad_pos)
    bad_top = WindowStats(mean_mo=0.012, pct_pos=0.75, top3=0.80, gates=True)
    assert not passes_smooth_constraints(bad_top)
    bad_gate = WindowStats(mean_mo=0.012, pct_pos=0.75, top3=0.50, gates=False)
    assert not passes_smooth_constraints(bad_gate)


def test_min_mean_mo_score_uses_minimum_and_never_sees_holdout_field():
    # Two IS windows — score is the min mean
    a = WindowStats(mean_mo=0.011, pct_pos=0.72, top3=0.50, gates=True, p2t=0.03)
    b = WindowStats(mean_mo=0.009, pct_pos=0.70, top3=0.52, gates=True, p2t=0.04)
    sc = min_mean_mo_score([a, b], min_pct_pos=0.70, max_top3=0.55, soft_top3=0.70)
    assert abs(sc - 0.009) < 1e-12
    # Soft top3: allow with penalty
    c = WindowStats(mean_mo=0.012, pct_pos=0.71, top3=0.60, gates=True, p2t=0.03)
    sc2 = min_mean_mo_score([a, c], min_pct_pos=0.70, max_top3=0.55, soft_top3=0.70)
    assert sc2 < 0.011  # penalized
    assert sc2 > -1e8
    # Hard fail soft
    d = WindowStats(mean_mo=0.012, pct_pos=0.71, top3=0.80, gates=True, p2t=0.03)
    assert min_mean_mo_score([a, d], soft_top3=0.70) == float("-inf")


def test_greedy_decorrelated_pick_respects_corr_cap():
    ids = ["A", "B", "C", "D"]
    scores = np.array([1.0, 0.9, 0.8, 0.7])
    corr = pd.DataFrame(
        [
            [1.0, 0.9, 0.1, 0.2],
            [0.9, 1.0, 0.15, 0.25],
            [0.1, 0.15, 1.0, 0.8],
            [0.2, 0.25, 0.8, 1.0],
        ],
        index=ids,
        columns=ids,
    )
    picked = greedy_decorrelated_pick(ids, scores, corr, n=3, max_corr=0.55)
    # A first; B corr with A=0.9 → skip; C ok; D corr with C=0.8 → skip
    assert picked == ["A", "C"]
    picked2 = greedy_decorrelated_pick(ids, scores, corr, n=2, max_corr=0.95)
    assert picked2 == ["A", "B"]


def test_expected_monthly_scales_with_vol():
    lo = expected_monthly_from_daily_vol(0.002, sharpe=1.0)
    hi = expected_monthly_from_daily_vol(0.005, sharpe=1.0)
    assert hi > lo
    assert 0.002 < hi < 0.03
