"""IS-only smooth-return portfolio selection helpers (no holdout tuning).

Objective: maximize min(year_mean_mo for IS years) subject to % positive months
and top-3 gain concentration constraints. Holdout metrics must never enter the
selection score.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowStats:
    mean_mo: float
    pct_pos: float
    top3: float
    gates: bool
    p2t: float = float("nan")


def passes_smooth_constraints(
    stats: WindowStats,
    *,
    min_pct_pos: float = 0.70,
    max_top3: float = 0.55,
    max_p2t: float = 0.085,
    require_gates: bool = True,
) -> bool:
    """Hard gates for a single IS window."""
    if require_gates and not stats.gates:
        return False
    if stats.mean_mo != stats.mean_mo:
        return False
    if stats.pct_pos != stats.pct_pos or stats.pct_pos < float(min_pct_pos):
        return False
    if stats.top3 != stats.top3 or stats.top3 > float(max_top3):
        return False
    if stats.p2t == stats.p2t and stats.p2t > float(max_p2t):
        return False
    return True


def min_mean_mo_score(
    windows: list[WindowStats],
    *,
    min_pct_pos: float = 0.70,
    max_top3: float = 0.55,
    max_p2t: float = 0.085,
    soft_top3: float | None = 0.70,
) -> float:
    """Selection score = min mean_mo across IS windows if constraints hold.

    If ``soft_top3`` is set and hard max_top3 fails but top3 <= soft_top3, allow
    with a penalty so ranking can still prefer lower concentration.
    Returns -inf when any window fails gates / %pos / soft top3 / p2t.
    """
    if not windows:
        return float("-inf")
    means: list[float] = []
    top_pen = 0.0
    for st in windows:
        if st.mean_mo != st.mean_mo or st.pct_pos != st.pct_pos or st.top3 != st.top3:
            return float("-inf")
        if (not st.gates) or st.pct_pos < float(min_pct_pos):
            return float("-inf")
        if st.p2t == st.p2t and st.p2t > float(max_p2t):
            return float("-inf")
        if st.top3 > float(max_top3):
            if soft_top3 is None or st.top3 > float(soft_top3):
                return float("-inf")
            # Soft-pass: small penalty in mean_mo units (top3 is 0-1)
            top_pen += 0.02 * (st.top3 - float(max_top3))
        means.append(float(st.mean_mo))
    return float(min(means) - top_pen)


def greedy_decorrelated_pick(
    ids: list[str],
    scores: np.ndarray,
    corr: pd.DataFrame,
    *,
    n: int,
    max_corr: float = 0.55,
) -> list[str]:
    """Greedy pick by descending score with pairwise |corr| cap.

    ``corr`` must be indexed/columned by the same ids (missing → treat as 1.0).
    """
    if n <= 0 or not ids:
        return []
    order = np.argsort(-np.asarray(scores, dtype=float))
    picked: list[str] = []
    for i in order:
        cand = ids[int(i)]
        ok = True
        for p in picked:
            if cand in corr.index and p in corr.columns:
                c = corr.loc[cand, p]
                c = 1.0 if c != c else float(c)
            else:
                c = 1.0
            if abs(c) > float(max_corr):
                ok = False
                break
        if ok:
            picked.append(cand)
        if len(picked) >= int(n):
            break
    return picked


def expected_monthly_from_daily_vol(daily_vol: float, sharpe: float = 1.0) -> float:
    """Rough expected monthly return from daily vol target and assumed Sharpe."""
    ann_vol = float(daily_vol) * np.sqrt(252.0)
    ann_ret = float(sharpe) * ann_vol
    return ann_ret / 12.0
