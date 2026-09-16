"""Causal equity-curve time-series momentum + monthly-budget vol targeting.

All scales use information available at t-1 (or earlier) for bar t.
Holdout metrics must never enter selection scores — confirmation only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.portfolio.smooth_select import WindowStats


def apply_equity_tsmom(
    port: pd.Series,
    *,
    lookback: int = 60,
    neg_scale: float = 0.25,
    pos_scale: float = 1.0,
    flat_band: float = 0.0,
    lo: float = 0.0,
    hi: float = 1.0,
) -> pd.Series:
    """Trend-follow the equity curve: lag lookback return → scale next bar.

    Positive lagged momentum → ``pos_scale``; negative → ``neg_scale``.
    ``|mom| < flat_band`` → midpoint. Scale is clipped to [lo, hi] and never
    increases leverage above ``hi`` (default 1.0 = no RF hike).
    """
    if port is None or len(port) < max(10, lookback + 3):
        return port if port is not None else pd.Series(dtype=float)
    eq = port.astype(float)
    r = eq.pct_change().fillna(0.0)
    mom = eq / eq.shift(int(lookback)) - 1.0
    mom_lag = mom.shift(1)
    mid = 0.5 * (float(pos_scale) + float(neg_scale))
    scale = pd.Series(mid, index=eq.index, dtype=float)
    band = float(flat_band)
    scale = scale.where(~(mom_lag > band), float(pos_scale))
    scale = scale.where(~(mom_lag < -band), float(neg_scale))
    # Unknown mom → neutral (mid), not full risk
    scale = scale.where(mom_lag == mom_lag, mid)
    scale = scale.clip(lower=float(lo), upper=float(hi)).fillna(mid)
    return (1.0 + r * scale).cumprod() * float(eq.iloc[0])


def daily_vol_for_monthly_budget(
    target_mo: float = 0.01,
    *,
    assumed_sharpe: float = 1.0,
) -> float:
    """Map a monthly return budget to a daily vol target under assumed Sharpe.

    ann_ret ≈ target_mo * 12; ann_vol = ann_ret / sharpe; daily = ann_vol / sqrt(252).
    """
    sharpe = max(float(assumed_sharpe), 1e-6)
    ann_ret = float(target_mo) * 12.0
    ann_vol = ann_ret / sharpe
    return ann_vol / np.sqrt(252.0)


def apply_monthly_budget_vt(
    port: pd.Series,
    *,
    target_mo: float = 0.01,
    lookback_months: int = 6,
    assumed_sharpe: float = 1.0,
    lo: float = 0.25,
    hi: float = 1.0,
) -> pd.Series:
    """Causal scale so trailing monthly vol · sharpe ≈ ``target_mo``.

    Uses completed months only (shifted). ``hi`` defaults to 1.0 (no leverage
    boost vs the input curve — RF stays fixed).
    """
    if port is None or len(port) < 40:
        return port if port is not None else pd.Series(dtype=float)
    eq = port.astype(float)
    r = eq.pct_change().fillna(0.0)
    m_last = eq.resample("ME").last()
    m_ret = m_last.pct_change().dropna()
    if len(m_ret) < 3:
        return eq
    # Target monthly *vol* so that vol * sharpe ≈ target_mo
    sharpe = max(float(assumed_sharpe), 1e-6)
    target_mo_vol = abs(float(target_mo)) / sharpe
    lb = max(3, int(lookback_months))
    trail = m_ret.rolling(lb, min_periods=3).std()
    trail_s = trail.shift(1)
    scale_m = (float(target_mo_vol) / trail_s.replace(0, np.nan)).clip(lo, hi)
    scale_m = scale_m.fillna(1.0)
    scale_bars = scale_m.reindex(eq.index, method="ffill").fillna(1.0)
    scale_bars = scale_bars.shift(1).fillna(1.0).clip(lo, hi)
    return (1.0 + r * scale_bars).cumprod() * float(eq.iloc[0])


def apply_daily_budget_vt(
    port: pd.Series,
    *,
    target_mo: float = 0.01,
    assumed_sharpe: float = 1.2,
    look: int = 60,
    lo: float = 0.25,
    hi: float = 1.0,
) -> pd.Series:
    """Daily trailing-vol target mapped from monthly budget (causal, hi≤1 default)."""
    from mt5_swing.portfolio.overlays import apply_vol_target

    daily = daily_vol_for_monthly_budget(target_mo, assumed_sharpe=assumed_sharpe)
    return apply_vol_target(port, daily, look=look, lo=lo, hi=hi)


@dataclass(frozen=True)
class CandidateScore:
    """IS selection score — holdout fields must stay out of ``score``."""

    score: float
    min_mean_mo: float
    max_top3: float
    min_pct_pos: float
    hard_pass: bool
    soft_pass: bool


def score_is_windows(
    windows: list[WindowStats],
    *,
    min_pct_pos: float = 0.70,
    max_top3_hard: float = 0.55,
    max_top3_soft: float = 0.70,
    max_p2t: float = 0.085,
) -> CandidateScore:
    """Score = min(mean_mo) − top3 penalty; require %pos / gates on every IS window.

    Ranking prefers higher min_mean_mo and lower max(top3). Returns -inf score
    when soft constraints fail.
    """
    if not windows:
        return CandidateScore(float("-inf"), float("nan"), float("nan"), float("nan"), False, False)
    means, tops, poss = [], [], []
    soft_ok = True
    hard_ok = True
    for st in windows:
        if st.mean_mo != st.mean_mo or st.pct_pos != st.pct_pos or st.top3 != st.top3:
            return CandidateScore(float("-inf"), float("nan"), float("nan"), float("nan"), False, False)
        if (not st.gates) or st.pct_pos < float(min_pct_pos):
            soft_ok = False
            hard_ok = False
        if st.p2t == st.p2t and st.p2t > float(max_p2t):
            soft_ok = False
            hard_ok = False
        if st.top3 > float(max_top3_hard):
            hard_ok = False
            if st.top3 > float(max_top3_soft):
                soft_ok = False
        means.append(float(st.mean_mo))
        tops.append(float(st.top3))
        poss.append(float(st.pct_pos))
    min_mo = float(min(means))
    max_t3 = float(max(tops))
    min_pos = float(min(poss))
    if not soft_ok:
        return CandidateScore(float("-inf"), min_mo, max_t3, min_pos, False, False)
    # Prefer lower concentration: small penalty in mean_mo units
    pen = 0.02 * max(0.0, max_t3 - float(max_top3_hard))
    return CandidateScore(min_mo - pen, min_mo, max_t3, min_pos, hard_ok, True)


def holdout_clears_promote(
    holdout: WindowStats,
    *,
    min_mean_mo: float = 0.01,
    min_pct_pos: float = 0.70,
    max_top3: float = 0.70,
    max_p2t: float = 0.085,
) -> bool:
    """Confirmation-only: holdout must clear mean/%pos (never used in selection)."""
    if holdout.mean_mo != holdout.mean_mo or holdout.pct_pos != holdout.pct_pos:
        return False
    if not holdout.gates:
        return False
    if holdout.mean_mo < float(min_mean_mo):
        return False
    if holdout.pct_pos < float(min_pct_pos):
        return False
    if holdout.top3 == holdout.top3 and holdout.top3 > float(max_top3):
        return False
    if holdout.p2t == holdout.p2t and holdout.p2t > float(max_p2t):
        return False
    return True


def years_each_clear(
    year_stats: dict[str, WindowStats],
    years: tuple[str, ...] = ("2024", "2025", "2026"),
    *,
    min_mean_mo: float = 0.01,
    min_pct_pos: float = 0.70,
) -> bool:
    """True iff every listed calendar year clears mean_mo and %pos (confirm path)."""
    for y in years:
        st = year_stats.get(y)
        if st is None or st.mean_mo != st.mean_mo or st.pct_pos != st.pct_pos:
            return False
        if not st.gates or st.mean_mo < float(min_mean_mo) or st.pct_pos < float(min_pct_pos):
            return False
    return True
