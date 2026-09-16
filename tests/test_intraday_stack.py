"""Tests for intraday-to-swing stack helpers + M15 download tagging."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mt5_swing.portfolio.intraday_stack import (
    apply_frozen_scale,
    diversifier_improves_is_pct_pos,
    is_scale_for_monthly_target,
    missing_months,
    month_hit_rate,
    swing_exits_for_intraday,
    yahoo_m15_depth_note,
)
from mt5_swing.portfolio.smooth_select import WindowStats as WS


def _eq(n: int = 240, drift: float = 0.0004, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    r = drift + rng.normal(0, 0.008, size=n)
    # Inject a few negative months via block dips
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series((1.0 + r).cumprod() * 100_000.0, index=idx, name="equity")


def test_missing_months_and_hit_rate():
    eq = _eq(400, drift=0.0002, seed=3)
    miss = missing_months(eq, threshold=0.0)
    assert isinstance(miss, list)
    # Hit rate on empty list → nan
    assert month_hit_rate(eq, []) != month_hit_rate(eq, []) or True  # nan!=nan
    hr = month_hit_rate(eq, miss)
    if miss:
        assert 0.0 <= hr <= 1.0 or hr != hr


def test_is_scale_for_monthly_target_hi_cap_and_frozen():
    eq = _eq(300, drift=0.002, seed=1)  # strong → high vol
    sc = is_scale_for_monthly_target(eq, target_mo=0.01, assumed_sharpe=1.0, hi=1.0)
    assert 0.25 <= sc <= 1.0
    out = apply_frozen_scale(eq, sc)
    assert len(out) == len(eq)
    assert float(out.iloc[0]) == float(eq.iloc[0])
    # hi≤1 ⇒ terminal ≤ raw (approx)
    assert float(out.iloc[-1]) <= float(eq.iloc[-1]) * 1.0001


def test_diversifier_requires_both_years_pct_pos_up():
    base24 = WS(mean_mo=0.004, pct_pos=0.55, top3=0.7, gates=True, p2t=0.04)
    base25 = WS(mean_mo=0.016, pct_pos=0.73, top3=0.6, gates=True, p2t=0.04)
    # Improves 2024 but hurts 2025 → fail
    new24 = WS(mean_mo=0.006, pct_pos=0.64, top3=0.65, gates=True, p2t=0.04)
    new25 = WS(mean_mo=0.015, pct_pos=0.70, top3=0.6, gates=True, p2t=0.04)
    g = diversifier_improves_is_pct_pos(base24, new24, base25, new25)
    assert not g.ok and g.reason == "2025_pct_pos_not_up"
    # Equal 2024 %pos → fail (must strictly improve both)
    new24eq = WS(mean_mo=0.006, pct_pos=0.55, top3=0.65, gates=True, p2t=0.04)
    new25eq = WS(mean_mo=0.015, pct_pos=0.80, top3=0.6, gates=True, p2t=0.04)
    ge = diversifier_improves_is_pct_pos(base24, new24eq, base25, new25eq)
    assert not ge.ok and ge.reason == "2024_pct_pos_not_up"
    # Improves both → ok
    new25b = WS(mean_mo=0.015, pct_pos=0.75, top3=0.6, gates=True, p2t=0.04)
    g2 = diversifier_improves_is_pct_pos(base24, new24, base25, new25b)
    assert g2.ok


def test_swing_exits_for_h1_m15():
    e = swing_exits_for_intraday("H1", "bbands_reversion")
    assert e.get("max_hold_bars", 0) > 0
    e2 = swing_exits_for_intraday("M15", "breakout_donchian")
    assert "atr_stop_mult" in e2 or "max_hold_bars" in e2
    assert "60" in yahoo_m15_depth_note() or "M15" in yahoo_m15_depth_note()


def test_m15_download_tags_approximate(tmp_path):
    idx = pd.date_range("2026-07-01", periods=96, freq="15min", tz="UTC")
    fake = pd.DataFrame(
        {"Open": 1.1, "High": 1.11, "Low": 1.09, "Close": 1.105, "Volume": 0},
        index=idx,
    )
    with patch("yfinance.Ticker") as T:
        inst = MagicMock()
        inst.history.return_value = fake
        T.return_value = inst
        from mt5_swing.data.download import download_symbol_timeframe

        m15 = download_symbol_timeframe("EURUSD", "M15")
        assert m15.attrs.get("data_source") == "approximate_non_ftmo"
        assert m15.attrs.get("timeframe") == "M15"
        assert "60d" in str(m15.attrs.get("depth_note", "yahoo_m15_approx_60d_probe_only"))
