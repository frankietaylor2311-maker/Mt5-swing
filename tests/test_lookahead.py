"""Look-ahead sabotage test: features/signals must not depend on future bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mt5_swing.data.loader import generate_sample_ohlc
from mt5_swing.features.indicators import apply_feature_pipeline, lag, sma
from mt5_swing.strategies.trend_ma_adx import TrendMAADX


def test_lag_rejects_negative():
    s = pd.Series([1.0, 2.0, 3.0])
    try:
        lag(s, -1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_sma_matches_manual_past_window():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(close, 3)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert abs(out.iloc[2] - 2.0) < 1e-12
    assert abs(out.iloc[4] - 4.0) < 1e-12


def test_lookahead_sabotage_future_corruption_does_not_change_past_features():
    """
    Sabotage: rewrite the *last* bars' OHLC to extreme values.
    Features on earlier bars (with signal_lag) must be unchanged — proves no look-ahead.
    """
    df = generate_sample_ohlc(n_bars=200, seed=99)
    feat_a = apply_feature_pipeline(df, signal_lag=1)

    df_bad = df.copy()
    # Corrupt the final 10 bars heavily
    df_bad.iloc[-10:, df_bad.columns.get_loc("close")] = 1e6
    df_bad.iloc[-10:, df_bad.columns.get_loc("high")] = 1e6
    df_bad.iloc[-10:, df_bad.columns.get_loc("low")] = 1e6
    feat_b = apply_feature_pipeline(df_bad, signal_lag=1)

    # Compare all rows except the last 10 + lag buffer (ATR/ADX windows)
    cutoff = -10 - 50  # generous buffer for rolling windows
    cols = ["sma_fast", "sma_slow", "atr", "adx", "rsi"]
    left = feat_a[cols].iloc[:cutoff]
    right = feat_b[cols].iloc[:cutoff]
    # Drop NaN warm-up
    mask = left.notna().all(axis=1) & right.notna().all(axis=1)
    assert mask.any()
    diff = (left.loc[mask] - right.loc[mask]).abs().max().max()
    assert diff == 0.0 or np.isnan(diff), f"Look-ahead detected: max abs diff={diff}"


def test_signals_stable_under_future_sabotage():
    df = generate_sample_ohlc(n_bars=300, seed=3)
    feat = apply_feature_pipeline(df, signal_lag=1)
    strat = TrendMAADX()
    sig_a = strat.generate_signals(feat)

    df2 = df.copy()
    df2.iloc[-5:, df2.columns.get_loc("close")] *= 10
    feat2 = apply_feature_pipeline(df2, signal_lag=1)
    sig_b = strat.generate_signals(feat2)

    # Signals before the sabotaged region (+ indicator window) must match
    n = len(sig_a) - 5 - 60
    assert (sig_a.iloc[:n].values == sig_b.iloc[:n].values).all()
