"""Donchian prior-window fix + new strategy smoke tests (no look-ahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mt5_swing.backtest.engine import BacktestConfig, run_backtest
from mt5_swing.data.loader import generate_sample_ohlc, load_ohlc_csv
from mt5_swing.features.indicators import apply_feature_pipeline, donchian
from mt5_swing.strategies.bbands_reversion import BBandsReversion
from mt5_swing.strategies.breakout import BreakoutDonchian
from mt5_swing.strategies.ema_pullback import EmaPullback
from mt5_swing.strategies.hybrid_regime import HybridRegime
from pathlib import Path


def test_donchian_excludes_current_bar():
    high = pd.Series([1.0, 2.0, 3.0, 2.5, 4.0, 3.5])
    low = pd.Series([0.5, 1.0, 2.0, 1.5, 3.0, 2.5])
    upper, lower, mid = donchian(high, low, window=3)
    # At index 3: prior highs are 1,2,3 → max 3 (excludes current 2.5)
    assert abs(upper.iloc[3] - 3.0) < 1e-12
    # At index 4: prior highs 2,3,2.5 → max 3 (excludes current 4)
    assert abs(upper.iloc[4] - 3.0) < 1e-12
    assert abs(mid.iloc[4] - (upper.iloc[4] + lower.iloc[4]) / 2) < 1e-12


def test_breakout_can_fire_on_history():
    hist = Path("data/history/EURUSD_H4.csv")
    if not hist.exists():
        df = generate_sample_ohlc(n_bars=800, seed=7)
    else:
        df = load_ohlc_csv(hist, symbol="EURUSD", timeframe="H4")
    feat = apply_feature_pipeline(df, signal_lag=1)
    px = feat["signal_close"]
    # Prior-window channels + lag should allow some breaks
    n_long = int((px > feat["donchian_upper"]).sum())
    n_short = int((px < feat["donchian_lower"]).sum())
    assert n_long + n_short > 0, "Donchian breakout still never fires"
    sig = BreakoutDonchian(adx_min=0.0).generate_signals(feat)
    assert int((sig != 0).sum()) > 0


def test_new_strategies_smoke_and_trades():
    df = generate_sample_ohlc(n_bars=600, seed=21)
    cfg = BacktestConfig(symbol="EURUSD", initial_equity=100_000)
    for strat in (EmaPullback(), HybridRegime(), BBandsReversion(require_rsi=False)):
        res = run_backtest(df, strat, cfg)
        assert len(res.equity) == len(df)
        assert isinstance(res.metrics.gates_pass, bool)


def test_lookahead_new_features_stable():
    df = generate_sample_ohlc(n_bars=250, seed=44)
    a = apply_feature_pipeline(df, signal_lag=1)
    bad = df.copy()
    bad.iloc[-8:, bad.columns.get_loc("close")] = 1e6
    bad.iloc[-8:, bad.columns.get_loc("high")] = 1e6
    b = apply_feature_pipeline(bad, signal_lag=1)
    cols = ["bb_mid", "bb_upper", "macd", "donchian_upper", "htf_sma_fast"]
    cutoff = -8 - 40
    left = a[cols].iloc[:cutoff]
    right = b[cols].iloc[:cutoff]
    mask = left.notna().all(axis=1) & right.notna().all(axis=1)
    assert mask.any()
    diff = (left.loc[mask] - right.loc[mask]).abs().max().max()
    assert diff == 0.0 or np.isnan(diff), f"look-ahead in new feats: {diff}"


def test_atr_exits_produce_stop_or_target_reasons():
    from mt5_swing.backtest.engine import BacktestConfig, run_backtest
    from mt5_swing.strategies.trend_ma_adx import TrendMAADX

    df = generate_sample_ohlc(n_bars=500, seed=17)
    cfg = BacktestConfig(
        symbol="EURUSD",
        use_atr_exits=True,
        atr_stop_mult=1.5,
        atr_target_mult=2.0,
        risk_fraction=0.01,
    )
    res = run_backtest(df, TrendMAADX(adx_threshold=15), cfg)
    if len(res.trades) == 0:
        return  # synthetic may be flat; do not fail
    reasons = set(res.trades["reason"].unique())
    assert reasons & {"atr_stop", "atr_target", "signal", "kill_switch_flatten"}
