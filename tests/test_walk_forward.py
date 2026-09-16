"""Walk-forward smoke tests."""

from __future__ import annotations

from mt5_swing.data.loader import generate_sample_ohlc
from mt5_swing.strategies.breakout import BreakoutDonchian
from mt5_swing.strategies.mean_reversion import MeanReversionRegime
from mt5_swing.strategies.trend_ma_adx import TrendMAADX
from mt5_swing.validation.walk_forward import WalkForwardConfig, run_walk_forward


def test_walk_forward_rolling_smoke():
    df = generate_sample_ohlc(n_bars=900, seed=11)
    wf = WalkForwardConfig(
        mode="rolling",
        train_bars=300,
        test_bars=50,
        step_bars=50,
        symbol="EURUSD",
    )
    result = run_walk_forward(df, TrendMAADX(), wf)
    assert len(result.folds) >= 2
    assert "max_drawdown" in result.aggregate_oos
    assert "gates_pass" in result.aggregate_oos
    # Must report honestly — do not require PASS
    assert isinstance(result.gates_pass, bool)


def test_walk_forward_anchored_smoke():
    df = generate_sample_ohlc(n_bars=900, seed=12)
    wf = WalkForwardConfig(
        mode="anchored",
        train_bars=300,
        test_bars=50,
        step_bars=100,
        symbol="EURUSD",
    )
    result = run_walk_forward(df, MeanReversionRegime(), wf)
    assert len(result.folds) >= 1
    assert result.mode == "anchored"


def test_breakout_backtest_runs():
    from mt5_swing.backtest.engine import BacktestConfig, run_backtest

    df = generate_sample_ohlc(n_bars=400, seed=5)
    res = run_backtest(df, BreakoutDonchian(), BacktestConfig(symbol="EURUSD"))
    assert len(res.equity) == len(df)
    assert res.metrics.start_equity == 10_000.0
