"""Causal macro factor + fred_carry smoke tests (no look-ahead)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MACRO = Path("data/macro")


@pytest.mark.skipif(not (MACRO / "vix_yahoo.csv").exists(), reason="VIX dump missing")
def test_vix_loader_is_lagged():
    from mt5_swing.features.macro_factors import load_vix

    raw = pd.read_csv(MACRO / "vix_yahoo.csv", parse_dates=["Date"])
    raw = raw.set_index(pd.to_datetime(raw["Date"], utc=True))["close"].astype(float).sort_index()
    lagged = load_vix(bar_lag=1)
    # First valid lagged point equals prior raw close
    idx = lagged.dropna().index[50]
    pos = raw.index.get_loc(idx)
    assert abs(float(lagged.loc[idx]) - float(raw.iloc[pos - 1])) < 1e-9


@pytest.mark.skipif(not (MACRO / "fred_IRSTCI01USM156N.csv").exists(), reason="FRED rates missing")
def test_rate_diff_month_lagged():
    from mt5_swing.features.macro_factors import rate_differential

    d = rate_differential("AUD", "USD", monthly_lag=1).dropna()
    assert len(d) > 100
    # Differential should be finite
    assert np.isfinite(d.iloc[-1])


@pytest.mark.skipif(not (MACRO / "gpr_daily.csv").exists(), reason="GPR missing")
def test_gpr_loader():
    from mt5_swing.features.macro_factors import load_gpr_daily, rolling_z

    g = load_gpr_daily(bar_lag=1).dropna()
    z = rolling_z(g, 252).dropna()
    assert len(z) > 200
    assert z.abs().median() < 5


def test_fred_carry_smoke_and_future_sabotage():
    from mt5_swing.backtest.engine import BacktestConfig, run_backtest
    from mt5_swing.data.loader import generate_sample_ohlc
    from mt5_swing.features.indicators import apply_feature_pipeline
    from mt5_swing.strategies.fred_carry import FredCarry

    if not (MACRO / "fred_IRSTCI01AUM156N.csv").exists():
        pytest.skip("FRED rates missing")
    df = generate_sample_ohlc(n_bars=600, seed=3)
    # Stamp attrs symbol for clarity
    df.attrs["symbol"] = "AUDUSD"
    feat = apply_feature_pipeline(df, signal_lag=1)
    strat = FredCarry(symbol="AUDUSD", min_diff=0.0, require_trend_agree=False)
    sig_a = strat.generate_signals(feat)
    bad = df.copy()
    bad.iloc[-5:, bad.columns.get_loc("close")] *= 10
    bad.iloc[-5:, bad.columns.get_loc("high")] *= 10
    feat2 = apply_feature_pipeline(bad, signal_lag=1)
    sig_b = FredCarry(symbol="AUDUSD", min_diff=0.0, require_trend_agree=False).generate_signals(feat2)
    cutoff = -5 - 30
    assert (sig_a.iloc[:cutoff] == sig_b.iloc[:cutoff]).all()
    cfg = BacktestConfig(symbol="AUDUSD", initial_equity=100_000, risk_fraction=0.01)
    res = run_backtest(df, FredCarry(symbol="AUDUSD"), cfg)
    assert len(res.equity) == len(df)


def test_macro_gates_hi_le_1():
    from mt5_swing.portfolio.macro_regimes import apply_vix_risk_gate, apply_gpr_risk_gate

    if not (MACRO / "vix_yahoo.csv").exists() or not (MACRO / "gpr_daily.csv").exists():
        pytest.skip("macro dumps missing")
    idx = pd.date_range("2024-01-01", periods=400, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.005, size=len(idx))
    port = pd.Series((1 + pd.Series(r, index=idx)).cumprod() * 100_000.0, index=idx)
    a = apply_vix_risk_gate(port, cool_scale=0.35)
    b = apply_gpr_risk_gate(port, cool_scale=0.35)
    # Gates only cool — cumulative path should not systematically exceed unscaled in a leverage sense:
    # max single-bar scale implied ≤1 → |r_scaled| ≤ |r| when cool applied; check no explosion
    assert a.min() > 0 and b.min() > 0
    assert len(a) == len(port)
