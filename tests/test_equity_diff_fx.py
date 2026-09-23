"""Tests for Hau–Rey equity-differential scholarly FX (causality / PIT)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.equity_indices import EQUITY_TICKERS, load_equity_panel
from mt5_swing.strategies.commodity_fx import align_daily_to_index
from mt5_swing.strategies.equity_diff_fx import (
    EquityDiffFxConfig,
    eq_diff_xs_returns,
    equity_diff_factor_returns,
    prepare_equity_diff_scores,
    trailing_equity_momentum,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_equity_signal_lag_at_least_one():
    cfg = EquityDiffFxConfig()
    assert cfg.signal_lag_days >= 1


def test_equity_mom_no_lookahead():
    """Future equity prices must not affect earlier lagged signals."""
    idx = pd.date_range("2020-01-01", periods=300, freq="B", tz="UTC")
    rng = np.random.default_rng(42)
    px = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)), index=idx, name="USD")
    cfg = EquityDiffFxConfig(formation_days=21, skip_days=0, signal_lag_days=1)
    mom = trailing_equity_momentum(px, formation_days=cfg.formation_days, skip_days=cfg.skip_days)
    mom_lag = mom.shift(cfg.signal_lag_days)

    px2 = px.copy()
    px2.iloc[-5:] *= 1.5
    mom2 = trailing_equity_momentum(px2, formation_days=cfg.formation_days, skip_days=cfg.skip_days)
    mom2_lag = mom2.shift(cfg.signal_lag_days)

    cut = -(cfg.formation_days + cfg.skip_days + cfg.signal_lag_days + 2)
    a = mom_lag.iloc[:cut].dropna()
    b = mom2_lag.iloc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 50
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True)


def test_prepare_scores_relative_and_lagged():
    idx = pd.date_range("2020-01-01", periods=200, freq="B", tz="UTC")
    panel = pd.DataFrame(
        {
            "USD": np.linspace(100, 200, 200),
            "EUR": np.linspace(100, 180, 200),
            "GBP": np.linspace(100, 220, 200),
            "JPY": np.linspace(100, 150, 200),
            "CAD": np.linspace(100, 190, 200),
            "AUD": np.linspace(100, 210, 200),
        },
        index=idx,
    )
    cfg = EquityDiffFxConfig(signal_lag_days=1, formation_days=21)
    rel = prepare_equity_diff_scores(panel, cfg=cfg, relative=True)
    loc = prepare_equity_diff_scores(panel, cfg=cfg, relative=False)
    assert set(rel.columns) <= {"EUR", "GBP", "JPY", "CAD", "AUD", "CHF"}
    assert "USD" not in rel.columns
    # After lag, first rows NaN
    assert rel.iloc[: cfg.signal_lag_days].isna().all().all()
    assert not np.allclose(rel.dropna().iloc[-1].values, loc.dropna().iloc[-1].values)


def test_eq_diff_xs_smoke_and_causal():
    idx = pd.date_range("2019-01-01", periods=500, freq="B", tz="UTC")
    rng = np.random.default_rng(7)
    panel = pd.DataFrame(
        {
            c: 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 500))
            for c in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF")
        },
        index=idx,
    )
    pret = pd.DataFrame(
        {
            "EURUSD": rng.normal(0, 0.004, 500),
            "GBPUSD": rng.normal(0, 0.004, 500),
            "AUDUSD": rng.normal(0, 0.005, 500),
            "NZDUSD": rng.normal(0, 0.005, 500),
            "USDJPY": rng.normal(0, 0.004, 500),
            "USDCAD": rng.normal(0, 0.004, 500),
            "USDCHF": rng.normal(0, 0.004, 500),
        },
        index=idx,
    )
    cfg = EquityDiffFxConfig(signal_lag_days=1)
    port = eq_diff_xs_returns(panel, pret, cfg=cfg)
    assert len(port) == len(pret)
    assert port.name == "eq_diff_xs"
    assert np.isfinite(port.fillna(0).sum())

    # Causality: zeroing last FX return shouldn't change earlier portfolio
    pret2 = pret.copy()
    pret2.iloc[-1] = 0.0
    port2 = eq_diff_xs_returns(panel, pret2, cfg=cfg)
    assert np.allclose(port.iloc[:-1].values, port2.iloc[:-1].values, equal_nan=True)

    # Mutating last equity closes shouldn't change early signals/returns
    panel2 = panel.copy()
    panel2.iloc[-3:] *= 1.4
    port3 = eq_diff_xs_returns(panel2, pret, cfg=cfg)
    cut = -(cfg.formation_days + cfg.signal_lag_days + 5)
    assert np.allclose(port.iloc[:cut].fillna(0).values, port3.iloc[:cut].fillna(0).values)


def test_align_yahoo_session_vs_fx_bar_times():
    cidx = pd.date_range("2020-01-01 04:00", periods=120, freq="B", tz="UTC")
    fidx = pd.date_range("2020-01-01 23:00", periods=100, freq="B", tz="UTC")
    panel = pd.DataFrame(
        {
            c: np.linspace(100, 100 + i * 10, 120)
            for i, c in enumerate(("USD", "EUR", "GBP", "JPY", "CAD", "AUD"))
        },
        index=cidx,
    )
    rng = np.random.default_rng(0)
    pret = pd.DataFrame(
        {
            s: rng.normal(0.0005, 0.005, 100)
            for s in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
        },
        index=fidx,
    )
    cfg = EquityDiffFxConfig(signal_lag_days=1)
    factors = equity_diff_factor_returns(panel, pret, rates=None, cfg=cfg)
    assert "eq_diff_xs" in factors
    assert (factors["eq_diff_xs"].fillna(0).abs() > 0).sum() > 5
    # Align helper joins calendar days
    sig = prepare_equity_diff_scores(panel, cfg=cfg, relative=True)
    aligned = align_daily_to_index(sig, fidx)
    assert aligned.index.equals(fidx)
    assert aligned.notna().any().any()


def test_load_equity_panel_if_cached():
    path = MACRO / "equity_yahoo_panel.csv"
    if not path.exists():
        pytest.skip("equity cache not yet downloaded")
    panel = load_equity_panel(download=False, pub_lag_days=1)
    assert panel.attrs.get("pub_lag_days") == 1
    assert "USD" in panel.columns
    assert len(panel.columns) >= 4


def test_equity_tickers_map_complete():
    assert EQUITY_TICKERS["USD"] == "^GSPC"
    assert EQUITY_TICKERS["EUR"] == "^GDAXI"
    assert EQUITY_TICKERS["GBP"] == "^FTSE"
    assert EQUITY_TICKERS["JPY"] == "^N225"
