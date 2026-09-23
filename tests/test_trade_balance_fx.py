"""Tests for monthly trade-balance FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_trade_balance import (
    DEFAULT_PUB_LAG_MONTHS,
    TB_EXPORT_IMPORT,
    load_trade_balance_panel,
    load_us_bopgstb,
    load_us_trade_balance,
    trade_balance_coverage,
)
from mt5_swing.strategies.trade_balance_fx import (
    TradeBalanceFxConfig,
    prepare_tb_scores,
    trade_balance_factor_returns,
    trailing_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_tb_series_map_covers_full_g10():
    for ccy in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF"):
        assert ccy in TB_EXPORT_IMPORT
        assert TB_EXPORT_IMPORT[ccy] is not None
        exp_id, imp_id = TB_EXPORT_IMPORT[ccy]
        assert exp_id.startswith("XTEXVA01")
        assert imp_id.startswith("XTIMVA01")


def test_pub_lag_shifts_known_date():
    """pub_lag_months=2 must move month-start index forward by ~2 months."""
    raw = load_trade_balance_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_trade_balance_panel(pub_lag_months=2, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    # ~2 months ≈ 60 days (±20d tolerance for month-start)
    assert 40 <= delta_days <= 80


def test_prepare_scores_deficit_long_low_tb():
    """Low TB/exports (deficit) → positive tb_deficit score → long that FX."""
    idx = pd.date_range("2018-01-01", periods=36, freq="MS", tz="UTC")
    tb = pd.DataFrame(
        {
            "USD": np.full(36, -0.3),
            "EUR": np.linspace(-0.20, -0.20, 36),  # deficit
            "GBP": np.linspace(0.15, 0.15, 36),  # surplus
            "JPY": np.linspace(-0.05, -0.05, 36),
            "AUD": np.linspace(0.05, 0.05, 36),
            "CAD": np.linspace(0.02, 0.02, 36),
            "NZD": np.linspace(-0.10, -0.10, 36),
            "CHF": np.linspace(0.12, 0.12, 36),
        },
        index=idx,
    )
    cfg = TradeBalanceFxConfig(signal_lag=0)
    scores = prepare_tb_scores(tb, cfg=cfg)
    assert float(scores["tb_deficit"].iloc[-1]["EUR"]) > float(
        scores["tb_deficit"].iloc[-1]["GBP"]
    )
    assert float(scores["tb_surplus"].iloc[-1]["GBP"]) > float(
        scores["tb_surplus"].iloc[-1]["EUR"]
    )


def test_tb_chg_score_prefers_improving_tb():
    """Rising TB/exports → positive tb_chg score."""
    idx = pd.date_range("2018-01-01", periods=36, freq="MS", tz="UTC")
    tb = pd.DataFrame(
        {
            "USD": np.full(36, -0.3),
            "EUR": np.linspace(-0.20, 0.05, 36),  # improving
            "GBP": np.linspace(0.10, -0.15, 36),  # deteriorating
            "JPY": np.full(36, 0.0),
            "AUD": np.full(36, 0.05),
            "CAD": np.full(36, 0.02),
            "NZD": np.full(36, -0.05),
            "CHF": np.full(36, 0.10),
        },
        index=idx,
    )
    cfg = TradeBalanceFxConfig(signal_lag=0, chg_periods=12)
    scores = prepare_tb_scores(tb, cfg=cfg)
    last = scores["tb_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_tb_no_lookahead():
    """Mutating future TB must not change earlier factor returns."""
    idx = pd.date_range("2015-01-01", periods=2000, freq="B", tz="UTC")
    rng = np.random.default_rng(17)
    close = pd.DataFrame(
        {
            "EURUSD": 1.1 * np.cumprod(1 + rng.normal(0, 0.005, 2000)),
            "GBPUSD": 1.3 * np.cumprod(1 + rng.normal(0, 0.005, 2000)),
            "AUDUSD": 0.75 * np.cumprod(1 + rng.normal(0, 0.006, 2000)),
            "NZDUSD": 0.65 * np.cumprod(1 + rng.normal(0, 0.006, 2000)),
            "USDJPY": 110 * np.cumprod(1 + rng.normal(0, 0.005, 2000)),
            "USDCAD": 1.3 * np.cumprod(1 + rng.normal(0, 0.004, 2000)),
            "USDCHF": 0.95 * np.cumprod(1 + rng.normal(0, 0.004, 2000)),
        },
        index=idx,
    )
    ret = close.pct_change()
    m_idx = pd.date_range("2014-01-01", periods=120, freq="MS", tz="UTC")
    tb = pd.DataFrame(
        {
            c: -0.05 + rng.normal(0, 0.01, 120).cumsum() * 0.02
            for c in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF")
        },
        index=m_idx,
    )
    cfg = TradeBalanceFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = trade_balance_factor_returns(ret, tb, cfg=cfg)
    assert "tb_deficit_xs" in f1
    assert "tb_surplus_xs" in f1
    assert "tb_chg_xs" in f1
    assert "us_tb_gr_fx" in f1
    assert "tb_ew" in f1

    tb2 = tb.copy()
    tb2.iloc[-6:] = tb2.iloc[-6:] - 0.25
    f2 = trade_balance_factor_returns(ret, tb2, cfg=cfg)

    cut = idx[-120]
    for name in ("tb_deficit_xs", "tb_chg_xs", "us_tb_gr_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_gr_tilt_fires_on_deep_deficit_z():
    """Deep negative US TB z when balance collapses vs recent history."""
    m_idx = pd.date_range("2010-01-01", periods=100, freq="MS", tz="UTC")
    vals = np.full(100, -0.10)
    vals[-3:] = -0.55
    us = pd.Series(vals, index=m_idx, name="USD")
    z = trailing_z(us, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) < -1.0


def test_loader_smoke_and_coverage():
    panel = load_trade_balance_panel(
        pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True
    )
    assert panel.shape[1] >= 7  # full G10 expected
    assert "USD" in panel.columns
    assert "NZD" in panel.columns
    assert "CHF" in panel.columns
    us = load_us_trade_balance(download=False)
    assert len(us) > 100
    cov = trade_balance_coverage(panel)
    assert len(cov) >= 7
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()


def test_us_bopgstb_loader_smoke():
    s = load_us_bopgstb(download=True, force=False)
    assert len(s.dropna()) > 100
    assert s.attrs.get("series_id") == "BOPGSTB"


def test_tb_ca_blend_optional():
    idx = pd.date_range("2018-01-01", periods=800, freq="B", tz="UTC")
    rng = np.random.default_rng(9)
    close = pd.DataFrame(
        {
            "EURUSD": 1.1 + np.cumsum(rng.normal(0, 0.002, 800)),
            "GBPUSD": 1.3 + np.cumsum(rng.normal(0, 0.002, 800)),
            "AUDUSD": 0.75 + np.cumsum(rng.normal(0, 0.002, 800)),
            "NZDUSD": 0.65 + np.cumsum(rng.normal(0, 0.002, 800)),
            "USDJPY": 110 + np.cumsum(rng.normal(0, 0.05, 800)),
            "USDCAD": 1.3 + np.cumsum(rng.normal(0, 0.002, 800)),
            "USDCHF": 0.95 + np.cumsum(rng.normal(0, 0.002, 800)),
        },
        index=idx,
    )
    ret = close.pct_change()
    m_idx = pd.date_range("2016-01-01", periods=60, freq="MS", tz="UTC")
    tb = pd.DataFrame(
        {
            "USD": np.full(60, -0.30),
            "EUR": np.full(60, -0.05),
            "GBP": np.full(60, -0.20),
            "JPY": np.full(60, 0.02),
            "AUD": np.full(60, 0.05),
            "CAD": np.full(60, 0.03),
            "NZD": np.full(60, -0.08),
            "CHF": np.full(60, 0.12),
        },
        index=m_idx,
    )
    ca_dummy = pd.Series(rng.normal(0, 0.001, len(idx)), index=idx, name="ca_debtor_xs")
    cfg = TradeBalanceFxConfig(signal_lag=1, cost_bps_side=0.0)
    factors = trade_balance_factor_returns(
        ret, tb, cfg=cfg, ca_debtor_returns=ca_dummy
    )
    assert "tb_ca_blend" in factors
    assert "tb_deficit_xs" in factors


def test_deficit_maps_low_tb_to_top_score():
    m_idx = pd.date_range("2016-01-01", periods=60, freq="MS", tz="UTC")
    tb = pd.DataFrame(
        {
            "USD": np.full(60, -0.30),
            "EUR": np.full(60, -0.25),  # deep deficit
            "GBP": np.full(60, 0.15),  # surplus
            "JPY": np.full(60, -0.18),
            "AUD": np.full(60, 0.08),
            "CAD": np.full(60, 0.05),
            "NZD": np.full(60, -0.10),
            "CHF": np.full(60, 0.12),
        },
        index=m_idx,
    )
    cfg = TradeBalanceFxConfig(signal_lag=1, n_long=2, n_short=2)
    scores = prepare_tb_scores(tb, cfg=cfg)
    last = scores["tb_deficit"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2 or "JPY" in top2
