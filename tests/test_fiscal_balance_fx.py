"""Tests for fiscal-balance / government-budget FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_fiscal_balance import (
    DEFAULT_PUB_LAG_MONTHS,
    FISCAL_GDP_SERIES,
    fiscal_coverage,
    load_fiscal_gdp_panel,
    load_us_fiscal_gdp,
    load_us_mts_surplus,
)
from mt5_swing.strategies.fiscal_balance_fx import (
    FiscalBalanceFxConfig,
    fiscal_balance_factor_returns,
    prepare_fiscal_scores,
    trailing_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_fiscal_series_map_covers_core_g10():
    for ccy in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD"):
        assert ccy in FISCAL_GDP_SERIES
        assert FISCAL_GDP_SERIES[ccy] is not None
        assert FISCAL_GDP_SERIES[ccy].startswith("GGNLBA")
    # Honest unmapped free-FRED gaps
    assert FISCAL_GDP_SERIES["NZD"] is None
    assert FISCAL_GDP_SERIES["CHF"] is None


def test_pub_lag_shifts_known_date():
    """pub_lag_months=15 must move year-start index forward by ~15 months."""
    raw = load_fiscal_gdp_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_fiscal_gdp_panel(pub_lag_months=15, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    # ~15 months ≈ 450 days (±45d tolerance for month-start expand)
    assert 400 <= delta_days <= 500


def test_prepare_scores_surplus_long_surplus():
    """High fiscal/GDP → positive surplus score → long that FX."""
    idx = pd.date_range("2018-01-01", periods=36, freq="MS", tz="UTC")
    fiscal = pd.DataFrame(
        {
            "USD": np.full(36, -5.0),
            "EUR": np.linspace(2.0, 2.0, 36),  # surplus
            "GBP": np.linspace(-8.0, -8.0, 36),  # deep deficit
            "JPY": np.linspace(-1.0, -1.0, 36),
            "AUD": np.linspace(-3.0, -3.0, 36),
            "CAD": np.linspace(-4.0, -4.0, 36),
        },
        index=idx,
    )
    cfg = FiscalBalanceFxConfig(signal_lag=0)
    scores = prepare_fiscal_scores(fiscal, cfg=cfg)
    assert float(scores["fiscal_surplus"].iloc[-1]["EUR"]) > float(
        scores["fiscal_surplus"].iloc[-1]["GBP"]
    )
    assert float(scores["fiscal_deficit"].iloc[-1]["GBP"]) > float(
        scores["fiscal_deficit"].iloc[-1]["EUR"]
    )


def test_fiscal_no_lookahead():
    """Mutating future fiscal must not change earlier factor returns."""
    idx = pd.date_range("2015-01-01", periods=2000, freq="B", tz="UTC")
    rng = np.random.default_rng(11)
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
    fiscal = pd.DataFrame(
        {
            c: rng.normal(-2.0 if c == "USD" else -1.0, 1.0, 120).cumsum() * 0.05
            for c in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD")
        },
        index=m_idx,
    )
    cfg = FiscalBalanceFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = fiscal_balance_factor_returns(ret, fiscal, cfg=cfg)
    assert "fiscal_surplus_xs" in f1
    assert "fiscal_deficit_xs" in f1
    assert "fiscal_chg_xs" in f1
    assert "us_fiscal_twin_fx" in f1
    assert "fiscal_ew" in f1

    fiscal2 = fiscal.copy()
    fiscal2.iloc[-6:] = fiscal2.iloc[-6:] + 10.0
    f2 = fiscal_balance_factor_returns(ret, fiscal2, cfg=cfg)

    cut = idx[-120]
    for name in ("fiscal_surplus_xs", "fiscal_chg_xs", "us_fiscal_twin_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_twin_tilt_fires_on_deep_deficit_z():
    """Deep negative US fiscal z when deficit spikes vs recent history."""
    m_idx = pd.date_range("2010-01-01", periods=100, freq="MS", tz="UTC")
    vals = np.full(100, -3.0)
    vals[-3:] = -15.0
    us = pd.Series(vals, index=m_idx, name="USD")
    z = trailing_z(us, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) < -1.0


def test_loader_smoke_and_coverage():
    panel = load_fiscal_gdp_panel(
        pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True
    )
    assert panel.shape[1] >= 5
    assert "USD" in panel.columns
    us = load_us_fiscal_gdp(download=False)
    assert len(us) > 40
    cov = fiscal_coverage(panel)
    assert len(cov) >= 5
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()


def test_mts_loader_smoke():
    s = load_us_mts_surplus(download=True, force=False)
    assert len(s.dropna()) > 100
    assert s.attrs.get("series_id") == "MTSDS133FMS"


def test_fiscal_ca_blend_optional():
    idx = pd.date_range("2018-01-01", periods=800, freq="B", tz="UTC")
    rng = np.random.default_rng(5)
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
    fiscal = pd.DataFrame(
        {
            "USD": np.full(60, -5.0),
            "EUR": np.full(60, 1.0),
            "GBP": np.full(60, -7.0),
            "JPY": np.full(60, -2.0),
            "AUD": np.full(60, -3.0),
            "CAD": np.full(60, -4.0),
        },
        index=m_idx,
    )
    ca_dummy = pd.Series(rng.normal(0, 0.001, len(idx)), index=idx, name="ca_surplus_xs")
    cfg = FiscalBalanceFxConfig(signal_lag=1, cost_bps_side=0.0)
    factors = fiscal_balance_factor_returns(
        ret, fiscal, cfg=cfg, ca_surplus_returns=ca_dummy
    )
    assert "fiscal_ca_blend" in factors
    assert "fiscal_surplus_xs" in factors


def test_surplus_maps_high_fiscal_to_top_score():
    m_idx = pd.date_range("2016-01-01", periods=60, freq="MS", tz="UTC")
    fiscal = pd.DataFrame(
        {
            "USD": np.full(60, -5.0),
            "EUR": np.full(60, 3.0),
            "GBP": np.full(60, -8.0),
            "JPY": np.full(60, 1.0),
            "AUD": np.full(60, -2.0),
            "CAD": np.full(60, -6.0),
        },
        index=m_idx,
    )
    cfg = FiscalBalanceFxConfig(signal_lag=1, n_long=2, n_short=2)
    scores = prepare_fiscal_scores(fiscal, cfg=cfg)
    last = scores["fiscal_surplus"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2 or "JPY" in top2
