"""Tests for BIS residential house-price FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_house_prices import (
    DEFAULT_PUB_LAG_MONTHS,
    HPI_SERIES,
    load_hpi_panel,
    load_us_hpi,
    hpi_coverage,
)
from mt5_swing.strategies.house_price_fx import (
    HousePriceFxConfig,
    prepare_house_price_scores,
    house_price_factor_returns,
    trailing_z,
    hpi_yoy_log_diff,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_hpi_series_map_core_g10():
    for ccy in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF"):
        assert ccy in HPI_SERIES
        assert HPI_SERIES[ccy] is not None
        assert HPI_SERIES[ccy].endswith("R628BIS")
    assert HPI_SERIES["EUR"] == "QDER628BIS"  # Germany proxy
    assert HPI_SERIES["USD"] == "QUSR628BIS"
    assert HPI_SERIES["NZD"] == "QNZR628BIS"
    assert HPI_SERIES["CHF"] == "QCHR628BIS"


def test_pub_lag_shifts_known_date():
    """pub_lag_months=4 must move quarter-start index forward by ~4 months."""
    raw = load_hpi_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_hpi_panel(pub_lag_months=4, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    # ~4 months ≈ 120 days (±35d tolerance for month-length)
    assert 85 <= delta_days <= 155


def test_prepare_scores_high_hpi_long_momentum():
    """Higher YoY HPI growth → higher high_hpi score (wealth/collateral prior)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    # EUR rising fast, GBP falling
    eur = np.exp(np.linspace(0.0, 0.6, 120)) * 100
    gbp = np.exp(np.linspace(0.0, -0.3, 120)) * 100
    panel = pd.DataFrame(
        {
            "USD": np.full(120, 100.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 100.0),
            "AUD": np.full(120, 100.0),
            "CAD": np.full(120, 100.0),
            "NZD": np.full(120, 100.0),
            "CHF": np.full(120, 100.0),
        },
        index=idx,
    )
    cfg = HousePriceFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_house_price_scores(panel, cfg=cfg)
    last = scores["high_hpi"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    last_lo = scores["low_hpi"].dropna(how="all").iloc[-1]
    assert float(last_lo["GBP"]) > float(last_lo["EUR"])


def test_hpi_chg_prefers_acceleration():
    """Accelerating YoY → positive hpi_chg score."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    # EUR: growth accelerates; GBP: decelerates
    # Build index from cumulative YoY path
    rng = np.random.default_rng(35)
    eur_yoy = np.concatenate([np.full(60, 0.02), np.linspace(0.02, 0.12, 60)])
    gbp_yoy = np.concatenate([np.full(60, 0.08), np.linspace(0.08, -0.02, 60)])
    # Reconstruct approx index via monthly log increments = yoy/12 (crude)
    eur = 100 * np.exp(np.cumsum(eur_yoy / 12))
    gbp = 100 * np.exp(np.cumsum(gbp_yoy / 12))
    panel = pd.DataFrame(
        {
            "USD": np.full(120, 100.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 100.0),
            "AUD": np.full(120, 100.0),
            "CAD": np.full(120, 100.0),
        },
        index=idx,
    )
    cfg = HousePriceFxConfig(signal_lag=0, chg_periods=12, yoy_periods=12)
    scores = prepare_house_price_scores(panel, cfg=cfg)
    last = scores["hpi_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_house_price_no_lookahead():
    """Mutating future HPI must not change earlier factor returns."""
    idx = pd.date_range("2015-01-01", periods=2000, freq="B", tz="UTC")
    rng = np.random.default_rng(35)
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
    m_idx = pd.date_range("2005-01-01", periods=260, freq="MS", tz="UTC")
    panel = pd.DataFrame(
        {
            c: 100.0 * np.exp(rng.normal(0, 0.005, 260).cumsum())
            for c in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF")
        },
        index=m_idx,
    )
    cfg = HousePriceFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = house_price_factor_returns(ret, panel, cfg=cfg)
    assert "high_hpi_xs" in f1
    assert "low_hpi_xs" in f1
    assert "high_hpi_z_xs" in f1
    assert "hpi_chg_xs" in f1
    assert "us_hpi_stress_fx" in f1
    assert "us_hpi_haven_usd" in f1
    assert "hpi_ew" in f1

    panel2 = panel.copy()
    panel2.iloc[-6:] = panel2.iloc[-6:] * 1.15
    f2 = house_price_factor_returns(ret, panel2, cfg=cfg)

    cut = idx[-120]
    for name in ("high_hpi_xs", "hpi_chg_xs", "us_hpi_stress_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_hpi_z_fires_on_depressed_growth():
    """Depressed US HPI YoY vs recent history → negative z."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    # Steady growth then crash
    vals = np.full(240, 100.0)
    for i in range(1, 228):
        vals[i] = vals[i - 1] * 1.005
    for i in range(228, 240):
        vals[i] = vals[i - 1] * 0.97
    us = pd.Series(vals, index=m_idx, name="USD")
    yoy = hpi_yoy_log_diff(pd.DataFrame({"USD": us}))["USD"]
    z = trailing_z(yoy, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) < -1.0


def test_loader_smoke_and_coverage():
    panel = load_hpi_panel(
        pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True
    )
    assert panel.shape[1] >= 6  # full G10 mapped
    assert "USD" in panel.columns
    assert "EUR" in panel.columns
    assert "NZD" in panel.columns
    assert "CHF" in panel.columns
    us = load_us_hpi(download=False)
    assert len(us) > 100
    cov = hpi_coverage(panel)
    assert len(cov) >= 6
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("EUR") == "QDER628BIS"
    assert panel.attrs.get("unit") == "index_level_ffill_monthly"
    assert panel.attrs.get("score_unit") == "yoy_log_diff"
    assert panel.attrs.get("unmapped", []) == []


def test_high_hpi_maps_to_top_score():
    """Deep high YoY HPI growth maps to top high_hpi scores."""
    m_idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = 100 * np.exp(np.linspace(0, 0.8, 120))
    gbp = 100 * np.exp(np.linspace(0, -0.2, 120))
    panel = pd.DataFrame(
        {
            "USD": np.full(120, 100.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": 100 * np.exp(np.linspace(0, 0.5, 120)),
            "AUD": np.full(120, 100.0),
            "CAD": np.full(120, 100.0),
        },
        index=m_idx,
    )
    cfg = HousePriceFxConfig(signal_lag=1, n_long=2, n_short=2)
    scores = prepare_house_price_scores(panel, cfg=cfg)
    last = scores["high_hpi"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2
