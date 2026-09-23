"""Tests for monetary / money-growth differential FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_money_growth import (
    DEFAULT_PUB_LAG_MONTHS,
    MONEY_GROWTH_SERIES,
    load_money_growth_panel,
    load_us_money_growth,
    money_coverage,
)
from mt5_swing.strategies.money_growth_fx import (
    MoneyGrowthFxConfig,
    money_growth_factor_returns,
    prepare_money_growth_scores,
    trailing_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_money_series_map_core_g10():
    for ccy in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD"):
        assert ccy in MONEY_GROWTH_SERIES
        assert MONEY_GROWTH_SERIES[ccy] is not None
        assert MONEY_GROWTH_SERIES[ccy].startswith("MABMM301")
        assert MONEY_GROWTH_SERIES[ccy].endswith("657S")
    assert MONEY_GROWTH_SERIES["EUR"] == "MABMM301EZM657S"
    assert MONEY_GROWTH_SERIES["USD"] == "MABMM301USM657S"
    # NZD/CHF intentionally unmapped on primary (657S stale 2018)
    assert MONEY_GROWTH_SERIES["NZD"] is None
    assert MONEY_GROWTH_SERIES["CHF"] is None


def test_pub_lag_shifts_known_date():
    """pub_lag_months=2 must move month-start index forward by ~2 months."""
    raw = load_money_growth_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_money_growth_panel(pub_lag_months=2, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    # ~2 months ≈ 60 days (±20d tolerance)
    assert 40 <= delta_days <= 80


def test_prepare_scores_low_growth_long_tight():
    """Lower money growth → higher low_money_growth score (tightness prior)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    growth = pd.DataFrame(
        {
            "USD": np.full(120, 5.0),
            "EUR": np.full(120, 1.0),  # tight
            "GBP": np.full(120, 10.0),  # loose
            "JPY": np.full(120, 0.5),
            "AUD": np.full(120, 4.0),
            "CAD": np.full(120, 6.0),
        },
        index=idx,
    )
    cfg = MoneyGrowthFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_money_growth_scores(growth, cfg=cfg)
    last = scores["low_money_growth"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    last_hi = scores["high_money_growth"].dropna(how="all").iloc[-1]
    assert float(last_hi["GBP"]) > float(last_hi["EUR"])


def test_money_growth_chg_prefers_deceleration():
    """Falling growth rate → positive money_growth_chg score."""
    idx = pd.date_range("2018-01-01", periods=36, freq="MS", tz="UTC")
    growth = pd.DataFrame(
        {
            "USD": np.full(36, 5.0),
            "EUR": np.linspace(12.0, 2.0, 36),  # decelerating
            "GBP": np.linspace(2.0, 12.0, 36),  # accelerating
            "JPY": np.full(36, 1.0),
            "AUD": np.full(36, 4.0),
            "CAD": np.full(36, 5.0),
        },
        index=idx,
    )
    cfg = MoneyGrowthFxConfig(signal_lag=0, chg_periods=12)
    scores = prepare_money_growth_scores(growth, cfg=cfg)
    last = scores["money_growth_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_money_no_lookahead():
    """Mutating future money growth must not change earlier factor returns."""
    idx = pd.date_range("2015-01-01", periods=2000, freq="B", tz="UTC")
    rng = np.random.default_rng(33)
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
    growth = pd.DataFrame(
        {
            c: 5.0 + rng.normal(0, 0.3, 260).cumsum()
            for c in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD")
        },
        index=m_idx,
    )
    cfg = MoneyGrowthFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = money_growth_factor_returns(ret, growth, cfg=cfg)
    assert "low_money_growth_xs" in f1
    assert "high_money_growth_xs" in f1
    assert "low_money_growth_z_xs" in f1
    assert "money_growth_chg_xs" in f1
    assert "us_money_stress_fx" in f1
    assert "us_money_haven_usd" in f1
    assert "money_ew" in f1

    growth2 = growth.copy()
    growth2.iloc[-6:] = growth2.iloc[-6:] + 8.0
    f2 = money_growth_factor_returns(ret, growth2, cfg=cfg)

    cut = idx[-120]
    for name in ("low_money_growth_xs", "money_growth_chg_xs", "us_money_stress_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_money_z_fires_on_elevated_growth():
    """Elevated US money growth vs recent history → positive z."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    vals = np.full(240, 4.0)
    vals[-12:] = 20.0  # surge
    us = pd.Series(vals, index=m_idx, name="USD")
    z = trailing_z(us, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) > 1.0


def test_loader_smoke_and_coverage():
    panel = load_money_growth_panel(
        pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True
    )
    assert panel.shape[1] >= 4  # core G10 minus NZD/CHF
    assert "USD" in panel.columns
    assert "EUR" in panel.columns
    assert "AUD" in panel.columns
    us = load_us_money_growth(download=False)
    assert len(us) > 100
    cov = money_coverage(panel)
    assert len(cov) >= 6
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("EUR") == "MABMM301EZM657S"
    assert panel.attrs.get("unit") == "oecd_mei_growth_rate_657"
    unmapped = panel.attrs.get("unmapped", [])
    assert "NZD" in unmapped or "CHF" in unmapped


def test_low_growth_maps_to_top_score():
    """Deep low growth maps to top low_money_growth scores."""
    m_idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    growth = pd.DataFrame(
        {
            "USD": np.full(120, 5.0),
            "EUR": np.full(120, -2.0),  # contraction / tight
            "GBP": np.full(120, 15.0),  # boom
            "JPY": np.full(120, 1.0),
            "AUD": np.full(120, 4.0),
            "CAD": np.full(120, 6.0),
        },
        index=m_idx,
    )
    cfg = MoneyGrowthFxConfig(signal_lag=1, n_long=2, n_short=2)
    scores = prepare_money_growth_scores(growth, cfg=cfg)
    last = scores["low_money_growth"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2
