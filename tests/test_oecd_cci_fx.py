"""Tests for OECD CCI consumer-confidence differential FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_oecd_cci import (
    CCI_SERIES,
    DEFAULT_PUB_LAG_MONTHS,
    cci_coverage,
    load_oecd_cci_panel,
    load_us_oecd_cci,
)
from mt5_swing.strategies.oecd_cci_fx import (
    OecdCciFxConfig,
    cci_yoy_diff,
    oecd_cci_factor_returns,
    prepare_oecd_cci_scores,
    trailing_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_cci_series_map_core_g10():
    for ccy in ("USD", "EUR", "GBP", "JPY", "AUD"):
        assert ccy in CCI_SERIES
        assert CCI_SERIES[ccy] is not None
    assert CCI_SERIES["USD"] == "USACSCICP02STSAM"
    assert CCI_SERIES["EUR"] == "CSCICP02EZM460S"  # EZ aggregate
    assert CCI_SERIES["GBP"] == "CSCICP02GBM460S"
    assert "CSCICP02" in CCI_SERIES["JPY"]
    # CAD/NZD/CHF intentionally unmapped on primary
    assert CCI_SERIES["CAD"] is None
    assert CCI_SERIES["NZD"] is None
    assert CCI_SERIES["CHF"] is None


def test_pub_lag_shifts_known_date():
    """pub_lag_months=2 must move month-start index forward by ~2 months."""
    raw = load_oecd_cci_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_oecd_cci_panel(pub_lag_months=2, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    # ~2 months ≈ 60 days (±20d tolerance)
    assert 40 <= delta_days <= 80


def test_prepare_scores_high_cci_long_strong():
    """Higher CCI YoY → higher high_cci score (sentiment prior)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    # Build levels so EUR YoY rises, GBP YoY falls
    eur = np.concatenate([np.full(60, 0.0), np.linspace(0.0, 12.0, 60)])
    gbp = np.concatenate([np.full(60, 0.0), np.linspace(0.0, -12.0, 60)])
    cci = pd.DataFrame(
        {
            "USD": np.full(120, 50.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 0.0),
            "AUD": np.full(120, 0.0),
            "CAD": np.full(120, 0.0),
        },
        index=idx,
    )
    cfg = OecdCciFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_oecd_cci_scores(cci, cfg=cfg)
    last = scores["high_cci"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    last_lo = scores["low_cci"].dropna(how="all").iloc[-1]
    assert float(last_lo["GBP"]) > float(last_lo["EUR"])


def test_cci_yoy_diff_and_chg_prefers_acceleration():
    """Rising YoY → positive cci_chg score."""
    idx = pd.date_range("2015-01-01", periods=48, freq="MS", tz="UTC")
    eur = np.concatenate([np.linspace(0, 2, 24), np.linspace(2, 8, 24)])
    gbp = np.concatenate([np.linspace(0, 6, 24), np.linspace(6, 7, 24)])
    cci = pd.DataFrame(
        {
            "USD": np.full(48, 50.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(48, 0.0),
            "AUD": np.full(48, 0.0),
            "CAD": np.full(48, 0.0),
        },
        index=idx,
    )
    yoy = cci_yoy_diff(cci, periods=12)
    assert yoy.shape[0] == 48
    cfg = OecdCciFxConfig(signal_lag=0, chg_periods=12, yoy_periods=12)
    scores = prepare_oecd_cci_scores(cci, cfg=cfg)
    last = scores["cci_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_oecd_cci_no_lookahead():
    """Mutating future CCI must not change earlier factor returns."""
    idx = pd.date_range("2015-01-01", periods=2000, freq="B", tz="UTC")
    rng = np.random.default_rng(38)
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
    cci = pd.DataFrame(
        {
            c: rng.normal(0, 2.0, 260).cumsum()
            for c in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD")
        },
        index=m_idx,
    )
    cfg = OecdCciFxConfig(signal_lag=0, cost_bps_side=1.5)
    f1 = oecd_cci_factor_returns(ret, cci, cfg=cfg)
    assert "high_cci_xs" in f1
    assert "low_cci_xs" in f1
    assert "high_cci_z_xs" in f1
    assert "cci_chg_xs" in f1
    assert "us_cci_weak_fx" in f1
    assert "us_cci_haven_usd" in f1
    assert "cci_ew" in f1

    cci2 = cci.copy()
    cci2.iloc[-6:] = cci2.iloc[-6:] + 5.0
    f2 = oecd_cci_factor_returns(ret, cci2, cfg=cfg)

    cut = idx[-120]
    for name in ("high_cci_xs", "cci_chg_xs", "us_cci_weak_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_cci_z_fires_on_depressed_yoy():
    """Depressed US CCI YoY vs recent history → negative z."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    vals = np.full(240, 100.0)
    vals[-24:] = np.linspace(100.0, 80.0, 24)
    us = pd.Series(vals, index=m_idx, name="USD")
    yoy = us.diff(12)
    z = trailing_z(yoy, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) < -1.0


def test_loader_smoke_and_coverage():
    panel = load_oecd_cci_panel(
        pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True
    )
    assert panel.shape[1] >= 4  # core mapped G10
    assert "USD" in panel.columns
    assert "EUR" in panel.columns
    assert "AUD" in panel.columns
    us = load_us_oecd_cci(download=False)
    assert len(us) > 100
    cov = cci_coverage(panel)
    assert len(cov) >= 6
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("EUR") == "CSCICP02EZM460S"
    assert panel.attrs.get("unit") == "oecd_mei_cci_balance_or_us_standardised"
    unmapped = panel.attrs.get("unmapped", [])
    assert "CAD" in unmapped or "NZD" in unmapped or "CHF" in unmapped


def test_high_cci_maps_to_top_score():
    """Deep high CCI YoY maps to top high_cci scores."""
    m_idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(60, 0.0), np.linspace(0.0, 15.0, 60)])
    gbp = np.concatenate([np.full(60, 0.0), np.linspace(0.0, -15.0, 60)])
    cci = pd.DataFrame(
        {
            "USD": np.full(120, 50.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 0.0),
            "AUD": np.full(120, 0.0),
            "CAD": np.full(120, 0.0),
        },
        index=m_idx,
    )
    cfg = OecdCciFxConfig(signal_lag=0, n_long=2, n_short=2)
    scores = prepare_oecd_cci_scores(cci, cfg=cfg)
    last = scores["high_cci"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2
