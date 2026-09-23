"""Tests for IMF IFS reserves / external-buffer FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_reserves import (
    DEFAULT_PUB_LAG_MONTHS,
    RESERVES_SERIES,
    load_reserves_panel,
    load_us_reserves,
    reserves_coverage,
)
from mt5_swing.strategies.reserves_fx import (
    ReservesFxConfig,
    prepare_reserves_scores,
    reserves_factor_returns,
    trailing_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_reserves_series_map_core_g10():
    for ccy in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD"):
        assert ccy in RESERVES_SERIES
        assert RESERVES_SERIES[ccy] is not None
        assert RESERVES_SERIES[ccy].startswith("TRESEG")
        assert RESERVES_SERIES[ccy].endswith("M052N")
    assert RESERVES_SERIES["EUR"] == "TRESEGDEM052N"  # Germany proxy
    assert RESERVES_SERIES["USD"] == "TRESEGUSM052N"
    # NZD/CHF intentionally unmapped on primary (404)
    assert RESERVES_SERIES["NZD"] is None
    assert RESERVES_SERIES["CHF"] is None


def test_pub_lag_shifts_known_date():
    """pub_lag_months=3 must move month-start index forward by ~3 months."""
    raw = load_reserves_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_reserves_panel(pub_lag_months=3, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    # ~3 months ≈ 90 days (±25d tolerance)
    assert 65 <= delta_days <= 115


def test_prepare_scores_high_reserves_long_buffer():
    """Higher reserves → higher high_reserves score (buffer prior)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    panel = pd.DataFrame(
        {
            "USD": np.full(120, 12.0),
            "EUR": np.full(120, 14.0),  # high buffer
            "GBP": np.full(120, 10.0),  # low buffer
            "JPY": np.full(120, 15.0),
            "AUD": np.full(120, 11.0),
            "CAD": np.full(120, 11.5),
        },
        index=idx,
    )
    cfg = ReservesFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_reserves_scores(panel, cfg=cfg)
    last = scores["high_reserves"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    last_lo = scores["low_reserves"].dropna(how="all").iloc[-1]
    assert float(last_lo["GBP"]) > float(last_lo["EUR"])


def test_reserves_chg_prefers_accumulation():
    """Rising log reserves → positive reserves_chg score."""
    idx = pd.date_range("2018-01-01", periods=36, freq="MS", tz="UTC")
    panel = pd.DataFrame(
        {
            "USD": np.full(36, 12.0),
            "EUR": np.linspace(10.0, 14.0, 36),  # accumulating
            "GBP": np.linspace(14.0, 10.0, 36),  # declining
            "JPY": np.full(36, 13.0),
            "AUD": np.full(36, 11.0),
            "CAD": np.full(36, 11.5),
        },
        index=idx,
    )
    cfg = ReservesFxConfig(signal_lag=0, chg_periods=12)
    scores = prepare_reserves_scores(panel, cfg=cfg)
    last = scores["reserves_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_reserves_no_lookahead():
    """Mutating future reserves must not change earlier factor returns."""
    idx = pd.date_range("2015-01-01", periods=2000, freq="B", tz="UTC")
    rng = np.random.default_rng(34)
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
            c: 12.0 + rng.normal(0, 0.02, 260).cumsum()
            for c in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD")
        },
        index=m_idx,
    )
    cfg = ReservesFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = reserves_factor_returns(ret, panel, cfg=cfg)
    assert "high_reserves_xs" in f1
    assert "low_reserves_xs" in f1
    assert "high_reserves_z_xs" in f1
    assert "reserves_chg_xs" in f1
    assert "us_reserves_stress_fx" in f1
    assert "us_reserves_haven_usd" in f1
    assert "reserves_ew" in f1

    panel2 = panel.copy()
    panel2.iloc[-6:] = panel2.iloc[-6:] + 2.0
    f2 = reserves_factor_returns(ret, panel2, cfg=cfg)

    cut = idx[-120]
    for name in ("high_reserves_xs", "reserves_chg_xs", "us_reserves_stress_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_reserves_z_fires_on_depressed_buffer():
    """Depressed US reserves vs recent history → negative z."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    vals = np.full(240, 12.0)
    vals[-12:] = 8.0  # drawdown
    us = pd.Series(vals, index=m_idx, name="USD")
    z = trailing_z(us, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) < -1.0


def test_loader_smoke_and_coverage():
    panel = load_reserves_panel(
        pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True
    )
    assert panel.shape[1] >= 4  # core G10 minus NZD/CHF
    assert "USD" in panel.columns
    assert "EUR" in panel.columns
    assert "AUD" in panel.columns
    us = load_us_reserves(download=False)
    assert len(us) > 100
    cov = reserves_coverage(panel)
    assert len(cov) >= 6
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("EUR") == "TRESEGDEM052N"
    assert panel.attrs.get("unit") == "log_usd_mn"
    unmapped = panel.attrs.get("unmapped", [])
    assert "NZD" in unmapped or "CHF" in unmapped


def test_high_reserves_maps_to_top_score():
    """Deep high reserves maps to top high_reserves scores."""
    m_idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    panel = pd.DataFrame(
        {
            "USD": np.full(120, 12.0),
            "EUR": np.full(120, 16.0),  # fat buffer
            "GBP": np.full(120, 9.0),  # thin
            "JPY": np.full(120, 15.0),
            "AUD": np.full(120, 11.0),
            "CAD": np.full(120, 11.5),
        },
        index=m_idx,
    )
    cfg = ReservesFxConfig(signal_lag=1, n_long=2, n_short=2)
    scores = prepare_reserves_scores(panel, cfg=cfg)
    last = scores["high_reserves"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2
