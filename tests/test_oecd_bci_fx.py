"""Tests for OECD BCI manufacturing / business-confidence differential FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_oecd_bci import (
    BCI_SERIES,
    DEFAULT_PUB_LAG_MONTHS,
    bci_coverage,
    load_oecd_bci_panel,
    load_us_oecd_bci,
)
from mt5_swing.strategies.oecd_bci_fx import (
    OecdBciFxConfig,
    bci_yoy_diff,
    oecd_bci_factor_returns,
    prepare_oecd_bci_scores,
    trailing_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_bci_series_map_core_g10():
    for ccy in ("USD", "EUR", "GBP", "CHF"):
        assert ccy in BCI_SERIES
        assert BCI_SERIES[ccy] is not None
    assert BCI_SERIES["USD"] == "BSCICP02USM460S"
    assert BCI_SERIES["EUR"] == "BSCICP02EZM460S"  # EZ aggregate
    assert BCI_SERIES["GBP"] == "BSCICP02GBM460S"
    assert BCI_SERIES["CHF"] == "BSCICP02CHM460S"  # live unlike CCI
    # AUD/CAD/NZD/JPY intentionally unmapped on primary
    assert BCI_SERIES["AUD"] is None
    assert BCI_SERIES["CAD"] is None
    assert BCI_SERIES["NZD"] is None
    assert BCI_SERIES["JPY"] is None


def test_pub_lag_shifts_known_date():
    """pub_lag_months=2 must move month-start index forward by ~2 months."""
    raw = load_oecd_bci_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_oecd_bci_panel(pub_lag_months=2, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    # ~2 months ≈ 60 days (±20d tolerance)
    assert 40 <= delta_days <= 80


def test_prepare_scores_high_bci_long_strong():
    """Higher BCI YoY → higher high_bci score (business-confidence prior)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    # Build levels so EUR YoY rises, GBP YoY falls
    eur = np.concatenate([np.full(60, 0.0), np.linspace(0.0, 12.0, 60)])
    gbp = np.concatenate([np.full(60, 0.0), np.linspace(0.0, -12.0, 60)])
    bci = pd.DataFrame(
        {
            "USD": np.full(120, 0.0),
            "EUR": eur,
            "GBP": gbp,
            "CHF": np.full(120, 0.0),
        },
        index=idx,
    )
    cfg = OecdBciFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_oecd_bci_scores(bci, cfg=cfg)
    last = scores["high_bci"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    last_lo = scores["low_bci"].dropna(how="all").iloc[-1]
    assert float(last_lo["GBP"]) > float(last_lo["EUR"])


def test_bci_yoy_diff_and_chg_prefers_acceleration():
    """Rising YoY → positive bci_chg score."""
    idx = pd.date_range("2015-01-01", periods=48, freq="MS", tz="UTC")
    eur = np.concatenate([np.linspace(0, 2, 24), np.linspace(2, 8, 24)])
    gbp = np.concatenate([np.linspace(0, 6, 24), np.linspace(6, 7, 24)])
    bci = pd.DataFrame(
        {
            "USD": np.full(48, 0.0),
            "EUR": eur,
            "GBP": gbp,
            "CHF": np.full(48, 0.0),
        },
        index=idx,
    )
    yoy = bci_yoy_diff(bci, periods=12)
    assert yoy.shape[0] == 48
    cfg = OecdBciFxConfig(signal_lag=0, chg_periods=12, yoy_periods=12)
    scores = prepare_oecd_bci_scores(bci, cfg=cfg)
    last = scores["bci_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_oecd_bci_no_lookahead():
    """Mutating future BCI must not change earlier factor returns."""
    idx = pd.date_range("2015-01-01", periods=2000, freq="B", tz="UTC")
    rng = np.random.default_rng(39)
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
    bci = pd.DataFrame(
        {
            c: rng.normal(0, 2.0, 260).cumsum()
            for c in ("USD", "EUR", "GBP", "CHF")
        },
        index=m_idx,
    )
    cfg = OecdBciFxConfig(signal_lag=0, cost_bps_side=1.5)
    f1 = oecd_bci_factor_returns(ret, bci, cfg=cfg)
    assert "high_bci_xs" in f1
    assert "low_bci_xs" in f1
    assert "high_bci_z_xs" in f1
    assert "bci_chg_xs" in f1
    assert "us_bci_weak_fx" in f1
    assert "us_bci_haven_usd" in f1
    assert "bci_ew" in f1

    bci2 = bci.copy()
    bci2.iloc[-6:] = bci2.iloc[-6:] + 5.0
    f2 = oecd_bci_factor_returns(ret, bci2, cfg=cfg)

    cut = idx[-120]
    for name in ("high_bci_xs", "bci_chg_xs", "us_bci_weak_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_bci_z_fires_on_depressed_yoy():
    """Depressed US BCI YoY vs recent history → negative z."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    vals = np.full(240, 10.0)
    vals[-24:] = np.linspace(10.0, -20.0, 24)
    us = pd.Series(vals, index=m_idx, name="USD")
    yoy = us.diff(12)
    z = trailing_z(yoy, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) < -1.0


def test_loader_smoke_and_coverage():
    panel = load_oecd_bci_panel(
        pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True
    )
    assert panel.shape[1] >= 3  # USD + foreign mapped
    assert "USD" in panel.columns
    assert "EUR" in panel.columns
    assert "CHF" in panel.columns  # live for BCI
    us = load_us_oecd_bci(download=False)
    assert len(us) > 100
    cov = bci_coverage(panel)
    assert len(cov) >= 6
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("EUR") == "BSCICP02EZM460S"
    assert panel.attrs.get("unit") == "oecd_mei_bci_balance"
    unmapped = panel.attrs.get("unmapped", [])
    assert "AUD" in unmapped or "JPY" in unmapped or "CAD" in unmapped or "NZD" in unmapped


def test_high_bci_maps_to_top_score():
    """Deep high BCI YoY maps to top high_bci scores."""
    m_idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(60, 0.0), np.linspace(0.0, 15.0, 60)])
    gbp = np.concatenate([np.full(60, 0.0), np.linspace(0.0, -15.0, 60)])
    bci = pd.DataFrame(
        {
            "USD": np.full(120, 0.0),
            "EUR": eur,
            "GBP": gbp,
            "CHF": np.full(120, 0.0),
        },
        index=m_idx,
    )
    cfg = OecdBciFxConfig(signal_lag=0, n_long=1, n_short=1)
    scores = prepare_oecd_bci_scores(bci, cfg=cfg)
    last = scores["high_bci"].dropna(how="all").iloc[-1].dropna()
    top1 = set(last.nlargest(1).index)
    assert "EUR" in top1


def test_n_long_short_a_priori_thin_panel():
    """Foreign BSCICP02 panel is 3 names → n_long=n_short=1 a priori."""
    cfg = OecdBciFxConfig()
    assert cfg.n_long == 1
    assert cfg.n_short == 1
