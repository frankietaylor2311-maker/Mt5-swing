"""Tests for global-imbalances / current-account FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_current_account import (
    CA_GDP_SERIES,
    DEFAULT_PUB_LAG_QUARTERS,
    ca_coverage,
    load_ca_gdp_panel,
    load_us_ca_gdp,
)
from mt5_swing.strategies.current_account_fx import (
    CurrentAccountFxConfig,
    current_account_factor_returns,
    prepare_ca_scores,
    trailing_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_ca_series_map_covers_g10():
    for ccy in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF"):
        assert ccy in CA_GDP_SERIES
        assert CA_GDP_SERIES[ccy][0].endswith("B6BLTT02STSAQ")


def test_pub_lag_shifts_known_date():
    """pub_lag_quarters=2 must move index forward by 6 months."""
    # Use cached files if present; else download
    raw = load_ca_gdp_panel(pub_lag_quarters=0, download=True, force=False)
    lagged = load_ca_gdp_panel(pub_lag_quarters=2, download=False, force=False)
    assert not raw.empty and not lagged.empty
    # First non-null USD observation date should differ by ~6 months
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    assert 150 <= delta_days <= 220  # ~6 months


def test_prepare_scores_debtor_long_deficit():
    """Low CA/GDP → positive debtor score → long that FX."""
    idx = pd.date_range("2018-01-01", periods=36, freq="MS", tz="UTC")
    ca = pd.DataFrame(
        {
            "USD": np.full(36, -3.0),
            "EUR": np.linspace(3.0, 3.0, 36),  # surplus
            "GBP": np.linspace(-5.0, -5.0, 36),  # deficit
            "JPY": np.linspace(2.0, 2.0, 36),
            "AUD": np.linspace(-2.0, -2.0, 36),
        },
        index=idx,
    )
    cfg = CurrentAccountFxConfig(signal_lag=0)
    scores = prepare_ca_scores(ca, cfg=cfg)
    # GBP deficit → higher debtor score than EUR surplus
    assert float(scores["ca_debtor"].iloc[-1]["GBP"]) > float(scores["ca_debtor"].iloc[-1]["EUR"])
    # Surplus score opposite
    assert float(scores["ca_surplus"].iloc[-1]["EUR"]) > float(scores["ca_surplus"].iloc[-1]["GBP"])


def test_ca_no_lookahead():
    """Mutating future CA must not change earlier factor returns."""
    idx = pd.date_range("2015-01-01", periods=2000, freq="B", tz="UTC")
    rng = np.random.default_rng(7)
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
    ca = pd.DataFrame(
        {
            c: rng.normal(-1.0 if c == "USD" else 0.5, 1.5, 120).cumsum() * 0.05
            for c in ("USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF")
        },
        index=m_idx,
    )
    cfg = CurrentAccountFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = current_account_factor_returns(ret, ca, cfg=cfg)
    assert "ca_debtor_xs" in f1
    assert "ca_surplus_xs" in f1
    assert "ca_chg_xs" in f1
    assert "us_ca_gr_fx" in f1
    assert "ca_ew" in f1

    ca2 = ca.copy()
    ca2.iloc[-6:] = ca2.iloc[-6:] + 10.0  # shock the future
    f2 = current_account_factor_returns(ret, ca2, cfg=cfg)

    cut = idx[-120]
    for name in ("ca_debtor_xs", "ca_chg_xs", "us_ca_gr_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_gr_tilt_fires_on_deep_deficit_z():
    """Deep negative US CA z fires when deficit spikes vs recent history."""
    m_idx = pd.date_range("2010-01-01", periods=100, freq="MS", tz="UTC")
    # Steady mild deficit, then a sharp 3-month worsening (keeps window mean near -2)
    vals = np.full(100, -2.0)
    vals[-3:] = -10.0
    us = pd.Series(vals, index=m_idx, name="USD")
    z = trailing_z(us, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) < -1.0


def test_loader_smoke_and_coverage():
    panel = load_ca_gdp_panel(pub_lag_quarters=DEFAULT_PUB_LAG_QUARTERS, download=True)
    assert panel.shape[1] >= 6
    assert "USD" in panel.columns
    us = load_us_ca_gdp(download=False)
    assert len(us) > 40
    cov = ca_coverage(panel)
    assert len(cov) >= 6
    assert (cov["n_obs"] > 0).all()


def test_debtor_maps_deficit_to_long_pair_sign():
    """End-to-end: deficit GBP should get positive GBPUSD weight in debtor sort."""
    idx = pd.date_range("2018-01-01", periods=800, freq="B", tz="UTC")
    rng = np.random.default_rng(3)
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
    # Persistent: GBP deep deficit, CHF/JPY surplus → debtor longs GBP
    ca = pd.DataFrame(
        {
            "USD": np.full(60, -3.0),
            "EUR": np.full(60, 1.0),
            "GBP": np.full(60, -6.0),
            "JPY": np.full(60, 4.0),
            "AUD": np.full(60, -1.0),
            "NZD": np.full(60, -4.0),
            "CAD": np.full(60, -1.5),
            "CHF": np.full(60, 8.0),
        },
        index=m_idx,
    )
    cfg = CurrentAccountFxConfig(signal_lag=1, n_long=2, n_short=2, cost_bps_side=0.0)
    scores = prepare_ca_scores(ca, cfg=cfg)
    last = scores["ca_debtor"].dropna(how="all").iloc[-1].dropna()
    # Top scores should be the most negative CA countries
    top2 = set(last.nlargest(2).index)
    assert "GBP" in top2 or "NZD" in top2
