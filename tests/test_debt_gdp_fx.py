"""Tests for government debt/GDP FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_debt_gdp import (
    DEFAULT_PUB_LAG_MONTHS,
    DEBT_GDP_SERIES,
    debt_coverage,
    load_debt_gdp_panel,
    load_us_debt_gdp,
    load_us_federal_debt_gdp_q,
)
from mt5_swing.strategies.debt_gdp_fx import (
    DebtGdpFxConfig,
    debt_gdp_factor_returns,
    prepare_debt_scores,
    trailing_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_debt_series_map_covers_core_g10():
    for ccy in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD"):
        assert ccy in DEBT_GDP_SERIES
        assert DEBT_GDP_SERIES[ccy] is not None
        assert DEBT_GDP_SERIES[ccy].startswith("GGGDTA")
    # Honest unmapped free-FRED gaps
    assert DEBT_GDP_SERIES["NZD"] is None
    assert DEBT_GDP_SERIES["CHF"] is None


def test_pub_lag_shifts_known_date():
    """pub_lag_months=15 must move year-start index forward by ~15 months."""
    raw = load_debt_gdp_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_debt_gdp_panel(pub_lag_months=15, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    # ~15 months ≈ 450 days (±45d tolerance for month-start expand)
    assert 400 <= delta_days <= 500


def test_prepare_scores_low_debt_long_low():
    """Low debt/GDP → positive low_debt score → long that FX."""
    idx = pd.date_range("2018-01-01", periods=36, freq="MS", tz="UTC")
    debt = pd.DataFrame(
        {
            "USD": np.full(36, 100.0),
            "EUR": np.linspace(40.0, 40.0, 36),  # low debt
            "GBP": np.linspace(120.0, 120.0, 36),  # high debt
            "JPY": np.linspace(200.0, 200.0, 36),
            "AUD": np.linspace(50.0, 50.0, 36),
            "CAD": np.linspace(80.0, 80.0, 36),
        },
        index=idx,
    )
    cfg = DebtGdpFxConfig(signal_lag=0)
    scores = prepare_debt_scores(debt, cfg=cfg)
    assert float(scores["low_debt"].iloc[-1]["EUR"]) > float(
        scores["low_debt"].iloc[-1]["GBP"]
    )
    assert float(scores["high_debt"].iloc[-1]["GBP"]) > float(
        scores["high_debt"].iloc[-1]["EUR"]
    )


def test_debt_chg_score_prefers_falling_debt():
    """Falling debt/GDP → positive debt_chg score."""
    idx = pd.date_range("2018-01-01", periods=36, freq="MS", tz="UTC")
    debt = pd.DataFrame(
        {
            "USD": np.full(36, 100.0),
            "EUR": np.linspace(80.0, 40.0, 36),  # falling
            "GBP": np.linspace(60.0, 100.0, 36),  # rising
            "JPY": np.full(36, 150.0),
            "AUD": np.full(36, 50.0),
            "CAD": np.full(36, 70.0),
        },
        index=idx,
    )
    cfg = DebtGdpFxConfig(signal_lag=0, chg_periods=12)
    scores = prepare_debt_scores(debt, cfg=cfg)
    last = scores["debt_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_debt_no_lookahead():
    """Mutating future debt must not change earlier factor returns."""
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
    debt = pd.DataFrame(
        {
            c: 60.0 + rng.normal(0, 1.0, 120).cumsum() * 0.2
            for c in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD")
        },
        index=m_idx,
    )
    cfg = DebtGdpFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = debt_gdp_factor_returns(ret, debt, cfg=cfg)
    assert "low_debt_xs" in f1
    assert "high_debt_xs" in f1
    assert "debt_chg_xs" in f1
    assert "us_debt_twin_fx" in f1
    assert "debt_ew" in f1

    debt2 = debt.copy()
    debt2.iloc[-6:] = debt2.iloc[-6:] + 30.0
    f2 = debt_gdp_factor_returns(ret, debt2, cfg=cfg)

    cut = idx[-120]
    for name in ("low_debt_xs", "debt_chg_xs", "us_debt_twin_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_twin_tilt_fires_on_elevated_debt_z():
    """Elevated positive US debt z when debt spikes vs recent history."""
    m_idx = pd.date_range("2010-01-01", periods=100, freq="MS", tz="UTC")
    vals = np.full(100, 80.0)
    vals[-3:] = 140.0
    us = pd.Series(vals, index=m_idx, name="USD")
    z = trailing_z(us, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) > 1.0


def test_loader_smoke_and_coverage():
    panel = load_debt_gdp_panel(
        pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True
    )
    assert panel.shape[1] >= 5
    assert "USD" in panel.columns
    us = load_us_debt_gdp(download=False)
    assert len(us) > 20
    cov = debt_coverage(panel)
    assert len(cov) >= 5
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()


def test_us_q_loader_smoke():
    s = load_us_federal_debt_gdp_q(download=True, force=False)
    assert len(s.dropna()) > 100
    assert s.attrs.get("series_id") == "GFDEGDQ188S"


def test_debt_fiscal_blend_optional():
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
    debt = pd.DataFrame(
        {
            "USD": np.full(60, 100.0),
            "EUR": np.full(60, 60.0),
            "GBP": np.full(60, 110.0),
            "JPY": np.full(60, 200.0),
            "AUD": np.full(60, 50.0),
            "CAD": np.full(60, 80.0),
        },
        index=m_idx,
    )
    fis_dummy = pd.Series(rng.normal(0, 0.001, len(idx)), index=idx, name="fiscal_surplus_xs")
    cfg = DebtGdpFxConfig(signal_lag=1, cost_bps_side=0.0)
    factors = debt_gdp_factor_returns(
        ret, debt, cfg=cfg, fiscal_surplus_returns=fis_dummy
    )
    assert "debt_fiscal_blend" in factors
    assert "low_debt_xs" in factors


def test_low_debt_maps_low_debt_to_top_score():
    m_idx = pd.date_range("2016-01-01", periods=60, freq="MS", tz="UTC")
    debt = pd.DataFrame(
        {
            "USD": np.full(60, 100.0),
            "EUR": np.full(60, 40.0),
            "GBP": np.full(60, 120.0),
            "JPY": np.full(60, 50.0),
            "AUD": np.full(60, 90.0),
            "CAD": np.full(60, 110.0),
        },
        index=m_idx,
    )
    cfg = DebtGdpFxConfig(signal_lag=1, n_long=2, n_short=2)
    scores = prepare_debt_scores(debt, cfg=cfg)
    last = scores["low_debt"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2 or "JPY" in top2
