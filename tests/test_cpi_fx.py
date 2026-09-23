"""Tests for OECD MEI / national CPI / inflation YoY differential FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_cpi import (
    DEFAULT_PUB_LAG_MONTHS,
    CPI_SERIES,
    load_cpi_panel,
    load_us_cpi,
    cpi_coverage,
)
from mt5_swing.strategies.cpi_fx import (
    CpiFxConfig,
    prepare_cpi_scores,
    trailing_z,
    cpi_factor_returns,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_cpi_series_map_full_g10():
    for ccy in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF"):
        assert ccy in CPI_SERIES
        assert CPI_SERIES[ccy] is not None
    assert CPI_SERIES["USD"] == "CPIAUCSL"
    assert CPI_SERIES["EUR"] == "CP0000EZ19M086NEST"
    assert CPI_SERIES["GBP"] == "GBRCPIALLMINMEI"
    assert CPI_SERIES["JPY"] == "JPNCPIALLMINMEI"
    assert CPI_SERIES["CAD"] == "CANCPIALLMINMEI"
    assert CPI_SERIES["AUD"] == "AUSCPIALLQINMEI"
    assert CPI_SERIES["NZD"] == "NZLCPIALLQINMEI"
    assert CPI_SERIES["CHF"] == "CHECPIALLMINMEI"


def test_pub_lag_shifts_known_date():
    """pub_lag_months=1 must move month-start index forward by ~2 months."""
    raw = load_cpi_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_cpi_panel(pub_lag_months=1, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    assert 20 <= delta_days <= 45


def test_prepare_scores_high_cpi_long_activity():
    """Higher CPI YoY → higher high_cpi score (economic-momentum prior)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    # EUR CPI rises (strong momentum), GBP falls
    eur = np.concatenate([np.full(60, 1.0), np.linspace(1.0, 8.0, 60)])
    gbp = np.concatenate([np.full(60, 1.0), np.linspace(1.0, -5.0, 60)])
    cu = pd.DataFrame(
        {
            "USD": np.full(120, 1.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 1.0),
            "CAD": np.full(120, 1.0),
            "AUD": np.full(120, 1.0),
            "NZD": np.full(120, 1.0),
            "CHF": np.full(120, 1.0),
        },
        index=idx,
    )
    cfg = CpiFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_cpi_scores(cu, cfg=cfg)
    last = scores["high_cpi"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    last_lo = scores["low_cpi"].dropna(how="all").iloc[-1]
    assert float(last_lo["GBP"]) > float(last_lo["EUR"])


def test_cpi_chg_prefers_acceleration():
    """Rising YoY → positive cpi_chg score (CPI acceleration)."""
    idx = pd.date_range("2015-01-01", periods=48, freq="MS", tz="UTC")
    # EUR YoY accelerates 2 → 20; GBP stays near 10→11
    eur_yoy = np.concatenate([np.zeros(12), np.full(24, 2.0), np.full(12, 20.0)])
    gbp_yoy = np.concatenate([np.zeros(12), np.full(24, 10.0), np.full(12, 11.0)])
    cu = pd.DataFrame(
        {
            "USD": np.full(48, 1.0),
            "EUR": eur_yoy,
            "GBP": gbp_yoy,
            "JPY": np.full(48, 1.0),
            "CAD": np.full(48, 1.0),
            "AUD": np.full(48, 1.0),
            "NZD": np.full(48, 1.0),
            "CHF": np.full(48, 1.0),
        },
        index=idx,
    )
    cfg = CpiFxConfig(signal_lag=0, chg_periods=12)
    scores = prepare_cpi_scores(cu, cfg=cfg)
    last = scores["cpi_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_cpi_no_lookahead():
    """Mutating future CPI must not change earlier factor returns."""
    idx = pd.date_range("2015-01-01", periods=2000, freq="B", tz="UTC")
    rng = np.random.default_rng(42)
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
    cu = pd.DataFrame(
        {
            c: rng.normal(1.0, 2.0, 260)
            for c in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF")
        },
        index=m_idx,
    )
    cfg = CpiFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = cpi_factor_returns(ret, cu, cfg=cfg)
    assert "high_cpi_xs" in f1
    assert "low_cpi_xs" in f1
    assert "high_cpi_z_xs" in f1
    assert "cpi_chg_xs" in f1
    assert "us_cpi_stress_fx" in f1
    assert "us_cpi_haven_usd" in f1
    assert "cpi_ew" in f1

    cu2 = cu.copy()
    cu2.iloc[-6:] = cu2.iloc[-6:] + 15.0
    f2 = cpi_factor_returns(ret, cu2, cfg=cfg)

    cut = idx[-120]
    for name in ("high_cpi_xs", "cpi_chg_xs", "us_cpi_stress_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_cpi_z_fires_on_depressed_growth():
    """Depressed US CPI YoY vs recent history → negative z."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    vals = np.concatenate([np.full(216, 1.0), np.full(24, -8.0)])
    us = pd.Series(vals, index=m_idx, name="USD")
    z = trailing_z(us, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) < -1.0


def test_loader_smoke_and_coverage():
    panel = load_cpi_panel(pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True)
    assert panel.shape[1] >= 7  # full G10 incl USD
    assert "USD" in panel.columns
    assert "EUR" in panel.columns
    assert "CHF" in panel.columns
    assert "NZD" in panel.columns
    assert "AUD" in panel.columns
    us = load_us_cpi(download=False)
    assert len(us) > 50
    cov = cpi_coverage(panel)
    assert len(cov) >= 8
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("EUR") == "CP0000EZ19M086NEST"
    assert panel.attrs.get("unit") == "cpi_yoy_pct"
    assert panel.attrs.get("score_basis") == "yoy_growth_derived_from_index"
    assert panel.attrs.get("stale") is True
    assert DEFAULT_PUB_LAG_MONTHS == 1
    # USD/EUR live; after pub_lag known dates reach into mid-2020s
    end = panel.dropna(how="all").index.max()
    assert end.year >= 2025


def test_high_cpi_maps_to_top_score():
    """High CPI YoY maps to top high_cpi scores."""
    m_idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(60, 1.0), np.linspace(1.0, 10.0, 60)])
    gbp = np.concatenate([np.full(60, 1.0), np.linspace(1.0, -2.0, 60)])
    cu = pd.DataFrame(
        {
            "USD": np.full(120, 1.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 1.0),
            "CAD": np.full(120, 1.0),
            "AUD": np.full(120, 1.0),
            "NZD": np.full(120, 1.0),
            "CHF": np.full(120, 1.0),
        },
        index=m_idx,
    )
    cfg = CpiFxConfig(signal_lag=0, n_long=2, n_short=2)
    scores = prepare_cpi_scores(cu, cfg=cfg)
    last = scores["high_cpi"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2


def test_n_long_short_a_priori_full_g10():
    """Full G10 foreign panel → n_long=n_short=2 a priori."""
    cfg = CpiFxConfig()
    assert cfg.n_long == 2
    assert cfg.n_short == 2
    assert cfg.signal_lag == 1


def test_quarterly_expand_no_intra_quarter_leak():
    """After pub lag, monthly values within a known quarter equal the lagged Q obs."""
    q_idx = pd.date_range("2020-01-01", periods=8, freq="QS", tz="UTC")
    raw = pd.DataFrame(
        {"AUD": np.arange(8, dtype=float) + 1.0},
        index=q_idx,
    )
    from mt5_swing.data.fred_cpi import _expand_quarterly_to_monthly

    monthly = _expand_quarterly_to_monthly(raw, pub_lag_months=1)
    assert monthly.index.min() >= pd.Timestamp("2020-02-01", tz="UTC")
    first_val = float(monthly["AUD"].dropna().iloc[0])
    s = monthly["AUD"].dropna()
    assert float(s.iloc[1]) == first_val
    assert float(s.iloc[2]) == first_val
