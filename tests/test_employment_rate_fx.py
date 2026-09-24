"""Tests for OECD/FRED employment-rate (LREM64TT) FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_employment_rate import (
    DEFAULT_PUB_LAG_MONTHS,
    ER_SERIES,
    load_employment_rate_panel,
    load_us_employment_rate,
    employment_rate_coverage,
)
from mt5_swing.strategies.employment_rate_fx import (
    EmploymentRateFxConfig,
    prepare_employment_rate_scores,
    trailing_z,
    employment_rate_factor_returns,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_er_series_map():
    assert ER_SERIES["USD"] == "LREM64TTUSM156S"
    assert ER_SERIES["JPY"] == "LREM64TTJPM156S"
    assert ER_SERIES["CAD"] == "LREM64TTCAM156S"
    assert ER_SERIES["AUD"] == "LREM64TTAUM156S"
    assert ER_SERIES["GBP"] == "LREM64TTGBQ156S"
    assert ER_SERIES["EUR"] == "LREM64TTDEQ156S"  # Germany proxy
    assert ER_SERIES["CHF"] == "LREM64TTCHQ156S"
    assert ER_SERIES["NZD"] == "LREM64TTNZQ156S"


def test_pub_lag_shifts_known_date():
    """pub_lag_months=2 must move month-start index forward by ~2 months."""
    raw = load_employment_rate_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_employment_rate_panel(pub_lag_months=2, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    assert 45 <= delta_days <= 75


def test_prepare_scores_high_er_long_engagement():
    """Higher ER → higher high_er score (labour-strength / engagement-stock prior)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    # EUR ER rises (engagement), GBP ER falls
    eur = np.concatenate([np.full(60, 70.0), np.linspace(70.0, 78.0, 60)])
    gbp = np.concatenate([np.full(60, 75.0), np.linspace(75.0, 68.0, 60)])
    er = pd.DataFrame(
        {
            "USD": np.full(120, 72.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 74.0),
            "CAD": np.full(120, 73.0),
            "AUD": np.full(120, 75.0),
            "CHF": np.full(120, 80.0),
            "NZD": np.full(120, 76.0),
        },
        index=idx,
    )
    cfg = EmploymentRateFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_employment_rate_scores(er, cfg=cfg)
    last = scores["high_er"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    last_lo = scores["low_er"].dropna(how="all").iloc[-1]
    assert float(last_lo["GBP"]) > float(last_lo["EUR"])


def test_er_chg_prefers_rising():
    """Rising ER → positive er_chg score (+Δ12 of level)."""
    idx = pd.date_range("2015-01-01", periods=48, freq="MS", tz="UTC")
    # EUR ER rises 70 → 78 over last year vs GBP stays near 75→75.5
    eur = np.concatenate([np.full(24, 70.0), np.full(12, 70.0), np.full(12, 78.0)])
    gbp = np.concatenate([np.full(24, 75.0), np.full(12, 75.0), np.full(12, 75.5)])
    er = pd.DataFrame(
        {
            "USD": np.full(48, 72.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(48, 74.0),
            "CAD": np.full(48, 73.0),
            "AUD": np.full(48, 75.0),
            "CHF": np.full(48, 80.0),
            "NZD": np.full(48, 76.0),
        },
        index=idx,
    )
    cfg = EmploymentRateFxConfig(signal_lag=0, chg_periods=12)
    scores = prepare_employment_rate_scores(er, cfg=cfg)
    last = scores["er_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_er_no_lookahead():
    """Mutating future ER must not change earlier factor returns."""
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
    er = pd.DataFrame(
        {
            c: rng.uniform(65.0, 85.0, 260)
            for c in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "NZD")
        },
        index=m_idx,
    )
    cfg = EmploymentRateFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = employment_rate_factor_returns(ret, er, cfg=cfg)
    assert "high_er_xs" in f1
    assert "low_er_xs" in f1
    assert "high_er_z_xs" in f1
    assert "er_chg_xs" in f1
    assert "us_er_stress_fx" in f1
    assert "us_er_haven_usd" in f1
    assert "er_ew" in f1

    er2 = er.copy()
    er2.iloc[-6:] = er2.iloc[-6:] + 5.0
    f2 = employment_rate_factor_returns(ret, er2, cfg=cfg)

    cut = idx[-120]
    for name in ("high_er_xs", "er_chg_xs", "us_er_stress_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_er_z_fires_on_depressed():
    """Depressed US ER vs recent history → negative z ≤ −1."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    vals = np.concatenate([np.full(216, 75.0), np.full(24, 68.0)])
    us = pd.Series(vals, index=m_idx, name="USD")
    z = trailing_z(us, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) < -1.0


def test_loader_smoke_and_coverage():
    panel = load_employment_rate_panel(pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True)
    assert panel.shape[1] >= 7  # full G10 mapped
    assert "USD" in panel.columns
    assert "EUR" in panel.columns
    assert "CHF" in panel.columns
    assert "NZD" in panel.columns
    us = load_us_employment_rate(download=False)
    assert len(us) > 50
    cov = employment_rate_coverage(panel)
    assert len(cov) >= 7
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("USD") == "LREM64TTUSM156S"
    assert panel.attrs.get("series_map", {}).get("EUR") == "LREM64TTDEQ156S"
    assert panel.attrs.get("unit") == "employment_rate_pct_level"
    assert panel.attrs.get("score_basis") == "er_level_pct"
    assert DEFAULT_PUB_LAG_MONTHS == 2


def test_high_er_maps_to_top_score():
    """High ER maps to top high_er scores."""
    m_idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(60, 70.0), np.linspace(70.0, 80.0, 60)])
    gbp = np.concatenate([np.full(60, 75.0), np.linspace(75.0, 68.0, 60)])
    er = pd.DataFrame(
        {
            "USD": np.full(120, 72.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 74.0),
            "CAD": np.full(120, 73.0),
            "AUD": np.full(120, 75.0),
            "CHF": np.full(120, 79.0),
            "NZD": np.full(120, 76.0),
        },
        index=m_idx,
    )
    cfg = EmploymentRateFxConfig(signal_lag=0, n_long=2, n_short=2)
    scores = prepare_employment_rate_scores(er, cfg=cfg)
    last = scores["high_er"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2


def test_n_long_short_a_priori():
    """Full G10 foreign panel → n_long=n_short=2; signal_lag=1; z_low=-1.0."""
    cfg = EmploymentRateFxConfig()
    assert cfg.n_long == 2
    assert cfg.n_short == 2
    assert cfg.signal_lag == 1
    assert cfg.z_low == -1.0
    assert cfg.usd_tilt == 0.5
