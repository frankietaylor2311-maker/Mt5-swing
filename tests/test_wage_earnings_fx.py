"""Tests for OECD/FRED manufacturing wage/earnings differential FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_wage_earnings import (
    DEFAULT_PUB_LAG_MONTHS,
    WAGE_SERIES,
    load_us_wage_earnings,
    load_wage_earnings_panel,
    wage_earnings_coverage,
)
from mt5_swing.strategies.wage_earnings_fx import (
    WageEarningsFxConfig,
    prepare_wage_earnings_scores,
    trailing_z,
    wage_earnings_factor_returns,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_wage_series_map():
    assert WAGE_SERIES["USD"] == "LCEAMN01USM657S"
    assert WAGE_SERIES["EUR"] == "LCEAMN01EZQ657S"
    assert WAGE_SERIES["GBP"] == "LCEAMN01GBM661S"
    assert WAGE_SERIES["JPY"] == "LCEAMN01JPM661S"
    assert WAGE_SERIES["AUD"] == "LCEAMN01AUQ661S"
    assert WAGE_SERIES["CAD"] == "LCEAMN01CAM657S"
    assert WAGE_SERIES["NZD"] == "LCEAMN01NZQ657S"
    assert WAGE_SERIES["CHF"] is None  # unmapped on FRED


def test_pub_lag_shifts_known_date():
    """pub_lag_months=2 must move month-start index forward by ~2 months."""
    raw = load_wage_earnings_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_wage_earnings_panel(pub_lag_months=2, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    assert 45 <= delta_days <= 75


def test_prepare_scores_low_wage_long_competitiveness():
    """Lower wage YoY → higher low_wage score (competitiveness prior)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    # EUR wage growth falls (competitive), GBP wage growth rises
    eur = np.concatenate([np.full(60, 3.0), np.linspace(3.0, 0.5, 60)])
    gbp = np.concatenate([np.full(60, 2.0), np.linspace(2.0, 8.0, 60)])
    wage = pd.DataFrame(
        {
            "USD": np.full(120, 2.5),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 1.0),
            "CAD": np.full(120, 2.0),
            "AUD": np.full(120, 2.5),
            "NZD": np.full(120, 2.0),
        },
        index=idx,
    )
    cfg = WageEarningsFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_wage_earnings_scores(wage, cfg=cfg)
    last = scores["low_wage"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    last_hi = scores["high_wage"].dropna(how="all").iloc[-1]
    assert float(last_hi["GBP"]) > float(last_hi["EUR"])


def test_wage_chg_prefers_deceleration():
    """Decelerating wage YoY → positive wage_chg score (−Δ12 of YoY)."""
    idx = pd.date_range("2015-01-01", periods=48, freq="MS", tz="UTC")
    # EUR wage YoY drops 6 → 1 over last year vs GBP stays near 3→2.5
    eur = np.concatenate([np.full(24, 6.0), np.full(12, 6.0), np.full(12, 1.0)])
    gbp = np.concatenate([np.full(24, 3.0), np.full(12, 3.0), np.full(12, 2.5)])
    wage = pd.DataFrame(
        {
            "USD": np.full(48, 2.5),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(48, 1.0),
            "CAD": np.full(48, 2.0),
            "AUD": np.full(48, 2.5),
            "NZD": np.full(48, 2.0),
        },
        index=idx,
    )
    cfg = WageEarningsFxConfig(signal_lag=0, chg_periods=12)
    scores = prepare_wage_earnings_scores(wage, cfg=cfg)
    last = scores["wage_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_wage_no_lookahead():
    """Mutating future wage must not change earlier factor returns."""
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
    wage = pd.DataFrame(
        {
            c: rng.uniform(-1.0, 8.0, 260)
            for c in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD")
        },
        index=m_idx,
    )
    cfg = WageEarningsFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = wage_earnings_factor_returns(ret, wage, cfg=cfg)
    assert "low_wage_growth_xs" in f1
    assert "high_wage_growth_xs" in f1
    assert "low_wage_z_xs" in f1
    assert "wage_chg_xs" in f1
    assert "us_wage_stress_fx" in f1
    assert "us_wage_haven_usd" in f1
    assert "wage_ew" in f1

    wage2 = wage.copy()
    wage2.iloc[-6:] = wage2.iloc[-6:] + 5.0
    f2 = wage_earnings_factor_returns(ret, wage2, cfg=cfg)

    cut = idx[-120]
    for name in ("low_wage_growth_xs", "wage_chg_xs", "us_wage_stress_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_wage_z_fires_on_elevated():
    """Elevated US wage YoY vs recent history → positive z ≥ +1."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    vals = np.concatenate([np.full(216, 2.0), np.full(24, 8.0)])
    us = pd.Series(vals, index=m_idx, name="USD")
    z = trailing_z(us, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) > 1.0


def test_loader_smoke_and_coverage():
    panel = load_wage_earnings_panel(pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True)
    assert panel.shape[1] >= 6  # USD + foreign excl CHF
    assert "USD" in panel.columns
    assert "EUR" in panel.columns
    assert "NZD" in panel.columns
    assert "CHF" not in panel.columns  # unmapped
    us = load_us_wage_earnings(download=False)
    assert len(us) > 50
    cov = wage_earnings_coverage(panel)
    assert len(cov) >= 7
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("USD") == "LCEAMN01USM657S"
    assert panel.attrs.get("unit") == "wage_earnings_yoy_pct"
    assert panel.attrs.get("score_basis") == "yoy_growth_pct"
    assert DEFAULT_PUB_LAG_MONTHS == 2
    assert "CHF" in panel.attrs.get("unmapped", [])


def test_low_wage_maps_to_top_score():
    """Low wage YoY maps to top low_wage scores."""
    m_idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(60, 4.0), np.linspace(4.0, 0.5, 60)])
    gbp = np.concatenate([np.full(60, 2.0), np.linspace(2.0, 7.0, 60)])
    wage = pd.DataFrame(
        {
            "USD": np.full(120, 2.5),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 1.0),
            "CAD": np.full(120, 2.0),
            "AUD": np.full(120, 2.5),
            "NZD": np.full(120, 2.0),
        },
        index=m_idx,
    )
    cfg = WageEarningsFxConfig(signal_lag=0, n_long=2, n_short=2)
    scores = prepare_wage_earnings_scores(wage, cfg=cfg)
    last = scores["low_wage"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2


def test_n_long_short_a_priori():
    """Available foreign panel → n_long=n_short=2; signal_lag=1 a priori."""
    cfg = WageEarningsFxConfig()
    assert cfg.n_long == 2
    assert cfg.n_short == 2
    assert cfg.signal_lag == 1
    assert cfg.z_high == 1.0
    assert cfg.usd_tilt == 0.5
