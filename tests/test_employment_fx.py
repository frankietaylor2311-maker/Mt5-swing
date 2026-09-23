"""Tests for OECD MEI employment-growth (LFEMTTTT) differential FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_employment import (
    DEFAULT_PUB_LAG_MONTHS,
    EMPLOYMENT_SERIES,
    employment_coverage,
    load_employment_panel,
    load_us_employment,
)
from mt5_swing.strategies.employment_fx import (
    EmploymentFxConfig,
    emp_yoy_pct,
    employment_factor_returns,
    prepare_employment_scores,
    trailing_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_employment_series_map_full_g10():
    for ccy in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF"):
        assert ccy in EMPLOYMENT_SERIES
        assert EMPLOYMENT_SERIES[ccy] is not None
    assert EMPLOYMENT_SERIES["USD"] == "LFEMTTTTUSM647S"
    assert EMPLOYMENT_SERIES["EUR"] == "LFEMTTTTDEQ647S"  # Germany proxy
    assert EMPLOYMENT_SERIES["GBP"] == "LFEMTTTTGBQ647S"
    assert EMPLOYMENT_SERIES["JPY"] == "LFEMTTTTJPM647S"
    assert EMPLOYMENT_SERIES["CAD"] == "LFEMTTTTCAM647S"
    assert EMPLOYMENT_SERIES["AUD"] == "LFEMTTTTAUM647S"
    assert EMPLOYMENT_SERIES["NZD"] == "LFEMTTTTNZQ647S"
    assert EMPLOYMENT_SERIES["CHF"] == "LFEMTTTTCHQ647S"


def test_pub_lag_shifts_known_date():
    """pub_lag_months=3 must move month-start index forward by ~3 months."""
    raw = load_employment_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_employment_panel(pub_lag_months=3, download=False, force=False)
    assert not raw.empty and not lagged.empty
    # Use a monthly series (USD) for clean lag check
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    # ~3 months ≈ 90 days (±30d tolerance)
    assert 60 <= delta_days <= 120


def test_prepare_scores_high_emp_long_strong():
    """Higher emp YoY → higher high_emp score (labour-strength prior)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    # Levels so EUR YoY rises, GBP YoY falls
    eur = np.concatenate([np.full(60, 100.0), np.linspace(100.0, 130.0, 60)])
    gbp = np.concatenate([np.full(60, 100.0), np.linspace(100.0, 70.0, 60)])
    emp = pd.DataFrame(
        {
            "USD": np.full(120, 100.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 100.0),
            "CAD": np.full(120, 100.0),
            "AUD": np.full(120, 100.0),
            "NZD": np.full(120, 100.0),
            "CHF": np.full(120, 100.0),
        },
        index=idx,
    )
    cfg = EmploymentFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_employment_scores(emp, cfg=cfg)
    last = scores["high_emp"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    last_lo = scores["low_emp"].dropna(how="all").iloc[-1]
    assert float(last_lo["GBP"]) > float(last_lo["EUR"])


def test_emp_yoy_and_chg_prefers_acceleration():
    """Rising YoY → positive emp_chg score."""
    idx = pd.date_range("2015-01-01", periods=48, freq="MS", tz="UTC")
    # Build levels from known YoY path so Δ12(YoY) is controlled:
    # months 0-11 base=100; 12-23: EUR +2%/y GBP +10%/y; 24-35 same;
    # 36-47: EUR +20%/y GBP +11%/y → Δ12 YoY: EUR +18pp, GBP +1pp.
    def levels_from_yoy(yoy_path: np.ndarray, base: float = 100.0) -> np.ndarray:
        out = np.empty(len(yoy_path))
        out[:12] = base
        for i in range(12, len(yoy_path)):
            out[i] = out[i - 12] * (1.0 + yoy_path[i])
        return out

    eur_yoy = np.concatenate([np.zeros(12), np.full(24, 0.02), np.full(12, 0.20)])
    gbp_yoy = np.concatenate([np.zeros(12), np.full(24, 0.10), np.full(12, 0.11)])
    emp = pd.DataFrame(
        {
            "USD": np.full(48, 100.0),
            "EUR": levels_from_yoy(eur_yoy),
            "GBP": levels_from_yoy(gbp_yoy),
            "JPY": np.full(48, 100.0),
            "CAD": np.full(48, 100.0),
            "AUD": np.full(48, 100.0),
            "NZD": np.full(48, 100.0),
            "CHF": np.full(48, 100.0),
        },
        index=idx,
    )
    yoy = emp_yoy_pct(emp, periods=12)
    assert yoy.shape[0] == 48
    cfg = EmploymentFxConfig(signal_lag=0, chg_periods=12, yoy_periods=12)
    scores = prepare_employment_scores(emp, cfg=cfg)
    last = scores["emp_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_employment_no_lookahead():
    """Mutating future employment must not change earlier factor returns."""
    idx = pd.date_range("2015-01-01", periods=2000, freq="B", tz="UTC")
    rng = np.random.default_rng(41)
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
    emp = pd.DataFrame(
        {
            c: 1e6 * np.cumprod(1 + rng.normal(0.001, 0.003, 260))
            for c in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF")
        },
        index=m_idx,
    )
    cfg = EmploymentFxConfig(signal_lag=0, cost_bps_side=1.5)
    f1 = employment_factor_returns(ret, emp, cfg=cfg)
    assert "high_emp_xs" in f1
    assert "low_emp_xs" in f1
    assert "high_emp_z_xs" in f1
    assert "emp_chg_xs" in f1
    assert "us_emp_stress_fx" in f1
    assert "us_emp_haven_usd" in f1
    assert "emp_ew" in f1

    emp2 = emp.copy()
    emp2.iloc[-6:] = emp2.iloc[-6:] * 1.15
    f2 = employment_factor_returns(ret, emp2, cfg=cfg)

    cut = idx[-120]
    for name in ("high_emp_xs", "emp_chg_xs", "us_emp_stress_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_emp_z_fires_on_depressed_growth():
    """Depressed US emp YoY vs recent history → negative z."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    # Steady growth then collapse
    vals = np.cumprod(np.concatenate([np.full(216, 1.002), np.full(24, 0.985)])) * 1e8
    us = pd.Series(vals, index=m_idx, name="USD")
    yoy = us.pct_change(12)
    z = trailing_z(yoy, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) < -1.0


def test_loader_smoke_and_coverage():
    panel = load_employment_panel(
        pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True
    )
    assert panel.shape[1] >= 7  # full G10 incl USD
    assert "USD" in panel.columns
    assert "EUR" in panel.columns
    assert "CHF" in panel.columns
    assert "NZD" in panel.columns
    us = load_us_employment(download=False)
    assert len(us) > 50
    cov = employment_coverage(panel)
    assert len(cov) >= 8
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("EUR") == "LFEMTTTTDEQ647S"
    assert panel.attrs.get("unit") == "employment_persons_level"
    assert panel.attrs.get("score_basis") == "yoy_pct_of_levels"
    assert DEFAULT_PUB_LAG_MONTHS == 3
    # PIT end should reach into 2025+ after pub lag
    end = panel.dropna(how="all").index.max()
    assert end.year >= 2025


def test_high_emp_maps_to_top_score():
    """High emp YoY maps to top high_emp scores."""
    m_idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(60, 100.0), np.linspace(100.0, 140.0, 60)])
    gbp = np.concatenate([np.full(60, 100.0), np.linspace(100.0, 90.0, 60)])
    emp = pd.DataFrame(
        {
            "USD": np.full(120, 100.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 100.0),
            "CAD": np.full(120, 100.0),
            "AUD": np.full(120, 100.0),
            "NZD": np.full(120, 100.0),
            "CHF": np.full(120, 100.0),
        },
        index=m_idx,
    )
    cfg = EmploymentFxConfig(signal_lag=0, n_long=2, n_short=2)
    scores = prepare_employment_scores(emp, cfg=cfg)
    last = scores["high_emp"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2


def test_n_long_short_a_priori_full_g10():
    """Full G10 foreign panel → n_long=n_short=2 a priori."""
    cfg = EmploymentFxConfig()
    assert cfg.n_long == 2
    assert cfg.n_short == 2


def test_quarterly_expand_no_intra_quarter_leak():
    """After pub lag, monthly values within a known quarter equal the lagged Q obs."""
    q_idx = pd.date_range("2020-01-01", periods=8, freq="QS", tz="UTC")
    raw = pd.DataFrame(
        {"EUR": np.arange(8, dtype=float) * 1e6 + 4e7},
        index=q_idx,
    )
    from mt5_swing.data.fred_employment import _expand_quarterly_to_monthly

    monthly = _expand_quarterly_to_monthly(raw, pub_lag_months=3)
    assert monthly.index.min() >= pd.Timestamp("2020-04-01", tz="UTC")
    first_val = float(monthly["EUR"].dropna().iloc[0])
    s = monthly["EUR"].dropna()
    assert float(s.iloc[1]) == first_val
    assert float(s.iloc[2]) == first_val
