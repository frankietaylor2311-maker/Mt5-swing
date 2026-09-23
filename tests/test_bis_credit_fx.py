"""Tests for BIS private credit-to-GDP / credit-gap FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_bis_credit import (
    BIS_CREDIT_GDP_SERIES,
    DEFAULT_PUB_LAG_MONTHS,
    credit_coverage,
    load_bis_credit_panel,
    load_us_bis_credit,
)
from mt5_swing.strategies.bis_credit_fx import (
    BisCreditFxConfig,
    bis_credit_factor_returns,
    credit_gap_from_level,
    prepare_credit_scores,
    trailing_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_credit_series_map_covers_full_g10():
    for ccy in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF"):
        assert ccy in BIS_CREDIT_GDP_SERIES
        assert BIS_CREDIT_GDP_SERIES[ccy] is not None
        assert BIS_CREDIT_GDP_SERIES[ccy].endswith("PAM770A")
    assert BIS_CREDIT_GDP_SERIES["EUR"] == "QXMPAM770A"  # euro-area, not Germany
    assert BIS_CREDIT_GDP_SERIES["USD"] == "QUSPAM770A"


def test_pub_lag_shifts_known_date():
    """pub_lag_months=5 must move quarter-start index forward by ~5 months."""
    raw = load_bis_credit_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_bis_credit_panel(pub_lag_months=5, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    # ~5 months ≈ 150 days (±45d tolerance for quarter→month expand)
    assert 100 <= delta_days <= 200


def test_prepare_scores_low_gap_long_negative_gap():
    """Negative credit gap (below trend) → positive low_credit_gap score."""
    idx = pd.date_range("2005-01-01", periods=240, freq="MS", tz="UTC")
    # Stable then diverge: EUR deleverages (falling), GBP leverages (rising)
    base = np.full(240, 100.0)
    eur = base.copy()
    eur[-60:] = np.linspace(100, 70, 60)  # falling → negative gap
    gbp = base.copy()
    gbp[-60:] = np.linspace(100, 140, 60)  # rising → positive gap
    credit = pd.DataFrame(
        {
            "USD": base,
            "EUR": eur,
            "GBP": gbp,
            "JPY": base,
            "AUD": base + 5.0,
            "CAD": base - 3.0,
            "NZD": base + 2.0,
            "CHF": base - 1.0,
        },
        index=idx,
    )
    cfg = BisCreditFxConfig(
        signal_lag=0, gap_lookback=180, gap_min_periods=60, z_window=60, min_periods=24
    )
    scores = prepare_credit_scores(credit, cfg=cfg)
    last = scores["low_credit_gap"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    # Level sort: low credit → long
    last_lvl = scores["low_credit"].dropna(how="all").iloc[-1]
    assert float(last_lvl["EUR"]) > float(last_lvl["GBP"])
    # High credit opposite
    last_hi = scores["high_credit"].dropna(how="all").iloc[-1]
    assert float(last_hi["GBP"]) > float(last_hi["EUR"])


def test_credit_chg_score_prefers_falling_credit():
    """Falling credit/GDP → positive credit_chg score."""
    idx = pd.date_range("2018-01-01", periods=36, freq="MS", tz="UTC")
    credit = pd.DataFrame(
        {
            "USD": np.full(36, 100.0),
            "EUR": np.linspace(120.0, 80.0, 36),  # falling
            "GBP": np.linspace(80.0, 120.0, 36),  # rising
            "JPY": np.full(36, 150.0),
            "AUD": np.full(36, 90.0),
            "CAD": np.full(36, 110.0),
            "NZD": np.full(36, 95.0),
            "CHF": np.full(36, 200.0),
        },
        index=idx,
    )
    cfg = BisCreditFxConfig(signal_lag=0, chg_periods=12)
    scores = prepare_credit_scores(credit, cfg=cfg)
    last = scores["credit_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_credit_gap_causal_trailing_mean():
    """Gap equals level − trailing mean; early NaN until min_periods."""
    idx = pd.date_range("2010-01-01", periods=100, freq="MS", tz="UTC")
    s = pd.Series(np.linspace(100, 150, 100), index=idx)
    panel = pd.DataFrame({"EUR": s})
    gap = credit_gap_from_level(panel, lookback=60, min_periods=60)
    assert gap["EUR"].iloc[:59].isna().all()
    assert np.isfinite(gap["EUR"].iloc[-1])
    # Rising series → last gap positive (above long trailing mean)
    assert float(gap["EUR"].iloc[-1]) > 0


def test_credit_no_lookahead():
    """Mutating future credit must not change earlier factor returns."""
    idx = pd.date_range("2015-01-01", periods=2000, freq="B", tz="UTC")
    rng = np.random.default_rng(32)
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
    # Extend credit past daily sample end so mutating last 6m cannot leak into cut
    m_idx = pd.date_range("2005-01-01", periods=260, freq="MS", tz="UTC")
    credit = pd.DataFrame(
        {
            c: 100.0 + rng.normal(0, 0.8, 260).cumsum()
            for c in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF")
        },
        index=m_idx,
    )
    cfg = BisCreditFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = bis_credit_factor_returns(ret, credit, cfg=cfg)
    assert "low_credit_gap_xs" in f1
    assert "low_credit_xs" in f1
    assert "high_credit_xs" in f1
    assert "low_credit_z_xs" in f1
    assert "credit_chg_xs" in f1
    assert "us_credit_stress_fx" in f1
    assert "us_credit_haven_usd" in f1
    assert "credit_ew" in f1

    credit2 = credit.copy()
    credit2.iloc[-6:] = credit2.iloc[-6:] + 25.0
    f2 = bis_credit_factor_returns(ret, credit2, cfg=cfg)

    cut = idx[-120]
    for name in ("low_credit_gap_xs", "credit_chg_xs", "us_credit_stress_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_credit_z_fires_on_elevated_gap():
    """Elevated US credit gap vs recent history → positive z."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    vals = np.full(240, 100.0)
    vals[-12:] = 160.0  # sharp credit boom
    us = pd.Series(vals, index=m_idx, name="USD")
    gap = us - us.rolling(180, min_periods=60).mean()
    z = trailing_z(gap, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) > 1.0


def test_loader_smoke_and_coverage():
    panel = load_bis_credit_panel(
        pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True
    )
    assert panel.shape[1] >= 7  # full G10 expected via Q*PAM770A
    assert "USD" in panel.columns
    assert "NZD" in panel.columns
    assert "CHF" in panel.columns
    assert "EUR" in panel.columns
    us = load_us_bis_credit(download=False)
    assert len(us) > 100
    # % of GDP scale, not absolute bn
    assert 20.0 < float(us.dropna().iloc[-1]) < 400.0
    cov = credit_coverage(panel)
    assert len(cov) >= 7
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("EUR") == "QXMPAM770A"
    assert panel.attrs.get("unit") == "percent_of_gdp"


def test_low_gap_maps_to_top_score():
    """Deep negative gap maps to top low_credit_gap scores."""
    m_idx = pd.date_range("2005-01-01", periods=240, freq="MS", tz="UTC")
    base = np.full(240, 120.0)
    eur = base.copy()
    eur[-48:] = 80.0  # deep deleveraging
    gbp = base.copy()
    gbp[-48:] = 180.0  # boom
    credit = pd.DataFrame(
        {
            "USD": base,
            "EUR": eur,
            "GBP": gbp,
            "JPY": base + 10.0,
            "AUD": base - 5.0,
            "CAD": base + 2.0,
            "NZD": base - 2.0,
            "CHF": base + 40.0,
        },
        index=m_idx,
    )
    cfg = BisCreditFxConfig(
        signal_lag=1, n_long=2, n_short=2, gap_lookback=180, gap_min_periods=60
    )
    scores = prepare_credit_scores(credit, cfg=cfg)
    last = scores["low_credit_gap"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2
