"""Tests for OECD MEI unit labour cost (ULQEUL01) differential FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_ulc import (
    DEFAULT_PUB_LAG_MONTHS,
    ULC_SERIES,
    load_ulc_panel,
    load_us_ulc,
    ulc_coverage,
)
from mt5_swing.strategies.ulc_fx import (
    UlcFxConfig,
    prepare_ulc_scores,
    trailing_z,
    ulc_factor_returns,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_ulc_series_map_full_g10():
    for ccy in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF"):
        assert ccy in ULC_SERIES
        assert ULC_SERIES[ccy] is not None
    assert ULC_SERIES["USD"] == "ULQEUL01USQ657S"
    assert ULC_SERIES["EUR"] == "ULQEUL01DEQ657S"  # Germany proxy
    assert ULC_SERIES["GBP"] == "ULQEUL01GBQ657S"
    assert ULC_SERIES["JPY"] == "ULQEUL01JPQ657S"
    assert ULC_SERIES["CAD"] == "ULQEUL01CAQ657S"
    assert ULC_SERIES["AUD"] == "ULQEUL01AUQ657S"
    assert ULC_SERIES["NZD"] == "ULQEUL01NZQ657S"
    assert ULC_SERIES["CHF"] == "ULQEUL01CHQ657S"


def test_pub_lag_shifts_known_date():
    """pub_lag_months=3 must move quarter-start index forward by ~3 months."""
    raw = load_ulc_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_ulc_panel(pub_lag_months=3, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    # ~3 months ≈ 90 days (±45d tolerance)
    assert 45 <= delta_days <= 135


def test_prepare_scores_low_ulc_long_cheap():
    """Lower ULC YoY → higher low_ulc score (competitiveness prior)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    # EUR ULC rises (expensive), GBP falls (cheap)
    eur = np.concatenate([np.full(60, 1.0), np.linspace(1.0, 8.0, 60)])
    gbp = np.concatenate([np.full(60, 1.0), np.linspace(1.0, -5.0, 60)])
    ulc = pd.DataFrame(
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
    cfg = UlcFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_ulc_scores(ulc, cfg=cfg)
    last = scores["low_ulc"].dropna(how="all").iloc[-1]
    assert float(last["GBP"]) > float(last["EUR"])
    last_hi = scores["high_ulc"].dropna(how="all").iloc[-1]
    assert float(last_hi["EUR"]) > float(last_hi["GBP"])


def test_ulc_chg_prefers_deceleration():
    """Falling YoY → positive ulc_chg score (competitiveness)."""
    idx = pd.date_range("2015-01-01", periods=48, freq="MS", tz="UTC")
    # EUR YoY decelerates 20 → 2; GBP stays near 10→11
    eur_yoy = np.concatenate([np.zeros(12), np.full(24, 20.0), np.full(12, 2.0)])
    gbp_yoy = np.concatenate([np.zeros(12), np.full(24, 10.0), np.full(12, 11.0)])
    ulc = pd.DataFrame(
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
    cfg = UlcFxConfig(signal_lag=0, chg_periods=12)
    scores = prepare_ulc_scores(ulc, cfg=cfg)
    last = scores["ulc_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_ulc_no_lookahead():
    """Mutating future ULC must not change earlier factor returns."""
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
    ulc = pd.DataFrame(
        {
            c: rng.normal(1.0, 2.0, 260)
            for c in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF")
        },
        index=m_idx,
    )
    cfg = UlcFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = ulc_factor_returns(ret, ulc, cfg=cfg)
    assert "low_ulc_xs" in f1
    assert "high_ulc_xs" in f1
    assert "low_ulc_z_xs" in f1
    assert "ulc_chg_xs" in f1
    assert "us_ulc_stress_fx" in f1
    assert "us_ulc_haven_usd" in f1
    assert "ulc_ew" in f1

    ulc2 = ulc.copy()
    ulc2.iloc[-6:] = ulc2.iloc[-6:] + 15.0
    f2 = ulc_factor_returns(ret, ulc2, cfg=cfg)

    cut = idx[-120]
    for name in ("low_ulc_xs", "ulc_chg_xs", "us_ulc_stress_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_ulc_z_fires_on_elevated_growth():
    """Elevated US ULC YoY vs recent history → positive z."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    vals = np.concatenate([np.full(216, 1.0), np.full(24, 8.0)])
    us = pd.Series(vals, index=m_idx, name="USD")
    z = trailing_z(us, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) > 1.0


def test_loader_smoke_and_coverage():
    panel = load_ulc_panel(pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True)
    assert panel.shape[1] >= 7  # full G10 incl USD
    assert "USD" in panel.columns
    assert "EUR" in panel.columns
    assert "CHF" in panel.columns
    assert "NZD" in panel.columns
    assert "AUD" in panel.columns
    us = load_us_ulc(download=False)
    assert len(us) > 50
    cov = ulc_coverage(panel)
    assert len(cov) >= 8
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("EUR") == "ULQEUL01DEQ657S"
    assert panel.attrs.get("unit") == "ulc_yoy_pct_as_reported"
    assert panel.attrs.get("score_basis") == "yoy_growth_as_reported"
    assert DEFAULT_PUB_LAG_MONTHS == 3
    end = panel.dropna(how="all").index.max()
    assert end.year >= 2025


def test_low_ulc_maps_to_top_score():
    """Low ULC YoY maps to top low_ulc scores."""
    m_idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(60, 1.0), np.linspace(1.0, 10.0, 60)])
    gbp = np.concatenate([np.full(60, 1.0), np.linspace(1.0, -2.0, 60)])
    ulc = pd.DataFrame(
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
    cfg = UlcFxConfig(signal_lag=0, n_long=2, n_short=2)
    scores = prepare_ulc_scores(ulc, cfg=cfg)
    last = scores["low_ulc"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "GBP" in top2


def test_n_long_short_a_priori_full_g10():
    """Full G10 foreign panel → n_long=n_short=2 a priori."""
    cfg = UlcFxConfig()
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
    from mt5_swing.data.fred_ulc import _expand_quarterly_to_monthly

    monthly = _expand_quarterly_to_monthly(raw, pub_lag_months=3)
    assert monthly.index.min() >= pd.Timestamp("2020-04-01", tz="UTC")
    first_val = float(monthly["AUD"].dropna().iloc[0])
    s = monthly["AUD"].dropna()
    assert float(s.iloc[1]) == first_val
    assert float(s.iloc[2]) == first_val
