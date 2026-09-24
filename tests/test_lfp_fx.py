"""Tests for OECD/FRED labour-force participation / activity-rate FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_lfp import (
    DEFAULT_PUB_LAG_MONTHS,
    LFP_SERIES,
    load_lfp_panel,
    load_us_lfp,
    lfp_coverage,
)
from mt5_swing.strategies.lfp_fx import (
    LfpFxConfig,
    prepare_lfp_scores,
    trailing_z,
    lfp_factor_returns,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_lfp_series_map():
    assert LFP_SERIES["USD"] == "LRAC64TTUSM156S"
    assert LFP_SERIES["JPY"] == "LRAC64TTJPM156S"
    assert LFP_SERIES["CAD"] == "LRAC64TTCAM156S"
    assert LFP_SERIES["AUD"] == "LRAC64TTAUM156S"
    assert LFP_SERIES["GBP"] == "LRAC64TTGBQ156S"
    assert LFP_SERIES["EUR"] == "LRAC64TTDEQ156S"  # Germany proxy
    assert LFP_SERIES["CHF"] == "LRAC64TTCHQ156S"
    assert LFP_SERIES["NZD"] == "LRAC64TTNZQ156S"


def test_pub_lag_shifts_known_date():
    """pub_lag_months=2 must move month-start index forward by ~2 months."""
    raw = load_lfp_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_lfp_panel(pub_lag_months=2, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    assert 45 <= delta_days <= 75


def test_prepare_scores_high_lfp_long_engagement():
    """Higher LFP → higher high_lfp score (labour-engagement prior)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    # EUR LFP rises (engagement), GBP LFP falls
    eur = np.concatenate([np.full(60, 70.0), np.linspace(70.0, 78.0, 60)])
    gbp = np.concatenate([np.full(60, 75.0), np.linspace(75.0, 68.0, 60)])
    lfp = pd.DataFrame(
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
    cfg = LfpFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_lfp_scores(lfp, cfg=cfg)
    last = scores["high_lfp"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    last_lo = scores["low_lfp"].dropna(how="all").iloc[-1]
    assert float(last_lo["GBP"]) > float(last_lo["EUR"])


def test_lfp_chg_prefers_rising():
    """Rising LFP → positive lfp_chg score (+Δ12 of level)."""
    idx = pd.date_range("2015-01-01", periods=48, freq="MS", tz="UTC")
    # EUR LFP rises 70 → 78 over last year vs GBP stays near 75→75.5
    eur = np.concatenate([np.full(24, 70.0), np.full(12, 70.0), np.full(12, 78.0)])
    gbp = np.concatenate([np.full(24, 75.0), np.full(12, 75.0), np.full(12, 75.5)])
    lfp = pd.DataFrame(
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
    cfg = LfpFxConfig(signal_lag=0, chg_periods=12)
    scores = prepare_lfp_scores(lfp, cfg=cfg)
    last = scores["lfp_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_lfp_no_lookahead():
    """Mutating future LFP must not change earlier factor returns."""
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
    lfp = pd.DataFrame(
        {
            c: rng.uniform(65.0, 85.0, 260)
            for c in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "NZD")
        },
        index=m_idx,
    )
    cfg = LfpFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = lfp_factor_returns(ret, lfp, cfg=cfg)
    assert "high_lfp_xs" in f1
    assert "low_lfp_xs" in f1
    assert "high_lfp_z_xs" in f1
    assert "lfp_chg_xs" in f1
    assert "us_lfp_stress_fx" in f1
    assert "us_lfp_haven_usd" in f1
    assert "lfp_ew" in f1

    lfp2 = lfp.copy()
    lfp2.iloc[-6:] = lfp2.iloc[-6:] + 5.0
    f2 = lfp_factor_returns(ret, lfp2, cfg=cfg)

    cut = idx[-120]
    for name in ("high_lfp_xs", "lfp_chg_xs", "us_lfp_stress_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_lfp_z_fires_on_depressed():
    """Depressed US LFP vs recent history → negative z ≤ −1."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    vals = np.concatenate([np.full(216, 75.0), np.full(24, 68.0)])
    us = pd.Series(vals, index=m_idx, name="USD")
    z = trailing_z(us, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) < -1.0


def test_loader_smoke_and_coverage():
    panel = load_lfp_panel(pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True)
    assert panel.shape[1] >= 7  # full G10 mapped
    assert "USD" in panel.columns
    assert "EUR" in panel.columns
    assert "CHF" in panel.columns
    assert "NZD" in panel.columns
    us = load_us_lfp(download=False)
    assert len(us) > 50
    cov = lfp_coverage(panel)
    assert len(cov) >= 7
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("USD") == "LRAC64TTUSM156S"
    assert panel.attrs.get("series_map", {}).get("EUR") == "LRAC64TTDEQ156S"
    assert panel.attrs.get("unit") == "lfp_activity_rate_pct_level"
    assert panel.attrs.get("score_basis") == "lfp_level_pct"
    assert DEFAULT_PUB_LAG_MONTHS == 2


def test_high_lfp_maps_to_top_score():
    """High LFP maps to top high_lfp scores."""
    m_idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(60, 70.0), np.linspace(70.0, 80.0, 60)])
    gbp = np.concatenate([np.full(60, 75.0), np.linspace(75.0, 68.0, 60)])
    lfp = pd.DataFrame(
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
    cfg = LfpFxConfig(signal_lag=0, n_long=2, n_short=2)
    scores = prepare_lfp_scores(lfp, cfg=cfg)
    last = scores["high_lfp"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2


def test_n_long_short_a_priori():
    """Full G10 foreign panel → n_long=n_short=2; signal_lag=1; z_low=-1.0."""
    cfg = LfpFxConfig()
    assert cfg.n_long == 2
    assert cfg.n_short == 2
    assert cfg.signal_lag == 1
    assert cfg.z_low == -1.0
    assert cfg.usd_tilt == 0.5
