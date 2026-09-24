"""Tests for OECD/FRED real GFCF / investment growth (NAEXKP04) FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_gfcf import (
    DEFAULT_PUB_LAG_MONTHS,
    GFCF_SERIES,
    load_gfcf_panel,
    load_us_gfcf,
    gfcf_coverage,
)
from mt5_swing.strategies.gfcf_fx import (
    GfcfFxConfig,
    prepare_gfcf_scores,
    trailing_z,
    gfcf_factor_returns,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_gfcf_series_map():
    assert GFCF_SERIES["USD"] == "NAEXKP04USQ657S"
    assert GFCF_SERIES["EUR"] == "NAEXKP04DEQ657S"  # Germany proxy
    assert GFCF_SERIES["GBP"] == "NAEXKP04GBQ657S"
    assert GFCF_SERIES["JPY"] == "NAEXKP04JPQ657S"
    assert GFCF_SERIES["CAD"] == "NAEXKP04CAQ657S"
    assert GFCF_SERIES["AUD"] == "NAEXKP04AUQ657S"
    assert GFCF_SERIES["CHF"] == "NAEXKP04CHQ657S"
    assert GFCF_SERIES["NZD"] == "NAEXKP04NZQ657S"


def test_pub_lag_shifts_known_date():
    """pub_lag_months=3 must move month-start index forward by ~3 months."""
    raw = load_gfcf_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_gfcf_panel(pub_lag_months=3, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    assert 75 <= delta_days <= 105


def test_prepare_scores_high_gfcf_long_growth():
    """Higher GFCF YoY → higher high_gfcf score (investment boom / economic-momentum prior)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    # EUR investment rises, GBP investment falls
    eur = np.concatenate([np.full(60, 1.5), np.linspace(1.5, 4.0, 60)])
    gbp = np.concatenate([np.full(60, 2.0), np.linspace(2.0, 0.0, 60)])
    gfcf = pd.DataFrame(
        {
            "USD": np.full(120, 2.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 1.0),
            "CAD": np.full(120, 2.2),
            "AUD": np.full(120, 2.5),
            "CHF": np.full(120, 1.8),
            "NZD": np.full(120, 2.3),
        },
        index=idx,
    )
    cfg = GfcfFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_gfcf_scores(gfcf, cfg=cfg)
    last = scores["high_gfcf"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    last_lo = scores["low_gfcf"].dropna(how="all").iloc[-1]
    assert float(last_lo["GBP"]) > float(last_lo["EUR"])


def test_gfcf_chg_prefers_acceleration():
    """Accelerating GFCF YoY → positive gfcf_chg score (+Δ12 of YoY)."""
    idx = pd.date_range("2015-01-01", periods=48, freq="MS", tz="UTC")
    # EUR GFCF YoY rises 1→4 over last year vs GBP stays near 2→2.2
    eur = np.concatenate([np.full(24, 1.0), np.full(12, 1.0), np.full(12, 4.0)])
    gbp = np.concatenate([np.full(24, 2.0), np.full(12, 2.0), np.full(12, 2.2)])
    gfcf = pd.DataFrame(
        {
            "USD": np.full(48, 2.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(48, 1.0),
            "CAD": np.full(48, 2.2),
            "AUD": np.full(48, 2.5),
            "CHF": np.full(48, 1.8),
            "NZD": np.full(48, 2.3),
        },
        index=idx,
    )
    cfg = GfcfFxConfig(signal_lag=0, chg_periods=12)
    scores = prepare_gfcf_scores(gfcf, cfg=cfg)
    last = scores["gfcf_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_gfcf_no_lookahead():
    """Mutating future GFCF must not change earlier factor returns."""
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
    gfcf = pd.DataFrame(
        {
            c: rng.uniform(-1.0, 5.0, 260)
            for c in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "NZD")
        },
        index=m_idx,
    )
    cfg = GfcfFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = gfcf_factor_returns(ret, gfcf, cfg=cfg)
    assert "high_gfcf_xs" in f1
    assert "low_gfcf_xs" in f1
    assert "high_gfcf_z_xs" in f1
    assert "gfcf_chg_xs" in f1
    assert "us_gfcf_stress_fx" in f1
    assert "us_gfcf_haven_usd" in f1
    assert "gfcf_ew" in f1

    gfcf2 = gfcf.copy()
    gfcf2.iloc[-6:] = gfcf2.iloc[-6:] + 3.0
    f2 = gfcf_factor_returns(ret, gfcf2, cfg=cfg)

    cut = idx[-120]
    for name in ("high_gfcf_xs", "gfcf_chg_xs", "us_gfcf_stress_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_gfcf_z_fires_on_depressed():
    """Depressed US GFCF YoY vs recent history → negative z ≤ −1."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    vals = np.concatenate([np.full(216, 2.5), np.full(24, -1.0)])
    us = pd.Series(vals, index=m_idx, name="USD")
    z = trailing_z(us, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) < -1.0


def test_loader_smoke_and_coverage():
    panel = load_gfcf_panel(pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True)
    assert panel.shape[1] >= 7  # full G10 mapped
    assert "USD" in panel.columns
    assert "EUR" in panel.columns
    assert "CHF" in panel.columns
    assert "NZD" in panel.columns
    us = load_us_gfcf(download=False)
    assert len(us) > 50
    cov = gfcf_coverage(panel)
    assert len(cov) >= 7
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("USD") == "NAEXKP04USQ657S"
    assert panel.attrs.get("series_map", {}).get("EUR") == "NAEXKP04DEQ657S"
    assert panel.attrs.get("unit") == "yoy_pct_compounded"
    assert panel.attrs.get("score_basis") == "gfcf_yoy_pct"
    assert DEFAULT_PUB_LAG_MONTHS == 3
    # YoY should be in a sensible percent range (not raw levels)
    last_us = float(panel["USD"].dropna().iloc[-1])
    assert -15.0 < last_us < 15.0


def test_high_gfcf_maps_to_top_score():
    """High GFCF YoY maps to top high_gfcf scores."""
    m_idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(60, 1.0), np.linspace(1.0, 5.0, 60)])
    gbp = np.concatenate([np.full(60, 2.0), np.linspace(2.0, 0.0, 60)])
    gfcf = pd.DataFrame(
        {
            "USD": np.full(120, 2.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 1.0),
            "CAD": np.full(120, 2.2),
            "AUD": np.full(120, 2.5),
            "CHF": np.full(120, 1.8),
            "NZD": np.full(120, 2.3),
        },
        index=m_idx,
    )
    cfg = GfcfFxConfig(signal_lag=0, n_long=2, n_short=2)
    scores = prepare_gfcf_scores(gfcf, cfg=cfg)
    last = scores["high_gfcf"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2


def test_n_long_short_a_priori():
    """Full G10 foreign panel → n_long=n_short=2; signal_lag=1; z_low=-1.0; pub_lag=3."""
    cfg = GfcfFxConfig()
    assert cfg.n_long == 2
    assert cfg.n_short == 2
    assert cfg.signal_lag == 1
    assert cfg.z_low == -1.0
    assert cfg.usd_tilt == 0.5
    assert DEFAULT_PUB_LAG_MONTHS == 3
