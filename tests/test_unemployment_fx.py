"""Tests for OECD/FRED unemployment-rate differential FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_unemployment import (
    DEFAULT_PUB_LAG_MONTHS,
    UNEMPLOYMENT_SERIES,
    load_unemployment_panel,
    load_us_unemployment,
    unemployment_coverage,
)
from mt5_swing.strategies.unemployment_fx import (
    UnemploymentFxConfig,
    prepare_unemployment_scores,
    trailing_z,
    unemployment_factor_returns,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_ur_series_map():
    assert UNEMPLOYMENT_SERIES["USD"] == "LRHUTTTTUSM156S"
    assert UNEMPLOYMENT_SERIES["EUR"] == "LRHUTTTTEZM156S"
    assert UNEMPLOYMENT_SERIES["GBP"] == "LRHUTTTTGBM156S"
    assert UNEMPLOYMENT_SERIES["JPY"] == "LRHUTTTTJPM156S"
    assert UNEMPLOYMENT_SERIES["AUD"] == "LRHUTTTTAUM156S"
    assert UNEMPLOYMENT_SERIES["CAD"] == "LRHUTTTTCAM156S"
    assert UNEMPLOYMENT_SERIES["CHF"] == "LMUNRRTTCHM156S"
    assert UNEMPLOYMENT_SERIES["NZD"] is None  # unmapped on FRED


def test_pub_lag_shifts_known_date():
    """pub_lag_months=1 must move month-start index forward by ~1 month."""
    raw = load_unemployment_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_unemployment_panel(pub_lag_months=1, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    assert 20 <= delta_days <= 45


def test_prepare_scores_low_ur_long_strong_labour():
    """Lower UR → higher low_ur score (labour-strength prior)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    # EUR UR falls (strong labour), GBP UR rises
    eur = np.concatenate([np.full(60, 8.0), np.linspace(8.0, 3.0, 60)])
    gbp = np.concatenate([np.full(60, 5.0), np.linspace(5.0, 12.0, 60)])
    ur = pd.DataFrame(
        {
            "USD": np.full(120, 5.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 4.0),
            "CAD": np.full(120, 6.0),
            "AUD": np.full(120, 5.5),
            "CHF": np.full(120, 3.0),
        },
        index=idx,
    )
    cfg = UnemploymentFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_unemployment_scores(ur, cfg=cfg)
    last = scores["low_ur"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    last_hi = scores["high_ur"].dropna(how="all").iloc[-1]
    assert float(last_hi["GBP"]) > float(last_hi["EUR"])


def test_ur_chg_prefers_falling():
    """Falling UR → positive ur_chg score (−Δ12 of level)."""
    idx = pd.date_range("2015-01-01", periods=48, freq="MS", tz="UTC")
    # EUR UR drops 10 → 4 over last year vs GBP stays near 6→5.5
    eur = np.concatenate([np.full(24, 10.0), np.full(12, 10.0), np.full(12, 4.0)])
    gbp = np.concatenate([np.full(24, 6.0), np.full(12, 6.0), np.full(12, 5.5)])
    ur = pd.DataFrame(
        {
            "USD": np.full(48, 5.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(48, 3.5),
            "CAD": np.full(48, 6.0),
            "AUD": np.full(48, 5.0),
            "CHF": np.full(48, 3.0),
        },
        index=idx,
    )
    cfg = UnemploymentFxConfig(signal_lag=0, chg_periods=12)
    scores = prepare_unemployment_scores(ur, cfg=cfg)
    last = scores["ur_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_ur_no_lookahead():
    """Mutating future UR must not change earlier factor returns."""
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
    ur = pd.DataFrame(
        {
            c: rng.uniform(3.0, 10.0, 260)
            for c in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF")
        },
        index=m_idx,
    )
    cfg = UnemploymentFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = unemployment_factor_returns(ret, ur, cfg=cfg)
    assert "low_ur_xs" in f1
    assert "high_ur_xs" in f1
    assert "low_ur_z_xs" in f1
    assert "ur_chg_xs" in f1
    assert "us_ur_stress_fx" in f1
    assert "us_ur_haven_usd" in f1
    assert "ur_ew" in f1

    ur2 = ur.copy()
    ur2.iloc[-6:] = ur2.iloc[-6:] + 5.0
    f2 = unemployment_factor_returns(ret, ur2, cfg=cfg)

    cut = idx[-120]
    for name in ("low_ur_xs", "ur_chg_xs", "us_ur_stress_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_ur_z_fires_on_elevated():
    """Elevated US UR vs recent history → positive z ≥ +1."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    vals = np.concatenate([np.full(216, 5.0), np.full(24, 12.0)])
    us = pd.Series(vals, index=m_idx, name="USD")
    z = trailing_z(us, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) > 1.0


def test_loader_smoke_and_coverage():
    panel = load_unemployment_panel(pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True)
    assert panel.shape[1] >= 6  # USD + foreign excl NZD
    assert "USD" in panel.columns
    assert "EUR" in panel.columns
    assert "CHF" in panel.columns
    assert "NZD" not in panel.columns  # unmapped
    us = load_us_unemployment(download=False)
    assert len(us) > 50
    cov = unemployment_coverage(panel)
    assert len(cov) >= 7
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("USD") == "LRHUTTTTUSM156S"
    assert panel.attrs.get("unit") == "unemployment_rate_pct_level"
    assert panel.attrs.get("score_basis") == "ur_level_pct"
    assert DEFAULT_PUB_LAG_MONTHS == 1
    assert "NZD" in panel.attrs.get("unmapped", [])


def test_low_ur_maps_to_top_score():
    """Low UR maps to top low_ur scores."""
    m_idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(60, 8.0), np.linspace(8.0, 2.0, 60)])
    gbp = np.concatenate([np.full(60, 5.0), np.linspace(5.0, 11.0, 60)])
    ur = pd.DataFrame(
        {
            "USD": np.full(120, 5.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 4.0),
            "CAD": np.full(120, 6.0),
            "AUD": np.full(120, 5.5),
            "CHF": np.full(120, 3.5),
        },
        index=m_idx,
    )
    cfg = UnemploymentFxConfig(signal_lag=0, n_long=2, n_short=2)
    scores = prepare_unemployment_scores(ur, cfg=cfg)
    last = scores["low_ur"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2


def test_n_long_short_a_priori():
    """Available foreign panel → n_long=n_short=2; signal_lag=1 a priori."""
    cfg = UnemploymentFxConfig()
    assert cfg.n_long == 2
    assert cfg.n_short == 2
    assert cfg.signal_lag == 1
    assert cfg.z_high == 1.0
