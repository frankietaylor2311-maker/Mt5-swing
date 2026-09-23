"""Tests for BIS REER FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_bis_reer import (
    BIS_REER_SERIES,
    DEFAULT_PUB_LAG_MONTHS,
    load_bis_reer_panel,
    load_us_bis_reer,
    reer_coverage,
)
from mt5_swing.strategies.bis_reer_fx import (
    BisReerFxConfig,
    bis_reer_factor_returns,
    prepare_reer_scores,
    trailing_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_reer_series_map_covers_full_g10():
    for ccy in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF"):
        assert ccy in BIS_REER_SERIES
        assert BIS_REER_SERIES[ccy] is not None
        assert BIS_REER_SERIES[ccy].startswith("RB")
        assert BIS_REER_SERIES[ccy].endswith("BIS")
    assert BIS_REER_SERIES["EUR"] == "RBXMBIS"  # euro-area, not Germany


def test_pub_lag_shifts_known_date():
    """pub_lag_months=2 must move month-start index forward by ~2 months."""
    raw = load_bis_reer_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_bis_reer_panel(pub_lag_months=2, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    assert 40 <= delta_days <= 80


def test_prepare_scores_cheap_long_low_reer_z():
    """Low REER (undervalued) → positive reer_cheap score → long that FX."""
    idx = pd.date_range("2015-01-01", periods=80, freq="MS", tz="UTC")
    # Stable then diverge: EUR cheap (low), GBP expensive (high)
    reer = pd.DataFrame(
        {
            "USD": np.full(80, 100.0),
            "EUR": np.concatenate([np.full(60, 100.0), np.linspace(100, 85, 20)]),
            "GBP": np.concatenate([np.full(60, 100.0), np.linspace(100, 115, 20)]),
            "JPY": np.full(80, 100.0),
            "AUD": np.full(80, 100.0),
            "CAD": np.full(80, 100.0),
            "NZD": np.full(80, 100.0),
            "CHF": np.full(80, 100.0),
        },
        index=idx,
    )
    cfg = BisReerFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_reer_scores(reer, cfg=cfg)
    last = scores["reer_cheap"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    # Momentum opposite: high z → long
    last_m = scores["reer_mom"].dropna(how="all").iloc[-1]
    assert float(last_m["GBP"]) > float(last_m["EUR"])


def test_reer_chg_score_prefers_falling_reer():
    """Falling REER (cheapening) → positive reer_chg score."""
    idx = pd.date_range("2018-01-01", periods=36, freq="MS", tz="UTC")
    reer = pd.DataFrame(
        {
            "USD": np.full(36, 100.0),
            "EUR": np.linspace(110, 90, 36),  # falling
            "GBP": np.linspace(90, 110, 36),  # rising
            "JPY": np.full(36, 100.0),
            "AUD": np.full(36, 100.0),
            "CAD": np.full(36, 100.0),
            "NZD": np.full(36, 100.0),
            "CHF": np.full(36, 100.0),
        },
        index=idx,
    )
    cfg = BisReerFxConfig(signal_lag=0, chg_periods=12)
    scores = prepare_reer_scores(reer, cfg=cfg)
    last = scores["reer_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_reer_no_lookahead():
    """Mutating future REER must not change earlier factor returns."""
    idx = pd.date_range("2015-01-01", periods=2000, freq="B", tz="UTC")
    rng = np.random.default_rng(31)
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
    reer = pd.DataFrame(
        {
            c: 100.0 + rng.normal(0, 0.5, 120).cumsum()
            for c in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF")
        },
        index=m_idx,
    )
    cfg = BisReerFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = bis_reer_factor_returns(ret, reer, cfg=cfg)
    assert "reer_cheap_xs" in f1
    assert "reer_cheap_xs_36" in f1
    assert "reer_mom_xs" in f1
    assert "reer_chg_xs" in f1
    assert "us_reer_strong_usd" in f1
    assert "us_reer_meanrev_fx" in f1
    assert "reer_ew" in f1

    reer2 = reer.copy()
    reer2.iloc[-6:] = reer2.iloc[-6:] + 15.0
    f2 = bis_reer_factor_returns(ret, reer2, cfg=cfg)

    cut = idx[-120]
    for name in ("reer_cheap_xs", "reer_chg_xs", "us_reer_meanrev_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_reer_z_fires_on_elevated_level():
    """Elevated US REER vs recent history → positive z."""
    m_idx = pd.date_range("2010-01-01", periods=100, freq="MS", tz="UTC")
    vals = np.full(100, 100.0)
    vals[-3:] = 120.0
    us = pd.Series(vals, index=m_idx, name="USD")
    z = trailing_z(us, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) > 1.0


def test_loader_smoke_and_coverage():
    panel = load_bis_reer_panel(
        pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True
    )
    assert panel.shape[1] >= 7  # full G10 expected
    assert "USD" in panel.columns
    assert "NZD" in panel.columns
    assert "CHF" in panel.columns
    assert "EUR" in panel.columns
    us = load_us_bis_reer(download=False)
    assert len(us) > 100
    cov = reer_coverage(panel)
    assert len(cov) >= 7
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("EUR") == "RBXMBIS"


def test_cheap_maps_low_z_to_top_score():
    """Deep undervaluation (low trailing z) maps to top cheap scores."""
    m_idx = pd.date_range("2014-01-01", periods=80, freq="MS", tz="UTC")
    base = np.full(80, 100.0)
    eur = base.copy()
    eur[-12:] = 88.0  # cheap
    gbp = base.copy()
    gbp[-12:] = 112.0  # expensive
    reer = pd.DataFrame(
        {
            "USD": base,
            "EUR": eur,
            "GBP": gbp,
            "JPY": base,
            "AUD": base + 2.0,
            "CAD": base - 1.0,
            "NZD": base + 1.0,
            "CHF": base - 2.0,
        },
        index=m_idx,
    )
    cfg = BisReerFxConfig(signal_lag=1, n_long=2, n_short=2, z_window=60, min_periods=24)
    scores = prepare_reer_scores(reer, cfg=cfg)
    last = scores["reer_cheap"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2
