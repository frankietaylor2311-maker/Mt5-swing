"""Tests for World Uncertainty Index (WUI) differential FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_wui import (
    DEFAULT_PUB_LAG_MONTHS,
    WUI_SERIES,
    load_us_wui,
    load_wui_panel,
    wui_coverage,
)
from mt5_swing.strategies.wui_fx import (
    WuiFxConfig,
    prepare_wui_scores,
    trailing_z,
    wui_factor_returns,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_wui_series_map_full_g10():
    for ccy in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF"):
        assert ccy in WUI_SERIES
        assert WUI_SERIES[ccy] is not None
    assert WUI_SERIES["USD"] == "WUIUSA"
    assert WUI_SERIES["EUR"] == "WUIDEU"  # Germany proxy
    assert WUI_SERIES["GBP"] == "WUIGBR"
    assert WUI_SERIES["JPY"] == "WUIJPN"
    assert WUI_SERIES["CAD"] == "WUICAN"
    assert WUI_SERIES["AUD"] == "WUIAUS"
    assert WUI_SERIES["NZD"] == "WUINZL"
    assert WUI_SERIES["CHF"] == "WUICHE"


def test_pub_lag_shifts_known_date():
    """pub_lag_months=4 must move quarter-start index forward by ~4 months."""
    raw = load_wui_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_wui_panel(pub_lag_months=4, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    # ~4 months ≈ 120 days (±30d tolerance)
    assert 90 <= delta_days <= 150


def test_prepare_scores_low_wui_long_calm():
    """Lower WUI level → higher low_wui score (calm-uncertainty prior)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    # EUR calm (falling), GBP stressed (rising)
    eur = np.linspace(50.0, 10.0, 120)
    gbp = np.linspace(10.0, 50.0, 120)
    wui = pd.DataFrame(
        {
            "USD": np.full(120, 20.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 25.0),
            "CAD": np.full(120, 25.0),
            "AUD": np.full(120, 25.0),
            "NZD": np.full(120, 25.0),
            "CHF": np.full(120, 25.0),
        },
        index=idx,
    )
    cfg = WuiFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_wui_scores(wui, cfg=cfg)
    last = scores["low_wui"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    last_hi = scores["high_wui"].dropna(how="all").iloc[-1]
    assert float(last_hi["GBP"]) > float(last_hi["EUR"])


def test_wui_chg_prefers_falling_uncertainty():
    """Falling WUI level → positive wui_chg score."""
    idx = pd.date_range("2015-01-01", periods=48, freq="MS", tz="UTC")
    # EUR falling sharply; GBP flat/rising
    eur = np.concatenate([np.linspace(40, 30, 24), np.linspace(30, 5, 24)])
    gbp = np.concatenate([np.linspace(20, 25, 24), np.linspace(25, 35, 24)])
    wui = pd.DataFrame(
        {
            "USD": np.full(48, 20.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(48, 25.0),
            "CAD": np.full(48, 25.0),
            "AUD": np.full(48, 25.0),
            "NZD": np.full(48, 25.0),
            "CHF": np.full(48, 25.0),
        },
        index=idx,
    )
    cfg = WuiFxConfig(signal_lag=0, chg_periods=12)
    scores = prepare_wui_scores(wui, cfg=cfg)
    last = scores["wui_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_wui_no_lookahead():
    """Mutating future WUI must not change earlier factor returns."""
    idx = pd.date_range("2015-01-01", periods=2000, freq="B", tz="UTC")
    rng = np.random.default_rng(40)
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
    wui = pd.DataFrame(
        {
            c: np.abs(rng.normal(20, 5.0, 260).cumsum()) + 5.0
            for c in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF")
        },
        index=m_idx,
    )
    cfg = WuiFxConfig(signal_lag=0, cost_bps_side=1.5)
    f1 = wui_factor_returns(ret, wui, cfg=cfg)
    assert "low_wui_xs" in f1
    assert "high_wui_xs" in f1
    assert "low_wui_z_xs" in f1
    assert "wui_chg_xs" in f1
    assert "us_wui_stress_fx" in f1
    assert "us_wui_haven_usd" in f1
    assert "wui_ew" in f1

    wui2 = wui.copy()
    wui2.iloc[-6:] = wui2.iloc[-6:] + 15.0
    f2 = wui_factor_returns(ret, wui2, cfg=cfg)

    cut = idx[-120]
    for name in ("low_wui_xs", "wui_chg_xs", "us_wui_stress_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_wui_z_fires_on_elevated_level():
    """Elevated US WUI vs recent history → positive z."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    vals = np.full(240, 15.0)
    vals[-24:] = np.linspace(15.0, 80.0, 24)
    us = pd.Series(vals, index=m_idx, name="USD")
    z = trailing_z(us, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) > 1.0


def test_loader_smoke_and_coverage():
    panel = load_wui_panel(
        pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True
    )
    assert panel.shape[1] >= 7  # full G10 incl USD
    assert "USD" in panel.columns
    assert "EUR" in panel.columns
    assert "CHF" in panel.columns
    assert "NZD" in panel.columns
    us = load_us_wui(download=False)
    assert len(us) > 50
    cov = wui_coverage(panel)
    assert len(cov) >= 8
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("EUR") == "WUIDEU"
    assert panel.attrs.get("unit") == "wui_index_level"
    assert panel.attrs.get("score_basis") == "levels"
    # PIT end should reach into 2025+ after pub lag on ~2026-04 raw
    end = panel.dropna(how="all").index.max()
    assert end.year >= 2025


def test_low_wui_maps_to_top_score():
    """Deep low WUI maps to top low_wui scores."""
    m_idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = np.full(120, 5.0)  # calm
    gbp = np.full(120, 60.0)  # stressed
    wui = pd.DataFrame(
        {
            "USD": np.full(120, 20.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 30.0),
            "CAD": np.full(120, 30.0),
            "AUD": np.full(120, 30.0),
            "NZD": np.full(120, 30.0),
            "CHF": np.full(120, 30.0),
        },
        index=m_idx,
    )
    cfg = WuiFxConfig(signal_lag=0, n_long=2, n_short=2)
    scores = prepare_wui_scores(wui, cfg=cfg)
    last = scores["low_wui"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2


def test_n_long_short_a_priori_full_g10():
    """Full G10 foreign panel → n_long=n_short=2 a priori."""
    cfg = WuiFxConfig()
    assert cfg.n_long == 2
    assert cfg.n_short == 2


def test_quarterly_expand_no_intra_quarter_leak():
    """After pub lag, monthly values within a known quarter equal the lagged Q obs."""
    # Build synthetic quarterly: only Q starts
    q_idx = pd.date_range("2020-01-01", periods=8, freq="QS", tz="UTC")
    raw = pd.DataFrame(
        {"USD": np.arange(8, dtype=float) * 10 + 10},
        index=q_idx,
    )
    # Mimic loader expand with pub_lag=4
    from mt5_swing.data.fred_wui import _expand_quarterly_to_monthly

    monthly = _expand_quarterly_to_monthly(raw, pub_lag_months=4)
    # First known month after lag of 2020-01-01 Q is ~2020-05-01
    assert monthly.index.min() >= pd.Timestamp("2020-05-01", tz="UTC")
    # Within the first known quarter-block, values are constant (ffill of lagged Q)
    first_val = float(monthly["USD"].dropna().iloc[0])
    # next two months of same known Q should match (ffill)
    s = monthly["USD"].dropna()
    assert float(s.iloc[1]) == first_val
    assert float(s.iloc[2]) == first_val
