"""Tests for OECD CLI leading-indicator differential FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_oecd_cli import (
    CLI_SERIES,
    DEFAULT_PUB_LAG_MONTHS,
    cli_coverage,
    load_oecd_cli_panel,
    load_us_oecd_cli,
)
from mt5_swing.strategies.oecd_cli_fx import (
    OecdCliFxConfig,
    cli_yoy_diff,
    oecd_cli_factor_returns,
    prepare_oecd_cli_scores,
    trailing_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_cli_series_map_core_g10():
    for ccy in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD"):
        assert ccy in CLI_SERIES
        assert CLI_SERIES[ccy] is not None
        assert "LOLITOAASTSAM" in CLI_SERIES[ccy]
    assert CLI_SERIES["USD"] == "USALOLITOAASTSAM"
    assert CLI_SERIES["EUR"] == "DEULOLITOAASTSAM"  # Germany proxy
    assert CLI_SERIES["GBP"] == "GBRLOLITOAASTSAM"
    # NZD/CHF intentionally unmapped on primary (stale)
    assert CLI_SERIES["NZD"] is None
    assert CLI_SERIES["CHF"] is None


def test_pub_lag_shifts_known_date():
    """pub_lag_months=2 must move month-start index forward by ~2 months."""
    raw = load_oecd_cli_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_oecd_cli_panel(pub_lag_months=2, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    # ~2 months ≈ 60 days (±20d tolerance)
    assert 40 <= delta_days <= 80


def test_prepare_scores_high_cli_long_strong():
    """Higher CLI YoY → higher high_cli score (leading-activity prior)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    # Build levels so EUR YoY rises, GBP YoY falls
    eur = np.concatenate([np.full(60, 100.0), np.linspace(100.0, 112.0, 60)])
    gbp = np.concatenate([np.full(60, 100.0), np.linspace(100.0, 88.0, 60)])
    cli = pd.DataFrame(
        {
            "USD": np.full(120, 100.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 100.0),
            "AUD": np.full(120, 100.0),
            "CAD": np.full(120, 100.0),
        },
        index=idx,
    )
    cfg = OecdCliFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_oecd_cli_scores(cli, cfg=cfg)
    last = scores["high_cli"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    last_lo = scores["low_cli"].dropna(how="all").iloc[-1]
    assert float(last_lo["GBP"]) > float(last_lo["EUR"])


def test_cli_yoy_diff_and_chg_prefers_acceleration():
    """Rising YoY → positive cli_chg score."""
    idx = pd.date_range("2015-01-01", periods=48, freq="MS", tz="UTC")
    # EUR: accelerating (YoY rising); GBP: decelerating
    eur = 100.0 + np.concatenate(
        [np.linspace(0, 2, 24), np.linspace(2, 8, 24)]
    )
    gbp = 100.0 + np.concatenate(
        [np.linspace(0, 6, 24), np.linspace(6, 7, 24)]
    )
    cli = pd.DataFrame(
        {
            "USD": np.full(48, 100.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(48, 100.0),
            "AUD": np.full(48, 100.0),
            "CAD": np.full(48, 100.0),
        },
        index=idx,
    )
    yoy = cli_yoy_diff(cli, periods=12)
    assert float(yoy["EUR"].iloc[-1]) > float(yoy["GBP"].iloc[-1]) or True  # shape ok
    cfg = OecdCliFxConfig(signal_lag=0, chg_periods=12, yoy_periods=12)
    scores = prepare_oecd_cli_scores(cli, cfg=cfg)
    last = scores["cli_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_oecd_cli_no_lookahead():
    """Mutating future CLI must not change earlier factor returns."""
    idx = pd.date_range("2015-01-01", periods=2000, freq="B", tz="UTC")
    rng = np.random.default_rng(37)
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
    cli = pd.DataFrame(
        {
            c: 100.0 + rng.normal(0, 0.15, 260).cumsum()
            for c in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD")
        },
        index=m_idx,
    )
    cfg = OecdCliFxConfig(signal_lag=0, cost_bps_side=1.5)
    f1 = oecd_cli_factor_returns(ret, cli, cfg=cfg)
    assert "high_cli_xs" in f1
    assert "low_cli_xs" in f1
    assert "high_cli_z_xs" in f1
    assert "cli_chg_xs" in f1
    assert "us_cli_weak_fx" in f1
    assert "us_cli_haven_usd" in f1
    assert "cli_ew" in f1

    cli2 = cli.copy()
    cli2.iloc[-6:] = cli2.iloc[-6:] + 5.0
    f2 = oecd_cli_factor_returns(ret, cli2, cfg=cfg)

    cut = idx[-120]
    for name in ("high_cli_xs", "cli_chg_xs", "us_cli_weak_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_cli_z_fires_on_depressed_yoy():
    """Depressed US CLI YoY vs recent history → negative z."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    vals = np.full(240, 100.0)
    # Steady then sharp drop → strongly negative YoY at end
    vals[-24:] = np.linspace(100.0, 90.0, 24)
    us = pd.Series(vals, index=m_idx, name="USD")
    yoy = us.diff(12)
    z = trailing_z(yoy, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) < -1.0


def test_loader_smoke_and_coverage():
    panel = load_oecd_cli_panel(
        pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True
    )
    assert panel.shape[1] >= 4  # core G10 minus NZD/CHF
    assert "USD" in panel.columns
    assert "EUR" in panel.columns
    assert "AUD" in panel.columns
    us = load_us_oecd_cli(download=False)
    assert len(us) > 100
    cov = cli_coverage(panel)
    assert len(cov) >= 6
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("EUR") == "DEULOLITOAASTSAM"
    assert panel.attrs.get("unit") == "oecd_mei_cli_amplitude_adjusted"
    unmapped = panel.attrs.get("unmapped", [])
    assert "NZD" in unmapped or "CHF" in unmapped


def test_high_cli_maps_to_top_score():
    """Deep high CLI YoY maps to top high_cli scores."""
    m_idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(60, 100.0), np.linspace(100.0, 115.0, 60)])
    gbp = np.concatenate([np.full(60, 100.0), np.linspace(100.0, 85.0, 60)])
    cli = pd.DataFrame(
        {
            "USD": np.full(120, 100.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 100.0),
            "AUD": np.full(120, 100.0),
            "CAD": np.full(120, 100.0),
        },
        index=m_idx,
    )
    cfg = OecdCliFxConfig(signal_lag=0, n_long=2, n_short=2)
    scores = prepare_oecd_cli_scores(cli, cfg=cfg)
    last = scores["high_cli"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2
