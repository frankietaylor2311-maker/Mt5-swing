"""Tests for BIS WS_DSR debt-service ratio differential FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_bis_dsr import (
    BIS_DSR_ISO2,
    DEFAULT_PUB_LAG_MONTHS,
    bis_dsr_coverage,
    load_bis_dsr_panel,
    load_us_bis_dsr,
)
from mt5_swing.strategies.bis_dsr_fx import (
    BisDsrFxConfig,
    bis_dsr_factor_returns,
    prepare_dsr_scores,
    trailing_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_dsr_iso2_map_g10_with_nzd_gap():
    for ccy in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"):
        assert ccy in BIS_DSR_ISO2
        assert BIS_DSR_ISO2[ccy] is not None
    assert BIS_DSR_ISO2["USD"] == "US"
    assert BIS_DSR_ISO2["EUR"] == "DE"  # Germany proxy
    assert BIS_DSR_ISO2["GBP"] == "GB"
    assert BIS_DSR_ISO2["NZD"] is None  # honest gap


def test_pub_lag_shifts_known_date():
    """pub_lag_months=5 must move quarter-start index forward by ~5 months."""
    raw = load_bis_dsr_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_bis_dsr_panel(pub_lag_months=5, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    # ~5 months ≈ 150 days (±45d tolerance)
    assert 100 <= delta_days <= 200


def test_prepare_scores_high_dsr_long_stressed():
    """Higher DSR → higher high_dsr score (risk-premia prior)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(60, 10.0), np.linspace(10.0, 30.0, 60)])
    gbp = np.concatenate([np.full(60, 10.0), np.linspace(10.0, 5.0, 60)])
    dsr = pd.DataFrame(
        {
            "USD": np.full(120, 12.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 12.0),
            "CAD": np.full(120, 15.0),
            "AUD": np.full(120, 14.0),
            "CHF": np.full(120, 13.0),
        },
        index=idx,
    )
    cfg = BisDsrFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_dsr_scores(dsr, cfg=cfg)
    last = scores["high_dsr"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    last_lo = scores["low_dsr"].dropna(how="all").iloc[-1]
    assert float(last_lo["GBP"]) > float(last_lo["EUR"])


def test_dsr_chg_prefers_acceleration():
    """Rising DSR → positive dsr_chg score."""
    idx = pd.date_range("2015-01-01", periods=48, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(12, 10.0), np.full(24, 12.0), np.full(12, 25.0)])
    gbp = np.concatenate([np.full(12, 10.0), np.full(24, 18.0), np.full(12, 19.0)])
    dsr = pd.DataFrame(
        {
            "USD": np.full(48, 12.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(48, 12.0),
            "CAD": np.full(48, 15.0),
            "AUD": np.full(48, 14.0),
            "CHF": np.full(48, 13.0),
        },
        index=idx,
    )
    cfg = BisDsrFxConfig(signal_lag=0, chg_periods=12)
    scores = prepare_dsr_scores(dsr, cfg=cfg)
    last = scores["dsr_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_bis_dsr_no_lookahead():
    """Mutating future DSR must not change earlier factor returns."""
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
    dsr = pd.DataFrame(
        {
            c: rng.normal(15.0, 2.0, 260)
            for c in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF")
        },
        index=m_idx,
    )
    cfg = BisDsrFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = bis_dsr_factor_returns(ret, dsr, cfg=cfg)
    assert "high_dsr_xs" in f1
    assert "low_dsr_xs" in f1
    assert "high_dsr_z_xs" in f1
    assert "dsr_chg_xs" in f1
    assert "us_dsr_stress_fx" in f1
    assert "us_dsr_haven_usd" in f1
    assert "dsr_ew" in f1

    dsr2 = dsr.copy()
    dsr2.iloc[-6:] = dsr2.iloc[-6:] + 10.0
    f2 = bis_dsr_factor_returns(ret, dsr2, cfg=cfg)

    cut = idx[-120]
    for name in ("high_dsr_xs", "dsr_chg_xs", "us_dsr_stress_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_dsr_z_fires_on_elevated():
    """Elevated US DSR vs recent history → positive z."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    vals = np.concatenate([np.full(216, 12.0), np.full(24, 22.0)])
    us = pd.Series(vals, index=m_idx, name="USD")
    z = trailing_z(us, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) > 1.0


def test_loader_smoke_and_coverage():
    panel = load_bis_dsr_panel(
        pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True
    )
    assert panel.shape[1] >= 6  # G10 minus NZD
    assert "USD" in panel.columns
    assert "EUR" in panel.columns
    assert "CHF" in panel.columns
    assert "AUD" in panel.columns
    assert "NZD" not in panel.columns or panel["NZD"].dropna().empty
    us = load_us_bis_dsr(download=False)
    assert len(us) > 50
    cov = bis_dsr_coverage(panel)
    assert len(cov) >= 7
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("EUR") == "BIS:WS_DSR:Q.DE.P"
    assert panel.attrs.get("series_map", {}).get("USD") == "BIS:WS_DSR:Q.US.P"
    assert panel.attrs.get("unit") == "dsr_pct_of_income"
    assert panel.attrs.get("score_basis") == "pnfs_dsr_level_pct"
    assert DEFAULT_PUB_LAG_MONTHS == 5
    assert "NZD" in panel.attrs.get("unmapped", [])
    end = panel.dropna(how="all").index.max()
    assert end.year >= 2025


def test_high_dsr_maps_to_top_score():
    """High DSR maps to top high_dsr scores."""
    m_idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(60, 10.0), np.linspace(10.0, 28.0, 60)])
    gbp = np.concatenate([np.full(60, 10.0), np.linspace(10.0, 8.0, 60)])
    dsr = pd.DataFrame(
        {
            "USD": np.full(120, 12.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 12.0),
            "CAD": np.full(120, 14.0),
            "AUD": np.full(120, 13.0),
            "CHF": np.full(120, 11.0),
        },
        index=m_idx,
    )
    cfg = BisDsrFxConfig(signal_lag=0, n_long=2, n_short=2)
    scores = prepare_dsr_scores(dsr, cfg=cfg)
    last = scores["high_dsr"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2


def test_n_long_short_a_priori():
    """Mapped foreign panel → n_long=n_short=2 a priori."""
    cfg = BisDsrFxConfig()
    assert cfg.n_long == 2
    assert cfg.n_short == 2
    assert cfg.signal_lag == 1


def test_quarterly_expand_no_intra_quarter_leak():
    """After pub lag, monthly values within a known quarter equal the lagged Q obs."""
    q_idx = pd.date_range("2020-01-01", periods=8, freq="QS", tz="UTC")
    raw = pd.DataFrame(
        {"GBP": np.arange(8, dtype=float) + 10.0},
        index=q_idx,
    )
    from mt5_swing.data.fred_bis_dsr import _expand_quarterly_to_monthly

    monthly = _expand_quarterly_to_monthly(raw, pub_lag_months=5)
    assert monthly.index.min() >= pd.Timestamp("2020-06-01", tz="UTC")
    first_val = float(monthly["GBP"].dropna().iloc[0])
    s = monthly["GBP"].dropna()
    assert float(s.iloc[1]) == first_val
    assert float(s.iloc[2]) == first_val
