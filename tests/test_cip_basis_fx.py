"""Tests for Du–Schreger CIP / U.S. Treasury premium FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.cip_basis import (
    DEFAULT_PUB_LAG_DAYS,
    DEFAULT_TENOR,
    TRADE_CURRENCIES,
    cip_coverage,
    ensure_cip_dataset,
    load_cip_panel,
    load_ust_premium,
)
from mt5_swing.strategies.cip_basis_fx import (
    CipBasisFxConfig,
    cip_basis_factor_returns,
    prepare_cip_scores,
    trailing_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_tenor_and_pub_lag_priors():
    assert DEFAULT_TENOR == "5y"
    assert DEFAULT_PUB_LAG_DAYS == 1
    assert set(TRADE_CURRENCIES) == {"EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}


def test_pub_lag_shifts_known_date():
    """pub_lag_days=1 must shift month-end sample forward by ≥1 calendar day."""
    ensure_cip_dataset(force=False)
    raw = load_cip_panel(pub_lag_days=0, download=False, frequency="daily")
    lagged = load_cip_panel(pub_lag_days=1, download=False, frequency="daily")
    assert not raw.empty and not lagged.empty
    # Pick a currency with dense history
    ccy = "EUR"
    r0 = raw[ccy].dropna().index.min()
    l0 = lagged[ccy].dropna().index.min()
    assert (l0 - r0).days == DEFAULT_PUB_LAG_DAYS


def test_prepare_scores_low_cip_long_negative():
    """Lower cip_govt → higher low_cip score (primary convenience / less-stress prior)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    # EUR CIP falls (improves); GBP CIP rises (worsens)
    eur = np.concatenate([np.full(60, 20.0), np.linspace(20.0, -40.0, 60)])
    gbp = np.concatenate([np.full(60, 10.0), np.linspace(10.0, 50.0, 60)])
    cip = pd.DataFrame(
        {
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 5.0),
            "CAD": np.full(120, 8.0),
            "AUD": np.full(120, 12.0),
            "CHF": np.full(120, -5.0),
            "NZD": np.full(120, 15.0),
        },
        index=idx,
    )
    cfg = CipBasisFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_cip_scores(cip, cfg=cfg)
    last = scores["low_cip"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    last_hi = scores["high_cip"].dropna(how="all").iloc[-1]
    assert float(last_hi["GBP"]) > float(last_hi["EUR"])


def test_weight_sign_prior_low_cip_maps_top_score():
    """Most negative CIP maps into top low_cip scores (long side of primary)."""
    m_idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(60, 10.0), np.linspace(10.0, -60.0, 60)])
    gbp = np.concatenate([np.full(60, 10.0), np.linspace(10.0, 40.0, 60)])
    cip = pd.DataFrame(
        {
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 5.0),
            "CAD": np.full(120, 8.0),
            "AUD": np.full(120, 12.0),
            "CHF": np.full(120, 0.0),
            "NZD": np.full(120, 15.0),
        },
        index=m_idx,
    )
    cfg = CipBasisFxConfig(signal_lag=0, n_long=2, n_short=2)
    scores = prepare_cip_scores(cip, cfg=cfg)
    last = scores["low_cip"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2


def test_cip_chg_prefers_improving():
    """Falling CIP (improving) → positive cip_chg score (−Δ12)."""
    idx = pd.date_range("2015-01-01", periods=48, freq="MS", tz="UTC")
    # EUR: flat then drop 30 bps; GBP: flat
    eur = np.concatenate([np.full(24, 20.0), np.full(12, 20.0), np.full(12, -10.0)])
    gbp = np.concatenate([np.full(24, 10.0), np.full(12, 10.0), np.full(12, 12.0)])
    cip = pd.DataFrame(
        {
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(48, 5.0),
            "CAD": np.full(48, 8.0),
            "AUD": np.full(48, 12.0),
            "CHF": np.full(48, 0.0),
            "NZD": np.full(48, 15.0),
        },
        index=idx,
    )
    cfg = CipBasisFxConfig(signal_lag=0, chg_periods=12)
    scores = prepare_cip_scores(cip, cfg=cfg)
    last = scores["cip_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_factor_returns_shape_and_no_lookahead():
    """factor_returns keys + mutating future CIP must not change earlier returns."""
    idx = pd.date_range("2015-01-01", periods=2000, freq="B", tz="UTC")
    rng = np.random.default_rng(7)
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
    cip = pd.DataFrame(
        {c: rng.normal(0.0, 25.0, 260) for c in TRADE_CURRENCIES},
        index=m_idx,
    )
    cfg = CipBasisFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = cip_basis_factor_returns(ret, cip, cfg=cfg)
    expected = {
        "low_cip_xs",
        "high_cip_xs",
        "low_cip_z_xs",
        "cip_chg_xs",
        "ust_premium_haven_usd",
        "ust_premium_stress_fx",
        "cip_ew",
    }
    assert expected.issubset(set(f1.keys()))
    for name, s in f1.items():
        assert isinstance(s, pd.Series)
        assert len(s) == len(ret)
        assert s.name == name

    cip2 = cip.copy()
    cip2.iloc[-6:] = cip2.iloc[-6:] + 40.0
    f2 = cip_basis_factor_returns(ret, cip2, cfg=cfg)

    cut = idx[-120]
    for name in ("low_cip_xs", "cip_chg_xs", "ust_premium_haven_usd"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_ust_premium_z_fires_on_elevated():
    """Elevated UST premium vs history → z ≥ +1."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    vals = np.concatenate([np.full(216, 5.0), np.full(24, 40.0)])
    prem = pd.Series(vals, index=m_idx, name="ust_premium")
    z = trailing_z(prem, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) > 1.0


def test_loader_smoke_and_coverage():
    panel = load_cip_panel(
        tenor=DEFAULT_TENOR,
        pub_lag_days=DEFAULT_PUB_LAG_DAYS,
        download=True,
        frequency="month_end",
    )
    assert panel.shape[1] >= 6
    for c in ("EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"):
        assert c in panel.columns
    assert panel.attrs.get("tenor") == "5y"
    assert panel.attrs.get("unit") == "bps"
    assert panel.attrs.get("pub_lag_days") == 1
    cov = cip_coverage(panel)
    assert len(cov) >= 7
    assert (cov["n_obs"] > 100).all()
    prem = load_ust_premium(panel)
    assert len(prem.dropna()) > 50
    # CIP in bps — sensible range for G10 gov
    last_eur = float(panel["EUR"].dropna().iloc[-1])
    assert -300.0 < last_eur < 300.0
    cfg = CipBasisFxConfig()
    assert cfg.n_long == 2 and cfg.n_short == 2
    assert cfg.signal_lag == 1
