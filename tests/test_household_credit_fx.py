"""Tests for BIS/FRED household-credit YoY (CRDQ*AHABIS) FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_household_credit import (
    DEFAULT_PUB_LAG_MONTHS,
    HHCRED_SERIES,
    load_household_credit_panel,
    load_us_household_credit,
    household_credit_coverage,
)
from mt5_swing.strategies.household_credit_fx import (
    HouseholdCreditFxConfig,
    prepare_household_credit_scores,
    trailing_z,
    household_credit_factor_returns,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_hhcred_series_map():
    assert HHCRED_SERIES["USD"] == "CRDQUSAHABIS"
    assert HHCRED_SERIES["EUR"] == "CRDQXMAHABIS"  # euro-area
    assert HHCRED_SERIES["GBP"] == "CRDQGBAHABIS"
    assert HHCRED_SERIES["JPY"] == "CRDQJPAHABIS"
    assert HHCRED_SERIES["CAD"] == "CRDQCAAHABIS"
    assert HHCRED_SERIES["AUD"] == "CRDQAUAHABIS"
    assert HHCRED_SERIES["CHF"] == "CRDQCHAHABIS"
    assert HHCRED_SERIES["NZD"] is None  # CRDQNZAHABIS 404


def test_pub_lag_shifts_known_date():
    """pub_lag_months=5 must shift known date forward by ~5 months."""
    raw = load_household_credit_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_household_credit_panel(pub_lag_months=5, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["USD"].dropna().index.min()
    l0 = lagged["USD"].dropna().index.min()
    delta_days = (l0 - r0).days
    assert 120 <= delta_days <= 200  # ~5 months


def test_prepare_scores_high_hhcred_long_growth():
    """Higher HH-credit YoY → higher high_hhcred score (consumer-demand prior)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(60, 1.5), np.linspace(1.5, 4.0, 60)])
    gbp = np.concatenate([np.full(60, 2.0), np.linspace(2.0, 0.0, 60)])
    hhcred = pd.DataFrame(
        {
            "USD": np.full(120, 2.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 1.0),
            "CAD": np.full(120, 2.2),
            "AUD": np.full(120, 2.5),
            "CHF": np.full(120, 1.8),
            "NZD": np.full(120, 2.1),
        },
        index=idx,
    )
    cfg = HouseholdCreditFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_household_credit_scores(hhcred, cfg=cfg)
    last = scores["high_hhcred"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    last_lo = scores["low_hhcred"].dropna(how="all").iloc[-1]
    assert float(last_lo["GBP"]) > float(last_lo["EUR"])


def test_hhcred_chg_prefers_acceleration():
    """Accelerating HH-credit YoY → positive hhcred_chg score (+Δ12 of YoY)."""
    idx = pd.date_range("2015-01-01", periods=48, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(24, 1.0), np.full(12, 1.0), np.full(12, 4.0)])
    gbp = np.concatenate([np.full(24, 2.0), np.full(12, 2.0), np.full(12, 2.2)])
    hhcred = pd.DataFrame(
        {
            "USD": np.full(48, 2.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(48, 1.0),
            "CAD": np.full(48, 2.2),
            "AUD": np.full(48, 2.5),
            "CHF": np.full(48, 1.8),
            "NZD": np.full(48, 2.1),
        },
        index=idx,
    )
    cfg = HouseholdCreditFxConfig(signal_lag=0, chg_periods=12)
    scores = prepare_household_credit_scores(hhcred, cfg=cfg)
    last = scores["hhcred_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_household_credit_no_lookahead():
    """Mutating future HH-credit YoY must not change earlier factor returns."""
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
    hhcred = pd.DataFrame(
        {
            c: rng.uniform(-20.0, 25.0, 260)
            for c in ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "NZD")
        },
        index=m_idx,
    )
    cfg = HouseholdCreditFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = household_credit_factor_returns(ret, hhcred, cfg=cfg)
    assert "high_hhcred_xs" in f1
    assert "low_hhcred_xs" in f1
    assert "high_hhcred_z_xs" in f1
    assert "hhcred_chg_xs" in f1
    assert "us_hhcred_stress_fx" in f1
    assert "us_hhcred_haven_usd" in f1
    assert "hhcred_ew" in f1

    hhcred2 = hhcred.copy()
    hhcred2.iloc[-6:] = hhcred2.iloc[-6:] + 5.0
    f2 = household_credit_factor_returns(ret, hhcred2, cfg=cfg)

    cut = idx[-120]
    for name in ("high_hhcred_xs", "hhcred_chg_xs", "us_hhcred_stress_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_hhcred_z_fires_on_depressed():
    """Depressed US HH-credit YoY vs recent history → negative z ≤ −1."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    vals = np.concatenate([np.full(216, 5.0), np.full(24, -15.0)])
    us = pd.Series(vals, index=m_idx, name="USD")
    z = trailing_z(us, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) < -1.0


def test_loader_smoke_and_coverage():
    panel = load_household_credit_panel(pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True)
    assert panel.shape[1] >= 6  # USD + foreign excl NZD
    assert "USD" in panel.columns
    assert "EUR" in panel.columns
    assert "GBP" in panel.columns
    assert "JPY" in panel.columns
    assert "CAD" in panel.columns
    assert "AUD" in panel.columns
    assert "CHF" in panel.columns
    assert "NZD" not in panel.columns
    unmapped = panel.attrs.get("unmapped", [])
    assert "NZD" in unmapped
    us = load_us_household_credit(download=False)
    assert len(us) > 50
    cov = household_credit_coverage(panel)
    assert len(cov) >= 7
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("USD") == "CRDQUSAHABIS"
    assert panel.attrs.get("series_map", {}).get("EUR") == "CRDQXMAHABIS"
    assert panel.attrs.get("unit") == "yoy_pct"
    assert panel.attrs.get("score_basis") == "household_credit_yoy_pct"
    assert panel.attrs.get("series_unit_caveat") == "abs_bn_to_yoy_not_pct_gdp"
    assert DEFAULT_PUB_LAG_MONTHS == 5
    # YoY should be in a sensible percent range
    last_us = float(panel["USD"].dropna().iloc[-1])
    assert -80.0 < last_us < 80.0
    # Fall-through documented
    assert "merchandise_trade_volume_404" in panel.attrs.get("fallthrough_from", [])


def test_high_hhcred_maps_to_top_score():
    """High HH-credit YoY maps to top high_hhcred scores."""
    m_idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(60, 1.0), np.linspace(1.0, 5.0, 60)])
    gbp = np.concatenate([np.full(60, 2.0), np.linspace(2.0, 0.0, 60)])
    hhcred = pd.DataFrame(
        {
            "USD": np.full(120, 2.0),
            "EUR": eur,
            "GBP": gbp,
            "JPY": np.full(120, 1.0),
            "CAD": np.full(120, 2.2),
            "AUD": np.full(120, 2.5),
            "CHF": np.full(120, 1.8),
            "NZD": np.full(120, 2.1),
        },
        index=m_idx,
    )
    cfg = HouseholdCreditFxConfig(signal_lag=0, n_long=2, n_short=2)
    scores = prepare_household_credit_scores(hhcred, cfg=cfg)
    last = scores["high_hhcred"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2


def test_n_long_short_a_priori():
    """NZD unmapped → n_long=n_short=2; signal_lag=1; z_low=-1.0; pub_lag=5."""
    cfg = HouseholdCreditFxConfig()
    assert cfg.n_long == 2
    assert cfg.n_short == 2
    assert cfg.signal_lag == 1
    assert cfg.z_low == -1.0
    assert cfg.usd_tilt == 0.5
    assert DEFAULT_PUB_LAG_MONTHS == 5
