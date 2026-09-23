"""Tests for country-GPR bilateral (rel-to-US) FX module (§45)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.country_gpr_bilateral import (
    DEFAULT_PUB_LAG_MONTHS,
    FOREIGN_CCY,
    country_gpr_bilateral_coverage,
    load_country_gpr_levels,
    relative_gpr_panel,
)
from mt5_swing.data.macro_uncertainty import load_country_gpr
from mt5_swing.strategies.country_gpr_bilateral_fx import (
    CountryGprBilateralFxConfig,
    country_gpr_bilateral_factor_returns,
    prepare_rel_gpr_scores,
    trailing_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_pub_lag_default_matches_section9():
    assert DEFAULT_PUB_LAG_MONTHS == 1


def test_relative_panel_is_home_minus_us():
    levels = load_country_gpr_levels(pub_lag_months=0, download=False)
    rel = relative_gpr_panel(levels, pub_lag_months=0, download=False)
    assert "USD" not in rel.columns
    assert "NZD" not in rel.columns
    assert "EUR" in rel.columns
    # Check arithmetic for a date where both EUR and USD exist
    common = levels[["EUR", "USD"]].dropna().index.intersection(rel["EUR"].dropna().index)
    assert len(common) > 50
    dt = common[-10]
    assert np.isclose(
        float(rel.loc[dt, "EUR"]),
        float(levels.loc[dt, "EUR"] - levels.loc[dt, "USD"]),
        equal_nan=False,
    )


def test_nzd_gap_documented():
    levels = load_country_gpr_levels(pub_lag_months=1, download=False)
    assert "NZD" not in levels.columns or levels["NZD"].dropna().empty
    rel = relative_gpr_panel(levels, download=False)
    cov = country_gpr_bilateral_coverage(rel)
    nzd = cov[cov["currency"] == "NZD"].iloc[0]
    assert int(nzd["n_obs"]) == 0
    assert nzd["mapped"] is False or nzd["mapped"] == False
    assert "NZD" in (levels.attrs.get("missing_currencies") or [])


def test_load_country_gpr_still_works_section9():
    """§45 helpers must not break §9 absolute loader."""
    panel = load_country_gpr(pub_lag_months=1, download=False)
    assert "USD" in panel.columns
    assert "EUR" in panel.columns
    assert panel.shape[0] > 100


def test_prepare_scores_low_rel_prefers_low_differential():
    """Lower (home−US) z → higher low_rel_gpr score."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    # EUR ends with low relative risk; JPY ends with high relative risk
    eur = np.concatenate([np.zeros(60), np.linspace(0.0, -2.0, 60)])
    jpy = np.concatenate([np.zeros(60), np.linspace(0.0, 3.0, 60)])
    gbp = np.full(120, 0.1)
    cad = np.full(120, 0.0)
    aud = np.full(120, -0.1)
    chf = np.full(120, 0.05)
    rel = pd.DataFrame(
        {"EUR": eur, "GBP": gbp, "JPY": jpy, "CAD": cad, "AUD": aud, "CHF": chf},
        index=idx,
    )
    cfg = CountryGprBilateralFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_rel_gpr_scores(rel, cfg=cfg)
    last = scores["low_rel_gpr"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["JPY"])
    last_hi = scores["high_rel_gpr"].dropna(how="all").iloc[-1]
    assert float(last_hi["JPY"]) > float(last_hi["EUR"])
    last_lvl = scores["low_rel_gpr_lvl"].dropna(how="all").iloc[-1]
    assert float(last_lvl["EUR"]) > float(last_lvl["JPY"])


def test_rel_gpr_chg_prefers_falling_differential():
    """Falling (home−US) → positive rel_gpr_chg score (−Δ12)."""
    idx = pd.date_range("2015-01-01", periods=48, freq="MS", tz="UTC")
    # EUR falling steadily → negative Δ12 → positive score (−Δ12)
    eur = np.linspace(3.0, -3.0, 48)
    # JPY rising steadily → positive Δ12 → negative score
    jpy = np.linspace(-3.0, 3.0, 48)
    rel = pd.DataFrame(
        {
            "EUR": eur,
            "GBP": np.zeros(48),
            "JPY": jpy,
            "CAD": np.zeros(48),
            "AUD": np.zeros(48),
            "CHF": np.zeros(48),
        },
        index=idx,
    )
    cfg = CountryGprBilateralFxConfig(signal_lag=0, chg_periods=12)
    scores = prepare_rel_gpr_scores(rel, cfg=cfg)
    last = scores["rel_gpr_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["JPY"])


def test_factor_returns_smoke_and_no_lookahead():
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
    rel = pd.DataFrame(
        {c: rng.normal(0.0, 0.5, 260) for c in FOREIGN_CCY},
        index=m_idx,
    )
    us = pd.Series(rng.normal(0.5, 0.3, 260), index=m_idx, name="USD")
    cfg = CountryGprBilateralFxConfig(signal_lag=0, cost_bps_side=1.5)
    f1 = country_gpr_bilateral_factor_returns(ret, rel, cfg=cfg, us_gpr=us)
    for name in (
        "low_rel_gpr_xs",
        "high_rel_gpr_xs",
        "low_rel_gpr_lvl_xs",
        "rel_gpr_chg_xs",
        "us_gpr_stress_fx",
        "us_gpr_haven_usd",
        "gpr_bilat_ew",
    ):
        assert name in f1

    rel2 = rel.copy()
    rel2.iloc[-6:] = rel2.iloc[-6:] + 2.0
    us2 = us.copy()
    us2.iloc[-6:] = us2.iloc[-6:] + 1.0
    f2 = country_gpr_bilateral_factor_returns(ret, rel2, cfg=cfg, us_gpr=us2)

    cut = idx[-120]
    for name in ("low_rel_gpr_xs", "rel_gpr_chg_xs", "us_gpr_stress_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(
            a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12
        )


def test_us_gpr_z_fires_on_elevated_levels():
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    vals = np.concatenate([np.full(216, 0.3), np.full(24, 2.5)])
    us = pd.Series(vals, index=m_idx, name="USD")
    z = trailing_z(us, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) > 1.0


def test_n_long_short_a_priori():
    cfg = CountryGprBilateralFxConfig()
    assert cfg.n_long == 2
    assert cfg.n_short == 2
    assert cfg.signal_lag == 0
    assert cfg.z_window == 60


def test_loader_smoke_live_panel():
    levels = load_country_gpr_levels(pub_lag_months=1, download=False)
    rel = relative_gpr_panel(levels, download=False)
    assert rel.shape[1] >= 5
    for c in ("EUR", "GBP", "JPY", "CAD", "AUD", "CHF"):
        assert c in rel.columns
        assert rel[c].dropna().shape[0] > 50
    end = rel.dropna(how="all").index.max()
    assert end.year >= 2025
    assert "USD" in levels.columns
