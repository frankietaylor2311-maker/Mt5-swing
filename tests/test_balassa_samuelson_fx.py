"""Tests for Balassa–Samuelson / productivity-adjusted real FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_macro_diff import (
    load_ip_level_panel,
    relative_productivity_vs_usd,
)
from mt5_swing.strategies.balassa_samuelson_fx import (
    BalassaSamuelsonFxConfig,
    balassa_samuelson_factor_returns,
    bs_gap_scores,
    bs_productivity_scores,
    bs_residual_scores,
    trailing_z,
)
from mt5_swing.strategies.ppp_real_fx import real_fx_panel

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_trailing_z_mean_near_zero():
    idx = pd.date_range("2010-01-01", periods=80, freq="MS", tz="UTC")
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"EUR": rng.normal(0, 1, 80)}, index=idx)
    z = trailing_z(x, lookback=36, min_periods=24)
    tail = z.dropna().iloc[-20:]
    assert abs(float(tail["EUR"].mean())) < 0.5


def test_bs_gap_long_when_cheap_vs_productivity():
    """Low real FX relative to high productivity → positive score (long)."""
    idx = pd.date_range("2010-01-01", periods=80, freq="MS", tz="UTC")
    # Stable productivity; real FX drops cheap at the end
    p = pd.DataFrame({"EUR": np.linspace(0.0, 0.1, 80)}, index=idx)
    q = pd.DataFrame({"EUR": np.full(80, 1.0)}, index=idx)
    q.iloc[-8:] = 0.7
    cfg = BalassaSamuelsonFxConfig(signal_lag=0, min_periods=24)
    score = bs_gap_scores(q, p, cfg=cfg, lookback=36)
    assert float(score.dropna().iloc[-1]["EUR"]) > 0


def test_bs_prod_high_productivity_positive_score():
    idx = pd.date_range("2010-01-01", periods=80, freq="MS", tz="UTC")
    p = pd.DataFrame({"EUR": np.concatenate([np.zeros(70), np.full(10, 1.0)])}, index=idx)
    cfg = BalassaSamuelsonFxConfig(signal_lag=0, min_periods=24)
    score = bs_productivity_scores(p, cfg=cfg, lookback=36)
    assert float(score.dropna().iloc[-1]["EUR"]) > 0


def test_bs_residual_no_lookahead():
    """Mutating future productivity must not change earlier residual scores."""
    idx = pd.date_range("2012-01-01", periods=90, freq="MS", tz="UTC")
    rng = np.random.default_rng(2)
    q = pd.DataFrame(
        {
            "EUR": np.exp(np.cumsum(rng.normal(0, 0.01, 90))),
            "GBP": np.exp(np.cumsum(rng.normal(0, 0.01, 90))),
        },
        index=idx,
    )
    p = pd.DataFrame(
        {
            "EUR": np.cumsum(rng.normal(0, 0.005, 90)),
            "GBP": np.cumsum(rng.normal(0, 0.005, 90)),
        },
        index=idx,
    )
    cfg = BalassaSamuelsonFxConfig(signal_lag=1, min_periods=24)
    s1 = bs_residual_scores(q, p, cfg=cfg, lookback=36)
    p2 = p.copy()
    p2.iloc[-5:] += 0.5
    s2 = bs_residual_scores(q, p2, cfg=cfg, lookback=36)
    cut = -12
    a = s1.iloc[:cut].dropna(how="all")
    b = s2.iloc[:cut].dropna(how="all")
    common = a.index.intersection(b.index)
    assert len(common) > 20
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True)


def test_bs_strategy_smoke_and_causal_returns():
    idx = pd.date_range("2016-01-01", periods=1800, freq="B", tz="UTC")
    rng = np.random.default_rng(4)
    close = pd.DataFrame(
        {
            "EURUSD": 1.1 * np.cumprod(1 + rng.normal(0, 0.005, 1800)),
            "GBPUSD": 1.3 * np.cumprod(1 + rng.normal(0, 0.005, 1800)),
            "AUDUSD": 0.75 * np.cumprod(1 + rng.normal(0, 0.006, 1800)),
            "NZDUSD": 0.65 * np.cumprod(1 + rng.normal(0, 0.006, 1800)),
            "USDJPY": 110 * np.cumprod(1 + rng.normal(0, 0.005, 1800)),
            "USDCAD": 1.3 * np.cumprod(1 + rng.normal(0, 0.004, 1800)),
            "USDCHF": 0.95 * np.cumprod(1 + rng.normal(0, 0.004, 1800)),
        },
        index=idx,
    )
    ret = close.pct_change()
    cpi_idx = pd.date_range("2015-01-01", periods=120, freq="MS", tz="UTC")
    cpi = pd.DataFrame(
        {
            c: 100 * np.cumprod(1 + rng.normal(0.002, 0.001, 120))
            for c in ("USD", "EUR", "GBP", "AUD", "NZD", "JPY", "CAD", "CHF")
        },
        index=cpi_idx,
    )
    # Relative productivity for currencies with IP in the real design
    rel_prod = pd.DataFrame(
        {
            "EUR": np.cumsum(rng.normal(0, 0.01, 120)),
            "GBP": np.cumsum(rng.normal(0, 0.01, 120)),
            "JPY": np.cumsum(rng.normal(0, 0.01, 120)),
            "CAD": np.cumsum(rng.normal(0, 0.01, 120)),
        },
        index=cpi_idx,
    )
    cfg = BalassaSamuelsonFxConfig(signal_lag=1, lookbacks=(60,), min_periods=36)
    factors = balassa_samuelson_factor_returns(close, ret, cpi, rel_prod, cfg=cfg)
    assert "bs_gap_60m" in factors
    assert "bs_resid_60m" in factors
    assert "bs_prod_60m" in factors
    assert "bs_ew" in factors
    port = factors["bs_gap_60m"]
    assert len(port) == len(ret)
    ret2 = ret.copy()
    ret2.iloc[-5:] = 0.05
    factors2 = balassa_samuelson_factor_returns(close, ret2, cpi, rel_prod, cfg=cfg)
    cut = -20
    assert np.allclose(
        port.iloc[:cut].fillna(0).values,
        factors2["bs_gap_60m"].iloc[:cut].fillna(0).values,
    )


def test_relative_productivity_definition():
    idx = pd.date_range("2020-01-01", periods=12, freq="MS", tz="UTC")
    ip = pd.DataFrame(
        {"USD": np.full(12, np.e), "EUR": np.full(12, np.e**2)},
        index=idx,
    )
    rel = relative_productivity_vs_usd(ip)
    assert "EUR" in rel.columns
    assert "USD" not in rel.columns
    assert float(rel["EUR"].iloc[0]) == pytest.approx(1.0)


def test_load_ip_level_panel_if_cached():
    if not (MACRO / "fred_INDPRO.csv").exists():
        pytest.skip("INDPRO cache missing")
    panel = load_ip_level_panel(download=False, pub_lag_months=2)
    assert "USD" in panel.columns
    assert panel.attrs.get("pub_lag_months") == 2
    assert panel.attrs.get("factor") == "ip_level"
    # Levels should be index magnitude, not YoY %
    assert float(panel["USD"].dropna().iloc[-1]) > 50


def test_real_fx_and_prod_align_for_gap():
    """Smoke: real_fx_panel + gap scores with synthetic aligned panels."""
    idx_fx = pd.date_range("2018-01-31", periods=60, freq="ME", tz="UTC")
    idx_m = pd.date_range("2018-01-01", periods=60, freq="MS", tz="UTC")
    fx = pd.DataFrame({"EUR": np.linspace(1.05, 1.15, 60)}, index=idx_fx)
    cpi = pd.DataFrame(
        {"USD": np.linspace(100, 120, 60), "EUR": np.linspace(100, 110, 60)},
        index=idx_m,
    )
    q = real_fx_panel(fx, cpi)
    p = pd.DataFrame({"EUR": np.linspace(0, 0.2, 60)}, index=idx_m)
    cfg = BalassaSamuelsonFxConfig(signal_lag=1, min_periods=24)
    sc = bs_gap_scores(q, p, cfg=cfg, lookback=36)
    assert sc.dropna(how="all").shape[0] > 5
