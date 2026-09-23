"""Tests for PPP / real-FX value scholarly module (Rogoff-style)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.backtest.ftmo_risk_sweep import sweep_scale_to_ftmo_budget
from mt5_swing.data.fred_macro_diff import load_cpi_level_panel
from mt5_swing.strategies.carry_rank import USD_PAIRS, currency_weights_to_pair_weights
from mt5_swing.strategies.ppp_real_fx import (
    PppRealFxConfig,
    monthly_nominal_fx,
    nominal_fx_usd_per_foreign,
    ppp_factor_returns,
    ppp_value_scores,
    real_fx_panel,
    real_fx_zscore,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_nominal_fx_inversion_usdjpy():
    idx = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
    close = pd.DataFrame(
        {
            "EURUSD": [1.10, 1.11, 1.09, 1.10, 1.12],
            "USDJPY": [150.0, 149.0, 151.0, 150.5, 148.0],
        },
        index=idx,
    )
    s = nominal_fx_usd_per_foreign(close)
    assert "EUR" in s.columns and "JPY" in s.columns
    assert float(s.iloc[0]["EUR"]) == pytest.approx(1.10)
    assert float(s.iloc[0]["JPY"]) == pytest.approx(1.0 / 150.0)


def test_real_fx_high_when_foreign_cpi_low():
    """Cheaper foreign goods basket (low CPI_f) → higher real FX q (rich FX)."""
    idx = pd.date_range("2020-01-01", periods=24, freq="MS", tz="UTC")
    fx = pd.DataFrame({"EUR": np.full(24, 1.1)}, index=idx.to_period("M").to_timestamp("M").tz_localize("UTC"))
    cpi = pd.DataFrame(
        {"USD": np.full(24, 100.0), "EUR": np.concatenate([np.full(12, 100.0), np.full(12, 80.0)])},
        index=idx,
    )
    q = real_fx_panel(fx, cpi)
    assert q.iloc[-1]["EUR"] > q.iloc[0]["EUR"]


def test_score_long_undervalued():
    """Negative z (cheap) → positive long-foreign score."""
    idx = pd.date_range("2015-01-01", periods=100, freq="ME", tz="UTC")
    # Mean-reverting real FX around 1.0 with a cheap spell at the end
    rng = np.random.default_rng(0)
    q = pd.DataFrame({"EUR": 1.0 + rng.normal(0, 0.02, 100)}, index=idx)
    q.iloc[-5:] = 0.85  # cheap
    cfg = PppRealFxConfig(signal_lag=0, min_periods=24)
    score = ppp_value_scores(q, cfg=cfg, lookback=36)
    # After enough history, cheap spell → positive score
    assert float(score.dropna().iloc[-1]["EUR"]) > 0


def test_ppp_no_lookahead_on_cpi():
    """Mutating future CPI must not change earlier scores."""
    idx_fx = pd.date_range("2018-01-31", periods=80, freq="ME", tz="UTC")
    idx_cpi = pd.date_range("2018-01-01", periods=80, freq="MS", tz="UTC")
    rng = np.random.default_rng(1)
    fx = pd.DataFrame(
        {
            "EUR": 1.1 + np.cumsum(rng.normal(0, 0.01, 80)),
            "GBP": 1.3 + np.cumsum(rng.normal(0, 0.01, 80)),
            "JPY": 0.009 + np.cumsum(rng.normal(0, 0.0001, 80)),
            "AUD": 0.7 + np.cumsum(rng.normal(0, 0.01, 80)),
        },
        index=idx_fx,
    )
    cpi = pd.DataFrame(
        {
            "USD": 100 * np.cumprod(1 + rng.normal(0.002, 0.001, 80)),
            "EUR": 100 * np.cumprod(1 + rng.normal(0.002, 0.001, 80)),
            "GBP": 100 * np.cumprod(1 + rng.normal(0.002, 0.001, 80)),
            "JPY": 100 * np.cumprod(1 + rng.normal(0.001, 0.001, 80)),
            "AUD": 100 * np.cumprod(1 + rng.normal(0.002, 0.001, 80)),
        },
        index=idx_cpi,
    )
    cfg = PppRealFxConfig(signal_lag=1, min_periods=24)
    q1 = real_fx_panel(fx, cpi)
    s1 = ppp_value_scores(q1, cfg=cfg, lookback=36)

    cpi2 = cpi.copy()
    cpi2.iloc[-3:] *= 1.2
    q2 = real_fx_panel(fx, cpi2)
    s2 = ppp_value_scores(q2, cfg=cfg, lookback=36)

    cut = -10
    a = s1.iloc[:cut].dropna(how="all")
    b = s2.iloc[:cut].dropna(how="all")
    common = a.index.intersection(b.index)
    assert len(common) > 20
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True)


def test_undervalued_long_maps_to_pair_sign():
    """Long EUR (undervalued) → positive EURUSD weight."""
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-31", tz="UTC")])
    ccy_w = pd.DataFrame({"EUR": [0.5], "JPY": [-0.5]}, index=idx)
    pw = currency_weights_to_pair_weights(ccy_w)
    assert float(pw.iloc[0]["EURUSD"]) > 0
    # JPY short → long USDJPY (USD_PAIRS JPY sign -1 → w_ccy * sign)
    sym, sign = USD_PAIRS["JPY"]
    assert sym == "USDJPY"
    assert float(pw.iloc[0]["USDJPY"]) == pytest.approx(-0.5 * sign)


def test_ppp_strategy_smoke_and_causal_returns():
    idx = pd.date_range("2016-01-01", periods=2000, freq="B", tz="UTC")
    rng = np.random.default_rng(3)
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
    # Monthly CPI levels spanning the sample
    cpi_idx = pd.date_range("2015-01-01", periods=120, freq="MS", tz="UTC")
    cpi = pd.DataFrame(
        {c: 100 * np.cumprod(1 + rng.normal(0.002, 0.001, 120)) for c in ("USD", "EUR", "GBP", "AUD", "NZD", "JPY", "CAD", "CHF")},
        index=cpi_idx,
    )
    cfg = PppRealFxConfig(signal_lag=1, lookbacks=(60,), min_periods=36)
    factors = ppp_factor_returns(close, ret, cpi, cfg=cfg)
    assert "ppp_xs_60m" in factors
    assert "ppp_ts_60m" in factors
    port = factors["ppp_xs_60m"]
    assert len(port) == len(ret)
    # Mutate last FX returns — early portfolio must be unchanged
    ret2 = ret.copy()
    ret2.iloc[-5:] = 0.05
    factors2 = ppp_factor_returns(close, ret2, cpi, cfg=cfg)
    cut = -20
    assert np.allclose(
        port.iloc[:cut].fillna(0).values,
        factors2["ppp_xs_60m"].iloc[:cut].fillna(0).values,
    )


def test_load_cpi_level_panel_if_cached():
    if not (MACRO / "fred_CPIAUCSL.csv").exists():
        pytest.skip("CPI cache missing")
    panel = load_cpi_level_panel(download=False, pub_lag_months=1)
    assert "USD" in panel.columns
    assert panel.attrs.get("pub_lag_months") == 1
    assert panel.attrs.get("factor") == "cpi_level"
    # Levels should be around CPI index magnitude, not YoY % (~0–10)
    assert float(panel["USD"].dropna().iloc[-1]) > 50


def test_risk_sweep_utilises_dd_budget():
    """Synthetic positive-drift series should scale up toward FTMO caps."""
    idx = pd.date_range("2020-01-01", periods=500, freq="B", tz="UTC")
    rng = np.random.default_rng(9)
    r = pd.Series(rng.normal(0.0003, 0.002, 500), index=idx)  # mild positive drift, low vol
    r_is = r.iloc[:-100]
    r_oos = r.iloc[-100:]
    sw = sweep_scale_to_ftmo_budget(r_is, r_oos, initial=100_000.0)
    assert sw.scale > 1.0
    assert sw.is_gates_pass
    # Near utilisation: at least one of static/daily close to cap
    assert sw.is_static_loss > 0.05 or sw.is_max_daily_dd > 0.025
    assert sw.oos_gates_pass is not None
