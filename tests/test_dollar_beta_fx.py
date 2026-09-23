"""Tests for Lustig–Verdelhan dollar-factor beta FX module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.dollar_beta_fx import (
    DollarBetaConfig,
    average_forward_discount,
    dollar_beta_afd_returns,
    dollar_beta_factor_returns,
    dollar_beta_xs_returns,
    monthly_rx_dollar,
    prepare_dollar_betas,
    _rank_sort_weights,
)


def _synth_panel(n: int = 2200, seed: int = 7) -> pd.DataFrame:
    """Long enough for 60m β (~1260 trading days + buffer)."""
    idx = pd.date_range("2010-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    # Common dollar factor + idiosyncratic
    common = rng.normal(0, 0.004, n)
    close = {}
    betas = {"EUR": 1.2, "GBP": 0.9, "AUD": 1.5, "NZD": 1.4, "JPY": 0.3, "CAD": 1.1, "CHF": 0.4}
    pairs = {
        "EUR": "EURUSD",
        "GBP": "GBPUSD",
        "AUD": "AUDUSD",
        "NZD": "NZDUSD",
        "JPY": "USDJPY",
        "CAD": "USDCAD",
        "CHF": "USDCHF",
    }
    # Build currency returns then map to pairs
    ccy_ret = {c: betas[c] * common + rng.normal(0, 0.003, n) for c in betas}
    # Pair returns: XXXUSD = ccy ret; USDXXX = -ccy ret
    for ccy, sym in pairs.items():
        if sym.startswith("USD"):
            close[sym] = 100 * np.cumprod(1 + (-ccy_ret[ccy]))
        else:
            close[sym] = 1.0 * np.cumprod(1 + ccy_ret[ccy])
    px = pd.DataFrame(close, index=idx)
    return px.pct_change()


def _synth_rates(idx_start: str = "2009-01-01", n_months: int = 200, seed: int = 1) -> pd.DataFrame:
    m_idx = pd.date_range(idx_start, periods=n_months, freq="MS", tz="UTC")
    rng = np.random.default_rng(seed)
    rates = pd.DataFrame(
        {
            c: 1.0 + 0.5 * rng.normal(0, 1, n_months).cumsum()
            for c in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")
        },
        index=m_idx,
    )
    return rates


def test_prepare_betas_shape_and_attrs():
    ret = _synth_panel(2200, seed=1)
    cfg = DollarBetaConfig(signal_lag=1, skip_months=1, beta_window_months=60, min_periods=36)
    beta = prepare_dollar_betas(ret, cfg=cfg)
    assert not beta.empty
    assert beta.shape[1] >= 5
    assert beta.attrs["beta_window_months"] == 60
    assert beta.attrs["skip_months"] == 1
    assert beta.attrs["signal_lag"] == 1
    # Early months before window should be NaN
    assert beta.iloc[:20].isna().all().all()
    assert beta.dropna(how="all").shape[0] > 20


def test_beta_at_t_ignores_t_return():
    """Mutating month-t currency return must not change β dated at t (skip≥1)."""
    ret = _synth_panel(2200, seed=11)
    cfg = DollarBetaConfig(signal_lag=0, skip_months=1, beta_window_months=60, min_periods=36)
    b1 = prepare_dollar_betas(ret, cfg=cfg)

    # Find a month-end with valid β
    valid = b1.dropna(how="all")
    assert len(valid) > 5
    cut = valid.index[len(valid) // 2]

    ret2 = ret.copy()
    # Blast returns in the cut month (decision month) — should not affect β at cut
    month_mask = (ret2.index.year == cut.year) & (ret2.index.month == cut.month)
    ret2.loc[month_mask] = ret2.loc[month_mask] + 0.05
    b2 = prepare_dollar_betas(ret2, cfg=cfg)

    assert cut in b1.index and cut in b2.index
    a = b1.loc[cut].dropna()
    b = b2.loc[cut].reindex(a.index).dropna()
    common = a.index.intersection(b.index)
    assert len(common) >= 3
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-10)


def test_hml_weights_sum_approx_zero():
    score = pd.DataFrame(
        {
            "EUR": [1.5, 0.2],
            "GBP": [1.2, 1.8],
            "AUD": [0.1, 0.5],
            "JPY": [-0.5, -0.2],
            "CAD": [0.8, 1.0],
            "CHF": [-1.0, -0.8],
        },
        index=pd.DatetimeIndex(["2020-01-31", "2020-02-29"], tz="UTC"),
    )
    w = _rank_sort_weights(score, n_long=2, n_short=2)
    assert not w.empty
    for _, row in w.iterrows():
        assert abs(float(row.sum())) < 1e-9
        assert abs(float(row[row > 0].sum()) - 0.5) < 1e-9
        assert abs(float(row[row < 0].sum()) + 0.5) < 1e-9


def test_afd_sign_flip():
    ret = _synth_panel(2200, seed=3)
    rates = _synth_rates()
    # Force AFD strongly positive then negative in two eras
    rates_pos = rates.copy()
    for c in rates_pos.columns:
        if c != "USD":
            rates_pos[c] = rates_pos["USD"] + 5.0
    rates_neg = rates.copy()
    for c in rates_neg.columns:
        if c != "USD":
            rates_neg[c] = rates_neg["USD"] - 5.0

    cfg = DollarBetaConfig(signal_lag=1, cost_bps_per_side=0.0)
    # Unconditional vs AFD-conditioned should differ when AFD flips
    xs = dollar_beta_xs_returns(ret, cfg=cfg, name="xs")
    afd_pos = dollar_beta_afd_returns(ret, rates_pos, cfg=cfg, name="afd")
    afd_neg = dollar_beta_afd_returns(ret, rates_neg, cfg=cfg, name="afd")

    # With constant positive AFD, afd ≈ xs on overlapping non-zero days
    common = xs.dropna().index.intersection(afd_pos.dropna().index)
    assert len(common) > 100
    # Correlation should be high when AFD always > 0
    corr_pos = xs.loc[common].corr(afd_pos.loc[common])
    assert corr_pos > 0.9

    # Negative AFD → flip → correlation with xs should be negative
    common2 = xs.dropna().index.intersection(afd_neg.dropna().index)
    corr_neg = xs.loc[common2].corr(afd_neg.loc[common2])
    assert corr_neg < -0.9


def test_factor_returns_finite_and_causal():
    ret = _synth_panel(2200, seed=5)
    rates = _synth_rates()
    cfg = DollarBetaConfig(signal_lag=1, cost_bps_per_side=1.5)
    f1 = dollar_beta_factor_returns(ret, rates=rates, cfg=cfg)
    assert "dollar_beta_xs" in f1
    assert "dollar_beta_xs_36m" in f1
    assert "dollar_beta_afd" in f1
    assert "dollar_rx_tsmom" in f1
    assert "dollar_ew" in f1
    for name, s in f1.items():
        assert np.isfinite(s.fillna(0.0)).all(), name
        assert s.dropna().shape[0] > 50, name

    # No look-ahead: mutate far future
    ret2 = ret.copy()
    cut_future = pd.Timestamp("2020-01-01", tz="UTC")
    ret2.loc[cut_future:] = ret2.loc[cut_future:] + 0.02
    f2 = dollar_beta_factor_returns(ret2, rates=rates, cfg=cfg)
    cut = pd.Timestamp("2017-06-30", tz="UTC")
    a = f1["dollar_beta_xs"].loc[:cut].dropna()
    b = f2["dollar_beta_xs"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 100
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_monthly_rx_dollar_and_afd():
    ret = _synth_panel(500, seed=9)
    rx = monthly_rx_dollar(ret)
    assert rx.name == "rx_dollar"
    assert rx.dropna().shape[0] > 10
    rates = _synth_rates(n_months=60)
    afd = average_forward_discount(rates)
    assert afd.name == "afd"
    assert afd.dropna().shape[0] > 5
