"""Tests for Menkhoff-style global FX realized-vol risk factor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.fx_realized_vol import (
    FxRealizedVolConfig,
    FxRealizedVolBasket,
    apply_fxrv_cool_to_weights,
    build_fxrv_state,
    daily_fx_vol_level,
    fx_realized_vol_factor_returns,
    fx_rv_innovation,
    fx_rv_zscore,
    trailing_fx_rv,
    usd_tilt_pair_weights_from_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
HISTORY = Path(__file__).resolve().parents[1] / "data" / "history"


def _synthetic_pair_ret(n: int = 800, seed: int = 7) -> pd.DataFrame:
    idx = pd.date_range("2018-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    # Inject a high-vol episode in the last 40 days
    base = rng.normal(0, 0.004, size=(n, 7))
    base[-40:] *= 4.0
    return pd.DataFrame(
        base,
        index=idx,
        columns=["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"],
    )


def test_daily_fx_vol_level_positive_and_rises_in_spike():
    pret = _synthetic_pair_ret()
    level = daily_fx_vol_level(pret)
    assert level.notna().sum() > 100
    assert float(level.iloc[-10:].mean()) > float(level.iloc[100:200].mean())


def test_dollar_neutral_abs_variant():
    pret = _synthetic_pair_ret()
    a = daily_fx_vol_level(pret, dollar_neutral_abs=False)
    b = daily_fx_vol_level(pret, dollar_neutral_abs=True)
    assert a.notna().sum() > 50 and b.notna().sum() > 50
    # Both positive; not identical in general
    assert float(a.dropna().mean()) > 0
    assert float(b.dropna().mean()) > 0


def test_trailing_rv_and_z_causal():
    """Future return shock must not change earlier RV/z."""
    pret = _synthetic_pair_ret()
    cfg = FxRealizedVolConfig(signal_lag=1, rv_windows=(21,))
    st1 = build_fxrv_state(pret, cfg=cfg, window=21)
    pret2 = pret.copy()
    pret2.iloc[-5:] *= 10.0
    st2 = build_fxrv_state(pret2, cfg=cfg, window=21)
    cut = -30
    for key in ("rv", "z"):
        a = st1[key].iloc[:cut]
        b = st2[key].iloc[:cut]
        common = a.dropna().index.intersection(b.dropna().index)
        assert len(common) > 50
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True)


def test_fx_rv_zscore_signal_lag():
    idx = pd.date_range("2020-01-01", periods=400, freq="B", tz="UTC")
    rng = np.random.default_rng(1)
    rv = pd.Series(rng.uniform(0.002, 0.01, 400), index=idx)
    z0 = fx_rv_zscore(rv, z_window=100, min_periods=40, signal_lag=0)
    z1 = fx_rv_zscore(rv, z_window=100, min_periods=40, signal_lag=1)
    # Lagged z equals shift of unlagged
    aligned = z0.shift(1).dropna()
    common = aligned.index.intersection(z1.dropna().index)
    assert len(common) > 50
    assert np.allclose(aligned.loc[common].values, z1.loc[common].values, equal_nan=True)


def test_usd_tilt_weights_high_z_long_usd():
    idx = pd.date_range("2021-01-01", periods=50, freq="B", tz="UTC")
    z = pd.Series(0.0, index=idx)
    z.iloc[-10:] = 2.0  # above z_high=1
    cfg = FxRealizedVolConfig(usd_tilt=0.5, z_high=1.0, z_low=0.0)
    cols = ["EURUSD", "USDJPY", "GBPUSD"]
    w = usd_tilt_pair_weights_from_z(z, cols, cfg=cfg)
    last = w.iloc[-1]
    # Long USD: short EURUSD, long USDJPY
    assert last["EURUSD"] < 0
    assert last["USDJPY"] > 0
    # Early flat-ish
    assert abs(float(w.iloc[5].sum())) < 1e-9


def test_carry_cool_reduces_gross_in_high_vol():
    idx = pd.date_range("2021-01-01", periods=60, freq="B", tz="UTC")
    base = pd.DataFrame(
        {"EURUSD": 0.25, "USDJPY": -0.25, "AUDUSD": 0.25, "GBPUSD": -0.25},
        index=idx,
    )
    z = pd.Series(0.0, index=idx)
    z.iloc[-15:] = 2.5
    cfg = FxRealizedVolConfig(cool=0.35, z_high=1.0, z_low=0.0)
    cooled = apply_fxrv_cool_to_weights(base, z, cfg=cfg)
    gross_early = float(cooled.iloc[10].abs().sum())
    gross_late = float(cooled.iloc[-1].abs().sum())
    assert gross_late < gross_early
    assert abs(gross_late / gross_early - 0.35) < 0.05


def test_innovation_causal():
    idx = pd.date_range("2019-01-01", periods=300, freq="B", tz="UTC")
    rng = np.random.default_rng(3)
    rv = pd.Series(rng.uniform(0.003, 0.008, 300), index=idx)
    rv.iloc[-20:] = 0.03
    inn1 = fx_rv_innovation(rv, ar_window=63, min_periods=40, signal_lag=1)
    rv2 = rv.copy()
    rv2.iloc[-3:] = 0.1
    inn2 = fx_rv_innovation(rv2, ar_window=63, min_periods=40, signal_lag=1)
    cut = -15
    a = inn1.iloc[:cut].dropna()
    b = inn2.iloc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 40
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True)


def test_factor_returns_smoke_and_no_lookahead():
    pret = _synthetic_pair_ret(900)
    # Synthetic rates panel (monthly)
    midx = pd.date_range("2018-01-01", periods=40, freq="MS", tz="UTC")
    rates = pd.DataFrame(
        {
            "USD": 1.0,
            "EUR": np.linspace(0.5, 1.5, 40),
            "GBP": np.linspace(1.0, 2.0, 40),
            "AUD": np.linspace(2.0, 3.0, 40),
            "JPY": np.linspace(0.1, 0.0, 40),
            "CAD": np.linspace(1.0, 1.5, 40),
            "CHF": np.linspace(0.0, -0.5, 40),
            "NZD": np.linspace(2.5, 3.5, 40),
        },
        index=midx,
    )
    cfg = FxRealizedVolConfig(signal_lag=1, cost_bps_side=1.5)
    factors = fx_realized_vol_factor_returns(pret, rates, cfg=cfg)
    assert "fxrv_usd_tilt_21d" in factors
    assert "carry_fxrv_cool_21d" in factors
    assert "carry_raw" in factors
    # Mutate last returns → earlier port unchanged
    factors1 = factors
    pret2 = pret.copy()
    pret2.iloc[-1] = 0.0
    factors2 = fx_realized_vol_factor_returns(pret2, rates, cfg=cfg)
    for name in ("fxrv_usd_tilt_21d", "carry_fxrv_cool_21d"):
        a = factors1[name].iloc[:-5]
        b = factors2[name].iloc[:-5]
        assert np.allclose(a.values, b.values, equal_nan=True)


def test_basket_helper():
    pret = _synthetic_pair_ret(500)
    basket = FxRealizedVolBasket(signal_lag=1, rv_windows=(21, 63))
    st = basket.state(pret, window=21)
    assert "z" in st and st["z"].notna().sum() > 50
    rets = basket.portfolio_returns(pret, rates=None)
    assert "fxrv_usd_tilt_21d" in rets
    assert "carry_fxrv_cool_21d" not in rets  # no rates


@pytest.mark.skipif(
    not (HISTORY / "EURUSD_D1.csv").exists(),
    reason="Yahoo FX history missing",
)
def test_real_yahoo_panel_rv_finite():
    from mt5_swing.data.loader import load_ohlc_csv

    closes = {}
    for sym in ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCAD", "USDCHF"]:
        path = HISTORY / f"{sym}_D1.csv"
        if path.exists():
            df = load_ohlc_csv(path, symbol=sym, timeframe="D1")
            closes[sym] = df["close"]
    px = pd.DataFrame(closes).sort_index().dropna(how="all")
    if px.index.tz is None:
        px.index = px.index.tz_localize("UTC")
    pret = px.pct_change()
    level = daily_fx_vol_level(pret)
    rv = trailing_fx_rv(level, window=21, min_periods=15)
    assert rv.dropna().mean() > 0
    assert rv.dropna().mean() < 0.05  # sane daily abs-return avg
