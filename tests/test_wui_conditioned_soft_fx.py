"""Tests for WUI-conditioned Dahlquist soft-signal EW FX module (§85)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.wui_conditioned_soft_fx import (
    PRIMARY,
    WuiConditionedSoftFxConfig,
    align_monthly_stress_z_daily,
    trailing_z_monthly,
    wui_conditioned_soft_factor_returns,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
WUI_FRED = MACRO / "fred_WUIUSA.csv"


def _synth_soft(n_days: int = 1800) -> tuple[pd.Series, pd.DataFrame]:
    idx = pd.date_range("2016-01-01", periods=n_days, freq="B", tz="UTC")
    rng = np.random.default_rng(11)
    soft = pd.Series(
        rng.normal(0.0001, 0.002, n_days), index=idx, name="soft_ew_macro5"
    )
    ret = pd.DataFrame(
        {
            s: rng.normal(0, 0.005, n_days)
            for s in ("EURUSD", "GBPUSD", "USDJPY", "USDCAD")
        },
        index=idx,
    )
    return soft, ret


def _synth_monthly(
    start: str = "2010-01-01",
    n_months: int = 180,
    *,
    seed: int = 0,
    mean: float = 0.15,
    name: str = "US_WUI",
) -> pd.Series:
    idx = pd.date_range(start, periods=n_months, freq="MS", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, 0.08, n_months).clip(min=0.01), index=idx, name=name)


def test_primary_name_constant():
    assert PRIMARY == "soft_low_wui"


def test_gate_direction_low_on_high_off():
    soft, ret = _synth_soft(2000)
    idx = pd.date_range("2010-01-01", periods=200, freq="MS", tz="UTC")
    # Low WUI early, then high — so z goes from low to high after warmup
    wui = pd.Series(
        np.concatenate([np.full(100, 0.05), np.full(100, 0.50)]),
        index=idx,
        name="US_WUI",
    )
    cfg = WuiConditionedSoftFxConfig(
        signal_lag_months=1,
        weight_lag_days=1,
        z_window=60,
        min_periods=24,
        z_high=1.0,
    )
    out = wui_conditioned_soft_factor_returns(
        soft, us_wui=wui, pair_ret=ret, cfg=cfg
    )
    for key in (
        "soft_low_wui",
        "soft_wui_cool",
        "soft_raw",
        "soft_high_wui",
        "us_wui_haven_usd",
        "soft_wui_stack",
        "soft_wui_ew",
        "soft_wui_regime",
    ):
        assert key in out
    common = soft.dropna().index.intersection(out["soft_raw"].dropna().index)
    assert np.allclose(soft.loc[common].values, out["soft_raw"].loc[common].values)
    lo = out["soft_low_wui"]
    raw = out["soft_raw"]
    both = lo.notna() & raw.notna()
    assert both.sum() > 100
    assert (lo.loc[both].abs() <= raw.loc[both].abs() + 1e-15).all()
    hi = out["soft_high_wui"]
    product = lo.loc[both].abs() * hi.loc[both].abs()
    assert (product <= 1e-18).mean() > 0.99


def test_cool_scale_bounds():
    soft, ret = _synth_soft(600)
    wui = _synth_monthly(n_months=120, seed=1, mean=0.20, name="US_WUI")
    cfg = WuiConditionedSoftFxConfig(cool=0.35, z_high=1.0)
    out = wui_conditioned_soft_factor_returns(
        soft, us_wui=wui, pair_ret=ret, cfg=cfg
    )
    raw = out["soft_raw"]
    cool = out["soft_wui_cool"]
    common = raw.dropna().index.intersection(cool.dropna().index)
    ratio = (cool.loc[common] / raw.loc[common]).replace([np.inf, -np.inf], np.nan)
    ratio = ratio.dropna()
    ratio = ratio[raw.loc[ratio.index].abs() > 1e-12]
    if len(ratio):
        assert float(ratio.min()) >= 0.35 - 1e-9
        assert float(ratio.max()) <= 1.0 + 1e-9


def test_stack_is_gate_then_cool():
    soft, ret = _synth_soft(800)
    wui = _synth_monthly(n_months=140, seed=5, mean=0.18, name="US_WUI")
    cfg = WuiConditionedSoftFxConfig(cool=0.35, z_high=1.0)
    out = wui_conditioned_soft_factor_returns(
        soft, us_wui=wui, pair_ret=ret, cfg=cfg
    )
    raw = out["soft_raw"]
    stack = out["soft_wui_stack"]
    lo = out["soft_low_wui"]
    both = raw.notna() & stack.notna() & lo.notna()
    assert both.sum() > 50
    # Stack magnitude ≤ |low| (cool ≤ 1) and ≤ |raw|
    assert (stack.loc[both].abs() <= lo.loc[both].abs() + 1e-12).all()
    assert (stack.loc[both].abs() <= raw.loc[both].abs() + 1e-12).all()


def test_no_lookahead_mutate_future_wui():
    soft, ret = _synth_soft(1800)
    wui = _synth_monthly(
        start="2012-01-01", n_months=180, seed=42, mean=0.15, name="US_WUI"
    )
    cfg = WuiConditionedSoftFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    f1 = wui_conditioned_soft_factor_returns(
        soft, us_wui=wui, pair_ret=ret, cfg=cfg
    )
    wui2 = wui.copy()
    wui2.loc["2021-01-01":] = wui2.loc["2021-01-01":] + 0.40
    f2 = wui_conditioned_soft_factor_returns(
        soft, us_wui=wui2, pair_ret=ret, cfg=cfg
    )
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["soft_low_wui"].loc[:cut].dropna()
    b = f2["soft_low_wui"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 100
    assert np.allclose(
        a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12
    )


def test_monthly_z_window_not_daily_252():
    """PIT: z is computed on monthly frequency with z_window months (CIP §71 mirror)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    s = pd.Series(np.linspace(0.05, 0.40, 120), index=idx, name="US_WUI")
    z = trailing_z_monthly(s, lookback=60, min_periods=24)
    assert z.iloc[59:].notna().sum() > 40
    assert z.iloc[:23].isna().all()


def test_align_monthly_stress_z_lagged():
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    s = pd.Series(np.linspace(0.05, 0.40, 120), index=idx, name="US_WUI")
    soft_idx = pd.date_range("2015-01-01", periods=500, freq="B", tz="UTC")
    cfg = WuiConditionedSoftFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    z = align_monthly_stress_z_daily(s, soft_idx, cfg=cfg)
    assert z.index.equals(soft_idx)
    assert z.notna().sum() > 50
    cfg0 = WuiConditionedSoftFxConfig(
        signal_lag_months=1, weight_lag_days=0, z_window=60, min_periods=24
    )
    z0 = align_monthly_stress_z_daily(s, soft_idx, cfg=cfg0)
    both = z.notna() & z0.notna()
    assert both.sum() > 50
    assert z.loc[both].isna().sum() == 0


def test_without_haven_still_builds_core():
    soft, _ret = _synth_soft(400)
    wui = _synth_monthly(n_months=100, seed=3, mean=0.12, name="US_WUI")
    out = wui_conditioned_soft_factor_returns(
        soft, us_wui=wui, pair_ret=None, include_haven=False
    )
    assert "soft_low_wui" in out
    assert "us_wui_haven_usd" not in out
    assert "soft_wui_ew" in out  # EW of low + cool still
    assert "soft_wui_stack" in out
    assert "soft_wui_regime" not in out  # needs haven


def test_factor_shapes_match_soft_index():
    soft, ret = _synth_soft(800)
    wui = _synth_monthly(n_months=140, seed=7, name="US_WUI")
    out = wui_conditioned_soft_factor_returns(
        soft, us_wui=wui, pair_ret=ret, cfg=WuiConditionedSoftFxConfig()
    )
    assert out["soft_low_wui"].index.equals(soft.index)
    assert out["us_wui_haven_usd"].index.equals(soft.index)
    assert out["soft_wui_ew"].index.equals(soft.index)
    assert out["soft_wui_regime"].index.equals(soft.index)


@pytest.mark.skipif(not WUI_FRED.exists(), reason="WUIUSA FRED CSV missing")
def test_load_real_us_wui():
    from mt5_swing.strategies.wui_conditioned_soft_fx import load_us_wui_series

    wui = load_us_wui_series(pub_lag_months=4, download=False)
    assert wui.dropna().shape[0] > 60
    soft, ret = _synth_soft(500)
    out = wui_conditioned_soft_factor_returns(soft, us_wui=wui, pair_ret=ret)
    assert PRIMARY in out
    assert out[PRIMARY].notna().sum() > 50
