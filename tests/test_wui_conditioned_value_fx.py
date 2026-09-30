"""Tests for WUI-conditioned Rogoff PPP / real-FX value module (§88)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.wui_conditioned_value_fx import (
    PRIMARY,
    WuiConditionedValueFxConfig,
    align_monthly_stress_z_daily,
    wui_conditioned_value_factor_returns,
)
from mt5_swing.strategies.wui_conditioned_soft_fx import (
    WuiConditionedSoftFxConfig,
    trailing_z_monthly,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
WUI_FRED = MACRO / "fred_WUIUSA.csv"


def _synth_panel(n_days: int = 2200) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic G10 closes + returns long enough for 60m PPP lookback."""
    idx = pd.date_range("2008-01-01", periods=n_days, freq="B", tz="UTC")
    rng = np.random.default_rng(17)
    close = {}
    for s, level in (
        ("EURUSD", 1.20),
        ("GBPUSD", 1.50),
        ("AUDUSD", 0.80),
        ("NZDUSD", 0.70),
        ("USDJPY", 110.0),
        ("USDCAD", 1.25),
        ("USDCHF", 0.95),
    ):
        ret = rng.normal(0, 0.005, n_days)
        close[s] = level * np.cumprod(1.0 + ret)
    px = pd.DataFrame(close, index=idx)
    ret = px.pct_change()
    return px, ret


def _synth_cpi(n_months: int = 220) -> pd.DataFrame:
    """Synthetic CPI levels (month-start) for G10 + USD."""
    idx = pd.date_range("2007-01-01", periods=n_months, freq="MS", tz="UTC")
    rng = np.random.default_rng(9)
    cols = {}
    for c in ("USD", "EUR", "GBP", "AUD", "NZD", "JPY", "CAD", "CHF"):
        growth = rng.normal(0.002, 0.001, n_months)
        cols[c] = 100.0 * np.cumprod(1.0 + growth)
    return pd.DataFrame(cols, index=idx)


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
    return pd.Series(
        rng.normal(mean, 0.08, n_months).clip(min=0.01), index=idx, name=name
    )


def test_primary_name_constant():
    assert PRIMARY == "value_low_wui"


def test_gate_direction_low_on_high_off():
    px, ret = _synth_panel(2400)
    cpi = _synth_cpi(240)
    idx = pd.date_range("2006-01-01", periods=240, freq="MS", tz="UTC")
    wui = pd.Series(
        np.concatenate([np.full(120, 0.05), np.full(120, 0.50)]),
        index=idx,
        name="US_WUI",
    )
    cfg = WuiConditionedValueFxConfig(
        signal_lag_months=1,
        weight_lag_days=1,
        z_window=60,
        min_periods=24,
        z_high=1.0,
        lookback=60,
    )
    out = wui_conditioned_value_factor_returns(px, ret, cpi, us_wui=wui, cfg=cfg)
    for key in (
        "value_low_wui",
        "value_wui_cool",
        "value_raw",
        "value_high_wui",
        "us_wui_haven_usd",
        "value_wui_stack",
        "value_wui_ew",
        "value_wui_regime",
    ):
        assert key in out
    lo = out["value_low_wui"]
    raw = out["value_raw"]
    both = lo.notna() & raw.notna()
    assert both.sum() > 100
    # Gated magnitude ≤ ungated (gate ∈ {0,1})
    assert (lo.loc[both].abs() <= raw.loc[both].abs() + 1e-12).mean() > 0.95
    hi = out["value_high_wui"]
    product = lo.loc[both].abs() * hi.loc[both].abs()
    assert (product <= 1e-10).mean() > 0.90


def test_cool_scale_bounds():
    """Cool shrinks exposure; costs on weight paths can push a few day ratios below cool."""
    px, ret = _synth_panel(1800)
    cpi = _synth_cpi(200)
    wui = _synth_monthly(
        start="2008-01-01", n_months=160, seed=1, mean=0.20, name="US_WUI"
    )
    cfg = WuiConditionedValueFxConfig(cool=0.35, z_high=1.0)
    out = wui_conditioned_value_factor_returns(px, ret, cpi, us_wui=wui, cfg=cfg)
    raw = out["value_raw"]
    cool = out["value_wui_cool"]
    both = raw.notna() & cool.notna()
    assert both.sum() > 100
    ok = (cool.loc[both].abs() <= raw.loc[both].abs() + 1e-9).mean()
    assert ok > 0.90
    common = raw.dropna().index.intersection(cool.dropna().index)
    ratio = (cool.loc[common] / raw.loc[common]).replace([np.inf, -np.inf], np.nan)
    ratio = ratio.dropna()
    ratio = ratio[raw.loc[ratio.index].abs() > 1e-4]
    if len(ratio) > 20:
        assert float(np.nanpercentile(ratio, 5)) >= 0.35 - 0.15
        assert float(np.nanpercentile(ratio, 95)) <= 1.0 + 0.15


def test_stack_is_gate_then_cool():
    """Stack = gate×cool on weights; vs low-gate, |stack| ≤ |lo| most days (costs)."""
    px, ret = _synth_panel(2000)
    cpi = _synth_cpi(220)
    wui = _synth_monthly(
        start="2008-01-01", n_months=180, seed=5, mean=0.18, name="US_WUI"
    )
    cfg = WuiConditionedValueFxConfig(cool=0.35, z_high=1.0)
    out = wui_conditioned_value_factor_returns(px, ret, cpi, us_wui=wui, cfg=cfg)
    raw = out["value_raw"]
    stack = out["value_wui_stack"]
    lo = out["value_low_wui"]
    both = raw.notna() & stack.notna() & lo.notna()
    assert both.sum() > 50
    assert (stack.loc[both].abs() <= lo.loc[both].abs() + 1e-9).mean() > 0.95
    assert (stack.loc[both].abs() <= raw.loc[both].abs() + 1e-9).mean() > 0.90


def test_no_lookahead_mutate_future_wui():
    """Mutating future WUI must not change earlier factor returns."""
    px, ret = _synth_panel(2200)
    cpi = _synth_cpi(220)
    wui = _synth_monthly(
        start="2008-01-01", n_months=200, seed=42, mean=0.15, name="US_WUI"
    )
    cfg = WuiConditionedValueFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    f1 = wui_conditioned_value_factor_returns(px, ret, cpi, us_wui=wui, cfg=cfg)
    wui2 = wui.copy()
    wui2.loc["2018-01-01":] = wui2.loc["2018-01-01":] + 0.40
    f2 = wui_conditioned_value_factor_returns(px, ret, cpi, us_wui=wui2, cfg=cfg)
    cut = pd.Timestamp("2016-06-30", tz="UTC")
    a = f1["value_low_wui"].loc[:cut].dropna()
    b = f2["value_low_wui"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 100
    assert np.allclose(
        a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12
    )


def test_monthly_z_window_not_daily_252():
    """PIT: z is computed on monthly frequency with z_window months."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    s = pd.Series(np.linspace(0.05, 0.40, 120), index=idx, name="US_WUI")
    z = trailing_z_monthly(s, lookback=60, min_periods=24)
    assert z.iloc[59:].notna().sum() > 40
    assert z.iloc[:23].isna().all()


def test_align_monthly_stress_z_lagged():
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    s = pd.Series(np.linspace(0.05, 0.40, 120), index=idx, name="US_WUI")
    daily_idx = pd.date_range("2015-01-01", periods=500, freq="B", tz="UTC")
    soft_cfg = WuiConditionedSoftFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    z = align_monthly_stress_z_daily(s, daily_idx, cfg=soft_cfg)
    assert z.index.equals(daily_idx)
    assert z.notna().sum() > 50
    assert pd.isna(z.iloc[0])


def test_monthly_z_window_prior():
    """Config must use monthly z_window=60 (CIP §71 / WUI §85–§87 mirror, not 252d)."""
    cfg = WuiConditionedValueFxConfig()
    assert cfg.z_window == 60
    assert cfg.min_periods == 24
    assert cfg.signal_lag_months == 1
    assert cfg.weight_lag_days == 1
    assert cfg.lookback == 60
    assert cfg.n_long == 2
    assert cfg.n_short == 2
    assert cfg.ppp_signal_lag == 1
    assert cfg.cool == 0.35
    assert cfg.z_high == 1.0
    assert cfg.pub_lag_months == 4


def test_without_haven_still_builds_core():
    px, ret = _synth_panel(1800)
    cpi = _synth_cpi(200)
    wui = _synth_monthly(
        start="2008-01-01", n_months=160, seed=3, mean=0.12, name="US_WUI"
    )
    out = wui_conditioned_value_factor_returns(
        px, ret, cpi, us_wui=wui, include_haven=False
    )
    assert "value_low_wui" in out
    assert "us_wui_haven_usd" not in out
    assert "value_wui_ew" in out  # EW of low + cool still
    assert "value_wui_stack" in out
    assert "value_wui_regime" not in out  # needs haven


def test_factor_shapes_match_index():
    px, ret = _synth_panel(1800)
    cpi = _synth_cpi(200)
    wui = _synth_monthly(
        start="2008-01-01", n_months=160, seed=7, name="US_WUI"
    )
    out = wui_conditioned_value_factor_returns(
        px, ret, cpi, us_wui=wui, cfg=WuiConditionedValueFxConfig()
    )
    assert out["value_low_wui"].index.equals(ret.index)
    assert out["us_wui_haven_usd"].index.equals(ret.index)
    assert out["value_wui_ew"].index.equals(ret.index)
    assert out["value_wui_regime"].index.equals(ret.index)


def test_empty_cpi_returns_empty():
    px, ret = _synth_panel(400)
    wui = _synth_monthly(n_months=80, seed=2, name="US_WUI")
    out = wui_conditioned_value_factor_returns(
        px, ret, pd.DataFrame(), us_wui=wui, cfg=WuiConditionedValueFxConfig()
    )
    assert out == {}


def test_thin_panel_returns_empty():
    px, ret = _synth_panel(400)
    ret2 = ret.iloc[:, :2]
    px2 = px.iloc[:, :2]
    cpi = _synth_cpi(80)
    wui = _synth_monthly(n_months=80, seed=2, name="US_WUI")
    out = wui_conditioned_value_factor_returns(px2, ret2, cpi, us_wui=wui)
    assert out == {}


@pytest.mark.skipif(not WUI_FRED.exists(), reason="WUIUSA FRED CSV missing")
def test_load_real_us_wui():
    from mt5_swing.strategies.wui_conditioned_value_fx import load_us_wui_series

    wui = load_us_wui_series(pub_lag_months=4, download=False)
    assert wui.dropna().shape[0] > 60
    px, ret = _synth_panel(1800)
    cpi = _synth_cpi(200)
    out = wui_conditioned_value_factor_returns(px, ret, cpi, us_wui=wui)
    assert PRIMARY in out
    assert out[PRIMARY].notna().sum() > 20
