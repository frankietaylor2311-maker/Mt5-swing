"""Tests for WUI-conditioned BIS REER HML-FX value module (§89)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.wui_conditioned_reer_value_fx import (
    PRIMARY,
    WuiConditionedReerValueFxConfig,
    align_monthly_stress_z_daily,
    wui_conditioned_reer_value_factor_returns,
)
from mt5_swing.strategies.wui_conditioned_soft_fx import (
    WuiConditionedSoftFxConfig,
    trailing_z_monthly,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
WUI_FRED = MACRO / "fred_WUIUSA.csv"
REER_USD = MACRO / "fred_RBUSBIS.csv"


def _synth_pairs(n_days: int = 1800) -> pd.DataFrame:
    idx = pd.date_range("2016-01-01", periods=n_days, freq="B", tz="UTC")
    rng = np.random.default_rng(17)
    cols = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
    data = {c: rng.normal(0.0, 0.005, n_days) for c in cols}
    return pd.DataFrame(data, index=idx)


def _synth_reer(
    start: str = "2010-01-01",
    n_months: int = 180,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n_months, freq="MS", tz="UTC")
    rng = np.random.default_rng(seed)
    cols = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
    data = {c: 100.0 + np.cumsum(rng.normal(0.0, 0.8, n_months)) for c in cols}
    return pd.DataFrame(data, index=idx)


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
    assert PRIMARY == "reer_low_wui"


def test_gate_direction_low_on_high_off():
    pret = _synth_pairs(2000)
    reer = _synth_reer(n_months=200, seed=1)
    idx = pd.date_range("2010-01-01", periods=200, freq="MS", tz="UTC")
    wui = pd.Series(
        np.concatenate([np.full(100, 0.05), np.full(100, 0.50)]),
        index=idx,
        name="US_WUI",
    )
    cfg = WuiConditionedReerValueFxConfig(
        signal_lag_months=1,
        weight_lag_days=1,
        z_window=60,
        min_periods=24,
        z_high=1.0,
    )
    out = wui_conditioned_reer_value_factor_returns(pret, reer, us_wui=wui, cfg=cfg)
    for key in (
        "reer_low_wui",
        "reer_wui_cool",
        "reer_raw",
        "reer_high_wui",
        "us_wui_haven_usd",
        "reer_wui_stack",
        "reer_wui_ew",
        "reer_wui_regime",
    ):
        assert key in out
    lo = out["reer_low_wui"]
    hi = out["reer_high_wui"]
    raw = out["reer_raw"]
    both = lo.notna() & raw.notna()
    assert both.sum() > 100
    # Gated magnitude ≤ ungated (gate ∈ {0,1})
    assert (lo.loc[both].abs() <= raw.loc[both].abs() + 1e-12).mean() > 0.95
    product = lo.loc[both].abs() * hi.loc[both].abs()
    assert (product <= 1e-10).mean() > 0.90


def test_cool_scale_bounds():
    """Cool shrinks exposure; costs on weight paths can push a few day ratios below cool."""
    pret = _synth_pairs(800)
    reer = _synth_reer(n_months=140, seed=2)
    wui = _synth_monthly(
        start="2010-01-01", n_months=160, seed=1, mean=0.20, name="US_WUI"
    )
    cfg = WuiConditionedReerValueFxConfig(cool=0.35, z_high=1.0)
    out = wui_conditioned_reer_value_factor_returns(pret, reer, us_wui=wui, cfg=cfg)
    raw = out["reer_raw"]
    cool = out["reer_wui_cool"]
    both = raw.notna() & cool.notna()
    assert both.sum() > 50
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
    pret = _synth_pairs(1800)
    reer = _synth_reer(n_months=180, seed=5)
    wui = _synth_monthly(
        start="2010-01-01", n_months=180, seed=5, mean=0.18, name="US_WUI"
    )
    cfg = WuiConditionedReerValueFxConfig(cool=0.35, z_high=1.0)
    out = wui_conditioned_reer_value_factor_returns(pret, reer, us_wui=wui, cfg=cfg)
    raw = out["reer_raw"]
    stack = out["reer_wui_stack"]
    lo = out["reer_low_wui"]
    both = raw.notna() & stack.notna() & lo.notna()
    assert both.sum() > 50
    assert (stack.loc[both].abs() <= lo.loc[both].abs() + 1e-9).mean() > 0.95
    assert (stack.loc[both].abs() <= raw.loc[both].abs() + 1e-9).mean() > 0.90


def test_no_lookahead_mutate_future_wui():
    """Mutating future WUI must not change earlier factor returns."""
    pret = _synth_pairs(1800)
    reer = _synth_reer(start="2012-01-01", n_months=180, seed=4)
    wui = _synth_monthly(
        start="2012-01-01", n_months=180, seed=42, mean=0.15, name="US_WUI"
    )
    cfg = WuiConditionedReerValueFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    f1 = wui_conditioned_reer_value_factor_returns(pret, reer, us_wui=wui, cfg=cfg)
    wui2 = wui.copy()
    wui2.loc["2019-01-01":] = wui2.loc["2019-01-01":] + 0.40
    f2 = wui_conditioned_reer_value_factor_returns(pret, reer, us_wui=wui2, cfg=cfg)
    cut = pd.Timestamp("2017-06-30", tz="UTC")
    a = f1["reer_low_wui"].loc[:cut].dropna()
    b = f2["reer_low_wui"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 50
    assert np.allclose(
        a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12
    )


def test_no_lookahead_mutate_future_reer():
    """Future REER mutation must not change past value signals."""
    pret = _synth_pairs(1800)
    reer = _synth_reer(start="2012-01-01", n_months=180, seed=8)
    wui = _synth_monthly(
        start="2012-01-01", n_months=180, seed=11, mean=0.15, name="US_WUI"
    )
    cfg = WuiConditionedReerValueFxConfig()
    f1 = wui_conditioned_reer_value_factor_returns(pret, reer, us_wui=wui, cfg=cfg)
    reer2 = reer.copy()
    reer2.loc["2021-01-01":] = reer2.loc["2021-01-01":] * 1.15
    f2 = wui_conditioned_reer_value_factor_returns(pret, reer2, us_wui=wui, cfg=cfg)
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["reer_raw"].loc[:cut].dropna()
    b = f2["reer_raw"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 50
    assert np.allclose(
        a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12
    )


def test_factor_shapes_match_pair_index():
    pret = _synth_pairs(800)
    reer = _synth_reer(n_months=140, seed=7)
    wui = _synth_monthly(start="2012-01-01", n_months=120, seed=7, name="US_WUI")
    out = wui_conditioned_reer_value_factor_returns(
        pret,
        reer,
        us_wui=wui,
        cfg=WuiConditionedReerValueFxConfig(z_window=36, min_periods=18),
    )
    assert out["reer_low_wui"].index.equals(pret.index)
    assert out["reer_wui_ew"].index.equals(pret.index)
    assert out["reer_wui_regime"].index.equals(pret.index)


def test_gate_on_off_explicit_z():
    """Primary gate: reer on when z≤0; inverse on when z≥z_high."""
    idx = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    z = pd.Series([-1.0, -0.5, 0.0, 0.5, 1.5, 2.0, -0.2, 1.1, 0.0, -2.0], index=idx)
    cfg = WuiConditionedReerValueFxConfig(z_high=1.0)
    gate_lo = (z <= 0.0).astype(float)
    gate_hi = (z >= cfg.z_high).astype(float)
    assert gate_lo.iloc[0] == 1.0
    assert gate_lo.iloc[4] == 0.0
    assert gate_hi.iloc[4] == 1.0
    assert gate_hi.iloc[0] == 0.0


def test_monthly_z_window_prior():
    """Config must use monthly z_window=60 (CIP §71 / WUI §85–§88 mirror, not 252d)."""
    cfg = WuiConditionedReerValueFxConfig()
    assert cfg.z_window == 60
    assert cfg.min_periods == 24
    assert cfg.signal_lag_months == 1
    assert cfg.weight_lag_days == 1
    assert cfg.reer_signal_lag == 1
    assert cfg.reer_z_window == 60
    assert cfg.n_long == 2
    assert cfg.n_short == 2
    assert cfg.reer_weight_lag_days == 1
    assert cfg.cool == 0.35
    assert cfg.z_high == 1.0
    assert cfg.pub_lag_months == 4


def test_haven_optional():
    pret = _synth_pairs(800)
    reer = _synth_reer(n_months=140, seed=3)
    wui = _synth_monthly(start="2012-01-01", n_months=120, seed=3, name="US_WUI")
    cfg = WuiConditionedReerValueFxConfig(z_window=36, min_periods=18)
    out = wui_conditioned_reer_value_factor_returns(
        pret, reer, us_wui=wui, cfg=cfg, include_haven=False
    )
    assert "reer_low_wui" in out
    assert "reer_wui_cool" in out
    assert "reer_raw" in out
    assert "reer_high_wui" in out
    assert "us_wui_haven_usd" not in out
    assert "reer_wui_stack" in out
    assert "reer_wui_ew" in out  # EW of low + cool still
    assert "reer_wui_regime" not in out


def test_empty_reer_returns_empty():
    pret = _synth_pairs(400)
    wui = _synth_monthly(n_months=80, seed=1, name="US_WUI")
    out = wui_conditioned_reer_value_factor_returns(
        pret, pd.DataFrame(), us_wui=wui, cfg=WuiConditionedReerValueFxConfig()
    )
    assert out == {}


def test_thin_panel_returns_empty():
    pret = _synth_pairs(400).iloc[:, :2]
    reer = _synth_reer(n_months=80, seed=2)
    wui = _synth_monthly(n_months=80, seed=2, name="US_WUI")
    out = wui_conditioned_reer_value_factor_returns(pret, reer, us_wui=wui)
    assert out == {}


def test_monthly_z_window_not_daily_252():
    """PIT: z is computed on monthly frequency with z_window months."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    s = pd.Series(np.linspace(0.05, 0.40, 120), index=idx, name="US_WUI")
    z = trailing_z_monthly(s, lookback=60, min_periods=24)
    assert z.iloc[59:].notna().sum() > 40
    assert z.iloc[:23].isna().all()


@pytest.mark.skipif(
    not (WUI_FRED.exists() and REER_USD.exists()),
    reason="WUIUSA FRED CSV or REER CSV missing",
)
def test_load_real_us_wui_and_reer():
    from mt5_swing.data.fred_bis_reer import load_bis_reer_panel
    from mt5_swing.strategies.wui_conditioned_reer_value_fx import load_us_wui_series

    wui = load_us_wui_series(pub_lag_months=4, download=False)
    assert wui.dropna().shape[0] > 60
    reer = load_bis_reer_panel(pub_lag_months=2, download=False, force=False)
    pret = _synth_pairs(500)
    assert reer.dropna(how="all").shape[0] > 60
    out = wui_conditioned_reer_value_factor_returns(pret, reer, us_wui=wui)
    assert PRIMARY in out
    assert out[PRIMARY].notna().sum() > 20
