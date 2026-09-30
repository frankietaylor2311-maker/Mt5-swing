"""Tests for ANFCI-conditioned Rogoff PPP / real-FX value module (§98)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.anfci_conditioned_value_fx import (
    PRIMARY,
    AnfciConditionedValueFxConfig,
    align_monthly_stress_z_daily,
    anfci_conditioned_value_factor_returns,
)
from mt5_swing.strategies.anfci_conditioned_soft_fx import (
    AnfciConditionedSoftFxConfig,
    trailing_z_monthly,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
ANFCI_FRED = MACRO / "fred_ANFCI.csv"


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
    name: str = "ANFCI",
) -> pd.Series:
    idx = pd.date_range(start, periods=n_months, freq="MS", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(mean, 0.08, n_months).clip(min=0.01), index=idx, name=name
    )


def test_primary_name_constant():
    assert PRIMARY == "value_low_anfci"


def test_gate_direction_low_on_high_off():
    px, ret = _synth_panel(2400)
    cpi = _synth_cpi(240)
    idx = pd.date_range("2006-01-01", periods=240, freq="MS", tz="UTC")
    anfci = pd.Series(
        np.concatenate([np.full(120, -0.5), np.full(120, 1.5)]),
        index=idx,
        name="ANFCI",
    )
    cfg = AnfciConditionedValueFxConfig(
        signal_lag_months=1,
        weight_lag_days=1,
        z_window=60,
        min_periods=24,
        z_high=1.0,
        lookback=60,
    )
    out = anfci_conditioned_value_factor_returns(px, ret, cpi, us_anfci=anfci, cfg=cfg)
    for key in (
        "value_low_anfci",
        "value_anfci_cool",
        "value_raw",
        "value_high_anfci",
        "us_anfci_haven_usd",
        "value_anfci_stack",
        "value_anfci_ew",
        "value_anfci_regime",
    ):
        assert key in out
    lo = out["value_low_anfci"]
    raw = out["value_raw"]
    both = lo.notna() & raw.notna()
    assert both.sum() > 100
    # Gated magnitude ≤ ungated (gate ∈ {0,1})
    assert (lo.loc[both].abs() <= raw.loc[both].abs() + 1e-12).mean() > 0.95
    hi = out["value_high_anfci"]
    product = lo.loc[both].abs() * hi.loc[both].abs()
    assert (product <= 1e-10).mean() > 0.90


def test_cool_scale_bounds():
    """Cool shrinks exposure; costs on weight paths can push a few day ratios below cool."""
    px, ret = _synth_panel(1800)
    cpi = _synth_cpi(200)
    anfci = _synth_monthly(
        start="2008-01-01", n_months=160, seed=1, mean=0.0, name="ANFCI"
    )
    cfg = AnfciConditionedValueFxConfig(cool=0.35, z_high=1.0)
    out = anfci_conditioned_value_factor_returns(px, ret, cpi, us_anfci=anfci, cfg=cfg)
    raw = out["value_raw"]
    cool = out["value_anfci_cool"]
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
    anfci = _synth_monthly(
        start="2008-01-01", n_months=180, seed=5, mean=0.0, name="ANFCI"
    )
    cfg = AnfciConditionedValueFxConfig(cool=0.35, z_high=1.0)
    out = anfci_conditioned_value_factor_returns(px, ret, cpi, us_anfci=anfci, cfg=cfg)
    raw = out["value_raw"]
    stack = out["value_anfci_stack"]
    lo = out["value_low_anfci"]
    both = raw.notna() & stack.notna() & lo.notna()
    assert both.sum() > 50
    assert (stack.loc[both].abs() <= lo.loc[both].abs() + 1e-9).mean() > 0.95
    assert (stack.loc[both].abs() <= raw.loc[both].abs() + 1e-9).mean() > 0.90


def test_no_lookahead_mutate_future_anfci():
    """Mutating future ANFCI must not change earlier factor returns."""
    px, ret = _synth_panel(2200)
    cpi = _synth_cpi(220)
    anfci = _synth_monthly(
        start="2008-01-01", n_months=200, seed=42, mean=0.0, name="ANFCI"
    )
    cfg = AnfciConditionedValueFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    f1 = anfci_conditioned_value_factor_returns(px, ret, cpi, us_anfci=anfci, cfg=cfg)
    anfci2 = anfci.copy()
    anfci2.loc["2018-01-01":] = anfci2.loc["2018-01-01":] + 0.40
    f2 = anfci_conditioned_value_factor_returns(px, ret, cpi, us_anfci=anfci2, cfg=cfg)
    cut = pd.Timestamp("2016-06-30", tz="UTC")
    a = f1["value_low_anfci"].loc[:cut].dropna()
    b = f2["value_low_anfci"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 100
    assert np.allclose(
        a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12
    )


def test_monthly_z_window_not_daily_252():
    """PIT: z is computed on monthly frequency with z_window months."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    s = pd.Series(np.linspace(0.05, 0.40, 120), index=idx, name="ANFCI")
    z = trailing_z_monthly(s, lookback=60, min_periods=24)
    assert z.iloc[59:].notna().sum() > 40
    assert z.iloc[:23].isna().all()


def test_align_monthly_stress_z_lagged():
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    s = pd.Series(np.linspace(0.05, 0.40, 120), index=idx, name="ANFCI")
    daily_idx = pd.date_range("2015-01-01", periods=500, freq="B", tz="UTC")
    soft_cfg = AnfciConditionedSoftFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    z = align_monthly_stress_z_daily(s, daily_idx, cfg=soft_cfg)
    assert z.index.equals(daily_idx)
    assert z.notna().sum() > 50
    assert pd.isna(z.iloc[0])


def test_monthly_z_window_prior():
    """Config must use monthly z_window=60 (CIP §71 / NFCI §90–§93 / ANFCI §95–§97 mirror, not 252d)."""
    cfg = AnfciConditionedValueFxConfig()
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
    assert cfg.pub_lag_days == 7


def test_without_haven_still_builds_core():
    px, ret = _synth_panel(1800)
    cpi = _synth_cpi(200)
    anfci = _synth_monthly(
        start="2008-01-01", n_months=160, seed=3, mean=0.0, name="ANFCI"
    )
    out = anfci_conditioned_value_factor_returns(
        px, ret, cpi, us_anfci=anfci, include_haven=False
    )
    assert "value_low_anfci" in out
    assert "us_anfci_haven_usd" not in out
    assert "value_anfci_ew" in out  # EW of low + cool still
    assert "value_anfci_stack" in out
    assert "value_anfci_regime" not in out  # needs haven


def test_factor_shapes_match_index():
    px, ret = _synth_panel(1800)
    cpi = _synth_cpi(200)
    anfci = _synth_monthly(
        start="2008-01-01", n_months=160, seed=7, name="ANFCI"
    )
    out = anfci_conditioned_value_factor_returns(
        px, ret, cpi, us_anfci=anfci, cfg=AnfciConditionedValueFxConfig()
    )
    assert out["value_low_anfci"].index.equals(ret.index)
    assert out["us_anfci_haven_usd"].index.equals(ret.index)
    assert out["value_anfci_ew"].index.equals(ret.index)
    assert out["value_anfci_regime"].index.equals(ret.index)


def test_empty_cpi_returns_empty():
    px, ret = _synth_panel(400)
    anfci = _synth_monthly(n_months=80, seed=2, name="ANFCI")
    out = anfci_conditioned_value_factor_returns(
        px, ret, pd.DataFrame(), us_anfci=anfci, cfg=AnfciConditionedValueFxConfig()
    )
    assert out == {}


def test_thin_panel_returns_empty():
    px, ret = _synth_panel(400)
    ret2 = ret.iloc[:, :2]
    px2 = px.iloc[:, :2]
    cpi = _synth_cpi(80)
    anfci = _synth_monthly(n_months=80, seed=2, name="ANFCI")
    out = anfci_conditioned_value_factor_returns(px2, ret2, cpi, us_anfci=anfci)
    assert out == {}


@pytest.mark.skipif(not ANFCI_FRED.exists(), reason="ANFCI FRED CSV missing")
def test_load_real_us_anfci():
    from mt5_swing.strategies.anfci_conditioned_value_fx import load_us_anfci_series

    anfci = load_us_anfci_series(pub_lag_days=7, download=False)
    assert anfci.dropna().shape[0] > 60
    px, ret = _synth_panel(1800)
    cpi = _synth_cpi(200)
    out = anfci_conditioned_value_factor_returns(px, ret, cpi, us_anfci=anfci)
    assert PRIMARY in out
    assert out[PRIMARY].notna().sum() > 20
