"""Tests for ANFCI-conditioned Lustig–Verdelhan carry FX module (§96)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.anfci_conditioned_carry_fx import (
    PRIMARY,
    AnfciConditionedCarryFxConfig,
    align_monthly_stress_z_daily,
    anfci_conditioned_carry_factor_returns,
)
from mt5_swing.strategies.anfci_conditioned_soft_fx import (
    AnfciConditionedSoftFxConfig,
    trailing_z_monthly,
    weekly_anfci_to_month_end,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
ANFCI_FRED = MACRO / "fred_ANFCI.csv"


def _synth_ret_rates(
    n_days: int = 1800,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2016-01-01", periods=n_days, freq="B", tz="UTC")
    rng = np.random.default_rng(42)
    ret = pd.DataFrame(
        {
            "EURUSD": rng.normal(0, 0.005, n_days),
            "GBPUSD": rng.normal(0, 0.005, n_days),
            "AUDUSD": rng.normal(0, 0.006, n_days),
            "NZDUSD": rng.normal(0, 0.006, n_days),
            "USDJPY": rng.normal(0, 0.005, n_days),
            "USDCAD": rng.normal(0, 0.004, n_days),
            "USDCHF": rng.normal(0, 0.004, n_days),
        },
        index=idx,
    )
    n_m = 120
    rates = pd.DataFrame(
        {
            c: 1.0 + 0.01 * rng.normal(0, 1, n_m).cumsum()
            for c in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")
        },
        index=pd.date_range("2014-01-01", periods=n_m, freq="MS", tz="UTC"),
    )
    return ret, rates


def _synth_weekly(
    start: str = "2010-01-01",
    n_weeks: int = 780,
    *,
    seed: int = 0,
    mean: float = 0.0,
    name: str = "ANFCI",
) -> pd.Series:
    idx = pd.date_range(start, periods=n_weeks, freq="W-FRI", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, 0.4, n_weeks), index=idx, name=name)


def _synth_monthly(
    start: str = "2010-01-01",
    n_months: int = 180,
    *,
    seed: int = 0,
    mean: float = 0.0,
    name: str = "ANFCI",
) -> pd.Series:
    idx = pd.date_range(start, periods=n_months, freq="MS", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, 0.4, n_months), index=idx, name=name)


def test_primary_name_constant():
    assert PRIMARY == "carry_low_anfci"


def test_weekly_to_month_end_last_obs():
    idx = pd.date_range("2020-01-03", periods=8, freq="W-FRI", tz="UTC")
    s = pd.Series(np.arange(8, dtype=float), index=idx, name="ANFCI")
    m = weekly_anfci_to_month_end(s)
    assert len(m) == 2
    assert float(m.iloc[0]) == 4.0
    assert float(m.iloc[1]) == 7.0


def test_gate_direction_low_on_high_off():
    ret, rates = _synth_ret_rates(2000)
    idx = pd.date_range("2010-01-01", periods=200, freq="MS", tz="UTC")
    anfci = pd.Series(
        np.concatenate([np.full(100, -0.5), np.full(100, 1.5)]),
        index=idx,
        name="ANFCI",
    )
    cfg = AnfciConditionedCarryFxConfig(
        signal_lag_months=1,
        weight_lag_days=1,
        z_window=60,
        min_periods=24,
        z_high=1.0,
    )
    out = anfci_conditioned_carry_factor_returns(
        ret, rates=rates, us_anfci=anfci, cfg=cfg
    )
    for key in (
        "carry_low_anfci",
        "carry_anfci_cool",
        "carry_raw",
        "carry_high_anfci",
        "us_anfci_haven_usd",
        "carry_anfci_stack",
        "carry_anfci_ew",
        "carry_anfci_regime",
    ):
        assert key in out
    lo = out["carry_low_anfci"]
    raw = out["carry_raw"]
    both = lo.notna() & raw.notna()
    assert both.sum() > 100
    # Gated magnitude ≤ ungated (gate ∈ {0,1})
    assert (lo.loc[both].abs() <= raw.loc[both].abs() + 1e-12).all()
    hi = out["carry_high_anfci"]
    product = lo.loc[both].abs() * hi.loc[both].abs()
    assert (product <= 1e-18).mean() > 0.99


def test_cool_scale_bounds():
    """Cool shrinks exposure; costs on weight paths can push a few day ratios below cool."""
    ret, rates = _synth_ret_rates(800)
    anfci = _synth_weekly(n_weeks=520, seed=1, mean=0.0, name="ANFCI")
    cfg = AnfciConditionedCarryFxConfig(cool=0.35, z_high=1.0)
    out = anfci_conditioned_carry_factor_returns(
        ret, rates=rates, us_anfci=anfci, cfg=cfg
    )
    raw = out["carry_raw"]
    cool = out["carry_anfci_cool"]
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
    ret, rates = _synth_ret_rates(1000)
    anfci = _synth_weekly(n_weeks=600, seed=5, mean=0.0, name="ANFCI")
    cfg = AnfciConditionedCarryFxConfig(cool=0.35, z_high=1.0)
    out = anfci_conditioned_carry_factor_returns(
        ret, rates=rates, us_anfci=anfci, cfg=cfg
    )
    raw = out["carry_raw"]
    stack = out["carry_anfci_stack"]
    lo = out["carry_low_anfci"]
    both = raw.notna() & stack.notna() & lo.notna()
    assert both.sum() > 50
    assert (stack.loc[both].abs() <= lo.loc[both].abs() + 1e-9).mean() > 0.95
    assert (stack.loc[both].abs() <= raw.loc[both].abs() + 1e-9).mean() > 0.90


def test_no_lookahead_mutate_future_anfci():
    """Mutating future ANFCI must not change earlier factor returns."""
    ret, rates = _synth_ret_rates(1800)
    anfci = _synth_weekly(
        start="2012-01-06", n_weeks=780, seed=42, mean=0.0, name="ANFCI"
    )
    cfg = AnfciConditionedCarryFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    f1 = anfci_conditioned_carry_factor_returns(
        ret, rates=rates, us_anfci=anfci, cfg=cfg
    )
    anfci2 = anfci.copy()
    anfci2.loc["2021-01-01":] = anfci2.loc["2021-01-01":] + 2.0
    f2 = anfci_conditioned_carry_factor_returns(
        ret, rates=rates, us_anfci=anfci2, cfg=cfg
    )
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["carry_low_anfci"].loc[:cut].dropna()
    b = f2["carry_low_anfci"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 100
    assert np.allclose(
        a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12
    )


def test_monthly_z_window_not_daily_252():
    """PIT: z is computed on monthly frequency with z_window months."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    s = pd.Series(np.linspace(-1.0, 1.5, 120), index=idx, name="ANFCI")
    z = trailing_z_monthly(s, lookback=60, min_periods=24)
    assert z.iloc[59:].notna().sum() > 40
    assert z.iloc[:23].isna().all()


def test_align_monthly_stress_z_lagged():
    anfci = _synth_weekly(start="2010-01-01", n_weeks=520, seed=9, mean=0.0)
    daily_idx = pd.date_range("2015-01-01", periods=500, freq="B", tz="UTC")
    soft_cfg = AnfciConditionedSoftFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    z = align_monthly_stress_z_daily(anfci, daily_idx, cfg=soft_cfg)
    assert z.index.equals(daily_idx)
    assert z.notna().sum() > 50
    soft_cfg0 = AnfciConditionedSoftFxConfig(
        signal_lag_months=1, weight_lag_days=0, z_window=60, min_periods=24
    )
    z0 = align_monthly_stress_z_daily(anfci, daily_idx, cfg=soft_cfg0)
    both = z.notna() & z0.notna()
    assert both.sum() > 50


def test_without_haven_still_builds_core():
    ret, rates = _synth_ret_rates(600)
    anfci = _synth_monthly(n_months=120, seed=3, mean=0.0, name="ANFCI")
    out = anfci_conditioned_carry_factor_returns(
        ret, rates=rates, us_anfci=anfci, include_haven=False
    )
    assert "carry_low_anfci" in out
    assert "us_anfci_haven_usd" not in out
    assert "carry_anfci_ew" in out  # EW of low + cool still
    assert "carry_anfci_stack" in out
    assert "carry_anfci_regime" not in out  # needs haven


def test_factor_shapes_match_index():
    ret, rates = _synth_ret_rates(800)
    anfci = _synth_weekly(n_weeks=600, seed=7, name="ANFCI")
    out = anfci_conditioned_carry_factor_returns(
        ret, rates=rates, us_anfci=anfci, cfg=AnfciConditionedCarryFxConfig()
    )
    assert out["carry_low_anfci"].index.equals(ret.index)
    assert out["us_anfci_haven_usd"].index.equals(ret.index)
    assert out["carry_anfci_ew"].index.equals(ret.index)
    assert out["carry_anfci_regime"].index.equals(ret.index)


def test_empty_rates_returns_empty():
    ret, _rates = _synth_ret_rates(100)
    anfci = _synth_monthly(n_months=80, seed=2, name="ANFCI")
    out = anfci_conditioned_carry_factor_returns(
        ret, rates=pd.DataFrame(), us_anfci=anfci
    )
    assert out == {}


@pytest.mark.skipif(not ANFCI_FRED.exists(), reason="ANFCI FRED CSV missing")
def test_load_real_us_anfci():
    from mt5_swing.strategies.anfci_conditioned_carry_fx import load_us_anfci_series

    anfci = load_us_anfci_series(pub_lag_days=7, download=False)
    assert anfci.dropna().shape[0] > 60
    ret, rates = _synth_ret_rates(600)
    out = anfci_conditioned_carry_factor_returns(
        ret, rates=rates, us_anfci=anfci
    )
    assert PRIMARY in out
    assert out[PRIMARY].notna().sum() > 50
