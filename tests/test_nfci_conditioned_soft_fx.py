"""Tests for NFCI-conditioned Dahlquist soft-signal EW FX module (§90)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.nfci_conditioned_soft_fx import (
    PRIMARY,
    NfciConditionedSoftFxConfig,
    align_monthly_stress_z_daily,
    trailing_z_monthly,
    weekly_nfci_to_month_end,
    nfci_conditioned_soft_factor_returns,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
NFCI_FRED = MACRO / "fred_NFCI.csv"


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


def _synth_weekly(
    start: str = "2010-01-01",
    n_weeks: int = 780,
    *,
    seed: int = 0,
    mean: float = 0.0,
    name: str = "NFCI",
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
    name: str = "NFCI",
) -> pd.Series:
    idx = pd.date_range(start, periods=n_months, freq="MS", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, 0.4, n_months), index=idx, name=name)


def test_primary_name_constant():
    assert PRIMARY == "soft_low_nfci"


def test_weekly_to_month_end_last_obs():
    idx = pd.date_range("2020-01-03", periods=8, freq="W-FRI", tz="UTC")
    s = pd.Series(np.arange(8, dtype=float), index=idx, name="NFCI")
    m = weekly_nfci_to_month_end(s)
    # Jan 2020 Fridays: 3,10,17,24,31 → last = 4th index value 4
    # Feb: 7,14,21,28 → last = 7
    assert len(m) == 2
    assert float(m.iloc[0]) == 4.0
    assert float(m.iloc[1]) == 7.0


def test_gate_direction_low_on_high_off():
    soft, ret = _synth_soft(2000)
    idx = pd.date_range("2010-01-01", periods=200, freq="MS", tz="UTC")
    # Low NFCI early, then high — so z goes from low to high after warmup
    nfci = pd.Series(
        np.concatenate([np.full(100, -0.5), np.full(100, 1.5)]),
        index=idx,
        name="NFCI",
    )
    cfg = NfciConditionedSoftFxConfig(
        signal_lag_months=1,
        weight_lag_days=1,
        z_window=60,
        min_periods=24,
        z_high=1.0,
    )
    out = nfci_conditioned_soft_factor_returns(
        soft, us_nfci=nfci, pair_ret=ret, cfg=cfg
    )
    for key in (
        "soft_low_nfci",
        "soft_nfci_cool",
        "soft_raw",
        "soft_high_nfci",
        "us_nfci_haven_usd",
        "soft_nfci_stack",
        "soft_nfci_ew",
        "soft_nfci_regime",
    ):
        assert key in out
    common = soft.dropna().index.intersection(out["soft_raw"].dropna().index)
    assert np.allclose(soft.loc[common].values, out["soft_raw"].loc[common].values)
    lo = out["soft_low_nfci"]
    raw = out["soft_raw"]
    both = lo.notna() & raw.notna()
    assert both.sum() > 100
    assert (lo.loc[both].abs() <= raw.loc[both].abs() + 1e-15).all()
    hi = out["soft_high_nfci"]
    product = lo.loc[both].abs() * hi.loc[both].abs()
    assert (product <= 1e-18).mean() > 0.99


def test_cool_scale_bounds():
    soft, ret = _synth_soft(600)
    nfci = _synth_weekly(n_weeks=520, seed=1, mean=0.0, name="NFCI")
    cfg = NfciConditionedSoftFxConfig(cool=0.35, z_high=1.0)
    out = nfci_conditioned_soft_factor_returns(
        soft, us_nfci=nfci, pair_ret=ret, cfg=cfg
    )
    raw = out["soft_raw"]
    cool = out["soft_nfci_cool"]
    common = raw.dropna().index.intersection(cool.dropna().index)
    ratio = (cool.loc[common] / raw.loc[common]).replace([np.inf, -np.inf], np.nan)
    ratio = ratio.dropna()
    ratio = ratio[raw.loc[ratio.index].abs() > 1e-12]
    if len(ratio):
        assert float(ratio.min()) >= 0.35 - 1e-9
        assert float(ratio.max()) <= 1.0 + 1e-9


def test_stack_is_gate_then_cool():
    soft, ret = _synth_soft(800)
    nfci = _synth_weekly(n_weeks=600, seed=5, mean=0.0, name="NFCI")
    cfg = NfciConditionedSoftFxConfig(cool=0.35, z_high=1.0)
    out = nfci_conditioned_soft_factor_returns(
        soft, us_nfci=nfci, pair_ret=ret, cfg=cfg
    )
    raw = out["soft_raw"]
    stack = out["soft_nfci_stack"]
    lo = out["soft_low_nfci"]
    both = raw.notna() & stack.notna() & lo.notna()
    assert both.sum() > 50
    # Stack magnitude ≤ |low| (cool ≤ 1) and ≤ |raw|
    assert (stack.loc[both].abs() <= lo.loc[both].abs() + 1e-12).all()
    assert (stack.loc[both].abs() <= raw.loc[both].abs() + 1e-12).all()


def test_no_lookahead_mutate_future_nfci():
    soft, ret = _synth_soft(1800)
    nfci = _synth_weekly(
        start="2012-01-06", n_weeks=780, seed=42, mean=0.0, name="NFCI"
    )
    cfg = NfciConditionedSoftFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    f1 = nfci_conditioned_soft_factor_returns(
        soft, us_nfci=nfci, pair_ret=ret, cfg=cfg
    )
    nfci2 = nfci.copy()
    nfci2.loc["2021-01-01":] = nfci2.loc["2021-01-01":] + 2.0
    f2 = nfci_conditioned_soft_factor_returns(
        soft, us_nfci=nfci2, pair_ret=ret, cfg=cfg
    )
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["soft_low_nfci"].loc[:cut].dropna()
    b = f2["soft_low_nfci"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 100
    assert np.allclose(
        a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12
    )


def test_monthly_z_window_not_daily_252():
    """PIT: z is computed on monthly frequency with z_window months (CIP §71 mirror)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    s = pd.Series(np.linspace(-1.0, 1.5, 120), index=idx, name="NFCI")
    z = trailing_z_monthly(s, lookback=60, min_periods=24)
    assert z.iloc[59:].notna().sum() > 40
    assert z.iloc[:23].isna().all()


def test_align_monthly_stress_z_lagged():
    # Weekly NFCI → month-end → monthly z
    nfci = _synth_weekly(start="2010-01-01", n_weeks=520, seed=9, mean=0.0)
    soft_idx = pd.date_range("2015-01-01", periods=500, freq="B", tz="UTC")
    cfg = NfciConditionedSoftFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    z = align_monthly_stress_z_daily(nfci, soft_idx, cfg=cfg)
    assert z.index.equals(soft_idx)
    assert z.notna().sum() > 50
    cfg0 = NfciConditionedSoftFxConfig(
        signal_lag_months=1, weight_lag_days=0, z_window=60, min_periods=24
    )
    z0 = align_monthly_stress_z_daily(nfci, soft_idx, cfg=cfg0)
    both = z.notna() & z0.notna()
    assert both.sum() > 50
    assert z.loc[both].isna().sum() == 0


def test_without_haven_still_builds_core():
    soft, _ret = _synth_soft(400)
    nfci = _synth_monthly(n_months=100, seed=3, mean=0.0, name="NFCI")
    out = nfci_conditioned_soft_factor_returns(
        soft, us_nfci=nfci, pair_ret=None, include_haven=False
    )
    assert "soft_low_nfci" in out
    assert "us_nfci_haven_usd" not in out
    assert "soft_nfci_ew" in out  # EW of low + cool still
    assert "soft_nfci_stack" in out
    assert "soft_nfci_regime" not in out  # needs haven


def test_factor_shapes_match_soft_index():
    soft, ret = _synth_soft(800)
    nfci = _synth_weekly(n_weeks=600, seed=7, name="NFCI")
    out = nfci_conditioned_soft_factor_returns(
        soft, us_nfci=nfci, pair_ret=ret, cfg=NfciConditionedSoftFxConfig()
    )
    assert out["soft_low_nfci"].index.equals(soft.index)
    assert out["us_nfci_haven_usd"].index.equals(soft.index)
    assert out["soft_nfci_ew"].index.equals(soft.index)
    assert out["soft_nfci_regime"].index.equals(soft.index)


@pytest.mark.skipif(not NFCI_FRED.exists(), reason="NFCI FRED CSV missing")
def test_load_real_us_nfci():
    from mt5_swing.strategies.nfci_conditioned_soft_fx import load_us_nfci_series

    nfci = load_us_nfci_series(pub_lag_days=7, download=False)
    assert nfci.dropna().shape[0] > 60
    soft, ret = _synth_soft(500)
    out = nfci_conditioned_soft_factor_returns(soft, us_nfci=nfci, pair_ret=ret)
    assert PRIMARY in out
    assert out[PRIMARY].notna().sum() > 50
