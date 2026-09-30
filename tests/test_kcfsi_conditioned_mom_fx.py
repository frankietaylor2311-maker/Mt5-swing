"""Tests for KCFSI-conditioned Menkhoff FX momentum module (§107)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.kcfsi_conditioned_mom_fx import (
    PRIMARY,
    KcfsiConditionedMomFxConfig,
    align_monthly_stress_z_daily,
    kcfsi_conditioned_mom_factor_returns,
)
from mt5_swing.strategies.kcfsi_conditioned_soft_fx import (
    KcfsiConditionedSoftFxConfig,
    trailing_z_monthly,
    monthly_kcfsi_to_month_end,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
KCFSI_FRED = MACRO / "fred_KCFSI.csv"


def _synth_ret(n_days: int = 1800) -> pd.DataFrame:
    idx = pd.date_range("2016-01-01", periods=n_days, freq="B", tz="UTC")
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            s: rng.normal(0, 0.005, n_days)
            for s in ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF")
        },
        index=idx,
    )


def _synth_monthly(
    start: str = "2010-01-01",
    n_months: int = 180,
    *,
    seed: int = 0,
    mean: float = 0.0,
    name: str = "KCFSI",
) -> pd.Series:
    idx = pd.date_range(start, periods=n_months, freq="MS", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, 0.4, n_months), index=idx, name=name)


def test_primary_name_constant():
    assert PRIMARY == "mom_low_kcfsi"


def test_monthly_to_month_end_last_obs():
    idx = pd.date_range("2020-01-01", periods=3, freq="MS", tz="UTC")
    s = pd.Series([1.0, 2.0, 3.0], index=idx, name="KCFSI")
    m = monthly_kcfsi_to_month_end(s)
    assert len(m) == 3
    assert float(m.iloc[0]) == 1.0
    assert float(m.iloc[1]) == 2.0
    assert float(m.iloc[2]) == 3.0


def test_gate_direction_low_on_high_off():
    ret = _synth_ret(2000)
    idx = pd.date_range("2010-01-01", periods=200, freq="MS", tz="UTC")
    kcfsi = pd.Series(
        np.concatenate([np.full(100, -0.5), np.full(100, 1.5)]),
        index=idx,
        name="KCFSI",
    )
    cfg = KcfsiConditionedMomFxConfig(
        signal_lag_months=1,
        weight_lag_days=1,
        z_window=60,
        min_periods=24,
        z_high=1.0,
    )
    out = kcfsi_conditioned_mom_factor_returns(ret, us_kcfsi=kcfsi, cfg=cfg)
    for key in (
        "mom_low_kcfsi",
        "mom_kcfsi_cool",
        "mom_raw",
        "mom_high_kcfsi",
        "us_kcfsi_haven_usd",
        "mom_kcfsi_stack",
        "mom_kcfsi_ew",
        "mom_kcfsi_regime",
    ):
        assert key in out
    lo = out["mom_low_kcfsi"]
    raw = out["mom_raw"]
    both = lo.notna() & raw.notna()
    assert both.sum() > 100
    # Gated magnitude ≤ ungated (gate ∈ {0,1})
    assert (lo.loc[both].abs() <= raw.loc[both].abs() + 1e-12).all()
    hi = out["mom_high_kcfsi"]
    product = lo.loc[both].abs() * hi.loc[both].abs()
    assert (product <= 1e-18).mean() > 0.99


def test_cool_scale_bounds():
    """Cool shrinks exposure; costs on weight paths can push a few day ratios below cool."""
    ret = _synth_ret(800)
    kcfsi = _synth_monthly(n_months=140, seed=1, mean=0.0, name="KCFSI")
    cfg = KcfsiConditionedMomFxConfig(cool=0.35, z_high=1.0)
    out = kcfsi_conditioned_mom_factor_returns(ret, us_kcfsi=kcfsi, cfg=cfg)
    raw = out["mom_raw"]
    cool = out["mom_kcfsi_cool"]
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
    ret = _synth_ret(1000)
    kcfsi = _synth_monthly(n_months=160, seed=5, mean=0.0, name="KCFSI")
    cfg = KcfsiConditionedMomFxConfig(cool=0.35, z_high=1.0)
    out = kcfsi_conditioned_mom_factor_returns(ret, us_kcfsi=kcfsi, cfg=cfg)
    raw = out["mom_raw"]
    stack = out["mom_kcfsi_stack"]
    lo = out["mom_low_kcfsi"]
    both = raw.notna() & stack.notna() & lo.notna()
    assert both.sum() > 50
    assert (stack.loc[both].abs() <= lo.loc[both].abs() + 1e-9).mean() > 0.95
    assert (stack.loc[both].abs() <= raw.loc[both].abs() + 1e-9).mean() > 0.90


def test_no_lookahead_mutate_future_kcfsi():
    """Mutating future KCFSI must not change earlier factor returns."""
    ret = _synth_ret(1800)
    kcfsi = _synth_monthly(
        start="2012-01-01", n_months=180, seed=42, mean=0.0, name="KCFSI"
    )
    cfg = KcfsiConditionedMomFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    f1 = kcfsi_conditioned_mom_factor_returns(ret, us_kcfsi=kcfsi, cfg=cfg)
    kcfsi2 = kcfsi.copy()
    kcfsi2.loc["2021-01-01":] = kcfsi2.loc["2021-01-01":] + 2.0
    f2 = kcfsi_conditioned_mom_factor_returns(ret, us_kcfsi=kcfsi2, cfg=cfg)
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["mom_low_kcfsi"].loc[:cut].dropna()
    b = f2["mom_low_kcfsi"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 100
    assert np.allclose(
        a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12
    )


def test_monthly_z_window_not_daily_252():
    """PIT: z is computed on monthly frequency with z_window months."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    s = pd.Series(np.linspace(-1.0, 1.5, 120), index=idx, name="KCFSI")
    z = trailing_z_monthly(s, lookback=60, min_periods=24)
    assert z.iloc[59:].notna().sum() > 40
    assert z.iloc[:23].isna().all()


def test_align_monthly_stress_z_lagged():
    kcfsi = _synth_monthly(start="2010-01-01", n_months=120, seed=9, mean=0.0)
    daily_idx = pd.date_range("2015-01-01", periods=500, freq="B", tz="UTC")
    soft_cfg = KcfsiConditionedSoftFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    z = align_monthly_stress_z_daily(kcfsi, daily_idx, cfg=soft_cfg)
    assert z.index.equals(daily_idx)
    assert z.notna().sum() > 50
    soft_cfg0 = KcfsiConditionedSoftFxConfig(
        signal_lag_months=1, weight_lag_days=0, z_window=60, min_periods=24
    )
    z0 = align_monthly_stress_z_daily(kcfsi, daily_idx, cfg=soft_cfg0)
    both = z.notna() & z0.notna()
    assert both.sum() > 50


def test_monthly_z_window_prior():
    """Config must use monthly z_window=60 (CIP §71 / NFCI / ANFCI / STLFSI / KCFSI mirror, not 252d)."""
    cfg = KcfsiConditionedMomFxConfig()
    assert cfg.z_window == 60
    assert cfg.min_periods == 24
    assert cfg.signal_lag_months == 1
    assert cfg.weight_lag_days == 1
    assert cfg.formation_days == 63
    assert cfg.skip_days == 21
    assert cfg.cool == 0.35
    assert cfg.z_high == 1.0
    assert cfg.pub_lag_months == 1


def test_without_haven_still_builds_core():
    ret = _synth_ret(600)
    kcfsi = _synth_monthly(n_months=120, seed=3, mean=0.0, name="KCFSI")
    out = kcfsi_conditioned_mom_factor_returns(
        ret, us_kcfsi=kcfsi, include_haven=False
    )
    assert "mom_low_kcfsi" in out
    assert "us_kcfsi_haven_usd" not in out
    assert "mom_kcfsi_ew" in out  # EW of low + cool still
    assert "mom_kcfsi_stack" in out
    assert "mom_kcfsi_regime" not in out  # needs haven


def test_factor_shapes_match_index():
    ret = _synth_ret(800)
    kcfsi = _synth_monthly(n_months=140, seed=7, name="KCFSI")
    out = kcfsi_conditioned_mom_factor_returns(
        ret, us_kcfsi=kcfsi, cfg=KcfsiConditionedMomFxConfig()
    )
    assert out["mom_low_kcfsi"].index.equals(ret.index)
    assert out["us_kcfsi_haven_usd"].index.equals(ret.index)
    assert out["mom_kcfsi_ew"].index.equals(ret.index)
    assert out["mom_kcfsi_regime"].index.equals(ret.index)


def test_thin_panel_returns_empty():
    ret = _synth_ret(100).iloc[:, :2]
    kcfsi = _synth_monthly(n_months=80, seed=2, name="KCFSI")
    out = kcfsi_conditioned_mom_factor_returns(ret, us_kcfsi=kcfsi)
    assert out == {}


@pytest.mark.skipif(not KCFSI_FRED.exists(), reason="KCFSI FRED CSV missing")
def test_load_real_us_kcfsi():
    from mt5_swing.strategies.kcfsi_conditioned_mom_fx import load_us_kcfsi_series

    kcfsi = load_us_kcfsi_series(pub_lag_months=1, download=False)
    assert kcfsi.dropna().shape[0] > 60
    ret = _synth_ret(600)
    out = kcfsi_conditioned_mom_factor_returns(ret, us_kcfsi=kcfsi)
    assert PRIMARY in out
    assert out[PRIMARY].notna().sum() > 50
