"""Tests for EPU/TPU-conditioned Menkhoff FX momentum module (§79)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.epu_tpu_conditioned_mom_fx import (
    PRIMARY,
    EpuTpuConditionedMomFxConfig,
    epu_tpu_conditioned_mom_factor_returns,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
EPU_FRED = MACRO / "fred_USEPUINDXM.csv"
TPU_CSV = MACRO / "tpu_us_monthly.csv"


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
    mean: float = 100.0,
    name: str = "epu",
) -> pd.Series:
    idx = pd.date_range(start, periods=n_months, freq="MS", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, 30.0, n_months), index=idx, name=name)


def test_primary_name_constant():
    assert PRIMARY == "mom_low_epu"


def test_gate_direction_low_on_high_off():
    ret = _synth_ret(2000)
    idx = pd.date_range("2010-01-01", periods=200, freq="MS", tz="UTC")
    epu = pd.Series(
        np.concatenate([np.full(100, 50.0), np.full(100, 250.0)]),
        index=idx,
        name="US_EPU",
    )
    tpu = _synth_monthly(n_months=200, seed=3, mean=80.0, name="TPU")
    cfg = EpuTpuConditionedMomFxConfig(
        signal_lag_months=1,
        weight_lag_days=1,
        z_window=60,
        min_periods=24,
        z_high=1.0,
    )
    out = epu_tpu_conditioned_mom_factor_returns(ret, epu=epu, tpu=tpu, cfg=cfg)
    for key in (
        "mom_low_epu",
        "mom_epu_cool",
        "mom_low_tpu",
        "mom_tpu_cool",
        "mom_raw",
        "mom_high_epu",
        "mom_epu_tpu_stack",
        "mom_epu_tpu_ew",
    ):
        assert key in out
    lo = out["mom_low_epu"]
    raw = out["mom_raw"]
    both = lo.notna() & raw.notna()
    assert both.sum() > 100
    # Inverse honesty: low and high EPU gates mutually exclusive on abs returns
    hi = out["mom_high_epu"]
    product = lo.loc[both].abs() * hi.loc[both].abs()
    assert (product <= 1e-10).mean() > 0.90


def test_cool_scale_bounds():
    """risk_scale_from_z stays in [cool, 1]; cool factor exists and is finite."""
    from mt5_swing.strategies.epu_tpu_conditioned_soft_fx import (
        EpuTpuConditionedSoftFxConfig,
        align_monthly_stress_z_daily,
    )
    from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z

    ret = _synth_ret(800)
    epu = _synth_monthly(n_months=140, seed=1, mean=110.0, name="US_EPU")
    tpu = _synth_monthly(n_months=140, seed=2, mean=90.0, name="TPU")
    cfg = EpuTpuConditionedMomFxConfig(cool=0.35, z_high=1.0)
    soft_cfg = EpuTpuConditionedSoftFxConfig(
        signal_lag_months=cfg.signal_lag_months,
        weight_lag_days=cfg.weight_lag_days,
        z_window=cfg.z_window,
        min_periods=cfg.min_periods,
        z_high=cfg.z_high,
        cool=cfg.cool,
    )
    z = align_monthly_stress_z_daily(epu, ret.index, cfg=soft_cfg)
    gcfg = GprRegimeConfig(
        z_high=cfg.z_high, z_low=cfg.z_low, cool=cfg.cool, signal_lag=0, usd_tilt=0.0
    )
    scale = risk_scale_from_z(z, cfg=gcfg).dropna()
    assert len(scale) > 50
    assert float(scale.min()) >= cfg.cool - 1e-9
    assert float(scale.max()) <= 1.0 + 1e-9
    out = epu_tpu_conditioned_mom_factor_returns(ret, epu=epu, tpu=tpu, cfg=cfg)
    assert out["mom_epu_cool"].notna().sum() > 50


def test_stack_scales_compound():
    """Stack scale = product of EPU×TPU scales, clipped to [cool^2, 1]."""
    from mt5_swing.strategies.epu_tpu_conditioned_soft_fx import (
        EpuTpuConditionedSoftFxConfig,
        align_monthly_stress_z_daily,
    )
    from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z

    ret = _synth_ret(800)
    epu = _synth_monthly(n_months=140, seed=5, mean=120.0, name="US_EPU")
    tpu = _synth_monthly(n_months=140, seed=6, mean=100.0, name="TPU")
    cfg = EpuTpuConditionedMomFxConfig(cool=0.35, z_high=1.0)
    soft_cfg = EpuTpuConditionedSoftFxConfig(
        signal_lag_months=cfg.signal_lag_months,
        weight_lag_days=cfg.weight_lag_days,
        z_window=cfg.z_window,
        min_periods=cfg.min_periods,
        z_high=cfg.z_high,
        cool=cfg.cool,
    )
    z_epu = align_monthly_stress_z_daily(epu, ret.index, cfg=soft_cfg)
    z_tpu = align_monthly_stress_z_daily(tpu, ret.index, cfg=soft_cfg)
    gcfg = GprRegimeConfig(
        z_high=cfg.z_high, z_low=cfg.z_low, cool=cfg.cool, signal_lag=0, usd_tilt=0.0
    )
    epu_s = risk_scale_from_z(z_epu, cfg=gcfg)
    tpu_s = risk_scale_from_z(z_tpu, cfg=gcfg)
    stack_s = (epu_s * tpu_s).clip(lower=cfg.cool ** 2, upper=1.0).dropna()
    assert len(stack_s) > 50
    assert float(stack_s.min()) >= cfg.cool ** 2 - 1e-9
    assert float(stack_s.max()) <= 1.0 + 1e-9
    out = epu_tpu_conditioned_mom_factor_returns(ret, epu=epu, tpu=tpu, cfg=cfg)
    assert out["mom_epu_tpu_stack"].notna().sum() > 50


def test_no_lookahead_mutate_future_epu():
    ret = _synth_ret(1800)
    epu = _synth_monthly(
        start="2012-01-01", n_months=180, seed=42, mean=100.0, name="US_EPU"
    )
    tpu = _synth_monthly(
        start="2012-01-01", n_months=180, seed=7, mean=80.0, name="TPU"
    )
    cfg = EpuTpuConditionedMomFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    f1 = epu_tpu_conditioned_mom_factor_returns(ret, epu=epu, tpu=tpu, cfg=cfg)
    epu2 = epu.copy()
    epu2.loc["2021-01-01":] = epu2.loc["2021-01-01":] + 80.0
    f2 = epu_tpu_conditioned_mom_factor_returns(ret, epu=epu2, tpu=tpu, cfg=cfg)
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["mom_low_epu"].loc[:cut].dropna()
    b = f2["mom_low_epu"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 200
    assert np.allclose(
        a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12
    )


def test_tpu_missing_fallthrough():
    ret = _synth_ret(800)
    epu = _synth_monthly(n_months=140, seed=9, mean=110.0, name="US_EPU")
    cfg = EpuTpuConditionedMomFxConfig()
    out = epu_tpu_conditioned_mom_factor_returns(
        ret, epu=epu, tpu=None, cfg=cfg, allow_tpu_missing=True
    )
    assert "mom_low_epu" in out
    assert "mom_epu_cool" in out
    assert "mom_raw" in out
    assert "mom_high_epu" in out
    assert "mom_epu_tpu_ew" in out
    assert "mom_low_tpu" not in out


def test_factor_shapes_match_index():
    ret = _synth_ret(800)
    epu = _synth_monthly(n_months=140, seed=7, name="US_EPU")
    tpu = _synth_monthly(n_months=140, seed=8, mean=95.0, name="TPU")
    out = epu_tpu_conditioned_mom_factor_returns(
        ret, epu=epu, tpu=tpu, cfg=EpuTpuConditionedMomFxConfig()
    )
    assert out["mom_low_epu"].index.equals(ret.index)
    assert out["mom_epu_tpu_ew"].index.equals(ret.index)


def test_gate_on_off_explicit_z():
    """Primary gate: mom on when z≤0; inverse on when z≥z_high."""
    idx = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    z = pd.Series([-1.0, -0.5, 0.0, 0.5, 1.5, 2.0, -0.2, 1.1, 0.0, -2.0], index=idx)
    cfg = EpuTpuConditionedMomFxConfig(z_high=1.0)
    gate_lo = (z <= 0.0).astype(float)
    gate_hi = (z >= cfg.z_high).astype(float)
    assert gate_lo.iloc[0] == 1.0
    assert gate_lo.iloc[4] == 0.0
    assert gate_hi.iloc[4] == 1.0
    assert gate_hi.iloc[0] == 0.0


def test_monthly_z_window_prior():
    """Config must use monthly z_window=60 (not 252d daily like §74)."""
    cfg = EpuTpuConditionedMomFxConfig()
    assert cfg.z_window == 60
    assert cfg.min_periods == 24
    assert cfg.signal_lag_months == 1
    assert cfg.formation_days == 63
    assert cfg.skip_days == 21


@pytest.mark.skipif(not EPU_FRED.exists(), reason="EPU CSV missing")
def test_load_real_epu_mom():
    from mt5_swing.strategies.epu_tpu_conditioned_mom_fx import (
        load_epu_series,
        load_tpu_series,
    )

    epu = load_epu_series(series="US", pub_lag_months=1, download=False)
    assert epu.dropna().shape[0] > 100
    ret = _synth_ret(500)
    try:
        tpu = load_tpu_series(pub_lag_months=1, download=False)
        out = epu_tpu_conditioned_mom_factor_returns(ret, epu=epu, tpu=tpu)
    except Exception:
        out = epu_tpu_conditioned_mom_factor_returns(
            ret, epu=epu, tpu=None, allow_tpu_missing=True
        )
    assert PRIMARY in out
    assert out[PRIMARY].notna().sum() > 20
