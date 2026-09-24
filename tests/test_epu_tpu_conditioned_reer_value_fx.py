"""Tests for EPU/TPU-conditioned BIS REER HML-FX value module (§77)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.epu_tpu_conditioned_reer_value_fx import (
    PRIMARY,
    EpuTpuConditionedReerValueFxConfig,
    epu_tpu_conditioned_reer_value_factor_returns,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
EPU_FRED = MACRO / "fred_USEPUINDXM.csv"
TPU_CSV = MACRO / "tpu_us_monthly.csv"
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
    mean: float = 100.0,
    name: str = "epu",
) -> pd.Series:
    idx = pd.date_range(start, periods=n_months, freq="MS", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, 30.0, n_months), index=idx, name=name)


def test_primary_name_constant():
    assert PRIMARY == "reer_low_epu"


def test_gate_direction_low_on_high_off():
    pret = _synth_pairs(2000)
    reer = _synth_reer(n_months=200, seed=1)
    idx = pd.date_range("2010-01-01", periods=200, freq="MS", tz="UTC")
    epu = pd.Series(
        np.concatenate([np.full(100, 50.0), np.full(100, 250.0)]),
        index=idx,
        name="US_EPU",
    )
    tpu = _synth_monthly(n_months=200, seed=3, mean=80.0, name="TPU")
    cfg = EpuTpuConditionedReerValueFxConfig(
        signal_lag_months=1,
        weight_lag_days=1,
        z_window=60,
        min_periods=24,
        z_high=1.0,
    )
    out = epu_tpu_conditioned_reer_value_factor_returns(
        pret, reer, epu=epu, tpu=tpu, cfg=cfg
    )
    for key in (
        "reer_low_epu",
        "reer_epu_cool",
        "reer_low_tpu",
        "reer_tpu_cool",
        "reer_raw",
        "reer_high_epu",
        "reer_epu_tpu_stack",
        "reer_epu_tpu_ew",
    ):
        assert key in out
    lo = out["reer_low_epu"]
    raw = out["reer_raw"]
    both = lo.notna() & raw.notna()
    assert both.sum() > 100
    assert (lo.loc[both].abs() <= raw.loc[both].abs() + 1e-12).mean() > 0.95
    hi = out["reer_high_epu"]
    product = lo.loc[both].abs() * hi.loc[both].abs()
    assert (product <= 1e-12).mean() > 0.95


def test_cool_scale_bounds():
    """Scale bounds on risk_scale_from_z (return ratios deviate when costs reapplied)."""
    from mt5_swing.strategies.epu_tpu_conditioned_soft_fx import (
        EpuTpuConditionedSoftFxConfig,
        align_monthly_stress_z_daily,
    )
    from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z

    pret = _synth_pairs(800)
    reer = _synth_reer(n_months=140, seed=2)
    epu = _synth_monthly(n_months=140, seed=1, mean=110.0, name="US_EPU")
    tpu = _synth_monthly(n_months=140, seed=2, mean=90.0, name="TPU")
    cfg = EpuTpuConditionedReerValueFxConfig(cool=0.35, z_high=1.0)
    soft_cfg = EpuTpuConditionedSoftFxConfig(
        signal_lag_months=cfg.signal_lag_months,
        weight_lag_days=cfg.weight_lag_days,
        z_window=cfg.z_window,
        min_periods=cfg.min_periods,
        z_high=cfg.z_high,
        cool=cfg.cool,
    )
    z = align_monthly_stress_z_daily(epu, pret.index, cfg=soft_cfg)
    gcfg = GprRegimeConfig(
        z_high=cfg.z_high, z_low=cfg.z_low, cool=cfg.cool, signal_lag=0, usd_tilt=0.0
    )
    scale = risk_scale_from_z(z, cfg=gcfg).dropna()
    assert len(scale) > 50
    assert float(scale.min()) >= cfg.cool - 1e-9
    assert float(scale.max()) <= 1.0 + 1e-9
    out = epu_tpu_conditioned_reer_value_factor_returns(
        pret, reer, epu=epu, tpu=tpu, cfg=cfg
    )
    assert out["reer_epu_cool"].notna().sum() > 50


def test_no_lookahead_mutate_future_epu():
    pret = _synth_pairs(1800)
    reer = _synth_reer(start="2012-01-01", n_months=180, seed=4)
    epu = _synth_monthly(
        start="2012-01-01", n_months=180, seed=42, mean=100.0, name="US_EPU"
    )
    tpu = _synth_monthly(
        start="2012-01-01", n_months=180, seed=7, mean=80.0, name="TPU"
    )
    cfg = EpuTpuConditionedReerValueFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    f1 = epu_tpu_conditioned_reer_value_factor_returns(
        pret, reer, epu=epu, tpu=tpu, cfg=cfg
    )
    epu2 = epu.copy()
    epu2.loc["2021-01-01":] = epu2.loc["2021-01-01":] + 80.0
    f2 = epu_tpu_conditioned_reer_value_factor_returns(
        pret, reer, epu=epu2, tpu=tpu, cfg=cfg
    )
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["reer_low_epu"].loc[:cut].dropna()
    b = f2["reer_low_epu"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 50
    assert np.allclose(
        a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12
    )


def test_tpu_missing_fallthrough():
    pret = _synth_pairs(800)
    reer = _synth_reer(n_months=140, seed=9)
    epu = _synth_monthly(n_months=140, seed=9, mean=110.0, name="US_EPU")
    cfg = EpuTpuConditionedReerValueFxConfig()
    out = epu_tpu_conditioned_reer_value_factor_returns(
        pret, reer, epu=epu, tpu=None, cfg=cfg, allow_tpu_missing=True
    )
    assert "reer_low_epu" in out
    assert "reer_epu_cool" in out
    assert "reer_raw" in out
    assert "reer_high_epu" in out
    assert "reer_epu_tpu_ew" in out
    assert "reer_low_tpu" not in out


def test_factor_shapes_match_pair_index():
    pret = _synth_pairs(800)
    reer = _synth_reer(n_months=140, seed=7)
    epu = _synth_monthly(n_months=140, seed=7, name="US_EPU")
    tpu = _synth_monthly(n_months=140, seed=8, mean=95.0, name="TPU")
    out = epu_tpu_conditioned_reer_value_factor_returns(
        pret, reer, epu=epu, tpu=tpu, cfg=EpuTpuConditionedReerValueFxConfig()
    )
    assert out["reer_low_epu"].index.equals(pret.index)
    assert out["reer_epu_tpu_ew"].index.equals(pret.index)


@pytest.mark.skipif(
    not (EPU_FRED.exists() and REER_USD.exists()),
    reason="EPU/REER CSV missing",
)
def test_load_real_epu_reer():
    from mt5_swing.data.fred_bis_reer import load_bis_reer_panel
    from mt5_swing.strategies.epu_tpu_conditioned_reer_value_fx import (
        load_epu_series,
        load_tpu_series,
    )

    epu = load_epu_series(series="US", pub_lag_months=1, download=False)
    reer = load_bis_reer_panel(pub_lag_months=2, download=False, force=False)
    pret = _synth_pairs(500)
    assert epu.dropna().shape[0] > 100
    assert reer.dropna(how="all").shape[0] > 60
    try:
        tpu = load_tpu_series(pub_lag_months=1, download=False)
        out = epu_tpu_conditioned_reer_value_factor_returns(
            pret, reer, epu=epu, tpu=tpu
        )
    except Exception:
        out = epu_tpu_conditioned_reer_value_factor_returns(
            pret, reer, epu=epu, tpu=None, allow_tpu_missing=True
        )
    assert PRIMARY in out
    assert out[PRIMARY].notna().sum() > 20
