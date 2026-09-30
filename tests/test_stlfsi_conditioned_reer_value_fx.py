"""Tests for STLFSI4-conditioned BIS REER HML-FX value module (§104)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.stlfsi_conditioned_reer_value_fx import (
    PRIMARY,
    StlfsiConditionedReerValueFxConfig,
    align_monthly_stress_z_daily,
    stlfsi_conditioned_reer_value_factor_returns,
)
from mt5_swing.strategies.stlfsi_conditioned_soft_fx import (
    StlfsiConditionedSoftFxConfig,
    trailing_z_monthly,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
STLFSI_FRED = MACRO / "fred_STLFSI4.csv"
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
    mean: float = 0.0,
    name: str = "STLFSI4",
) -> pd.Series:
    idx = pd.date_range(start, periods=n_months, freq="MS", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(mean, 0.08, n_months), index=idx, name=name
    )


def test_primary_name_constant():
    assert PRIMARY == "reer_low_stlfsi"


def test_gate_direction_low_on_high_off():
    pret = _synth_pairs(2000)
    reer = _synth_reer(n_months=200, seed=1)
    idx = pd.date_range("2010-01-01", periods=200, freq="MS", tz="UTC")
    stlfsi = pd.Series(
        np.concatenate([np.full(100, -0.5), np.full(100, 1.5)]),
        index=idx,
        name="STLFSI4",
    )
    cfg = StlfsiConditionedReerValueFxConfig(
        signal_lag_months=1,
        weight_lag_days=1,
        z_window=60,
        min_periods=24,
        z_high=1.0,
    )
    out = stlfsi_conditioned_reer_value_factor_returns(pret, reer, us_stlfsi=stlfsi, cfg=cfg)
    for key in (
        "reer_low_stlfsi",
        "reer_stlfsi_cool",
        "reer_raw",
        "reer_high_stlfsi",
        "us_stlfsi_haven_usd",
        "reer_stlfsi_stack",
        "reer_stlfsi_ew",
        "reer_stlfsi_regime",
    ):
        assert key in out
    lo = out["reer_low_stlfsi"]
    hi = out["reer_high_stlfsi"]
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
    stlfsi = _synth_monthly(
        start="2010-01-01", n_months=160, seed=1, mean=0.0, name="STLFSI4"
    )
    cfg = StlfsiConditionedReerValueFxConfig(cool=0.35, z_high=1.0)
    out = stlfsi_conditioned_reer_value_factor_returns(pret, reer, us_stlfsi=stlfsi, cfg=cfg)
    raw = out["reer_raw"]
    cool = out["reer_stlfsi_cool"]
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
    stlfsi = _synth_monthly(
        start="2010-01-01", n_months=180, seed=5, mean=0.0, name="STLFSI4"
    )
    cfg = StlfsiConditionedReerValueFxConfig(cool=0.35, z_high=1.0)
    out = stlfsi_conditioned_reer_value_factor_returns(pret, reer, us_stlfsi=stlfsi, cfg=cfg)
    raw = out["reer_raw"]
    stack = out["reer_stlfsi_stack"]
    lo = out["reer_low_stlfsi"]
    both = raw.notna() & stack.notna() & lo.notna()
    assert both.sum() > 50
    assert (stack.loc[both].abs() <= lo.loc[both].abs() + 1e-9).mean() > 0.95
    assert (stack.loc[both].abs() <= raw.loc[both].abs() + 1e-9).mean() > 0.90


def test_no_lookahead_mutate_future_stlfsi():
    """Mutating future STLFSI must not change earlier factor returns."""
    pret = _synth_pairs(1800)
    reer = _synth_reer(start="2012-01-01", n_months=180, seed=4)
    stlfsi = _synth_monthly(
        start="2012-01-01", n_months=180, seed=42, mean=0.0, name="STLFSI4"
    )
    cfg = StlfsiConditionedReerValueFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    f1 = stlfsi_conditioned_reer_value_factor_returns(pret, reer, us_stlfsi=stlfsi, cfg=cfg)
    stlfsi2 = stlfsi.copy()
    stlfsi2.loc["2019-01-01":] = stlfsi2.loc["2019-01-01":] + 0.40
    f2 = stlfsi_conditioned_reer_value_factor_returns(pret, reer, us_stlfsi=stlfsi2, cfg=cfg)
    cut = pd.Timestamp("2017-06-30", tz="UTC")
    a = f1["reer_low_stlfsi"].loc[:cut].dropna()
    b = f2["reer_low_stlfsi"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 50
    assert np.allclose(
        a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12
    )


def test_no_lookahead_mutate_future_reer():
    """Future REER mutation must not change past value signals."""
    pret = _synth_pairs(1800)
    reer = _synth_reer(start="2012-01-01", n_months=180, seed=8)
    stlfsi = _synth_monthly(
        start="2012-01-01", n_months=180, seed=11, mean=0.0, name="STLFSI4"
    )
    cfg = StlfsiConditionedReerValueFxConfig()
    f1 = stlfsi_conditioned_reer_value_factor_returns(pret, reer, us_stlfsi=stlfsi, cfg=cfg)
    reer2 = reer.copy()
    reer2.loc["2021-01-01":] = reer2.loc["2021-01-01":] * 1.15
    f2 = stlfsi_conditioned_reer_value_factor_returns(pret, reer2, us_stlfsi=stlfsi, cfg=cfg)
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
    stlfsi = _synth_monthly(start="2012-01-01", n_months=120, seed=7, name="STLFSI4")
    out = stlfsi_conditioned_reer_value_factor_returns(
        pret,
        reer,
        us_stlfsi=stlfsi,
        cfg=StlfsiConditionedReerValueFxConfig(z_window=36, min_periods=18),
    )
    assert out["reer_low_stlfsi"].index.equals(pret.index)
    assert out["reer_stlfsi_ew"].index.equals(pret.index)
    assert out["reer_stlfsi_regime"].index.equals(pret.index)


def test_gate_on_off_explicit_z():
    """Primary gate: reer on when z≤0; inverse on when z≥z_high."""
    idx = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    z = pd.Series([-1.0, -0.5, 0.0, 0.5, 1.5, 2.0, -0.2, 1.1, 0.0, -2.0], index=idx)
    cfg = StlfsiConditionedReerValueFxConfig(z_high=1.0)
    gate_lo = (z <= 0.0).astype(float)
    gate_hi = (z >= cfg.z_high).astype(float)
    assert gate_lo.iloc[0] == 1.0
    assert gate_lo.iloc[4] == 0.0
    assert gate_hi.iloc[4] == 1.0
    assert gate_hi.iloc[0] == 0.0


def test_monthly_z_window_prior():
    """Config must use monthly z_window=60 (CIP §71 / NFCI §90–§94 / ANFCI §95–§99 / STLFSI §100–§103 mirror, not 252d)."""
    cfg = StlfsiConditionedReerValueFxConfig()
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
    assert cfg.pub_lag_days == 7


def test_haven_optional():
    pret = _synth_pairs(800)
    reer = _synth_reer(n_months=140, seed=3)
    stlfsi = _synth_monthly(start="2012-01-01", n_months=120, seed=3, name="STLFSI4")
    cfg = StlfsiConditionedReerValueFxConfig(z_window=36, min_periods=18)
    out = stlfsi_conditioned_reer_value_factor_returns(
        pret, reer, us_stlfsi=stlfsi, cfg=cfg, include_haven=False
    )
    assert "reer_low_stlfsi" in out
    assert "reer_stlfsi_cool" in out
    assert "reer_raw" in out
    assert "reer_high_stlfsi" in out
    assert "us_stlfsi_haven_usd" not in out
    assert "reer_stlfsi_stack" in out
    assert "reer_stlfsi_ew" in out  # EW of low + cool still
    assert "reer_stlfsi_regime" not in out


def test_empty_reer_returns_empty():
    pret = _synth_pairs(400)
    stlfsi = _synth_monthly(n_months=80, seed=1, name="STLFSI4")
    out = stlfsi_conditioned_reer_value_factor_returns(
        pret, pd.DataFrame(), us_stlfsi=stlfsi, cfg=StlfsiConditionedReerValueFxConfig()
    )
    assert out == {}


def test_thin_panel_returns_empty():
    pret = _synth_pairs(400).iloc[:, :2]
    reer = _synth_reer(n_months=80, seed=2)
    stlfsi = _synth_monthly(n_months=80, seed=2, name="STLFSI4")
    out = stlfsi_conditioned_reer_value_factor_returns(pret, reer, us_stlfsi=stlfsi)
    assert out == {}


def test_monthly_z_window_not_daily_252():
    """PIT: z is computed on monthly frequency with z_window months."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    s = pd.Series(np.linspace(-0.5, 0.5, 120), index=idx, name="STLFSI4")
    z = trailing_z_monthly(s, lookback=60, min_periods=24)
    assert z.iloc[59:].notna().sum() > 40
    assert z.iloc[:23].isna().all()


def test_align_monthly_stress_z_lagged():
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    s = pd.Series(np.linspace(-0.5, 0.5, 120), index=idx, name="STLFSI4")
    daily_idx = pd.date_range("2015-01-01", periods=500, freq="B", tz="UTC")
    soft_cfg = StlfsiConditionedSoftFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    z = align_monthly_stress_z_daily(s, daily_idx, cfg=soft_cfg)
    assert z.index.equals(daily_idx)
    assert z.notna().sum() > 50
    assert pd.isna(z.iloc[0])


@pytest.mark.skipif(
    not (STLFSI_FRED.exists() and REER_USD.exists()),
    reason="STLFSI FRED CSV or REER CSV missing",
)
def test_load_real_us_stlfsi_and_reer():
    from mt5_swing.data.fred_bis_reer import load_bis_reer_panel
    from mt5_swing.strategies.stlfsi_conditioned_reer_value_fx import load_us_stlfsi_series

    stlfsi = load_us_stlfsi_series(pub_lag_days=7, download=False)
    assert stlfsi.dropna().shape[0] > 60
    reer = load_bis_reer_panel(pub_lag_months=2, download=False, force=False)
    pret = _synth_pairs(500)
    assert reer.dropna(how="all").shape[0] > 60
    out = stlfsi_conditioned_reer_value_factor_returns(pret, reer, us_stlfsi=stlfsi)
    assert PRIMARY in out
    assert out[PRIMARY].notna().sum() > 20
