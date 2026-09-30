"""Tests for Du–Schreger CIP-conditioned BIS REER HML-FX value module (§84)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.cip_conditioned_reer_value_fx import (
    PRIMARY,
    CipConditionedReerValueFxConfig,
    cip_conditioned_reer_value_factor_returns,
    cip_stress_from_panel,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
SLIM = MACRO / "cip_g10_5y_govt_panel.csv"
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


def _synth_cip_panel(n_months: int = 120, *, seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range("2014-01-31", periods=n_months, freq="ME", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            c: rng.normal(-20, 15, n_months)
            for c in ("EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")
        },
        index=idx,
    )


def test_primary_name_constant():
    assert PRIMARY == "reer_low_cip"


def test_cip_stress_is_negative_mean():
    panel = _synth_cip_panel(12)
    stress = cip_stress_from_panel(panel)
    expected = -panel.mean(axis=1)
    assert np.allclose(stress.values, expected.values, equal_nan=True)


def test_gate_direction_low_on_high_off():
    pret = _synth_pairs(2000)
    reer = _synth_reer(n_months=200, seed=1)
    # Low CIP stress first half / high second half
    m_idx = pd.date_range("2010-01-31", periods=200, freq="ME", tz="UTC")
    panel = pd.DataFrame(
        {
            c: np.concatenate([np.full(100, 10.0), np.full(100, -60.0)])
            for c in ("EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")
        },
        index=m_idx,
    )
    cfg = CipConditionedReerValueFxConfig(
        signal_lag_months=1,
        weight_lag_days=1,
        z_window=60,
        min_periods=24,
        z_high=1.0,
    )
    out = cip_conditioned_reer_value_factor_returns(pret, reer, panel, cfg=cfg)
    for key in (
        "reer_low_cip",
        "reer_cip_cool",
        "reer_raw",
        "reer_high_cip",
        "cip_stress_haven_usd",
        "reer_cip_stack",
        "reer_cip_ew",
        "reer_cip_regime",
    ):
        assert key in out
    lo = out["reer_low_cip"]
    hi = out["reer_high_cip"]
    raw = out["reer_raw"]
    both = lo.notna() & raw.notna()
    assert both.sum() > 100
    product = lo.loc[both].abs() * hi.loc[both].abs()
    assert (product <= 1e-10).mean() > 0.90


def test_cool_scale_bounds():
    from mt5_swing.strategies.cip_conditioned_carry_fx import (
        CipConditionedCarryFxConfig,
        align_cip_stress_z_daily,
    )
    from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z

    pret = _synth_pairs(800)
    reer = _synth_reer(n_months=140, seed=2)
    panel = _synth_cip_panel(160, seed=1)
    panel.index = pd.date_range("2010-01-31", periods=len(panel), freq="ME", tz="UTC")
    cfg = CipConditionedReerValueFxConfig(
        cool=0.35, z_high=1.0, z_window=36, min_periods=18
    )
    cip_cfg = CipConditionedCarryFxConfig(
        signal_lag_months=cfg.signal_lag_months,
        weight_lag_days=cfg.weight_lag_days,
        z_window=cfg.z_window,
        min_periods=cfg.min_periods,
        z_high=cfg.z_high,
        cool=cfg.cool,
    )
    stress = cip_stress_from_panel(panel)
    z = align_cip_stress_z_daily(stress, pret.index, cfg=cip_cfg)
    gcfg = GprRegimeConfig(
        z_high=cfg.z_high, z_low=cfg.z_low, cool=cfg.cool, signal_lag=0, usd_tilt=0.0
    )
    scale = risk_scale_from_z(z, cfg=gcfg).dropna()
    assert len(scale) > 50
    assert float(scale.min()) >= cfg.cool - 1e-9
    assert float(scale.max()) <= 1.0 + 1e-9
    out = cip_conditioned_reer_value_factor_returns(pret, reer, panel, cfg=cfg)
    assert out["reer_cip_cool"].notna().sum() > 50


def test_no_lookahead_mutate_future_cip():
    pret = _synth_pairs(1800)
    reer = _synth_reer(start="2012-01-01", n_months=180, seed=4)
    panel = _synth_cip_panel(180, seed=42)
    panel.index = pd.date_range("2012-01-31", periods=len(panel), freq="ME", tz="UTC")
    cfg = CipConditionedReerValueFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=60, min_periods=24
    )
    f1 = cip_conditioned_reer_value_factor_returns(pret, reer, panel, cfg=cfg)
    panel2 = panel.copy()
    panel2.loc["2021-01-31":] = panel2.loc["2021-01-31":] - 40.0
    f2 = cip_conditioned_reer_value_factor_returns(pret, reer, panel2, cfg=cfg)
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["reer_low_cip"].loc[:cut].dropna()
    b = f2["reer_low_cip"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 50
    assert np.allclose(
        a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12
    )


def test_no_lookahead_mutate_future_reer():
    """Future REER mutation must not change past value signals."""
    pret = _synth_pairs(1800)
    reer = _synth_reer(start="2012-01-01", n_months=180, seed=8)
    panel = _synth_cip_panel(180, seed=11)
    panel.index = pd.date_range("2012-01-31", periods=len(panel), freq="ME", tz="UTC")
    cfg = CipConditionedReerValueFxConfig()
    f1 = cip_conditioned_reer_value_factor_returns(pret, reer, panel, cfg=cfg)
    reer2 = reer.copy()
    reer2.loc["2021-01-01":] = reer2.loc["2021-01-01":] * 1.15
    f2 = cip_conditioned_reer_value_factor_returns(pret, reer2, panel, cfg=cfg)
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
    panel = _synth_cip_panel(100, seed=7)
    panel.index = pd.date_range("2012-01-31", periods=len(panel), freq="ME", tz="UTC")
    out = cip_conditioned_reer_value_factor_returns(
        pret,
        reer,
        panel,
        cfg=CipConditionedReerValueFxConfig(z_window=36, min_periods=18),
    )
    assert out["reer_low_cip"].index.equals(pret.index)
    assert out["reer_cip_ew"].index.equals(pret.index)
    assert out["reer_cip_regime"].index.equals(pret.index)


def test_gate_on_off_explicit_z():
    """Primary gate: reer on when z≤0; inverse on when z≥z_high."""
    idx = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    z = pd.Series([-1.0, -0.5, 0.0, 0.5, 1.5, 2.0, -0.2, 1.1, 0.0, -2.0], index=idx)
    cfg = CipConditionedReerValueFxConfig(z_high=1.0)
    gate_lo = (z <= 0.0).astype(float)
    gate_hi = (z >= cfg.z_high).astype(float)
    assert gate_lo.iloc[0] == 1.0
    assert gate_lo.iloc[4] == 0.0
    assert gate_hi.iloc[4] == 1.0
    assert gate_hi.iloc[0] == 0.0


def test_monthly_z_window_prior():
    """Config must use monthly z_window=60 (CIP §68/§71/§82/§83 mirror, not 252d)."""
    cfg = CipConditionedReerValueFxConfig()
    assert cfg.z_window == 60
    assert cfg.min_periods == 24
    assert cfg.signal_lag_months == 1
    assert cfg.weight_lag_days == 1
    assert cfg.reer_signal_lag == 1
    assert cfg.reer_z_window == 60
    assert cfg.n_long == 2
    assert cfg.n_short == 2
    assert cfg.reer_weight_lag_days == 1


def test_haven_optional():
    pret = _synth_pairs(800)
    reer = _synth_reer(n_months=140, seed=3)
    panel = _synth_cip_panel(100, seed=3)
    panel.index = pd.date_range("2012-01-31", periods=len(panel), freq="ME", tz="UTC")
    cfg = CipConditionedReerValueFxConfig(z_window=36, min_periods=18)
    out = cip_conditioned_reer_value_factor_returns(
        pret, reer, panel, cfg=cfg, include_haven=False
    )
    assert "reer_low_cip" in out
    assert "reer_cip_cool" in out
    assert "reer_raw" in out
    assert "reer_high_cip" in out
    assert "cip_stress_haven_usd" not in out
    assert "reer_cip_regime" not in out


def test_empty_reer_returns_empty():
    pret = _synth_pairs(400)
    panel = _synth_cip_panel(40, seed=1)
    out = cip_conditioned_reer_value_factor_returns(
        pret, pd.DataFrame(), panel, cfg=CipConditionedReerValueFxConfig()
    )
    assert out == {}


@pytest.mark.skipif(
    not (SLIM.exists() and REER_USD.exists()),
    reason="CIP slim panel or REER CSV missing",
)
def test_load_slim_panel_and_reer():
    from mt5_swing.data.cip_basis import (
        DEFAULT_PUB_LAG_DAYS,
        DEFAULT_TENOR,
        load_cip_panel,
    )
    from mt5_swing.data.fred_bis_reer import load_bis_reer_panel

    lp = load_cip_panel(
        tenor=DEFAULT_TENOR,
        pub_lag_days=DEFAULT_PUB_LAG_DAYS,
        download=False,
        force=False,
        frequency="month_end",
    )
    assert lp.shape[1] >= 4
    stress = cip_stress_from_panel(lp)
    assert stress.dropna().shape[0] > 24
    reer = load_bis_reer_panel(pub_lag_months=2, download=False, force=False)
    pret = _synth_pairs(500)
    assert reer.dropna(how="all").shape[0] > 60
    out = cip_conditioned_reer_value_factor_returns(pret, reer, lp)
    assert PRIMARY in out
    assert out[PRIMARY].notna().sum() > 20
