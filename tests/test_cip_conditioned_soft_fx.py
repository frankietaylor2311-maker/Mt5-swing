"""Tests for CIP-stress-conditioned Dahlquist soft-signal EW FX module (§71)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.cip_conditioned_carry_fx import (
    CipConditionedCarryFxConfig,
    cip_stress_from_panel,
)
from mt5_swing.strategies.cip_conditioned_soft_fx import (
    PRIMARY,
    cip_conditioned_soft_factor_returns,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
SLIM = MACRO / "cip_g10_5y_govt_panel.csv"


def _synth_panel(n_months: int = 96) -> pd.DataFrame:
    idx = pd.date_range("2014-01-31", periods=n_months, freq="ME", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            c: rng.normal(-20, 15, n_months)
            for c in ("EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")
        },
        index=idx,
    )


def _synth_soft(n_days: int = 1800) -> tuple[pd.Series, pd.DataFrame]:
    idx = pd.date_range("2016-01-01", periods=n_days, freq="B", tz="UTC")
    rng = np.random.default_rng(11)
    soft = pd.Series(rng.normal(0.0001, 0.002, n_days), index=idx, name="soft_ew_macro5")
    ret = pd.DataFrame(
        {s: rng.normal(0, 0.005, n_days) for s in ("EURUSD", "GBPUSD", "USDJPY", "USDCAD")},
        index=idx,
    )
    return soft, ret


def test_primary_name_constant():
    assert PRIMARY == "soft_low_cip_stress"


def test_gate_direction_low_on_high_off():
    soft, ret = _synth_soft(2000)
    # Force stress so z is controllable: use override series with known pattern
    # Build monthly stress that produces low then high z after warmup.
    m_idx = pd.date_range("2014-01-31", periods=120, freq="ME", tz="UTC")
    # Low stress early (negative), then jump high
    stress = pd.Series(
        np.concatenate([np.full(60, -5.0), np.full(60, 40.0)]),
        index=m_idx,
        name="cip_stress",
    )
    panel = _synth_panel(120)
    cfg = CipConditionedCarryFxConfig(
        signal_lag_months=1, weight_lag_days=1, z_window=24, min_periods=12, z_high=1.0
    )
    out = cip_conditioned_soft_factor_returns(
        soft, panel, pair_ret=ret, cfg=cfg, cip_stress_override=stress
    )
    for key in (
        "soft_low_cip_stress",
        "soft_cip_cool",
        "soft_raw",
        "soft_high_cip_stress",
        "cip_stress_haven_usd",
        "soft_cip_ew",
    ):
        assert key in out
    # soft_raw identity
    common = soft.dropna().index.intersection(out["soft_raw"].dropna().index)
    assert np.allclose(soft.loc[common].values, out["soft_raw"].loc[common].values)
    # Gate magnitude: |gated| <= |raw| wherever both defined (gate ∈ {0,1})
    lo = out["soft_low_cip_stress"]
    raw = out["soft_raw"]
    both = lo.notna() & raw.notna()
    assert both.sum() > 100
    assert (lo.loc[both].abs() <= raw.loc[both].abs() + 1e-15).all()
    # Inverse honesty: high-stress gate and low-stress gate never both 1 on same day
    # (except when raw==0); check product of absolute gated series <= eps when raw nonzero
    hi = out["soft_high_cip_stress"]
    product = lo.loc[both].abs() * hi.loc[both].abs()
    # At most one of lo/hi can be non-zero for a given day (z cannot be both ≤0 and ≥1)
    assert (product <= 1e-18).mean() > 0.99


def test_cool_scale_bounds():
    soft, ret = _synth_soft(600)
    panel = _synth_panel(96)
    cfg = CipConditionedCarryFxConfig(cool=0.35, z_high=1.0)
    out = cip_conditioned_soft_factor_returns(soft, panel, pair_ret=ret, cfg=cfg)
    raw = out["soft_raw"]
    cool = out["soft_cip_cool"]
    common = raw.dropna().index.intersection(cool.dropna().index)
    # |cool| <= |raw| when same sign (scale in [0.35,1]); allow zeros
    ratio = (cool.loc[common] / raw.loc[common]).replace([np.inf, -np.inf], np.nan)
    ratio = ratio.dropna()
    ratio = ratio[raw.loc[ratio.index].abs() > 1e-12]
    if len(ratio):
        assert float(ratio.min()) >= 0.35 - 1e-9
        assert float(ratio.max()) <= 1.0 + 1e-9


def test_no_lookahead_mutate_future_cip():
    soft, ret = _synth_soft(1800)
    rng = np.random.default_rng(42)
    m_idx = pd.date_range("2014-01-31", periods=120, freq="ME", tz="UTC")
    panel = pd.DataFrame(
        {c: rng.normal(-15, 20, 120) for c in ("EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")},
        index=m_idx,
    )
    cfg = CipConditionedCarryFxConfig(signal_lag_months=1, weight_lag_days=1)
    f1 = cip_conditioned_soft_factor_returns(soft, panel, pair_ret=ret, cfg=cfg)
    panel2 = panel.copy()
    panel2.loc["2021-01-31":] = panel2.loc["2021-01-31":] - 40.0
    f2 = cip_conditioned_soft_factor_returns(soft, panel2, pair_ret=ret, cfg=cfg)
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["soft_low_cip_stress"].loc[:cut].dropna()
    b = f2["soft_low_cip_stress"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 200
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_factor_shapes_match_soft_index():
    soft, ret = _synth_soft(800)
    panel = _synth_panel(60)
    out = cip_conditioned_soft_factor_returns(
        soft, panel, pair_ret=ret, cfg=CipConditionedCarryFxConfig()
    )
    assert out["soft_low_cip_stress"].index.equals(soft.index)
    assert out["cip_stress_haven_usd"].index.equals(soft.index)


def test_without_haven_still_builds_core():
    soft, _ret = _synth_soft(400)
    panel = _synth_panel(60)
    out = cip_conditioned_soft_factor_returns(
        soft, panel, pair_ret=None, include_haven=False
    )
    assert "soft_low_cip_stress" in out
    assert "cip_stress_haven_usd" not in out
    assert "soft_cip_ew" in out  # EW of low + cool still


@pytest.mark.skipif(not SLIM.exists(), reason="CIP slim panel missing")
def test_load_slim_panel_stress():
    from mt5_swing.data.cip_basis import (
        DEFAULT_PUB_LAG_DAYS,
        DEFAULT_TENOR,
        load_cip_panel,
    )

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
