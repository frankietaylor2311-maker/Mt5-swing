"""Tests for NY Fed ACM term-premium / USD risk-appetite FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_acm_term_premium import (
    load_acm_tp_bundle,
    load_acm_tp_series,
    tp_change,
)
from mt5_swing.strategies.funding_liquidity_fx import (
    FundingLiquidityFxConfig,
    align_and_z,
    usd_tilt_weights_from_z,
)
from mt5_swing.strategies.acm_term_premium_fx import (
    AcmTermPremiumFxConfig,
    acm_tp_factor_returns,
    risk_fx_tilt_weights_from_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_tp_change_name():
    s = pd.Series(
        [1.0, 1.2, 0.9],
        index=pd.date_range("2020-01-02", periods=3, freq="B", tz="UTC"),
        name="THREEFYTP10",
    )
    d = tp_change(s)
    assert d.name == "d_THREEFYTP10"
    assert np.isclose(d.iloc[1], 0.2)


def test_align_and_z_pit_lag():
    idx = pd.date_range("2015-01-01", periods=400, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    raw = pd.Series(
        rng.normal(0.5, 0.3, 400),
        index=idx,
        name="THREEFYTP10",
    )
    cfg = FundingLiquidityFxConfig(signal_lag=1, z_window=60, min_periods=30)
    z = align_and_z(raw, idx, cfg=cfg)
    assert z.index.equals(idx)
    assert z.dropna().shape[0] > 50
    assert z.dropna().index.min() > idx.min()


def test_usd_tilt_fires_on_high_z():
    idx = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    z = pd.Series([0.0, 0.5, 1.5, 2.0, 0.2, -0.5, 1.1, 0.0, 0.0, 0.0], index=idx)
    cfg = FundingLiquidityFxConfig(usd_tilt=0.5, z_high=1.0, binary_usd=True)
    cols = ["EURUSD", "USDJPY"]
    w = usd_tilt_weights_from_z(z, cols, cfg=cfg)
    assert w.loc[idx[2], "EURUSD"] < 0  # long USD
    assert w.loc[idx[2], "USDJPY"] > 0
    assert w.loc[idx[0], "EURUSD"] == 0.0


def test_honesty_stress_fx_flips_sign():
    """acm_tp_stress_fx must be opposite of USD haven tilt on elevated z."""
    idx = pd.date_range("2020-01-01", periods=5, freq="B", tz="UTC")
    z = pd.Series([0.0, 1.5, 2.0, 0.2, -0.5], index=idx)
    cfg = AcmTermPremiumFxConfig(usd_tilt=0.5, z_high=1.0, binary_usd=True)
    cols = ["EURUSD", "USDJPY"]
    haven = usd_tilt_weights_from_z(
        z, cols, cfg=FundingLiquidityFxConfig(usd_tilt=0.5, z_high=1.0, binary_usd=True)
    )
    stress = risk_fx_tilt_weights_from_z(z, cols, cfg=cfg)
    assert stress.loc[idx[1], "EURUSD"] > 0  # long EUR = short USD
    assert stress.loc[idx[1], "USDJPY"] < 0
    assert np.isclose(
        stress.loc[idx[1], "EURUSD"], -haven.loc[idx[1], "EURUSD"], atol=1e-12
    )


def test_acm_tp_no_lookahead():
    """Mutating future ACM TP must not change earlier factor returns."""
    idx = pd.date_range("2016-01-01", periods=1800, freq="B", tz="UTC")
    rng = np.random.default_rng(42)
    close = pd.DataFrame(
        {
            "EURUSD": 1.1 * np.cumprod(1 + rng.normal(0, 0.005, 1800)),
            "GBPUSD": 1.3 * np.cumprod(1 + rng.normal(0, 0.005, 1800)),
            "AUDUSD": 0.75 * np.cumprod(1 + rng.normal(0, 0.006, 1800)),
            "NZDUSD": 0.65 * np.cumprod(1 + rng.normal(0, 0.006, 1800)),
            "USDJPY": 110 * np.cumprod(1 + rng.normal(0, 0.005, 1800)),
            "USDCAD": 1.3 * np.cumprod(1 + rng.normal(0, 0.004, 1800)),
            "USDCHF": 0.95 * np.cumprod(1 + rng.normal(0, 0.004, 1800)),
        },
        index=idx,
    )
    ret = close.pct_change()
    tp = pd.Series(rng.normal(0.5, 0.4, 1800), index=idx, name="THREEFYTP10")
    tp5 = pd.Series(rng.normal(0.3, 0.3, 1800), index=idx, name="THREEFYTP5")
    m_idx = pd.date_range("2014-01-01", periods=96, freq="MS", tz="UTC")
    rates = pd.DataFrame(
        {
            c: 1.0 + 0.01 * rng.normal(0, 1, 96).cumsum()
            for c in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")
        },
        index=m_idx,
    )
    cfg = AcmTermPremiumFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = acm_tp_factor_returns(
        ret, acm_tp10=tp, acm_tp5=tp5, d_acm_tp10=tp_change(tp), rates=rates, cfg=cfg
    )
    for key in (
        "acm_tp_usd",
        "acm_tp5_usd",
        "acm_tp_chg_usd",
        "acm_tp_lvl_usd",
        "acm_tp_stress_fx",
        "carry_acm_tp_cool",
        "carry_acm_tp_loose",
        "acm_tp_ew",
    ):
        assert key in f1

    tp2 = tp.copy()
    tp2.loc["2021-01-01":] = tp2.loc["2021-01-01":] + 2.0
    f2 = acm_tp_factor_returns(
        ret, acm_tp10=tp2, acm_tp5=tp5, d_acm_tp10=tp_change(tp2), rates=rates, cfg=cfg
    )
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["acm_tp_usd"].loc[:cut].dropna()
    b = f2["acm_tp_usd"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 200
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_factor_shapes_match_index():
    idx = pd.date_range("2018-01-01", periods=800, freq="B", tz="UTC")
    rng = np.random.default_rng(7)
    ret = pd.DataFrame(
        {s: rng.normal(0, 0.005, 800) for s in ("EURUSD", "GBPUSD", "USDJPY", "USDCAD")},
        index=idx,
    )
    tp = pd.Series(rng.normal(0.4, 0.3, 800), index=idx, name="THREEFYTP10")
    out = acm_tp_factor_returns(ret, acm_tp10=tp, cfg=AcmTermPremiumFxConfig())
    assert out["acm_tp_usd"].index.equals(idx)
    assert out["acm_tp_stress_fx"].index.equals(idx)


@pytest.mark.skipif(
    not (MACRO / "fred_THREEFYTP10.csv").exists(), reason="ACM TP10 cache missing"
)
def test_load_acm_tp_from_cache():
    s = load_acm_tp_series("THREEFYTP10", download=False, pub_lag_days=1)
    assert len(s.dropna()) > 100
    assert s.attrs.get("pub_lag_days") == 1
    assert s.attrs.get("frequency") == "daily"


@pytest.mark.skipif(
    not (MACRO / "fred_THREEFYTP10.csv").exists(),
    reason="ACM TP10 cache missing",
)
def test_bundle_keys():
    b = load_acm_tp_bundle(download=False, include_tp5=True)
    assert "ACM_TP10" in b and len(b["ACM_TP10"]) > 100
    assert "dACM_TP10" in b
    assert b["_meta"]["daily_pub_lag_days"] == 1
