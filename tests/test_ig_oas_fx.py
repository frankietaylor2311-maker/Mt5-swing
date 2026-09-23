"""Tests for ICE BofA IG OAS / credit risk-appetite FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_ig_oas import (
    load_ig_oas_bundle,
    load_oas_series,
    oas_change,
)
from mt5_swing.strategies.funding_liquidity_fx import (
    FundingLiquidityFxConfig,
    align_and_z,
    usd_tilt_weights_from_z,
)
from mt5_swing.strategies.ig_oas_fx import (
    IgOasFxConfig,
    ig_oas_factor_returns,
    risk_fx_tilt_weights_from_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_oas_change_name():
    s = pd.Series(
        [1.0, 1.2, 0.9],
        index=pd.date_range("2020-01-02", periods=3, freq="B", tz="UTC"),
        name="BAMLC0A0CM",
    )
    d = oas_change(s)
    assert d.name == "d_BAMLC0A0CM"
    assert np.isclose(d.iloc[1], 0.2)


def test_align_and_z_pit_lag():
    idx = pd.date_range("2015-01-01", periods=400, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    raw = pd.Series(
        np.abs(rng.normal(1.2, 0.3, 400)),
        index=idx,
        name="BAMLC0A0CM",
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
    """oas_stress_fx must be opposite of USD haven tilt on elevated z."""
    idx = pd.date_range("2020-01-01", periods=5, freq="B", tz="UTC")
    z = pd.Series([0.0, 1.5, 2.0, 0.2, -0.5], index=idx)
    cfg = IgOasFxConfig(usd_tilt=0.5, z_high=1.0, binary_usd=True)
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


def test_ig_oas_no_lookahead():
    """Mutating future IG OAS must not change earlier factor returns."""
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
    ig = pd.Series(np.abs(rng.normal(1.2, 0.25, 1800)), index=idx, name="BAMLC0A0CM")
    hy = pd.Series(np.abs(rng.normal(4.0, 0.8, 1800)), index=idx, name="BAMLH0A0HYM2")
    m_idx = pd.date_range("2014-01-01", periods=96, freq="MS", tz="UTC")
    rates = pd.DataFrame(
        {
            c: 1.0 + 0.01 * rng.normal(0, 1, 96).cumsum()
            for c in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")
        },
        index=m_idx,
    )
    cfg = IgOasFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = ig_oas_factor_returns(
        ret, ig_oas=ig, hy_oas=hy, d_ig_oas=oas_change(ig), rates=rates, cfg=cfg
    )
    for key in (
        "ig_oas_usd",
        "hy_oas_usd",
        "ig_oas_chg_usd",
        "ig_oas_lvl_usd",
        "oas_stress_fx",
        "carry_ig_oas_cool",
        "carry_ig_oas_loose",
        "oas_ew",
    ):
        assert key in f1

    ig2 = ig.copy()
    ig2.loc["2021-01-01":] = ig2.loc["2021-01-01":] + 2.0
    f2 = ig_oas_factor_returns(
        ret, ig_oas=ig2, hy_oas=hy, d_ig_oas=oas_change(ig2), rates=rates, cfg=cfg
    )
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["ig_oas_usd"].loc[:cut].dropna()
    b = f2["ig_oas_usd"].loc[:cut].dropna()
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
    ig = pd.Series(np.abs(rng.normal(1.0, 0.2, 800)), index=idx, name="BAMLC0A0CM")
    out = ig_oas_factor_returns(ret, ig_oas=ig, cfg=IgOasFxConfig())
    assert out["ig_oas_usd"].index.equals(idx)
    assert out["oas_stress_fx"].index.equals(idx)


@pytest.mark.skipif(not (MACRO / "fred_BAMLC0A0CM.csv").exists(), reason="IG OAS cache missing")
def test_load_ig_oas_from_cache():
    s = load_oas_series("BAMLC0A0CM", download=False, pub_lag_days=1)
    assert len(s.dropna()) > 100
    assert s.attrs.get("pub_lag_days") == 1
    assert s.attrs.get("frequency") == "daily"


@pytest.mark.skipif(
    not (MACRO / "fred_BAMLC0A0CM.csv").exists(),
    reason="IG OAS cache missing",
)
def test_bundle_keys():
    b = load_ig_oas_bundle(download=False, include_hy=True, include_bbb=True)
    assert "IG_OAS" in b and len(b["IG_OAS"]) > 100
    assert "dIG_OAS" in b
    assert b["_meta"]["daily_pub_lag_days"] == 1
