"""Tests for funding-liquidity / financial-conditions FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_funding_liquidity import (
    load_funding_liquidity_bundle,
    load_nfci_series,
    load_funding_spread,
    weekly_change,
)
from mt5_swing.strategies.funding_liquidity_fx import (
    FundingLiquidityFxConfig,
    align_and_z,
    funding_liquidity_factor_returns,
    trailing_z,
    usd_tilt_weights_from_level_gate,
    usd_tilt_weights_from_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_trailing_z_and_align_lag():
    idx = pd.date_range("2015-01-01", periods=400, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    raw = pd.Series(rng.normal(0, 0.2, 80), index=pd.date_range("2014-01-01", periods=80, freq="W-FRI", tz="UTC"), name="NFCI")
    cfg = FundingLiquidityFxConfig(signal_lag=1, z_window=60, min_periods=30)
    z = align_and_z(raw, idx, cfg=cfg)
    assert z.index.equals(idx)
    assert z.dropna().shape[0] > 50
    # First valid z should be after start (rolling + lag)
    assert z.dropna().index.min() > idx.min()


def test_usd_tilt_fires_on_high_z():
    idx = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    z = pd.Series([0.0, 0.5, 1.5, 2.0, 0.2, -0.5, 1.1, 0.0, 0.0, 0.0], index=idx)
    cfg = FundingLiquidityFxConfig(usd_tilt=0.5, z_high=1.0, binary_usd=True)
    cols = ["EURUSD", "USDJPY"]
    w = usd_tilt_weights_from_z(z, cols, cfg=cfg)
    # High z days: short EURUSD (+long USD), long USDJPY
    assert w.loc[idx[2], "EURUSD"] < 0
    assert w.loc[idx[2], "USDJPY"] > 0
    assert w.loc[idx[0], "EURUSD"] == 0.0


def test_nfci_level_gate():
    idx = pd.date_range("2020-01-01", periods=5, freq="B", tz="UTC")
    lvl = pd.Series([-0.2, 0.0, 0.1, 0.5, -0.1], index=idx, name="NFCI")
    cfg = FundingLiquidityFxConfig(nfci_level_cut=0.0, usd_tilt=0.5)
    w = usd_tilt_weights_from_level_gate(lvl, ["EURUSD", "USDJPY"], cfg=cfg)
    assert w.loc[idx[0], "EURUSD"] == 0.0  # loose
    assert w.loc[idx[2], "EURUSD"] < 0  # tight → long USD


def test_funding_no_lookahead():
    """Mutating future NFCI must not change earlier factor returns."""
    idx = pd.date_range("2016-01-01", periods=1800, freq="B", tz="UTC")
    rng = np.random.default_rng(11)
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
    w_idx = pd.date_range("2014-01-01", periods=400, freq="W-FRI", tz="UTC")
    nfci = pd.Series(rng.normal(-0.3, 0.4, 400), index=w_idx, name="NFCI")
    cpff = pd.Series(
        np.abs(rng.normal(0.2, 0.1, 1800)),
        index=idx,
        name="CPFF",
    )
    m_idx = pd.date_range("2014-01-01", periods=96, freq="MS", tz="UTC")
    rates = pd.DataFrame(
        {c: 1.0 + 0.01 * rng.normal(0, 1, 96).cumsum() for c in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")},
        index=m_idx,
    )
    cfg = FundingLiquidityFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = funding_liquidity_factor_returns(
        ret, nfci=nfci, d_nfci=weekly_change(nfci), cpff=cpff, rates=rates, cfg=cfg
    )
    assert "nfci_usd" in f1
    assert "nfci_lvl_usd" in f1
    assert "nfci_chg_usd" in f1
    assert "cpff_usd" in f1
    assert "carry_nfci_cool" in f1
    assert "carry_nfci_loose" in f1
    assert "funding_ew" in f1

    nfci2 = nfci.copy()
    nfci2.loc["2021-01-01":] = nfci2.loc["2021-01-01":] + 2.0
    f2 = funding_liquidity_factor_returns(
        ret, nfci=nfci2, d_nfci=weekly_change(nfci2), cpff=cpff, rates=rates, cfg=cfg
    )
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["nfci_usd"].loc[:cut].dropna()
    b = f2["nfci_usd"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 200
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_weekly_change_name():
    s = pd.Series([1.0, 1.2, 0.9], index=pd.date_range("2020-01-03", periods=3, freq="W-FRI", tz="UTC"), name="NFCI")
    d = weekly_change(s)
    assert d.name == "d_NFCI"
    assert np.isclose(d.iloc[1], 0.2)


@pytest.mark.skipif(not (MACRO / "fred_NFCI.csv").exists(), reason="NFCI cache missing")
def test_load_nfci_from_cache():
    s = load_nfci_series("NFCI", download=False, pub_lag_days=7)
    assert len(s.dropna()) > 500
    assert s.attrs.get("pub_lag_days") == 7
    assert s.attrs.get("frequency") == "weekly"


@pytest.mark.skipif(not (MACRO / "fred_CPFF.csv").exists(), reason="CPFF cache missing")
def test_load_cpff_from_cache():
    s = load_funding_spread("CPFF", download=False, pub_lag_days=1)
    assert len(s.dropna()) > 500
    assert s.attrs.get("pub_lag_days") == 1


@pytest.mark.skipif(
    not (MACRO / "fred_NFCI.csv").exists() or not (MACRO / "fred_CPFF.csv").exists(),
    reason="funding FRED caches missing",
)
def test_bundle_keys():
    b = load_funding_liquidity_bundle(download=False)
    for k in ("NFCI", "ANFCI", "dNFCI", "CPFF", "TEDRATE", "BAA10Y", "FUNDING_SPREAD"):
        assert k in b
        assert len(b[k]) > 100
