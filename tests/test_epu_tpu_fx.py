"""Tests for Baker–Bloom–Davis EPU / TPU FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.macro_uncertainty import (
    load_country_epu,
    load_epu,
    load_tpu,
)
from mt5_swing.strategies.epu_tpu_fx import (
    EpuTpuFxConfig,
    epu_tpu_factor_returns,
    prepare_country_epu_signal,
    trailing_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_trailing_z_shape():
    idx = pd.date_range("2010-01-01", periods=80, freq="MS", tz="UTC")
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"EUR": rng.normal(100, 20, 80)}, index=idx)
    z = trailing_z(x, lookback=24, min_periods=12)
    assert z.shape == x.shape
    assert z.dropna().shape[0] > 40


def test_country_epu_signal_lag_and_relative():
    idx = pd.date_range("2015-01-01", periods=72, freq="MS", tz="UTC")
    rng = np.random.default_rng(3)
    panel = pd.DataFrame(
        {
            "EUR": 100 + rng.normal(0, 10, 72).cumsum() * 0.1,
            "GBP": 100 + rng.normal(0, 10, 72).cumsum() * 0.1,
            "AUD": 100 + rng.normal(0, 10, 72).cumsum() * 0.1,
            "JPY": 100 + rng.normal(0, 10, 72).cumsum() * 0.1,
            "CAD": 100 + rng.normal(0, 10, 72).cumsum() * 0.1,
            "USD": 100 + rng.normal(0, 10, 72).cumsum() * 0.1,
        },
        index=idx,
    )
    cfg = EpuTpuFxConfig(signal_lag_months=1, z_window=24, min_periods=12)
    sig = prepare_country_epu_signal(panel, cfg=cfg, relative=False)
    # After lag, first non-null should be later than z without lag
    assert sig.dropna(how="all").index.min() > panel.index.min()
    rel = prepare_country_epu_signal(panel, cfg=cfg, relative=True)
    assert "USD" not in rel.columns
    assert "EUR" in rel.columns


def test_epu_tpu_no_lookahead():
    """Mutating future country EPU must not change earlier factor returns."""
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
    m_idx = pd.date_range("2014-01-01", periods=96, freq="MS", tz="UTC")
    country = pd.DataFrame(
        {c: 80 + rng.normal(0, 15, 96).cumsum() * 0.05 for c in ("EUR", "GBP", "AUD", "JPY", "CAD", "USD")},
        index=m_idx,
    )
    us = country["USD"].copy()
    us.name = "US_EPU"
    tpu = (us * 1.2 + rng.normal(0, 5, 96)).rename("TPU")
    tpu.index = m_idx
    gepu = (us * 0.9 + 10).rename("GEPU")
    gepu.index = m_idx
    rates = pd.DataFrame(
        {c: 1.0 + 0.01 * rng.normal(0, 1, 96).cumsum() for c in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")},
        index=m_idx,
    )
    cfg = EpuTpuFxConfig(signal_lag_months=1, signal_lag_days=1, cost_bps_side=1.5)
    f1 = epu_tpu_factor_returns(
        ret,
        country_epu=country,
        us_epu=us,
        tpu=tpu,
        gepu=gepu,
        rates=rates,
        cfg=cfg,
    )
    assert "country_epu_xs" in f1
    assert "epu_diff_xs" in f1
    assert "epu_us_usd" in f1
    assert "tpu_us_usd" in f1
    assert "gepu_usd" in f1
    assert "carry_epu_cool" in f1
    assert "carry_tpu_cool" in f1
    assert "epu_ew" in f1
    # Mutate only post-2021 months; compare through 2019 so rolling z (60m)
    # and signal_lag cannot see the future perturbation.
    country2 = country.copy()
    country2.loc["2021-01-01":] = country2.loc["2021-01-01":] + 50.0
    f2 = epu_tpu_factor_returns(
        ret,
        country_epu=country2,
        us_epu=us,
        tpu=tpu,
        gepu=gepu,
        rates=rates,
        cfg=cfg,
    )
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["country_epu_xs"].loc[:cut].dropna()
    b = f2["country_epu_xs"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 200
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_high_epu_ranks_short_side():
    """Currency with highest EPU z should be on the short side of the sort."""
    idx = pd.date_range("2018-01-01", periods=60, freq="MS", tz="UTC")
    # Stable low for EUR/GBP/AUD/CAD; exploding JPY EPU
    panel = pd.DataFrame(
        {
            "EUR": np.full(60, 100.0),
            "GBP": np.full(60, 100.0),
            "AUD": np.full(60, 100.0),
            "CAD": np.full(60, 100.0),
            "JPY": np.linspace(100, 300, 60),
            "USD": np.full(60, 100.0),
        },
        index=idx,
    )
    cfg = EpuTpuFxConfig(signal_lag_months=0, z_window=24, min_periods=12, n_long=1, n_short=1)
    sig = prepare_country_epu_signal(panel, cfg=cfg)
    last = sig.drop(columns=["USD"], errors="ignore").dropna(how="all").iloc[-1]
    assert last.idxmax() == "JPY"


@pytest.mark.skipif(
    not (MACRO / "epu_all_country.xlsx").exists()
    and not (MACRO / "epu_country_monthly.csv").exists(),
    reason="Country EPU cache not present",
)
def test_load_country_epu_from_cache():
    panel = load_country_epu(download=False, pub_lag_months=1)
    assert "EUR" in panel.columns
    assert "GBP" in panel.columns
    assert "USD" in panel.columns
    assert len(panel.dropna(how="all")) > 100
    assert panel.attrs.get("pub_lag_months") == 1
    assert "NZD" in panel.attrs.get("missing_currencies", [])


@pytest.mark.skipif(
    not (MACRO / "epu_categorical.xlsx").exists()
    and not (MACRO / "tpu_trade_uncertainty.xlsx").exists()
    and not (MACRO / "tpu_us_monthly.csv").exists(),
    reason="TPU cache not present",
)
def test_load_tpu_from_cache():
    tpu = load_tpu(download=False, pub_lag_months=1)
    assert len(tpu.dropna()) > 200
    assert tpu.attrs.get("pub_lag_months") == 1
    # Categorical series should reach mid-2020s
    assert tpu.index.max().year >= 2024


@pytest.mark.skipif(
    not (MACRO / "fred_USEPUINDXM.csv").exists(),
    reason="FRED US EPU cache not present",
)
def test_load_us_epu_fred():
    us = load_epu(series="US", download=False, pub_lag_months=1)
    assert len(us.dropna()) > 200
    assert us.name == "US_EPU"
