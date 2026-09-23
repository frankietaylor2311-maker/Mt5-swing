"""Tests for real-rate / breakeven differential FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_real_rates import (
    DEFAULT_DAILY_PUB_LAG_DAYS,
    US_BREAKEVEN_SERIES,
    US_NOMINAL_10Y,
    US_REAL_SERIES,
    load_foreign_real_proxy_panel,
    load_real_rate_panels,
    load_us_breakeven,
    load_us_real_rate,
    real_rate_coverage,
)
from mt5_swing.strategies.real_rate_fx import (
    RealRateFxConfig,
    align_and_z_daily,
    real_rate_factor_returns,
    trailing_z_series,
    usd_or_fx_tilt_weights,
)


def test_us_series_ids_are_fred_tips_and_be():
    assert US_REAL_SERIES == "DFII10"
    assert US_BREAKEVEN_SERIES == "T10YIE"
    assert US_NOMINAL_10Y == "DGS10"


def test_daily_pub_lag_shifts_known_date():
    raw = load_us_real_rate(pub_lag_days=0, download=True, force=False)
    lagged = load_us_real_rate(pub_lag_days=DEFAULT_DAILY_PUB_LAG_DAYS, download=False)
    assert not raw.empty and not lagged.empty
    r0 = raw.dropna().index.min()
    l0 = lagged.dropna().index.min()
    delta_days = (l0 - r0).days
    assert 1 <= delta_days <= 2


def test_foreign_proxy_is_lt_minus_cpi_not_linker():
    panel = load_foreign_real_proxy_panel(("EUR", "GBP", "USD"), download=True)
    assert "EUR" in panel.columns and "USD" in panel.columns
    note = (
        str(panel.attrs.get("definition", ""))
        + str(panel.attrs.get("honest_note", ""))
        + str(panel.attrs.get("source", ""))
    ).lower()
    assert "proxy" in note or "lt" in note
    assert panel.dropna(how="all").shape[0] > 100


def test_usd_tilt_long_usd_when_z_high():
    """High US real z → RID → long USD (short EURUSD)."""
    idx = pd.date_range("2020-01-01", periods=100, freq="B", tz="UTC")
    z = pd.Series(0.0, index=idx)
    z.iloc[-20:] = 2.0
    cfg = RealRateFxConfig(signal_lag_days=0, binary_tilt=True, usd_tilt=0.5)
    w = usd_or_fx_tilt_weights(z, ["EURUSD", "USDJPY"], cfg=cfg, mode="usd")
    assert float(w["EURUSD"].iloc[-1]) < 0
    assert float(w["USDJPY"].iloc[-1]) > 0


def test_fx_tilt_opposite_of_usd():
    idx = pd.date_range("2020-01-01", periods=50, freq="B", tz="UTC")
    z = pd.Series(2.0, index=idx)
    cfg = RealRateFxConfig(signal_lag_days=0, usd_tilt=0.5)
    w_usd = usd_or_fx_tilt_weights(z, ["EURUSD"], cfg=cfg, mode="usd")
    w_fx = usd_or_fx_tilt_weights(z, ["EURUSD"], cfg=cfg, mode="fx")
    assert float(w_usd["EURUSD"].iloc[-1]) == pytest.approx(
        -float(w_fx["EURUSD"].iloc[-1]), abs=1e-12
    )


def test_no_lookahead_real_rate_shock():
    """Mutating future DFII10 must not change earlier factor returns."""
    idx = pd.date_range("2015-01-01", periods=2000, freq="B", tz="UTC")
    rng = np.random.default_rng(17)
    close = pd.DataFrame(
        {
            "EURUSD": 1.1 * np.cumprod(1 + rng.normal(0, 0.005, 2000)),
            "GBPUSD": 1.3 * np.cumprod(1 + rng.normal(0, 0.005, 2000)),
            "AUDUSD": 0.75 * np.cumprod(1 + rng.normal(0, 0.006, 2000)),
            "NZDUSD": 0.65 * np.cumprod(1 + rng.normal(0, 0.006, 2000)),
            "USDJPY": 110 * np.cumprod(1 + rng.normal(0, 0.005, 2000)),
            "USDCAD": 1.3 * np.cumprod(1 + rng.normal(0, 0.004, 2000)),
            "USDCHF": 0.95 * np.cumprod(1 + rng.normal(0, 0.004, 2000)),
        },
        index=idx,
    )
    ret = close.pct_change()
    d_idx = pd.date_range("2014-01-01", periods=800, freq="B", tz="UTC")
    us_real = pd.Series(rng.normal(1.0, 0.5, 800), index=d_idx, name="DFII10")
    us_be = pd.Series(rng.normal(2.0, 0.3, 800), index=d_idx, name="T10YIE")
    m_idx = pd.date_range("2014-01-01", periods=100, freq="MS", tz="UTC")
    rr_diff = pd.DataFrame(
        {
            "EUR": rng.normal(0.0, 1.0, 100),
            "GBP": rng.normal(0.0, 1.0, 100),
            "AUD": rng.normal(0.0, 1.0, 100),
            "JPY": rng.normal(0.0, 1.0, 100),
            "CAD": rng.normal(0.0, 1.0, 100),
            "CHF": rng.normal(0.0, 1.0, 100),
            "NZD": rng.normal(0.0, 1.0, 100),
        },
        index=m_idx,
    )
    cfg = RealRateFxConfig(signal_lag_days=1, signal_lag_months=1, cost_bps_side=1.5)
    f1 = real_rate_factor_returns(
        ret, us_real=us_real, us_be=us_be, rr_diff=rr_diff, cfg=cfg
    )
    assert "us_real_usd" in f1
    assert "rr_xs" in f1
    assert "rr_ew" in f1

    us_real2 = us_real.copy()
    us_real2.iloc[-40:] = us_real2.iloc[-40:] + 2.0
    rr2 = rr_diff.copy()
    rr2.iloc[-10:] = rr2.iloc[-10:] + 1.0
    f2 = real_rate_factor_returns(
        ret, us_real=us_real2, us_be=us_be, rr_diff=rr2, cfg=cfg
    )
    cut = d_idx[-120]
    for name in ("us_real_usd", "us_real_chg_usd", "rr_xs"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(
            a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12
        )


def test_align_and_z_applies_signal_lag():
    idx = pd.date_range("2020-01-01", periods=300, freq="B", tz="UTC")
    macro = pd.Series(
        np.linspace(0.0, 1.0, 200),
        index=pd.date_range("2019-06-01", periods=200, freq="B", tz="UTC"),
        name="DFII10",
    )
    cfg0 = RealRateFxConfig(signal_lag_days=0, z_window_daily=60, min_periods_daily=20)
    cfg1 = RealRateFxConfig(signal_lag_days=1, z_window_daily=60, min_periods_daily=20)
    z0 = align_and_z_daily(macro, idx, cfg=cfg0)
    z1 = align_and_z_daily(macro, idx, cfg=cfg1)
    common = z0.dropna().index.intersection(z1.dropna().index)
    assert len(common) > 50
    shifted = z0.shift(1)
    overlap = common.intersection(shifted.dropna().index)
    assert np.allclose(
        z1.loc[overlap].values, shifted.loc[overlap].values, equal_nan=True, atol=1e-12
    )


def test_loader_smoke_and_coverage():
    panels = load_real_rate_panels(download=True)
    assert "us_real_d" in panels and "rr_diff_vs_usd" in panels
    assert panels["us_real_d"].dropna().shape[0] > 1000
    assert panels["rr_diff_vs_usd"].shape[1] >= 4
    cov = real_rate_coverage(panels)
    assert len(cov) >= 5
    n_col = "n_obs" if "n_obs" in cov.columns else "n"
    assert (cov[n_col] > 0).all()
    be = load_us_breakeven(download=False)
    assert len(be) > 1000


def test_trailing_z_series_unit():
    idx = pd.date_range("2020-01-01", periods=100, freq="B", tz="UTC")
    s = pd.Series(np.arange(100, dtype=float), index=idx)
    z = trailing_z_series(s, lookback=20, min_periods=10)
    assert z.dropna().shape[0] >= 80
    assert float(z.iloc[-1]) > 0
