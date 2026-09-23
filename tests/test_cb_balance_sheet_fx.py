"""Tests for central-bank balance-sheet / QE differential FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_cb_balance_sheet import (
    CB_ASSETS_SERIES,
    DEFAULT_WEEKLY_PUB_LAG_DAYS,
    cb_coverage,
    load_bs_gdp_ratio,
    load_cb_assets,
    load_cb_yoy_panel,
    us_vs_peer_yoy_diff,
    yoy_growth,
)
from mt5_swing.strategies.cb_balance_sheet_fx import (
    CbBalanceSheetFxConfig,
    align_and_z,
    cb_balance_sheet_factor_returns,
    trailing_z,
    usd_or_fx_tilt_weights,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_cb_series_map_covers_g3():
    for ccy in ("USD", "EUR", "JPY"):
        assert ccy in CB_ASSETS_SERIES
    assert CB_ASSETS_SERIES["USD"][0] == "WALCL"
    assert CB_ASSETS_SERIES["EUR"][0] == "ECBASSETSW"
    assert CB_ASSETS_SERIES["JPY"][0] == "JPNASSETS"


def test_weekly_pub_lag_shifts_known_date():
    raw = load_cb_assets("USD", weekly_pub_lag_days=0, download=True, force=False)
    lagged = load_cb_assets(
        "USD", weekly_pub_lag_days=DEFAULT_WEEKLY_PUB_LAG_DAYS, download=False
    )
    assert not raw.empty and not lagged.empty
    r0 = raw.dropna().index.min()
    l0 = lagged.dropna().index.min()
    delta_days = (l0 - r0).days
    assert 6 <= delta_days <= 8


def test_yoy_growth_periods_by_frequency():
    idx = pd.date_range("2018-01-03", periods=120, freq="W-WED", tz="UTC")
    s = pd.Series(np.linspace(100.0, 200.0, 120), index=idx, name="USD")
    s.attrs["frequency"] = "weekly"
    g = yoy_growth(s)
    assert g.attrs["yoy_periods"] == 52
    # After 52 weeks, growth should be finite
    assert g.dropna().shape[0] >= 60


def test_us_peer_diff_sign():
    idx = pd.date_range("2020-01-01", periods=10, freq="W-WED", tz="UTC")
    panel = pd.DataFrame(
        {"USD": np.full(10, 0.20), "EUR": np.full(10, 0.05), "JPY": np.full(10, 0.05)},
        index=idx,
    )
    diff = us_vs_peer_yoy_diff(panel)
    assert float(diff.iloc[-1]) == pytest.approx(0.15, abs=1e-9)


def test_pb_tilt_long_foreign_when_z_high():
    """High Fed BS z → portfolio-balance → long foreign (short EURUSD)."""
    idx = pd.date_range("2020-01-01", periods=100, freq="B", tz="UTC")
    z = pd.Series(0.0, index=idx)
    z.iloc[-20:] = 2.0
    cfg = CbBalanceSheetFxConfig(signal_lag=0, binary_tilt=True, usd_tilt=0.5)
    w = usd_or_fx_tilt_weights(z, ["EURUSD", "USDJPY"], cfg=cfg, mode="pb")
    # Long foreign on EURUSD = positive EURUSD weight when PB fires
    assert float(w["EURUSD"].iloc[-1]) > 0
    # Long foreign on USDJPY = negative USDJPY (short USD)
    assert float(w["USDJPY"].iloc[-1]) < 0


def test_haven_tilt_opposite_of_pb():
    idx = pd.date_range("2020-01-01", periods=50, freq="B", tz="UTC")
    z = pd.Series(2.0, index=idx)
    cfg = CbBalanceSheetFxConfig(signal_lag=0, usd_tilt=0.5)
    w_pb = usd_or_fx_tilt_weights(z, ["EURUSD"], cfg=cfg, mode="pb")
    w_hv = usd_or_fx_tilt_weights(z, ["EURUSD"], cfg=cfg, mode="haven")
    assert float(w_pb["EURUSD"].iloc[-1]) == pytest.approx(
        -float(w_hv["EURUSD"].iloc[-1]), abs=1e-12
    )


def test_no_lookahead_walcl_shock():
    """Mutating future WALCL YoY must not change earlier factor returns."""
    idx = pd.date_range("2015-01-01", periods=2000, freq="B", tz="UTC")
    rng = np.random.default_rng(11)
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
    w_idx = pd.date_range("2014-01-01", periods=400, freq="W-WED", tz="UTC")
    yoy = pd.Series(rng.normal(0.05, 0.08, 400), index=w_idx, name="USD")
    panel = pd.DataFrame(
        {
            "USD": yoy,
            "EUR": yoy * 0.5 + 0.02,
            "JPY": yoy * 0.3,
        }
    )
    cfg = CbBalanceSheetFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = cb_balance_sheet_factor_returns(
        ret,
        walcl_yoy=yoy,
        walcl_level=pd.Series(np.linspace(1e6, 2e6, 400), index=w_idx, name="USD"),
        us_peer_diff=us_vs_peer_yoy_diff(panel),
        yoy_panel=panel,
        cfg=cfg,
    )
    assert "walcl_pb_fx" in f1
    assert "bs_diff_pb_fx" in f1
    assert "bs_peer_xs" in f1
    assert "bs_ew" in f1

    yoy2 = yoy.copy()
    yoy2.iloc[-30:] = yoy2.iloc[-30:] + 0.5
    panel2 = panel.copy()
    panel2["USD"] = yoy2
    f2 = cb_balance_sheet_factor_returns(
        ret,
        walcl_yoy=yoy2,
        walcl_level=pd.Series(np.linspace(1e6, 2e6, 400), index=w_idx, name="USD"),
        us_peer_diff=us_vs_peer_yoy_diff(panel2),
        yoy_panel=panel2,
        cfg=cfg,
    )
    # Shock is on the last 30 weekly points (~last ~7 months of w_idx).
    # Cut well before that so ffill of the shocked tail cannot leak.
    cut = w_idx[-80]  # ~18 months before weekly end
    for name in ("walcl_pb_fx", "bs_diff_pb_fx", "bs_peer_xs"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_align_and_z_applies_signal_lag():
    idx = pd.date_range("2020-01-01", periods=300, freq="B", tz="UTC")
    macro = pd.Series(
        np.linspace(0.0, 1.0, 100),
        index=pd.date_range("2019-01-01", periods=100, freq="W-WED", tz="UTC"),
        name="WALCL",
    )
    cfg0 = CbBalanceSheetFxConfig(signal_lag=0, z_window=60, min_periods=20)
    cfg1 = CbBalanceSheetFxConfig(signal_lag=1, z_window=60, min_periods=20)
    z0 = align_and_z(macro, idx, cfg=cfg0)
    z1 = align_and_z(macro, idx, cfg=cfg1)
    # signal_lag=1 shifts by one trading day vs lag=0
    common = z0.dropna().index.intersection(z1.dropna().index)
    assert len(common) > 50
    # Most dates: z1[t] == z0[t-1]
    shifted = z0.shift(1)
    overlap = common.intersection(shifted.dropna().index)
    assert np.allclose(
        z1.loc[overlap].values, shifted.loc[overlap].values, equal_nan=True, atol=1e-12
    )


def test_loader_smoke_and_coverage():
    panel = load_cb_yoy_panel(("USD", "EUR", "JPY"), download=True)
    assert panel.shape[1] >= 3
    assert "USD" in panel.columns
    assert panel.dropna(how="all").shape[0] > 100
    cov = cb_coverage(panel)
    assert len(cov) >= 3
    assert (cov["n_obs"] > 0).all()
    # BS/GDP optional path
    ratio = load_bs_gdp_ratio("USD", download=True)
    assert len(ratio) > 40
