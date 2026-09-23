"""Tests for yield-curve / term-structure FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_yields import (
    FRED_LT_GOVT_MONTHLY,
    load_curve_panels,
    load_lt_yield_panel,
    yield_coverage_table,
)
from mt5_swing.strategies.yield_curve_fx import (
    YieldCurveFxConfig,
    apply_signal_lag,
    prepare_curve_scores,
    trailing_z,
    yield_curve_factor_returns,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_trailing_z_mean_near_zero():
    idx = pd.date_range("2010-01-01", periods=80, freq="MS", tz="UTC")
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"EUR": rng.normal(0, 1, 80)}, index=idx)
    z = trailing_z(x, lookback=36, min_periods=24)
    tail = z.dropna().iloc[-20:]
    assert abs(float(tail["EUR"].mean())) < 0.5


def test_slope_xs_long_steep_currency():
    """Currency with steeper slope_diff should get positive score / long weight."""
    idx = pd.date_range("2015-01-01", periods=40, freq="MS", tz="UTC")
    # EUR steep, GBP flat
    slope_diff = pd.DataFrame(
        {
            "EUR": np.linspace(0.5, 2.0, 40),
            "GBP": np.linspace(-1.0, -0.5, 40),
            "AUD": np.zeros(40),
            "CAD": np.zeros(40),
        },
        index=idx,
    )
    cfg = YieldCurveFxConfig(signal_lag=0, n_long=1, n_short=1)
    scores = prepare_curve_scores(
        slope_diff,
        lt_diff=slope_diff * 0.5,
        st_diff=slope_diff * 0.1,
        cfg=cfg,
    )
    last = scores["curve_slope_xs"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])


def test_slope_x_carry_no_lookahead():
    """Mutating future slope must not change earlier interaction scores."""
    idx = pd.date_range("2012-01-01", periods=90, freq="MS", tz="UTC")
    rng = np.random.default_rng(3)
    slope = pd.DataFrame(
        {c: rng.normal(0, 0.5, 90) for c in ("EUR", "GBP", "AUD", "CAD")},
        index=idx,
    )
    st = pd.DataFrame(
        {c: rng.normal(0, 0.3, 90) for c in ("EUR", "GBP", "AUD", "CAD")},
        index=idx,
    )
    cfg = YieldCurveFxConfig(signal_lag=1, min_periods=24, z_window=36)
    s1 = prepare_curve_scores(slope, slope, st, cfg=cfg)["slope_x_carry"]
    slope2 = slope.copy()
    slope2.iloc[-5:] += 5.0
    s2 = prepare_curve_scores(slope2, slope2, st, cfg=cfg)["slope_x_carry"]
    cut = -12
    a = s1.iloc[:cut].dropna(how="all")
    b = s2.iloc[:cut].dropna(how="all")
    common = a.index.intersection(b.index)
    assert len(common) > 20
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True)


def test_apply_signal_lag_shifts_availability():
    idx = pd.date_range("2020-01-01", periods=6, freq="MS", tz="UTC")
    s = pd.DataFrame({"EUR": np.arange(6.0)}, index=idx)
    lagged = apply_signal_lag(s, 1)
    # Value 0 observed Jan becomes available Feb
    assert lagged.loc["2020-02-01", "EUR"] == 0.0
    assert np.isnan(lagged.iloc[0]["EUR"])


def test_factor_returns_smoke_and_causal():
    idx = pd.date_range("2016-01-01", periods=1800, freq="B", tz="UTC")
    rng = np.random.default_rng(7)
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
    m_idx = pd.date_range("2015-01-01", periods=120, freq="MS", tz="UTC")
    slope_diff = pd.DataFrame(
        {c: rng.normal(0, 0.4, 120) for c in ("EUR", "GBP", "AUD", "NZD", "JPY", "CAD", "CHF")},
        index=m_idx,
    )
    lt_diff = slope_diff + rng.normal(0, 0.1, slope_diff.shape)
    st_diff = slope_diff * 0.5 + rng.normal(0, 0.1, slope_diff.shape)
    cfg = YieldCurveFxConfig(signal_lag=1, cost_bps_side=1.5)
    factors = yield_curve_factor_returns(ret, slope_diff, lt_diff, st_diff, cfg=cfg)
    assert "curve_slope_xs" in factors
    assert "curve_lt_xs" in factors
    assert "slope_x_carry" in factors
    assert "curve_ew" in factors
    assert "uip_st_xs" in factors
    # Causal: mutating last month of slope should not change early returns
    r1 = factors["curve_slope_xs"].copy()
    slope2 = slope_diff.copy()
    slope2.iloc[-3:] += 10.0
    r2 = yield_curve_factor_returns(ret, slope2, lt_diff, st_diff, cfg=cfg)["curve_slope_xs"]
    cut = r1.index[-60]
    a = r1.loc[:cut].dropna()
    b = r2.loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 200
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


@pytest.mark.skipif(
    not (MACRO / "fred_IRLTLT01USM156N.csv").exists()
    and not (MACRO / f"fred_{FRED_LT_GOVT_MONTHLY['USD']}.csv").exists(),
    reason="FRED LT yield CSV not cached",
)
def test_load_lt_panel_from_cache():
    # May download if missing — prefer cache
    lt = load_lt_yield_panel(download=True, pub_lag_months=1)
    assert "USD" in lt.columns
    assert "EUR" in lt.columns
    assert lt.attrs.get("pub_lag_months") == 1
    # After 1m pub lag, last raw observation is not usable at its calendar month
    assert len(lt.dropna(how="all")) > 100


@pytest.mark.skipif(
    not any(MACRO.glob("fred_IRLTLT01*.csv")),
    reason="No LT yield CSVs cached yet",
)
def test_curve_panels_slope_definition():
    panels = load_curve_panels(download=True, pub_lag_months=1)
    for key in ("lt", "st", "slope", "slope_diff", "lt_diff", "st_diff"):
        assert key in panels
        assert not panels[key].empty
    # slope ≈ lt − st on overlapping non-null cells
    common = panels["lt"].columns.intersection(panels["st"].columns)
    sample = panels["lt"][common].dropna(how="all").iloc[-20:]
    st_al = panels["st"][common].reindex(sample.index)
    expected = sample - st_al
    got = panels["slope"][common].reindex(sample.index)
    mask = expected.notna() & got.notna()
    assert np.allclose(expected.values[mask.values], got.values[mask.values], atol=1e-9)


def test_coverage_table_shape():
    # download=False may still work if files exist; else may fail per-row
    try:
        cov = yield_coverage_table(download=False)
    except Exception:
        cov = yield_coverage_table(download=True)
    assert "currency" in cov.columns
    assert "lt_series" in cov.columns
    assert set(FRED_LT_GOVT_MONTHLY.keys()).issubset(set(cov["currency"]))
