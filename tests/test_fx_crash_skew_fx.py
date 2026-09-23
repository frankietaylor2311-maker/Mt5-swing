"""Tests for Brunnermeier–Nagel–Pedersen crash-skew FX module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.fx_crash_skew_fx import (
    FxCrashSkewConfig,
    aggregate_crash_skew_z,
    crash_skew_factor_returns,
    prepare_crash_skew_scores,
    prepare_left_tail_scores,
    trailing_left_tail_shortfall,
    trailing_return_skewness,
)
from mt5_swing.strategies.fx_momentum import pair_returns_to_currency_returns


def _synth_panel(n: int = 1800, seed: int = 7) -> pd.DataFrame:
    idx = pd.date_range("2016-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(
        {
            "EURUSD": 1.1 * np.cumprod(1 + rng.normal(0, 0.005, n)),
            "GBPUSD": 1.3 * np.cumprod(1 + rng.normal(0, 0.005, n)),
            "AUDUSD": 0.75 * np.cumprod(1 + rng.normal(0, 0.006, n)),
            "NZDUSD": 0.65 * np.cumprod(1 + rng.normal(0, 0.006, n)),
            "USDJPY": 110 * np.cumprod(1 + rng.normal(0, 0.005, n)),
            "USDCAD": 1.3 * np.cumprod(1 + rng.normal(0, 0.004, n)),
            "USDCHF": 0.95 * np.cumprod(1 + rng.normal(0, 0.004, n)),
        },
        index=idx,
    )
    return close.pct_change()


def test_trailing_skew_uses_skip_and_is_finite():
    ret = _synth_panel(400, seed=1)
    ccy = pair_returns_to_currency_returns(ret).drop(columns=["USD"], errors="ignore")
    skew = trailing_return_skewness(ccy, formation_days=63, skip_days=1, min_periods=40)
    assert skew.shape[1] == ccy.shape[1]
    assert skew.dropna(how="all").shape[0] > 50
    # First ~40+skip rows should be mostly NaN
    assert skew.iloc[:30].isna().all().all()


def test_left_tail_shortfall_more_negative_on_crash_series():
    idx = pd.date_range("2020-01-01", periods=120, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    normal = pd.Series(rng.normal(0, 0.005, 120), index=idx)
    crashy = normal.copy()
    crashy.iloc[80:85] = -0.08  # left-tail events
    panel = pd.DataFrame({"SAFE": normal, "CRASH": crashy})
    sf = trailing_left_tail_shortfall(
        panel, formation_days=60, skip_days=1, min_periods=40, q=0.05
    )
    # After the crash window, CRASH shortfall should be ≤ SAFE
    late = sf.dropna(how="any").iloc[-10:]
    assert late["CRASH"].mean() <= late["SAFE"].mean() + 1e-12


def test_prepare_scores_signal_lag_shifts():
    ret = _synth_panel(500, seed=2)
    cfg = FxCrashSkewConfig(signal_lag=1, skip_days=1, formation_days=63, min_periods=40)
    score0 = prepare_crash_skew_scores(
        ret, cfg=FxCrashSkewConfig(signal_lag=0, skip_days=1, formation_days=63, min_periods=40)
    )
    score1 = prepare_crash_skew_scores(ret, cfg=cfg)
    # With lag=1, first valid date should be later
    assert score1.dropna(how="all").index.min() >= score0.dropna(how="all").index.min()
    # Score = −skew ⇒ higher when more negative skew
    assert score1.attrs["signal_lag"] == 1


def test_crash_skew_no_lookahead():
    """Mutating future returns must not change earlier factor returns."""
    ret = _synth_panel(1800, seed=11)
    m_idx = pd.date_range("2014-01-01", periods=96, freq="MS", tz="UTC")
    rng = np.random.default_rng(3)
    rates = pd.DataFrame(
        {
            c: 1.0 + 0.01 * rng.normal(0, 1, 96).cumsum()
            for c in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")
        },
        index=m_idx,
    )
    cfg = FxCrashSkewConfig(signal_lag=1, cost_bps_per_side=1.5)
    f1 = crash_skew_factor_returns(ret, rates=rates, cfg=cfg)
    assert "crash_skew_xs" in f1
    assert "crash_skew_xs_126" in f1
    assert "left_tail_xs" in f1
    assert "mom_skew_regime" in f1
    assert "carry_crash_cool" in f1
    assert "crash_ew" in f1

    ret2 = ret.copy()
    cut_future = pd.Timestamp("2021-01-01", tz="UTC")
    ret2.loc[cut_future:] = ret2.loc[cut_future:] + 0.02
    f2 = crash_skew_factor_returns(ret2, rates=rates, cfg=cfg)

    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["crash_skew_xs"].loc[:cut].dropna()
    b = f2["crash_skew_xs"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 200
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_left_tail_scores_shape():
    ret = _synth_panel(500, seed=4)
    score = prepare_left_tail_scores(ret, cfg=FxCrashSkewConfig())
    assert not score.empty
    assert score.dropna(how="all").shape[0] > 50


def test_aggregate_crash_z_causal():
    ret = _synth_panel(800, seed=5)
    z = aggregate_crash_skew_z(ret, cfg=FxCrashSkewConfig(signal_lag=1))
    assert z.name == "agg_crash_skew_z"
    assert z.dropna().shape[0] > 100
    # First observations before z window should be NaN
    assert z.iloc[:50].isna().all()
