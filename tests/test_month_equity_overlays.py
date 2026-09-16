"""Causal checks for month-aware / equity-curve / runup overlays."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mt5_swing.portfolio.overlays import (
    apply_equity_curve_target,
    apply_month_aware_scale,
    apply_runup_throttle,
)


def _eq_from_returns(rets: np.ndarray, start="2024-01-01", freq="D") -> pd.Series:
    idx = pd.date_range(start, periods=len(rets), freq=freq, tz="UTC")
    return pd.Series(np.cumprod(1.0 + rets) * 100_000.0, index=idx)


def test_month_aware_never_increases_vs_unit_scale():
    rng = np.random.default_rng(7)
    # Strong first month then mild
    rets = np.concatenate([np.full(31, 0.003), rng.normal(0.0002, 0.002, 200)])
    eq = _eq_from_returns(rets)
    out = apply_month_aware_scale(eq, strong_mo=0.02, after_strong=0.5, dd_trigger=0.99, after_dd=1.0)
    # Overlay scales <=1 so final return should not exceed raw when all positive-ish;
    # more rigorously: bar scales never > 1 → |scaled_r| <= |r| when r>0 growth slower or equal
    r_raw = eq.pct_change().fillna(0)
    r_out = out.pct_change().fillna(0)
    # where raw positive, scaled should be <= raw (downscale only)
    mask = r_raw > 0
    assert (r_out[mask] <= r_raw[mask] + 1e-12).all()


def test_month_aware_uses_completed_month_only():
    # Constant tiny returns except a huge return on last day of January
    idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    rets = np.full(60, 0.0001)
    # Jan 31 is index 30
    rets[30] = 0.10  # huge January close spike
    eq = pd.Series(np.cumprod(1.0 + rets) * 100_000.0, index=idx)
    out = apply_month_aware_scale(eq, strong_mo=0.02, after_strong=0.4, dd_trigger=0.99, after_dd=1.0)
    # February bars should be scaled; first few January bars (no prior month) stay ~unscaled
    r_raw = eq.pct_change()
    r_out = out.pct_change()
    # mid-January (no completed prior strong month yet with our ffill of 0)
    assert abs(r_out.iloc[10] - r_raw.iloc[10]) < 1e-12
    # mid-February should be cooled if January was strong
    feb = out.index[out.index.month == 2]
    assert len(feb) > 5
    i = out.index.get_loc(feb[5])
    assert r_out.iloc[i] < r_raw.iloc[i] - 1e-15 or abs(r_raw.iloc[i]) < 1e-15


def test_runup_throttle_causal_last_bar():
    idx = pd.date_range("2024-01-01", periods=80, freq="D", tz="UTC")
    rets = np.full(80, 0.002)
    eq = pd.Series(np.cumprod(1.0 + rets) * 100_000.0, index=idx)
    a = apply_runup_throttle(eq, trail_bars=20, runup_thresh=0.03, cool_scale=0.5)
    # Mutate last equity point's preceding path equivalently by changing only last raw return source:
    # changing last bar of *input* return shouldn't need future — throttle uses lag.
    eq2 = eq.copy()
    # bump only the final equity level abruptly (as if last return larger)
    eq2.iloc[-1] = eq2.iloc[-2] * 1.05
    b = apply_runup_throttle(eq2, trail_bars=20, runup_thresh=0.03, cool_scale=0.5)
    # The scale applied to last return uses trail_lag at t-1, identical for a and b
    # until last bar's return differs — scale series at last bar equal
    # Check penultimate return identical
    assert abs(a.pct_change().iloc[-2] - b.pct_change().iloc[-2]) < 1e-12


def test_equity_curve_target_hi_cap():
    rng = np.random.default_rng(1)
    rets = rng.normal(0.001, 0.01, 400)
    eq = _eq_from_returns(rets)
    out = apply_equity_curve_target(eq, target_mo_vol=0.05, lookback_months=4, lo=0.25, hi=1.0)
    r_raw = eq.pct_change().fillna(0)
    r_out = out.pct_change().fillna(0)
    mask = r_raw > 0
    assert (r_out[mask] <= r_raw[mask] + 1e-9).all()


def test_mtd_gain_clip_caps_after_tau_and_never_leverages():
    from mt5_swing.portfolio.overlays import apply_mtd_gain_clip

    # Steady +0.5%/day → MTD exceeds 3% quickly within a month
    rets = np.full(40, 0.005)
    eq = _eq_from_returns(rets)
    out = apply_mtd_gain_clip(eq, tau=0.03, after_clip=0.0)
    r_raw = eq.pct_change().fillna(0)
    r_out = out.pct_change().fillna(0)
    # Never leverages: |scaled| <= |raw| when raw>0
    mask = r_raw > 0
    assert (r_out[mask] <= r_raw[mask] + 1e-12).all()
    # After MTD crosses tau, some later same-month bars should be zeroed
    assert (r_out.abs() < 1e-15).sum() >= 3


def test_mtd_gain_clip_is_causal_no_future_peek():
    from mt5_swing.portfolio.overlays import apply_mtd_gain_clip

    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, 90)
    eq = _eq_from_returns(rets)
    # Mutating a future bar must not change past scaled returns
    out1 = apply_mtd_gain_clip(eq, tau=0.02, after_clip=0.25)
    eq2 = eq.copy()
    eq2.iloc[-1] = eq2.iloc[-1] * 1.5
    out2 = apply_mtd_gain_clip(eq2, tau=0.02, after_clip=0.25)
    assert np.allclose(out1.iloc[:-1].pct_change().fillna(0), out2.iloc[:-1].pct_change().fillna(0))


def test_mtd_loss_halt_never_leverages():
    from mt5_swing.portfolio.overlays import apply_mtd_loss_halt

    rng = np.random.default_rng(3)
    rets = rng.normal(0.0005, 0.01, 90)
    eq = _eq_from_returns(rets)
    out = apply_mtd_loss_halt(eq, tau=0.02, after_halt=0.0)
    r_raw = eq.pct_change().fillna(0)
    r_out = out.pct_change().fillna(0)
    mask = r_raw > 0
    assert (r_out[mask] <= r_raw[mask] + 1e-12).all()


def test_mtd_loss_halt_is_causal_no_future_peek():
    from mt5_swing.portfolio.overlays import apply_mtd_loss_halt

    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.012, 90)
    eq = _eq_from_returns(rets)
    out1 = apply_mtd_loss_halt(eq, tau=0.015, after_halt=0.25)
    eq2 = eq.copy()
    eq2.iloc[-1] = eq2.iloc[-1] * 0.5  # mutate future last bar
    out2 = apply_mtd_loss_halt(eq2, tau=0.015, after_halt=0.25)
    assert np.allclose(
        out1.iloc[:-1].pct_change().fillna(0),
        out2.iloc[:-1].pct_change().fillna(0),
    )


def test_mtd_loss_halt_flattens_after_large_intra_month_drop():
    from mt5_swing.portfolio.overlays import apply_mtd_loss_halt

    # Steady mild then a large drop early in month → later bars flattened
    rets = np.concatenate([np.full(5, 0.001), np.array([-0.04]), np.full(25, 0.002)])
    eq = _eq_from_returns(rets)
    out = apply_mtd_loss_halt(eq, tau=0.02, after_halt=0.0)
    r_raw = eq.pct_change().fillna(0)
    r_out = out.pct_change().fillna(0)
    # After the drop, MTD through prior bar < -tau → later same-month bars ~0
    assert (r_out.iloc[8:].abs() < 1e-14).sum() >= 5
    # Drop bar itself still applies (scale decided on prior MTD)
    assert abs(r_out.iloc[6] - r_raw.iloc[6]) < 1e-12 or r_out.iloc[6] == 0.0


def test_after_loss_throttle_losing_jan_scales_feb_not_jan():
    from mt5_swing.portfolio.overlays import apply_after_loss_throttle

    idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    rets = np.full(60, 0.001)
    # Make January a clear loser: large negative mid-month
    rets[10] = -0.08
    eq = pd.Series(np.cumprod(1.0 + rets) * 100_000.0, index=idx)
    out = apply_after_loss_throttle(eq, after_loss=0.5, lo=0.25)
    r_raw = eq.pct_change().fillna(0)
    r_out = out.pct_change().fillna(0)
    # Mid-January (no completed prior losing month): unscaled
    assert abs(r_out.iloc[5] - r_raw.iloc[5]) < 1e-12
    # Mid-February should be scaled down vs raw when raw > 0
    feb = out.index[out.index.month == 2]
    assert len(feb) > 5
    i = out.index.get_loc(feb[5])
    if r_raw.iloc[i] > 1e-15:
        assert r_out.iloc[i] < r_raw.iloc[i] - 1e-15
        assert abs(r_out.iloc[i] - 0.5 * r_raw.iloc[i]) < 1e-10


def test_after_loss_throttle_causal_mutate_feb_last():
    from mt5_swing.portfolio.overlays import apply_after_loss_throttle

    idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    rets = np.full(60, 0.001)
    rets[10] = -0.08  # losing January
    eq = pd.Series(np.cumprod(1.0 + rets) * 100_000.0, index=idx)
    out1 = apply_after_loss_throttle(eq, after_loss=0.5, lo=0.25)
    eq2 = eq.copy()
    # Mutate last February bar — must not change January scaled returns
    feb_mask = eq2.index.month == 2
    feb_idx = eq2.index[feb_mask]
    eq2.loc[feb_idx[-1]] = eq2.loc[feb_idx[-1]] * 1.2
    out2 = apply_after_loss_throttle(eq2, after_loss=0.5, lo=0.25)
    jan1 = out1.loc[out1.index.month == 1].pct_change().fillna(0)
    jan2 = out2.loc[out2.index.month == 1].pct_change().fillna(0)
    assert np.allclose(jan1, jan2)
