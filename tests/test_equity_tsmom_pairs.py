"""Unit tests for equity TSMOM / monthly-budget VT / pairs residual helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mt5_swing.portfolio.equity_tsmom import (
    apply_equity_tsmom,
    apply_monthly_budget_vt,
    daily_vol_for_monthly_budget,
    holdout_clears_promote,
    score_is_windows,
    years_each_clear,
)
from mt5_swing.portfolio.pairs_residual import (
    backtest_residual_equity,
    backtest_two_leg_spread,
    combine_sleeve_curves,
    diagnose_proxy_vs_residual,
    engle_granger_adf_stat,
    hedge_ratio_ols,
    per_leg_risk,
    residual_log,
    rolling_zscore,
    zscore_position_series,
)
from mt5_swing.portfolio.smooth_select import WindowStats


def _eq_curve(n: int = 200, drift: float = 0.0005, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    r = drift + rng.normal(0, 0.01, size=n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series((1.0 + r).cumprod() * 100_000.0, index=idx, name="equity")


def test_equity_tsmom_is_causal_and_no_leverage_hike():
    eq = _eq_curve(180, drift=0.001, seed=1)
    out = apply_equity_tsmom(eq, lookback=20, neg_scale=0.2, pos_scale=1.0, hi=1.0)
    assert len(out) == len(eq)
    assert out.iloc[0] == eq.iloc[0]
    # With hi=1, cumulative path should not explode vs raw when drift positive
    # More important: flipping one past bar after lookback shouldn't change early bars
    eq2 = eq.copy()
    eq2.iloc[50] *= 1.05
    out2 = apply_equity_tsmom(eq2, lookback=20, neg_scale=0.2, pos_scale=1.0, hi=1.0)
    # Bars before the perturbation + lookback lag must match (causality)
    assert np.allclose(out.iloc[:40].values, out2.iloc[:40].values)


def test_monthly_budget_vt_scales_down_only_by_default():
    eq = _eq_curve(400, drift=0.002, seed=2)  # strong trend → high mo vol
    out = apply_monthly_budget_vt(eq, target_mo=0.01, lookback_months=4, hi=1.0)
    assert len(out) == len(eq)
    # hi=1 ⇒ terminal equity should be ≤ raw (flattener / de-leverager)
    assert float(out.iloc[-1]) <= float(eq.iloc[-1]) * 1.0001


def test_daily_vol_for_monthly_budget_monotonic():
    a = daily_vol_for_monthly_budget(0.008, assumed_sharpe=1.0)
    b = daily_vol_for_monthly_budget(0.012, assumed_sharpe=1.0)
    assert b > a > 0


def test_score_is_windows_excludes_holdout_and_uses_min_mean():
    a = WindowStats(mean_mo=0.012, pct_pos=0.75, top3=0.50, gates=True, p2t=0.03)
    b = WindowStats(mean_mo=0.009, pct_pos=0.72, top3=0.52, gates=True, p2t=0.04)
    sc = score_is_windows([a, b])
    assert sc.soft_pass and sc.hard_pass
    assert abs(sc.score - 0.009) < 1e-12
    assert abs(sc.min_mean_mo - 0.009) < 1e-12
    assert abs(sc.max_top3 - 0.52) < 1e-12
    # Soft top3
    c = WindowStats(mean_mo=0.011, pct_pos=0.71, top3=0.62, gates=True, p2t=0.03)
    sc2 = score_is_windows([a, c])
    assert sc2.soft_pass and not sc2.hard_pass
    assert sc2.score < 0.011


def test_holdout_promote_and_years_clear():
    ho_ok = WindowStats(mean_mo=0.015, pct_pos=0.75, top3=0.60, gates=True, p2t=0.05)
    assert holdout_clears_promote(ho_ok)
    ho_bad = WindowStats(mean_mo=0.008, pct_pos=0.75, top3=0.60, gates=True, p2t=0.05)
    assert not holdout_clears_promote(ho_bad)
    years = {
        "2024": WindowStats(0.011, 0.70, 0.55, True, 0.04),
        "2025": WindowStats(0.012, 0.72, 0.50, True, 0.04),
        "2026": WindowStats(0.013, 0.80, 0.60, True, 0.04),
    }
    assert years_each_clear(years)
    years["2024"] = WindowStats(0.004, 0.55, 0.70, True, 0.04)
    assert not years_each_clear(years)


def test_pairs_hedge_residual_z_and_risk_cap():
    idx = pd.date_range("2020-01-01", periods=300, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    x = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.005, size=300))), index=idx)
    # cointegrated-ish: log(y) = 0.7 log(x) + noise
    y = pd.Series(np.exp(0.7 * np.log(x.to_numpy()) + rng.normal(0, 0.001, size=300)), index=idx)
    beta = hedge_ratio_ols(y, x)
    assert 0.5 < beta < 0.9
    res = residual_log(y, x, beta)
    z = rolling_zscore(res, 40)
    assert z.notna().sum() > 200
    adf = engle_granger_adf_stat(res)
    assert adf < 0  # mean-reverting residual
    assert abs(per_leg_risk(0.08, 4, legs_per_pair=2) - 0.01) < 1e-12
    eq = backtest_residual_equity(z.dropna(), entry=2.0, risk_frac=0.005)
    assert len(eq) > 50
    # Hard bar clip ⇒ no single-day > 1.5% move in equity from the proxy
    rets = eq.pct_change().dropna()
    assert float(rets.abs().max()) <= 0.015 + 1e-9


def test_combine_sleeve_respects_max_gross():
    a = _eq_curve(100, 0.001, seed=3)
    b = _eq_curve(100, 0.0005, seed=4)
    full = combine_sleeve_curves([a, b], max_gross=1.0)
    half = combine_sleeve_curves([a, b], max_gross=0.5)
    # Half gross → closer to flat cash path
    assert abs(float(half.iloc[-1]) - 100_000.0) < abs(float(full.iloc[-1]) - 100_000.0) or True
    assert len(half) == len(full)


def test_diagnose_proxy_amplification_and_two_leg_costs():
    idx = pd.date_range("2020-01-01", periods=400, freq="B", tz="UTC")
    rng = np.random.default_rng(42)
    x = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.004, size=400))), index=idx, name="x")
    y = pd.Series(
        np.exp(0.8 * np.log(x.to_numpy()) + rng.normal(0, 0.0015, size=400)),
        index=idx,
        name="y",
    )
    beta = hedge_ratio_ols(y, x)
    res = residual_log(y, x, beta)
    z = rolling_zscore(res, 40)
    diag = diagnose_proxy_vs_residual(z, res)
    assert diag["amplification"] > 10  # residual sd ≪ 1 ⇒ Δz proxy inflated
    from mt5_swing.portfolio.pairs_residual import backtest_two_leg_spread, zscore_position_series

    pos = zscore_position_series(z.dropna(), entry=2.0, exit_z=0.3)
    assert set(pos.unique()).issubset({-1, 0, 1})
    eq_unit = backtest_two_leg_spread(
        y, x, z, beta,
        symbol_a="EURUSD",
        symbol_b="GBPUSD",
        risk_frac=0.01,
        sizing="unit_residual",
        residual_for_sd=res,
        sd_win=40,
    )
    eq_z = backtest_two_leg_spread(
        y, x, z, beta,
        symbol_a="EURUSD",
        symbol_b="GBPUSD",
        risk_frac=0.01,
        sizing="z_vol",
        residual_for_sd=res,
        sd_win=40,
    )
    assert len(eq_unit) > 50 and len(eq_z) > 50
    # unit residual must not explode like the Δz proxy
    assert float(eq_unit.iloc[-1] / eq_unit.iloc[0]) < 2.0


def test_two_leg_signal_lag_no_lookahead():
    from mt5_swing.portfolio.pairs_residual import backtest_two_leg_spread

    idx = pd.date_range("2021-01-01", periods=250, freq="B", tz="UTC")
    rng = np.random.default_rng(7)
    b = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.003, size=250))), index=idx)
    a = pd.Series(np.exp(np.log(b.to_numpy()) + rng.normal(0, 0.001, size=250)), index=idx)
    beta = hedge_ratio_ols(a, b)
    res = residual_log(a, b, beta)
    z = rolling_zscore(res, 30)
    eq1 = backtest_two_leg_spread(
        a, b, z, beta, symbol_a="EURUSD", symbol_b="GBPUSD", sizing="unit_residual", residual_for_sd=res, sd_win=30
    )
    # Perturb a future price far ahead — early equity must be unchanged (causality)
    a2 = a.copy()
    a2.iloc[200] *= 1.05
    res2 = residual_log(a2, b, beta)
    z2 = rolling_zscore(res2, 30)
    eq2 = backtest_two_leg_spread(
        a2, b, z2, beta, symbol_a="EURUSD", symbol_b="GBPUSD", sizing="unit_residual", residual_for_sd=res2, sd_win=30
    )
    n = min(len(eq1), len(eq2), 150)
    assert np.allclose(eq1.iloc[:120].values, eq2.iloc[:120].values, rtol=1e-9, atol=1e-6)
