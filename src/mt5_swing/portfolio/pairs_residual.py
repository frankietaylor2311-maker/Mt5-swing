"""Cointegrated FX residual / pairs helpers with hard risk caps.

Hedge ratios and entry thresholds must be fit on IS only. Holdout is
confirmation. Residual trading targets monthly distribution (mean_mo / %pos /
top3), not peak total return.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairSpec:
    a: str
    b: str
    timeframe: str = "D1"


DEFAULT_PAIRS: tuple[PairSpec, ...] = (
    PairSpec("EURUSD", "GBPUSD"),
    PairSpec("AUDUSD", "NZDUSD"),
    PairSpec("EURUSD", "USDCHF"),
    PairSpec("AUDCAD", "NZDCAD"),
    PairSpec("EURJPY", "GBPJPY"),
    PairSpec("EURCHF", "USDCHF"),
    PairSpec("GBPCAD", "USDCAD"),
    PairSpec("EURAUD", "AUDUSD"),
)


def hedge_ratio_ols(y: pd.Series, x: pd.Series) -> float:
    """OLS beta of log(y) on log(x); fail-closed to 1.0."""
    both = pd.concat([y.rename("y"), x.rename("x")], axis=1, sort=True).dropna()
    if len(both) < 30:
        return 1.0
    ly = np.log(both["y"].to_numpy(dtype=float))
    lx = np.log(both["x"].to_numpy(dtype=float))
    var = float(np.var(lx))
    if var < 1e-18:
        return 1.0
    return float(np.cov(ly, lx)[0, 1] / var)


def residual_log(a: pd.Series, b: pd.Series, beta: float) -> pd.Series:
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1, sort=True).dropna()
    return (np.log(both["a"]) - float(beta) * np.log(both["b"])).rename("residual")


def rolling_zscore(res: pd.Series, win: int) -> pd.Series:
    mu = res.rolling(int(win), min_periods=int(win)).mean()
    sd = res.rolling(int(win), min_periods=int(win)).std(ddof=0)
    return ((res - mu) / sd.replace(0, np.nan)).rename("z")


def engle_granger_adf_stat(res: pd.Series) -> float:
    """Rough ADF t-stat on residual (no lag selection) — research screen only.

    More negative ⇒ stronger mean-reversion evidence. Returns +inf if too short.
    """
    r = res.dropna()
    if len(r) < 50:
        return float("inf")
    y = r.diff().dropna()
    x = r.shift(1).loc[y.index]
    # Δr_t = γ r_{t-1} + ε
    x = x.to_numpy(dtype=float)
    y = y.to_numpy(dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    varx = float(np.dot(x, x))
    if varx < 1e-18:
        return float("inf")
    gamma = float(np.dot(x, y) / varx)
    resid = y - gamma * x
    se = float(np.sqrt(np.dot(resid, resid) / max(len(resid) - 2, 1) / varx))
    if se < 1e-18:
        return float("inf")
    return gamma / se


def per_leg_risk(rf: float, n_pairs: int, *, legs_per_pair: int = 2) -> float:
    """Hard cap: RF split across all live legs (no hike)."""
    n = max(int(n_pairs) * int(legs_per_pair), 1)
    return float(rf) / float(n)


def pair_sleeve_weight(n_pairs: int, *, sleeve_budget: float = 0.30) -> float:
    """Equal weight within a pairs sleeve that uses at most ``sleeve_budget`` of book."""
    n = max(int(n_pairs), 1)
    return float(sleeve_budget) / float(n)


def backtest_residual_equity(
    z: pd.Series,
    *,
    entry: float = 2.0,
    exit_z: float = 0.30,
    risk_frac: float = 0.005,
    initial: float = 100_000.0,
    cost_bps: float = 2.0,
    max_abs_bar_ret: float = 0.015,
) -> pd.Series:
    """Dollar equity from z-score MR with signal_lag=1 and hard per-bar return clip.

    ``risk_frac`` is the hard risk map from Δz → return (research proxy). Real
    two-leg fills should use the bar engine when promoting; this path is for
    fast IS screening of monthly distribution.
    """
    if z is None or len(z) < 40:
        return pd.Series(dtype=float)
    sig = pd.Series(0, index=z.index, dtype=int)
    last = 0
    for i in range(len(z)):
        zi = z.iloc[i]
        if zi != zi:
            last = 0
        elif zi > float(entry):
            last = -1
        elif zi < -float(entry):
            last = 1
        elif abs(zi) < float(exit_z):
            last = 0
        sig.iloc[i] = last
    pos = sig.shift(1).fillna(0)
    dz = z.diff().fillna(0.0)
    raw = pos * dz
    eq = float(initial)
    curve = np.empty(len(raw), dtype=float)
    for i in range(len(raw)):
        if i > 0 and pos.iloc[i] != pos.iloc[i - 1]:
            eq *= 1.0 - float(cost_bps) * 1e-4
        r = float(raw.iloc[i]) * float(risk_frac)
        r = max(-float(max_abs_bar_ret), min(float(max_abs_bar_ret), r))
        eq *= 1.0 + r
        curve[i] = eq
    return pd.Series(curve, index=z.index, name="equity")


def combine_sleeve_curves(
    curves: list[pd.Series],
    *,
    weights: np.ndarray | None = None,
    initial: float = 100_000.0,
    max_gross: float = 1.0,
) -> pd.Series:
    """Weighted mix with gross exposure clipped to ``max_gross`` (≤1 ⇒ no hike)."""
    if not curves:
        return pd.Series(dtype=float)
    eq = pd.concat(curves, axis=1, sort=True).sort_index().ffill().dropna(how="any")
    if eq.empty:
        return eq
    n = eq.shape[1]
    if weights is None:
        w = np.ones(n, dtype=float) / n
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != n:
            w = np.ones(n, dtype=float) / n
        w = np.maximum(w, 0.0)
        s = w.sum()
        w = (w / s) if s > 0 else np.ones(n) / n
    # Scale so sum(w) <= max_gross
    gross = float(max(min(max_gross, 1.0), 0.0))
    w = w * gross
    norms = eq / eq.iloc[0]
    # Idle cash earns 0 — residual weight (1-gross) stays in cash
    port = (norms * w).sum(axis=1) + (1.0 - gross)
    return port * float(initial)
