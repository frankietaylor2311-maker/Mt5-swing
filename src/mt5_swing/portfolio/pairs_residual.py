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


def zscore_position_series(
    z: pd.Series,
    *,
    entry: float = 2.0,
    exit_z: float = 0.30,
) -> pd.Series:
    """Causal MR position from z (+1 long residual, -1 short). No lag yet."""
    if z is None or len(z) == 0:
        return pd.Series(dtype=int)
    sig = np.zeros(len(z), dtype=int)
    last = 0
    vals = z.to_numpy(dtype=float)
    for i, zi in enumerate(vals):
        if zi != zi:
            last = 0
        elif zi > float(entry):
            last = -1
        elif zi < -float(entry):
            last = 1
        elif abs(zi) < float(exit_z):
            last = 0
        sig[i] = last
    return pd.Series(sig, index=z.index, name="pos_raw")


def diagnose_proxy_vs_residual(
    z: pd.Series,
    residual: pd.Series,
    *,
    risk_frac: float = 0.01,
) -> dict:
    """Quantify Δz proxy amplification vs true log-residual returns.

    Proxy maps ``pos * Δz * risk_frac``; true residual map is
    ``pos * Δres * risk_frac``. Ratio mean|Δz|/mean|Δres| ≈ 1/sd(res).
    """
    both = pd.concat([z.rename("z"), residual.rename("res")], axis=1, sort=True).dropna()
    if len(both) < 40:
        return {"n": len(both), "amplification": float("nan")}
    dz = both["z"].diff().abs()
    dr = both["res"].diff().abs()
    mean_dz = float(dz.mean())
    mean_dr = float(dr.mean())
    amp = mean_dz / mean_dr if mean_dr > 1e-18 else float("inf")
    sd = float(both["res"].std())
    return {
        "n": int(len(both)),
        "mean_abs_dz": mean_dz,
        "mean_abs_dres": mean_dr,
        "amplification": amp,
        "residual_sd": sd,
        "implied_proxy_vs_unit_notional": amp,
        "note": (
            "Δz proxy ≈ true_residual_pnl * (1/rolling_sd); "
            f"amplification≈{amp:.1f}x vs unit-notional residual map at risk_frac={risk_frac}"
        ),
    }


def _round_lot(x: float, *, min_lot: float = 0.01, max_lot: float = 50.0) -> float:
    if x <= 0 or x != x:
        return 0.0
    stepped = max(float(min_lot), min(float(max_lot), round(x / float(min_lot)) * float(min_lot)))
    return float(stepped)


def backtest_two_leg_spread(
    close_a: pd.Series,
    close_b: pd.Series,
    z: pd.Series,
    beta: float,
    *,
    symbol_a: str,
    symbol_b: str,
    entry: float = 2.0,
    exit_z: float = 0.30,
    risk_frac: float = 0.01,
    initial: float = 100_000.0,
    spread_pips_a: float = 1.2,
    spread_pips_b: float = 1.4,
    commission_per_lot: float = 7.0,
    slippage_pips: float = 0.5,
    max_lot: float = 50.0,
    min_lot: float = 0.01,
    sizing: str = "z_vol",  # z_vol | unit_residual
    max_abs_bar_ret: float = 0.02,
    residual_for_sd: pd.Series | None = None,
    sd_win: int = 40,
) -> pd.Series:
    """Real two-leg FX spread MR with both legs, spreads, commission, slippage.

    Positions follow lagged z (signal_lag=1). Exits are z-exit only (no ATR
    stops — those fought MR in the prior negative two-leg basket).

    Sizing modes:
    - ``z_vol``: dollar notional on A = risk_frac * equity / rolling_sd(residual)
      so 1σ residual move ≈ risk_frac equity (matches Δz-proxy economics).
    - ``unit_residual``: notional_A = risk_frac * equity (true log-residual map;
      exposes proxy amplification as illusory when sd ≪ 1).

    Hedge: lots_b sized so |dollar log-exposure_B| = |beta| * dollar log-exposure_A.
    """
    from mt5_swing.data.symbols import get_symbol_meta, pip_value_per_lot

    if close_a is None or close_b is None or z is None or len(z) < 40:
        return pd.Series(dtype=float)

    both = pd.concat(
        [close_a.rename("a"), close_b.rename("b"), z.rename("z")],
        axis=1,
        sort=True,
    ).dropna()
    if len(both) < 40:
        return pd.Series(dtype=float)

    if residual_for_sd is None:
        residual_for_sd = residual_log(both["a"], both["b"], float(beta))
    sd_series = (
        residual_for_sd.reindex(both.index)
        .rolling(int(sd_win), min_periods=int(sd_win))
        .std(ddof=0)
        .replace(0, np.nan)
    )

    raw = zscore_position_series(both["z"], entry=entry, exit_z=exit_z)
    pos = raw.shift(1).fillna(0).astype(int)

    meta_a = get_symbol_meta(symbol_a)
    meta_b = get_symbol_meta(symbol_b)
    pip_a = meta_a.pip_size
    pip_b = meta_b.pip_size

    eq = float(initial)
    curve = np.empty(len(both), dtype=float)
    lots_a = 0.0
    lots_b = 0.0
    dir_a = 0
    dir_b = 0
    prev_pos = 0
    pa = both["a"].to_numpy(dtype=float)
    pb = both["b"].to_numpy(dtype=float)
    pos_arr = pos.to_numpy(dtype=int)
    sds = sd_series.to_numpy(dtype=float)

    def _leg_pnl_usd(direction: int, lots: float, d_px: float, px: float, sym: str, meta, pip: float) -> float:
        if direction == 0 or lots <= 0:
            return 0.0
        pv = pip_value_per_lot(sym, float(px) if px > 0 else None)
        return float(direction * lots * (d_px / pip) * pv)

    def _turn_cost(lots: float, spread_pips: float, px: float, sym: str) -> float:
        if lots <= 0:
            return 0.0
        pv = pip_value_per_lot(sym, float(px) if px > 0 else None)
        spread_slip = (0.5 * float(spread_pips) + float(slippage_pips)) * pv * lots
        comm = 0.5 * float(commission_per_lot) * lots
        return float(spread_slip + comm)

    for i in range(len(both)):
        if i > 0 and dir_a != 0 and lots_a > 0:
            d_a = pa[i] - pa[i - 1]
            d_b = pb[i] - pb[i - 1]
            pnl = _leg_pnl_usd(dir_a, lots_a, d_a, pa[i], symbol_a, meta_a, pip_a)
            pnl += _leg_pnl_usd(dir_b, lots_b, d_b, pb[i], symbol_b, meta_b, pip_b)
            r = pnl / eq if eq > 0 else 0.0
            r = max(-float(max_abs_bar_ret), min(float(max_abs_bar_ret), r))
            eq *= 1.0 + r

        target = int(pos_arr[i])
        if target != prev_pos:
            if prev_pos != 0 and lots_a > 0:
                eq -= _turn_cost(lots_a, spread_pips_a, pa[i], symbol_a)
                eq -= _turn_cost(lots_b, spread_pips_b, pb[i], symbol_b)
                lots_a = lots_b = 0.0
                dir_a = dir_b = 0
            if target != 0 and eq > 0 and pa[i] > 0 and pb[i] > 0:
                sd = sds[i]
                if sizing == "unit_residual":
                    notional = float(risk_frac) * eq
                else:
                    if sd != sd or sd < 1e-8:
                        notional = 0.0
                    else:
                        notional = float(risk_frac) * eq / float(sd)
                if notional > 0:
                    la = notional / (meta_a.contract_size * pa[i])
                    lb = abs(float(beta)) * notional / (meta_b.contract_size * pb[i])
                    la = _round_lot(la, min_lot=min_lot, max_lot=max_lot)
                    lb = _round_lot(lb, min_lot=min_lot, max_lot=max_lot)
                    if la > 0 and lb > 0:
                        dir_a = 1 if target > 0 else -1
                        dir_b = -1 if target > 0 else 1
                        lots_a, lots_b = la, lb
                        eq -= _turn_cost(lots_a, spread_pips_a, pa[i], symbol_a)
                        eq -= _turn_cost(lots_b, spread_pips_b, pb[i], symbol_b)
            if target == 0:
                prev_pos = 0
            elif lots_a > 0:
                prev_pos = target
            else:
                prev_pos = 0
                dir_a = dir_b = 0

        curve[i] = max(eq, 1.0)

    return pd.Series(curve, index=both.index, name="equity")
