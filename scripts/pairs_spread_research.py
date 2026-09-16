#!/usr/bin/env python3
"""Pairs / spread mean-reversion across correlated FX (IS-only param pick).

Builds a residual series z = log(A) - hedge*log(B), trades synthetic OHLC of the
spread with mean-reversion, then maps P&L onto a dollar equity curve with FTMO
gates. Holdout never used for hedge ratio or thresholds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mt5_swing.backtest.metrics import compute_metrics
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.strategies.base import Signal

HIST = ROOT / "data" / "history"
REPORTS = ROOT / "reports"

PAIRS = [
    ("EURUSD", "GBPUSD"),
    ("AUDUSD", "NZDUSD"),
    ("EURUSD", "USDCHF"),
    ("AUDCAD", "NZDCAD"),
    ("EURJPY", "GBPJPY"),
]


def load_h4(sym: str) -> pd.DataFrame:
    p = HIST / f"{sym}_H4.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return load_ohlc_csv(p, symbol=sym, timeframe="H4")


def align_pair(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"a": a["close"], "b": b["close"], "ha": a["high"], "la": a["low"],
                       "hb": b["high"], "lb": b["low"]}).dropna()
    return df


def hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    # OLS on log prices, causal IS only
    ly, lx = np.log(y.values), np.log(x.values)
    var = np.var(lx)
    if var < 1e-18:
        return 1.0
    return float(np.cov(ly, lx)[0, 1] / var)


def residual(df: pd.DataFrame, beta: float) -> pd.Series:
    return np.log(df["a"]) - beta * np.log(df["b"])


def zscore(res: pd.Series, win: int) -> pd.Series:
    mu = res.rolling(win, min_periods=win).mean()
    sd = res.rolling(win, min_periods=win).std(ddof=0)
    return (res - mu) / sd.replace(0, np.nan)


def backtest_spread(
    z: pd.Series,
    *,
    entry: float = 2.0,
    exit_z: float = 0.25,
    risk_frac: float = 0.005,
    initial: float = 100_000.0,
    cost_bps: float = 2.0,
) -> pd.Series:
    """Dollar equity from z-score MR; signal_lag=1 via z already from lagged closes conceptually.
    We shift signals by 1 bar for execution."""
    sig = pd.Series(0, index=z.index, dtype=int)
    last = 0
    for i in range(len(z)):
        zi = z.iloc[i]
        if zi != zi:
            last = 0
        elif zi > entry:
            last = -1  # short spread
        elif zi < -entry:
            last = 1
        elif abs(zi) < exit_z:
            last = 0
        sig.iloc[i] = last
    # lag signal 1 bar
    pos = sig.shift(1).fillna(0)
    # P&L ≈ -pos * d(z) scaled — use residual diff as return proxy
    # Better: use change in residual as PnL units
    dz = z.diff().fillna(0)
    # When long residual, profit when residual rises
    raw = pos * dz
    # Scale to ~risk: target risk_frac of equity per unit z-move of ~1
    eq = initial
    curve = []
    for i in range(len(raw)):
        # cost on position change
        if i > 0 and pos.iloc[i] != pos.iloc[i - 1]:
            eq *= 1.0 - cost_bps * 1e-4
        # map dz to return: risk_frac * dz (clipped)
        r = float(raw.iloc[i]) * risk_frac
        r = max(-0.02, min(0.02, r))
        eq *= 1.0 + r
        curve.append(eq)
    return pd.Series(curve, index=z.index, name="equity")


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    rows = []
    for sa, sb in PAIRS:
        try:
            a, b = load_h4(sa), load_h4(sb)
        except FileNotFoundError as e:
            rows.append({"pair": f"{sa}/{sb}", "error": str(e)})
            continue
        df = align_pair(a, b)
        end = df.index.max()
        cut = end - pd.Timedelta(days=365)
        research, holdout = df.loc[df.index < cut], df.loc[df.index >= cut]
        # IS: first 60% of research for hedge + entry grid
        cut_i = int(len(research) * 0.6)
        is_df = research.iloc[:cut_i]
        oos_df = research.iloc[cut_i:]
        best = None
        for win in (40, 60, 80):
            for entry in (1.5, 2.0, 2.5):
                beta = hedge_ratio(is_df["a"], is_df["b"])
                z_is = zscore(residual(is_df, beta), win)
                # require enough excursions
                if (z_is.abs() > entry).sum() < 8:
                    continue
                # score on IS equity sharpe proxy
                eq_is = backtest_spread(z_is.dropna(), entry=entry)
                if len(eq_is) < 50:
                    continue
                rets = eq_is.pct_change().dropna()
                sharpe = float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(252 * 6))
                score = sharpe + 0.5 * float(eq_is.iloc[-1] / eq_is.iloc[0] - 1)
                cand = {"win": win, "entry": entry, "beta": beta, "score": score, "is_sharpe": sharpe}
                if best is None or score > best["score"]:
                    best = cand
        if best is None:
            rows.append({"pair": f"{sa}/{sb}", "error": "no IS candidate"})
            continue
        # OOS on remaining research with frozen params
        beta, win, entry = best["beta"], best["win"], best["entry"]
        z_oos = zscore(residual(oos_df, beta), win)
        eq_oos = backtest_spread(z_oos.dropna(), entry=entry)
        m_oos = compute_metrics(
            eq_oos, None, max_dd_gate=0.10, daily_dd_gate=0.05,
            max_loss_mode="static_initial", daily_loss_mode="ftmo_initial",
            daily_tz="Europe/Prague", initial_equity=100_000,
        )
        z_ho = zscore(residual(holdout, beta), win)
        eq_ho = backtest_spread(z_ho.dropna(), entry=entry)
        m_ho = compute_metrics(
            eq_ho, None, max_dd_gate=0.10, daily_dd_gate=0.05,
            max_loss_mode="static_initial", daily_loss_mode="ftmo_initial",
            daily_tz="Europe/Prague", initial_equity=100_000,
        )
        # monthly on holdout
        mo = eq_ho.resample("ME").last().pct_change().dropna()
        rows.append({
            "pair": f"{sa}/{sb}",
            "win": win,
            "entry": entry,
            "beta": round(beta, 4),
            "is_sharpe": round(best["is_sharpe"], 3),
            "oos_ret": round(float(m_oos.total_return), 4),
            "oos_gates": bool(m_oos.gates_pass),
            "holdout_ret": round(float(m_ho.total_return), 4),
            "holdout_gates": bool(m_ho.gates_pass),
            "holdout_mean_mo": round(float(mo.mean()), 4) if len(mo) else None,
            "holdout_pct_pos": round(float((mo > 0).mean()), 3) if len(mo) else None,
            "data_source": "approximate_non_ftmo",
        })
        print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "pairs_spread_quest.csv", index=False)
    lines = ["# Pairs spread mean-reversion quest", "", "IS-only hedge/thresholds; holdout confirmation.", ""]
    for _, row in out.iterrows():
        lines.append("- " + ", ".join(f"{k}={v}" for k, v in row.items()))
    md = lines
    (REPORTS / "pairs_spread_quest.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Wrote reports/pairs_spread_quest.md")


if __name__ == "__main__":
    main()
