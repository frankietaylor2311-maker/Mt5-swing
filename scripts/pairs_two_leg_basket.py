#!/usr/bin/env python3
"""Real two-leg pairs MR: opposite positions on A/B from IS-fit z-score.

Uses the real bar backtester per leg (spreads/commission/ATR sizing) so costs
are realistic. Hedge ratio + entry from IS only; holdout confirmation only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mt5_swing.backtest.engine import BacktestConfig, run_backtest
from mt5_swing.backtest.metrics import compute_metrics
from mt5_swing.config import load_config
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.strategies.base import Signal

HIST = ROOT / "data" / "history"
REPORTS = ROOT / "reports"

PAIRS = [
    ("EURUSD", "GBPUSD"),
    ("AUDUSD", "NZDUSD"),
    ("EURJPY", "GBPJPY"),
    ("AUDCAD", "NZDCAD"),
]


class ZScorePairsLeg:
    """Single-leg strategy driven by shared z-score series (aligned index)."""

    name = "pairs_zscore_leg"

    def __init__(self, z: pd.Series, *, side: str, entry: float = 2.0, exit_z: float = 0.3):
        self.z = z
        self.side = side  # "a" or "b"
        self.entry = float(entry)
        self.exit_z = float(exit_z)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        z = self.z.reindex(data.index)
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        last = int(Signal.FLAT)
        for i in range(len(sig)):
            zi = z.iloc[i]
            if zi != zi:
                last = int(Signal.FLAT)
            elif zi > self.entry:
                # short residual: short A, long B
                last = int(Signal.SHORT) if self.side == "a" else int(Signal.LONG)
            elif zi < -self.entry:
                last = int(Signal.LONG) if self.side == "a" else int(Signal.SHORT)
            elif abs(zi) < self.exit_z:
                last = int(Signal.FLAT)
            sig.iloc[i] = last
        return sig.astype(int)


def hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    ly, lx = np.log(y.values), np.log(x.values)
    var = np.var(lx)
    if var < 1e-18:
        return 1.0
    return float(np.cov(ly, lx)[0, 1] / var)


def zscore(res: pd.Series, win: int) -> pd.Series:
    mu = res.rolling(win, min_periods=win).mean()
    sd = res.rolling(win, min_periods=win).std(ddof=0)
    return (res - mu) / sd.replace(0, np.nan)


def bt_for(symbol: str, cfg: dict, risk_fraction: float, max_lot: float) -> BacktestConfig:
    risk, bt = cfg.get("risk", {}), cfg.get("backtest", {})
    return BacktestConfig(
        symbol=symbol,
        initial_equity=float(bt.get("initial_equity", 100_000)),
        commission_per_lot=float(bt.get("commission_per_lot", 7.0)),
        slippage_pips=float(bt.get("slippage_pips", 0.5)),
        default_spread_pips=float(cfg.get("default_spreads_pips", {}).get(symbol, bt.get("default_spread_pips", 1.2))),
        signal_lag=1,
        max_dd=0.10,
        daily_dd=0.05,
        risk_fraction=risk_fraction,
        atr_stop_mult=2.0,
        sizing="atr",
        vol_target=True,
        max_lot=max_lot,
        max_loss_mode="static_initial",
        daily_loss_mode="ftmo_initial",
        daily_tz="Europe/Prague",
        use_atr_exits=True,
        atr_target_mult=3.0,
        no_same_bar_exit=True,
        max_hold_bars=48,
    )


def eval_pair(sa: str, sb: str, cfg: dict, risk_fraction: float = 0.04, max_lot: float = 50.0) -> dict:
    a = load_ohlc_csv(HIST / f"{sa}_H4.csv", symbol=sa, timeframe="H4")
    b = load_ohlc_csv(HIST / f"{sb}_H4.csv", symbol=sb, timeframe="H4")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    end = common.max()
    cut = end - pd.Timedelta(days=365)
    research_idx = common[common < cut]
    holdout_idx = common[common >= cut]
    is_cut = research_idx[int(len(research_idx) * 0.6)]
    is_idx = research_idx[research_idx < is_cut]
    oos_idx = research_idx[research_idx >= is_cut]

    best = None
    for win in (40, 60, 80):
        for entry in (1.75, 2.0, 2.5):
            beta = hedge_ratio(a.loc[is_idx, "close"], b.loc[is_idx, "close"])
            res = np.log(a["close"]) - beta * np.log(b["close"])
            z = zscore(res, win)
            # IS score via simple count of mean-reverting excursions (not holdout)
            z_is = z.loc[is_idx].dropna()
            if (z_is.abs() > entry).sum() < 10:
                continue
            # quick IS proxy: correlation of -sign(z) with future dz
            fut = z_is.diff().shift(-1)
            score = float((-np.sign(z_is.clip(-3, 3)) * fut).dropna().mean())
            cand = {"win": win, "entry": entry, "beta": beta, "score": score}
            if best is None or score > best["score"]:
                best = cand
    if best is None:
        return {"pair": f"{sa}/{sb}", "error": "no IS fit"}

    beta, win, entry = best["beta"], best["win"], best["entry"]
    res = np.log(a["close"]) - beta * np.log(b["close"])
    z_all = zscore(res, win)

    def port_on(idx: pd.DatetimeIndex) -> tuple[pd.Series, object]:
        # warmup 250 bars before idx start if available
        start = idx.min()
        warm_start_pos = max(0, common.get_loc(start) - 250) if start in common else 0
        if isinstance(warm_start_pos, slice):
            warm_start_pos = warm_start_pos.start or 0
        run_idx = common[warm_start_pos:]
        run_idx = run_idx[run_idx <= idx.max()]
        aa, bb = a.loc[run_idx], b.loc[run_idx]
        zz = z_all.loc[run_idx]
        rf = risk_fraction / 2
        ra = run_backtest(aa, ZScorePairsLeg(zz, side="a", entry=entry), bt_for(sa, cfg, rf, max_lot))
        rb = run_backtest(bb, ZScorePairsLeg(zz, side="b", entry=entry), bt_for(sb, cfg, rf, max_lot))
        ea, eb = ra.equity.loc[idx], rb.equity.loc[idx]
        both = pd.concat([ea, eb], axis=1, sort=True).ffill().dropna()
        norms = both / both.iloc[0]
        port = norms.mean(axis=1) * 100_000
        m = compute_metrics(
            port, None, max_dd_gate=0.10, daily_dd_gate=0.05,
            max_loss_mode="static_initial", daily_loss_mode="ftmo_initial",
            daily_tz="Europe/Prague", initial_equity=100_000,
        )
        return port, m

    # OOS research confirmation
    port_oos, m_oos = port_on(oos_idx)
    port_ho, m_ho = port_on(holdout_idx)
    mo = port_ho.resample("ME").last().pct_change().dropna()
    return {
        "pair": f"{sa}/{sb}",
        "win": win,
        "entry": entry,
        "beta": round(beta, 4),
        "oos_ret": round(float(m_oos.total_return), 4),
        "oos_gates": bool(m_oos.gates_pass),
        "oos_sharpe": round(float(m_oos.sharpe), 3),
        "holdout_ret": round(float(m_ho.total_return), 4),
        "holdout_gates": bool(m_ho.gates_pass),
        "holdout_sharpe": round(float(m_ho.sharpe), 3),
        "holdout_mean_mo": round(float(mo.mean()), 4) if len(mo) else None,
        "holdout_pct_pos": round(float((mo > 0).mean()), 3) if len(mo) else None,
        "holdout_top3_share": round(float(mo[mo > 0].nlargest(3).sum() / mo[mo > 0].sum()), 3)
        if len(mo[mo > 0]) and mo[mo > 0].sum() > 0
        else None,
        "data_source": "approximate_non_ftmo",
    }


def main() -> None:
    cfg = load_config(ROOT / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    rows = []
    for sa, sb in PAIRS:
        print("pair", sa, sb, flush=True)
        try:
            row = eval_pair(sa, sb, cfg, risk_fraction=float(__import__("os").environ.get("RISK_FRACTION", "0.05")))
        except Exception as e:
            row = {"pair": f"{sa}/{sb}", "error": str(e)}
        print(row, flush=True)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "pairs_two_leg_quest.csv", index=False)
    lines = ["# Real two-leg pairs MR quest", "", "IS-only beta/entry; real costs; holdout confirmation.", ""]
    for r in rows:
        lines.append("- " + ", ".join(f"{k}={v}" for k, v in r.items()))
    (REPORTS / "pairs_two_leg_quest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote reports/pairs_two_leg_quest.md")


if __name__ == "__main__":
    main()
