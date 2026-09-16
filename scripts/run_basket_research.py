#!/usr/bin/env python3
"""
Equal-weight basket of top interim candidates (IS-selected params from CSV).
Members ranked by OOS research only; holdout reported separately (never for selection).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mt5_swing.backtest.engine import run_backtest
from mt5_swing.backtest.metrics import compute_metrics
from mt5_swing.config import load_config
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.strategies.registry import get_strategy

spec = importlib.util.spec_from_file_location("rfr", ROOT / "scripts" / "run_ftmo_research.py")
rfr = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rfr)


def main() -> None:
    cfg = load_config(ROOT / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    summary = ROOT / "reports" / "walk_forward_summary.csv"
    if not summary.exists():
        print("No walk_forward_summary.csv — run run_ftmo_research.py first")
        return
    df = pd.read_csv(summary)
    cand = df[
        (df["oos_trades"] >= 15)
        & (df["oos_gates_pass"] == True)  # noqa: E712
        & (df["oos_profitable"] == True)  # noqa: E712
    ].copy()
    cand = cand.sort_values(["oos_return", "oos_sharpe"], ascending=False).head(4)
    if cand.empty:
        print("No OOS candidates with >=15 trades")
        return

    holdout_days = int(cfg.get("walk_forward", {}).get("holdout_days", 365))
    hold_curves = []
    oos_notes = []
    n_legs = max(len(cand), 1)
    for _, row in cand.iterrows():
        path = rfr.resolve_csv(row["symbol"], row["timeframe"])
        if path is None:
            continue
        ohlc = load_ohlc_csv(path, symbol=row["symbol"], timeframe=row["timeframe"])
        _research, holdout = rfr.split_holdout(ohlc, holdout_days)
        params = json.loads(row["params"]) if isinstance(row["params"], str) else {}
        if not params.get("session_hours"):
            params["session_hours"] = None
        strat = get_strategy(row["strategy"], **params)
        bt = rfr.bt_cfg(cfg, row["symbol"])
        bt.risk_fraction = float(cfg.get("risk", {}).get("risk_fraction", 0.01)) / n_legs
        if len(holdout) < 50:
            continue
        res = run_backtest(holdout, strat, bt)
        hold_curves.append(
            res.equity.rename(f"{row['symbol']}_{row['timeframe']}_{row['strategy']}")
        )
        oos_notes.append(
            f"{row['symbol']} {row['timeframe']} {row['strategy']}: "
            f"OOS={row['oos_return']:.2%} n={int(row['oos_trades'])} "
            f"holdout_leg={res.metrics.total_return:.2%} gates={res.metrics.gates_pass}"
        )

    if not hold_curves:
        print("No holdout curves")
        return
    eq = pd.concat(hold_curves, axis=1).sort_index().ffill().dropna(how="all")
    norms = eq / eq.iloc[0]
    initial = float(cfg.get("backtest", {}).get("initial_equity", 100_000))
    port = norms.mean(axis=1) * initial
    port.name = "basket_equity"
    m = compute_metrics(
        port,
        None,
        max_dd_gate=0.10,
        daily_dd_gate=0.05,
        max_loss_mode="static_initial",
        daily_loss_mode="ftmo_initial",
        daily_tz="Europe/Prague",
        initial_equity=initial,
    )
    out = ROOT / "reports" / "basket_holdout.md"
    lines = [
        "# Basket holdout (confirmation only)",
        "",
        "Members chosen by **OOS research** rank (≥15 trades, profitable, gates). "
        "Holdout never used for selection. Per-leg risk = risk_fraction / n_legs.",
        "",
        "- data_source: approximate_non_ftmo (unless FTMO exports present)",
        f"- legs: {len(hold_curves)}",
        f"- holdout basket return: {m.total_return:.2%}",
        f"- static loss: {m.static_loss_from_initial:.2%}",
        f"- max daily loss: {m.max_daily_dd:.2%}",
        f"- gates: {'PASS' if m.gates_pass else 'FAIL'}",
        f"- sharpe: {m.sharpe:.2f}",
        "",
        "## Legs",
        "",
    ]
    for n in oos_notes:
        lines.append(f"- {n}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    port.to_csv(ROOT / "reports" / "basket_holdout_equity.csv")
    print(out.read_text())


if __name__ == "__main__":
    main()
