#!/usr/bin/env python3
"""Stress-test preferred dual-confirm legs at multiple risk fractions.

Uses IS-selected params from configs/best_*.yaml. Holdout is confirmation only —
risk levels are swept for gate safety, not for cherry-picking winners on holdout.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

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
    cfg_path = Path(os.environ.get("BEST_CFG", ROOT / "configs" / "best_interim_approximate.yaml"))
    best = yaml.safe_load(cfg_path.read_text())
    candidates = best.get("candidates") or []
    if not candidates:
        print("No candidates in", cfg_path)
        return
    fx_only = os.environ.get("FX_ONLY", "1").lower() in ("1", "true", "yes")
    if fx_only:
        candidates = [c for c in candidates if c["symbol"] not in ("XAUUSD", "XAGUSD")]
    risks = [
        float(x)
        for x in os.environ.get("RISK_SWEEP", "0.01,0.015,0.02,0.025").split(",")
        if x.strip()
    ]
    trail = float(os.environ.get("ATR_TRAIL_MULT", "0") or 0)
    weight_mode = os.environ.get("WEIGHTS", best.get("weights", "equal")).strip().lower()
    cfg = load_config(ROOT / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    holdout_days = int(cfg.get("walk_forward", {}).get("holdout_days", 365))
    initial = float(cfg.get("backtest", {}).get("initial_equity", 100_000))
    n_legs = max(len(candidates), 1)
    lines = [
        "# Preferred basket risk stress (confirmation)",
        "",
        f"- config: `{cfg_path.name}`",
        f"- data_source: {best.get('data_source', 'approximate_non_ftmo')}",
        f"- fx_only: {fx_only}",
        f"- legs: {n_legs}",
        f"- atr_trail_mult: {trail}",
        f"- weights: {weight_mode}",
        "- Note: params from IS grids; holdout never used to choose risk/weights.",
        "",
        "| Risk/trade | Basket ret | Static loss | Daily loss | Gates | Sharpe |",
        "|---:|---:|---:|---:|:---:|---:|",
    ]
    for rf in risks:
        hold_curves = []
        used = []
        for c in candidates:
            path = rfr.resolve_csv(c["symbol"], c["timeframe"])
            if path is None:
                continue
            ohlc = load_ohlc_csv(path, symbol=c["symbol"], timeframe=c["timeframe"])
            _r, holdout = rfr.split_holdout(ohlc, holdout_days)
            params = dict(c.get("params") or {})
            if not params.get("session_hours"):
                params["session_hours"] = None
            strat = get_strategy(c["strategy"], **params)
            bt = rfr.bt_cfg_for(cfg, c["symbol"], c["strategy"])
            bt.risk_fraction = rf / n_legs
            if trail > 0:
                bt.atr_trail_mult = trail
            if len(holdout) < 50:
                continue
            res = run_backtest(holdout, strat, bt)
            hold_curves.append(
                res.equity.rename(f"{c['symbol']}_{c['timeframe']}_{c['strategy']}")
            )
            used.append(c)
        if not hold_curves:
            continue
        eq = pd.concat(hold_curves, axis=1, sort=True).sort_index().ffill().dropna(how="any")
        if eq.empty:
            continue
        norms = eq / eq.iloc[0]
        if weight_mode == "oos_sharpe":
            sh = np.array([max(float(c.get("oos_sharpe") or 0.01), 0.01) for c in used], dtype=float)
            w = sh / sh.sum()
            port = (norms * w).sum(axis=1) * initial
        else:
            port = norms.mean(axis=1) * initial
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
        lines.append(
            f"| {rf:.1%} | {m.total_return:.2%} | {m.static_loss_from_initial:.2%} | "
            f"{m.max_daily_dd:.2%} | {'PASS' if m.gates_pass else 'FAIL'} | {m.sharpe:.2f} |"
        )
    lines.append("")
    lines.append("## Legs (params IS-selected)")
    lines.append("")
    for c in candidates:
        w = c.get("weight")
        wtxt = f" w={float(w):.3f}" if w is not None else ""
        lines.append(
            f"- {c['symbol']} {c['timeframe']} {c['strategy']}: "
            f"OOS={float(c.get('oos_return', 0)):.2%} hold={float(c.get('holdout_return', 0)):.2%}{wtxt}"
        )
    out = ROOT / "reports" / "basket_risk_stress.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text())


if __name__ == "__main__":
    main()
