#!/usr/bin/env python3
"""Build preferred FX basket from OOS research (a priori / dual-confirm).

Default interim mode (INTERIM_FX4=1): use locked dual FX4 legs + optional
OOS-Sharpe portfolio weights + risk_fraction from ftmo_2step.yaml (0.025).

Pure mode (INTERIM_FX4=0): dual ≥2 OOS winners/symbol, ≤1 leg/symbol,
rank OOS Sharpe→return, min OOS return 0.1%.

Holdout is confirmation only — never used for selection or weights.
"""
from __future__ import annotations

import importlib.util
import json
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

INTERIM_FX4 = [
    ("USDCHF", "H4", "bbands_reversion"),
    ("GBPUSD", "H4", "breakout_donchian"),
    ("CADJPY", "H4", "mean_reversion_regime"),
    ("AUDCAD", "H4", "mean_reversion_regime"),
]


def pick_preferred(df: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    if os.environ.get("INTERIM_FX4", "1").lower() in ("1", "true", "yes"):
        rows = []
        for sym, tf, st in INTERIM_FX4[:n]:
            hit = df[(df["symbol"] == sym) & (df["timeframe"] == tf) & (df["strategy"] == st)]
            if hit.empty:
                raise SystemExit(f"Missing interim leg {sym} {tf} {st} in summary")
            rows.append(hit.iloc[0])
        return pd.DataFrame(rows)

    fx = df[~df["symbol"].isin(["XAUUSD", "XAGUSD"])].copy()
    prof = fx[
        (fx["oos_trades"] >= 10)
        & (fx["oos_gates_pass"] == True)  # noqa: E712
        & (fx["oos_profitable"] == True)  # noqa: E712
    ]
    breadth = prof.groupby("symbol").size()
    dual = set(breadth[breadth >= 2].index)
    cand = fx[
        (fx["symbol"].isin(dual))
        & (fx["oos_trades"] >= 15)
        & (fx["oos_gates_pass"] == True)  # noqa: E712
        & (fx["oos_profitable"] == True)  # noqa: E712
        & (fx["oos_sharpe"] > 0)
        & (fx["oos_return"] >= 0.001)
        & (fx["is_return"] > 0)
    ].copy()
    cand = cand.sort_values(["oos_sharpe", "oos_return", "oos_trades"], ascending=False)
    picked, seen = [], set()
    for _, row in cand.iterrows():
        if row["symbol"] in seen:
            continue
        seen.add(row["symbol"])
        picked.append(row)
        if len(picked) >= n:
            break
    return pd.DataFrame(picked) if picked else cand.head(0)


def eval_basket(cand: pd.DataFrame, risk_fraction: float, tag: str) -> dict:
    cfg = load_config(ROOT / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    holdout_days = int(cfg.get("walk_forward", {}).get("holdout_days", 365))
    initial = float(cfg.get("backtest", {}).get("initial_equity", 100_000))
    n_legs = max(len(cand), 1)
    weight_mode = os.environ.get("WEIGHTS", "oos_sharpe").strip().lower()
    hold_curves, notes, candidates = [], [], []
    sharpes = []
    for _, row in cand.iterrows():
        path = rfr.resolve_csv(row["symbol"], row["timeframe"])
        if path is None:
            continue
        ohlc = load_ohlc_csv(path, symbol=row["symbol"], timeframe=row["timeframe"])
        _r, holdout = rfr.split_holdout(ohlc, holdout_days)
        params = json.loads(row["params"]) if isinstance(row["params"], str) else {}
        if not params.get("session_hours"):
            params["session_hours"] = None
        strat = get_strategy(row["strategy"], **params)
        bt = rfr.bt_cfg_for(cfg, row["symbol"], row["strategy"])
        bt.risk_fraction = risk_fraction / n_legs
        # Optional per-leg exits (IS-selected); also INTERIM_EXITS env JSON
        exits = {}
        if "exits" in row and isinstance(row["exits"], dict):
            exits = row["exits"]
        elif os.environ.get("INTERIM_EXITS"):
            import json as _json
            em = _json.loads(os.environ["INTERIM_EXITS"])
            exits = em.get(f"{row['symbol']}|{row['timeframe']}|{row['strategy']}", {})
        for k, v in (exits or {}).items():
            setattr(bt, k, v)
        if len(holdout) < 50:
            continue
        res = run_backtest(holdout, strat, bt)
        hold_curves.append(res.equity.rename(f"{row['symbol']}_{row['timeframe']}_{row['strategy']}"))
        sharpes.append(max(float(row["oos_sharpe"] or 0), 0.01))
        notes.append(
            f"- {row['symbol']} {row['timeframe']} {row['strategy']}: "
            f"OOS={row['oos_return']:.2%} Sh={row['oos_sharpe']:.2f} "
            f"hold_leg={res.metrics.total_return:.2%} n={int(res.metrics.n_trades)}"
        )
        candidates.append(
            {
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "strategy": row["strategy"],
                "params": params,
                "oos_return": float(row["oos_return"]),
                "oos_sharpe": float(row["oos_sharpe"]),
                "oos_trades": int(row["oos_trades"]),
                "holdout_return": float(res.metrics.total_return),
            }
        )
    if not hold_curves:
        return {"error": "no curves"}
    if weight_mode == "oos_sharpe":
        w = np.array(sharpes, dtype=float)
        w = w / w.sum()
    else:
        w = np.ones(len(hold_curves)) / len(hold_curves)
    for i, c in enumerate(candidates):
        c["weight"] = float(w[i])
        notes[i] += f" w={w[i]:.3f}"
    eq = pd.concat(hold_curves, axis=1, sort=True).sort_index().ffill().dropna(how="any")
    norms = eq / eq.iloc[0]
    port = (norms * w).sum(axis=1) * initial
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
    md = [
        f"# Preferred {tag} @{risk_fraction:.1%} (weights={weight_mode})",
        "",
        "Interim dual FX4 or pure Sharpe-first; weights from OOS only. Holdout confirmation only.",
        "",
        f"- return: {m.total_return:.2%}",
        f"- static loss: {m.static_loss_from_initial:.2%}",
        f"- daily loss: {m.max_daily_dd:.2%}",
        f"- gates: {'PASS' if m.gates_pass else 'FAIL'}",
        f"- sharpe: {m.sharpe:.2f}",
        f"- risk_fraction: {risk_fraction:.1%}",
        f"- weight_mode: {weight_mode}",
        "",
        "## Legs",
        "",
        *notes,
        "",
    ]
    out = ROOT / "reports" / f"basket_{tag}.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    port.to_csv(ROOT / "reports" / f"basket_{tag}_equity.csv")
    return {
        "tag": tag,
        "return": float(m.total_return),
        "static": float(m.static_loss_from_initial),
        "daily": float(m.max_daily_dd),
        "gates": bool(m.gates_pass),
        "sharpe": float(m.sharpe),
        "candidates": candidates,
        "md_path": str(out),
        "weight_mode": weight_mode,
    }


def main() -> None:
    summary = ROOT / "reports" / "walk_forward_summary.csv"
    df = pd.read_csv(summary)
    n = int(os.environ.get("N_LEGS", "4"))
    cfg = load_config(ROOT / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    rf = float(os.environ.get("RISK_FRACTION") or cfg.get("risk", {}).get("risk_fraction", 0.025))
    tag = os.environ.get("BASKET_TAG", "rebuild_probe")
    cand = pick_preferred(df, n=n)
    if cand.empty:
        print("No preferred candidates")
        return
    print("Selected legs:")
    print(cand[["symbol", "timeframe", "strategy", "oos_return", "oos_sharpe", "oos_trades"]].to_string(index=False))
    result = eval_basket(cand, rf, tag)
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))
    if result.get("candidates") and os.environ.get("WRITE_CFG", "0") in ("1", "true", "yes"):
        cfg_out = {
            "data_source": "approximate_non_ftmo",
            "challenge": "FTMO_2-Step",
            "basket_tag": tag,
            "risk": {
                "risk_fraction": rf,
                "atr_stop_mult": 2.0,
                "atr_target_mult": 3.0,
                "atr_trail_mult": 0.0,
                "vol_target": False,
            },
            "weights": result.get("weight_mode", "oos_sharpe"),
            "methodology": "Interim dual FX4 ≤1/symbol; weights ∝ OOS Sharpe; risk from ftmo_2step default unless overridden.",
            "disclaimer": "Yahoo FX ≠ FTMO MT5. No go-live without ftmo_mt5_export.",
            "candidates": result["candidates"],
        }
        cfg_path = ROOT / "configs" / "best_interim_approximate.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg_out, sort_keys=False), encoding="utf-8")
        print("Wrote", cfg_path)


if __name__ == "__main__":
    main()
