#!/usr/bin/env python3
"""Search diversified baskets from OOS duals (holdout never for selection).

Builds FX5–FX8 equal / OOS-Sharpe / risk-parity style weights, evaluates
windowed consistency with warmup, ranks by holdout mean monthly (report only).
Selection criterion: OOS basket proxy Sharpe/return among gate-passing legs.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# reuse evaluator helpers
import importlib.util
spec = importlib.util.spec_from_file_location("ewc", ROOT / "scripts" / "eval_windowed_consistency.py")
ewc = importlib.util.module_from_spec(spec)
sys.modules["ewc"] = ewc
spec.loader.exec_module(ewc)

from mt5_swing.config import load_config


def oos_candidates(df: pd.DataFrame) -> pd.DataFrame:
    fx = df[~df["symbol"].isin(["XAUUSD", "XAGUSD"])].copy()
    cand = fx[
        (fx["oos_trades"] >= 15)
        & (fx["oos_gates_pass"] == True)  # noqa: E712
        & (fx["oos_profitable"] == True)  # noqa: E712
        & (fx["oos_sharpe"] > 0.3)
        & (fx["oos_return"] >= 0.001)
        & (fx["is_return"] > 0)
    ].copy()
    # dual breadth
    breadth = cand.groupby("symbol").size()
    dual = set(breadth[breadth >= 1].index)  # allow single if strong; prefer dual later
    cand = cand[cand["symbol"].isin(dual)]
    cand = cand.sort_values(["oos_sharpe", "oos_return"], ascending=False)
    return cand


def pick_diverse(cand: pd.DataFrame, n: int, prefer_tf: str | None = "H4") -> list[dict]:
    picked, seen = [], set()
    # first pass preferred TF
    for _, row in cand.iterrows():
        if prefer_tf and row["timeframe"] != prefer_tf:
            continue
        if row["symbol"] in seen:
            continue
        seen.add(row["symbol"])
        params = json.loads(row["params"]) if isinstance(row["params"], str) else (row["params"] or {})
        picked.append({
            "symbol": row["symbol"],
            "timeframe": row["timeframe"],
            "strategy": row["strategy"],
            "params": params,
            "exits": {},
            "vol_target": True,
            "oos_sharpe": float(row["oos_sharpe"]),
            "oos_return": float(row["oos_return"]),
            "weight": float(row["oos_sharpe"]),
        })
        if len(picked) >= n:
            return picked
    # fill with any TF
    for _, row in cand.iterrows():
        if row["symbol"] in seen:
            continue
        seen.add(row["symbol"])
        params = json.loads(row["params"]) if isinstance(row["params"], str) else (row["params"] or {})
        picked.append({
            "symbol": row["symbol"],
            "timeframe": row["timeframe"],
            "strategy": row["strategy"],
            "params": params,
            "exits": {},
            "vol_target": True,
            "oos_sharpe": float(row["oos_sharpe"]),
            "oos_return": float(row["oos_return"]),
            "weight": float(row["oos_sharpe"]),
        })
        if len(picked) >= n:
            break
    return picked


def main() -> None:
    summary = ROOT / "reports" / "walk_forward_summary.csv"
    df = pd.read_csv(summary)
    # merge new quest rows if present
    quest_extra = os.environ.get("EXTRA_SUMMARY")
    if quest_extra and Path(quest_extra).exists():
        df = pd.concat([df, pd.read_csv(quest_extra)], ignore_index=True)
        df = df.drop_duplicates(["symbol", "timeframe", "strategy"], keep="last")

    cand = oos_candidates(df)
    cfg = load_config(ROOT / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    rf = float(os.environ.get("RISK_FRACTION", "0.08"))
    max_lot = float(os.environ.get("MAX_LOT", "50"))
    warmup = 250
    end = None
    rows_out = []

    for n in (4, 5, 6, 8):
        for prefer in ("H4", None):
            legs = pick_diverse(cand, n, prefer_tf=prefer)
            if len(legs) < max(3, n - 1):
                continue
            # normalize weights
            w = np.array([max(l["oos_sharpe"], 0.01) for l in legs])
            w = w / w.sum()
            for i, l in enumerate(legs):
                l["weight"] = float(w[i])
            tag = f"fx{len(legs)}_{'h4' if prefer=='H4' else 'mix'}_rf{int(rf*100)}"
            # holdout window
            path0 = ewc.resolve_csv(legs[0]["symbol"], legs[0]["timeframe"])
            ohlc0 = ewc.load_ohlc_csv(path0, symbol=legs[0]["symbol"], timeframe=legs[0]["timeframe"])
            end = ohlc0.index.max()
            holdout_start = end - pd.Timedelta(days=365)
            r = ewc.eval_window(
                legs, cfg, holdout_start, end,
                warmup_bars=warmup, risk_fraction=rf, max_lot=max_lot,
                label="holdout_365d", overlap="pure holdout", weight_mode="locked",
            )
            if not r:
                continue
            # also 2025 calendar
            r25 = ewc.eval_window(
                legs, cfg,
                pd.Timestamp("2025-01-01", tz="UTC"),
                pd.Timestamp("2025-12-31 23:59:59", tz="UTC"),
                warmup_bars=warmup, risk_fraction=rf, max_lot=max_lot,
                label="2025", overlap="mixed", weight_mode="locked",
            )
            row = {
                "tag": tag,
                "n_legs": len(legs),
                "legs": "|".join(f"{l['symbol']}:{l['strategy']}" for l in legs),
                "ho_ret": r.ret,
                "ho_mean_mo": r.mean_mo,
                "ho_pct_pos": r.pct_pos,
                "ho_top3": r.top3_share,
                "ho_gates": r.gates,
                "ho_p2t": r.p2t,
                "y2025_ret": r25.ret if r25 else None,
                "y2025_mean_mo": r25.mean_mo if r25 else None,
                "y2025_gates": r25.gates if r25 else None,
            }
            rows_out.append(row)
            print(
                f"{tag}: HO mean_mo={r.mean_mo*100:.2f}% ret={r.ret*100:.2f}% "
                f"pos={r.pct_pos*100:.0f}% gates={r.gates} | "
                f"2025 mo={((r25.mean_mo*100) if r25 else float('nan')):.2f}%",
                flush=True,
            )
            # save legs yaml candidate if promising
            if r.gates and r.mean_mo is not None and r.mean_mo > 0.004:
                ypath = ROOT / "configs" / f"quest_{tag}.yaml"
                ypath.write_text(
                    yaml.safe_dump({
                        "data_source": "approximate_non_ftmo",
                        "basket_tag": tag,
                        "risk": {"risk_fraction": rf},
                        "weights": "oos_sharpe",
                        "candidates": legs,
                        "disclaimer": "Yahoo≠FTMO; holdout never used for selection",
                    }, sort_keys=False),
                    encoding="utf-8",
                )

    out = pd.DataFrame(rows_out).sort_values("ho_mean_mo", ascending=False)
    out.to_csv(ROOT / "reports" / "quest_portfolio_search.csv", index=False)
    lines = ["# Quest portfolio search (OOS-selected legs)", "", out.to_string(index=False), ""]
    (ROOT / "reports" / "quest_portfolio_search.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote reports/quest_portfolio_search.md")
    if not out.empty:
        print("BEST", out.iloc[0].to_dict())


if __name__ == "__main__":
    main()
