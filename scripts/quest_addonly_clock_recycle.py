#!/usr/bin/env python3
"""Wave: 2024-positive dual add-only + D1/H4 budgets + causal max-concurrent recycle.

Selection on 2024 only. Holdout is confirmation. signal_lag=1. No extra leverage
(exposure after recycle clipped to 1). approximate_non_ftmo unless ftmo exports exist.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location("ewc", ROOT / "scripts" / "eval_windowed_consistency.py")
ewc = importlib.util.module_from_spec(spec)
sys.modules["ewc"] = ewc
assert spec.loader is not None
spec.loader.exec_module(ewc)

from mt5_swing.backtest.engine import run_backtest
from mt5_swing.backtest.metrics import compute_metrics
from mt5_swing.config import load_config
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.portfolio.overlays import (
    INITIAL,
    apply_vol_target,
    causal_max_concurrent_recycle,
    clock_budget_weights,
    combine_weighted,
    daily_return_corr,
    occupancy_frame,
)
from mt5_swing.strategies.registry import get_strategy

WARMUP = 250
RF = float(os.environ.get("RISK_FRACTION", "0.08"))
MAX_LOT = float(os.environ.get("MAX_LOT", "50"))
N_REF = 5  # locked basket size; per-leg risk = RF/N_REF (do not raise)
RISK_PER = RF / N_REF
VT = 0.0025
LOCKED_SYMS = {"USDCHF", "GBPUSD", "CADJPY", "AUDCAD", "GBPCAD"}

MR = {
    "mean_reversion_regime",
    "bbands_reversion",
    "cci_reversion",
    "stoch_reversion",
    "willr_reversion",
}


def data_source() -> str:
    if any((ROOT / "data" / "ftmo").glob("*.csv")):
        return "ftmo_mt5_export"
    return "approximate_non_ftmo"


def locked_legs() -> list[dict]:
    locked = yaml.safe_load((ROOT / "configs" / "quest_one_pct_candidate.yaml").read_text())
    legs = []
    for c in locked["candidates"]:
        params = dict(c.get("params") or {})
        if not params.get("session_hours"):
            params["session_hours"] = None
        legs.append(
            {
                "symbol": c["symbol"],
                "timeframe": c["timeframe"],
                "strategy": c["strategy"],
                "params": params,
                "exits": dict(c.get("exits") or {}),
                "vol_target": bool(c.get("vol_target", False)),
                "oos_sharpe": float(c.get("oos_sharpe") or 0.01),
                "weight": float(c.get("weight") or c.get("oos_sharpe") or 0.01),
                "add_only": False,
            }
        )
    w = np.array([max(l["oos_sharpe"], 0.01) for l in legs], dtype=float)
    w = w / w.sum()
    for i, l in enumerate(legs):
        l["weight"] = float(w[i])
    return legs


def _row_to_leg(row: pd.Series, add_only: bool = True) -> dict:
    params = json.loads(row["params"]) if isinstance(row["params"], str) else (row["params"] or {})
    if not params.get("session_hours"):
        params["session_hours"] = None
    strat = str(row["strategy"])
    return {
        "symbol": row["symbol"],
        "timeframe": row["timeframe"],
        "strategy": strat,
        "params": params,
        "exits": {},
        "vol_target": strat not in MR,
        "oos_sharpe": float(row["oos_sharpe"] or 0.01),
        "oos_return": float(row["oos_return"] or 0.0),
        "oos_trades": int(row["oos_trades"] or 0),
        "weight": float(row["oos_sharpe"] or 0.01),
        "add_only": add_only,
    }


def dual_confirm_symbols(df: pd.DataFrame) -> set[str]:
    ok = df[
        (df["oos_gates_pass"] == True)  # noqa: E712
        & (df["oos_profitable"] == True)  # noqa: E712
        & (df["oos_trades"] >= 10)
        & (df["oos_sharpe"] > 0.3)
        & (~df["symbol"].isin(["XAUUSD", "XAGUSD"]))
    ]
    by = ok.groupby(["symbol", "timeframe"]).size().unstack(fill_value=0)
    if "H4" not in by.columns or "D1" not in by.columns:
        return set()
    return set(by[(by["H4"] > 0) & (by["D1"] > 0)].index)


def candidate_duals(df: pd.DataFrame, dual: set[str]) -> list[dict]:
    """One best-OOS-Sharpe meaningful leg per unused dual-confirm symbol."""
    ok = df[
        (df["symbol"].isin(dual - LOCKED_SYMS))
        & (df["oos_gates_pass"] == True)  # noqa: E712
        & (df["oos_profitable"] == True)  # noqa: E712
        & (df["oos_trades"] >= 10)
        & (df["oos_sharpe"] > 0.3)
        & (df["oos_return"] >= 0.001)
        & (df["is_return"] > 0)
    ].copy()
    ok = ok.sort_values(["oos_sharpe", "oos_return"], ascending=False)
    picked, seen = [], set()
    for _, row in ok.iterrows():
        if row["symbol"] in seen:
            continue
        path = ewc.resolve_csv(row["symbol"], row["timeframe"])
        if path is None:
            continue
        seen.add(row["symbol"])
        picked.append(_row_to_leg(row, add_only=True))
    return picked


def windows_for(end: pd.Timestamp):
    holdout_start = end - pd.Timedelta(days=365)
    out = []
    for year in range(2024, end.year + 1):
        s = pd.Timestamp(f"{year}-01-01", tz="UTC")
        e = min(pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC"), end)
        if s > end:
            continue
        ov = "research/WF" if year < holdout_start.year else ("mixed" if year == holdout_start.year else "pure holdout")
        out.append((f"{year}", s, e, ov))
    out.append(("holdout_365d", holdout_start, end, "pure holdout"))
    out.append(("roll12_end", end - pd.Timedelta(days=365), end, "pure holdout"))
    out.append(("roll12_m6", end - pd.Timedelta(days=365 + 182), end - pd.Timedelta(days=182), "mixed"))
    return out


def window_leg(leg: dict, cfg: dict, start: pd.Timestamp, end: pd.Timestamp):
    path = ewc.resolve_csv(leg["symbol"], leg["timeframe"])
    ohlc = load_ohlc_csv(path, symbol=leg["symbol"], timeframe=leg["timeframe"])
    ohlc = ohlc.loc[ohlc.index <= end]
    pre = ohlc.loc[ohlc.index < start]
    win = ohlc.loc[(ohlc.index >= start) & (ohlc.index <= end)]
    warm = pre.iloc[-WARMUP:] if len(pre) >= WARMUP else pre
    ohlc_run = pd.concat([warm, win])
    if len(ohlc_run) < WARMUP + 30 or len(win) < 20:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0
    bt = ewc.make_bt(cfg, leg, RISK_PER, MAX_LOT)
    params = dict(leg.get("params") or {})
    if not params.get("session_hours"):
        params["session_hours"] = None
    if leg["strategy"] == "carry_proxy" and "symbol" not in params:
        params["symbol"] = leg["symbol"]
    strat = get_strategy(leg["strategy"], **params)
    res = run_backtest(ohlc_run, strat, bt)
    eq = res.equity.loc[(res.equity.index >= start) & (res.equity.index <= end)]
    pos = res.positions.loc[(res.positions.index >= start) & (res.positions.index <= end)]
    n_trades = 0
    if res.trades is not None and len(res.trades):
        tr = res.trades
        if isinstance(tr, pd.DataFrame) and "exit_time" in tr.columns:
            n_trades = int(((tr["exit_time"] >= start) & (tr["exit_time"] <= end)).sum())
        else:
            n_trades = len(tr)
    name = f"{leg['symbol']}_{leg['timeframe']}_{leg['strategy']}"
    return eq.rename(name), pos.rename(name), n_trades


# cache: (symbol, tf, strategy, start.iso, end.iso) -> (eq, pos, n)
_CACHE: dict[tuple, tuple] = {}


def cached_leg(leg, cfg, start, end):
    key = (leg["symbol"], leg["timeframe"], leg["strategy"], str(start), str(end), json.dumps(leg.get("params"), sort_keys=True))
    if key not in _CACHE:
        _CACHE[key] = window_leg(leg, cfg, start, end)
    return _CACHE[key]


def pack_legs(legs, cfg, start, end):
    curves, poss, trades = [], [], 0
    for leg in legs:
        eq, pos, nt = cached_leg(leg, cfg, start, end)
        if eq.empty or pos.empty:
            return [], [], 0
        curves.append(eq)
        poss.append(pos)
        trades += nt
    return curves, poss, trades


def aligned_frames(curves, poss):
    eq = pd.concat(curves, axis=1, sort=True).sort_index().ffill().dropna(how="any")
    if eq.empty:
        return eq, pd.DataFrame(), pd.DataFrame()
    occ = occupancy_frame(poss, eq.index)
    occ = occ.reindex(eq.index).fillna(0.0)
    rets = eq.pct_change().fillna(0.0)
    return eq, rets, occ


@dataclass
class EvalRow:
    idea: str
    params: str
    window: str
    ret: float
    mean_mo: float
    median_mo: float
    pct_pos: float
    top3: float
    gates: bool
    p2t: float
    static: float
    daily: float
    n_months: int
    trades: int
    overlap: str
    n_legs: int


def eval_port(port: pd.Series, label: str, overlap: str, trades: int, idea: str, params: str, n_legs: int) -> EvalRow | None:
    if port is None or len(port) < 30:
        return None
    port = (port / float(port.iloc[0])) * INITIAL
    m = compute_metrics(
        port, None, max_dd_gate=0.10, daily_dd_gate=0.05,
        max_loss_mode="static_initial", daily_loss_mode="ftmo_initial",
        daily_tz="Europe/Prague", initial_equity=INITIAL,
    )
    ms = ewc.monthly_stats(port)
    peak = port.cummax()
    p2t = float(((peak - port) / peak).max())
    return EvalRow(
        idea=idea, params=params, window=label,
        ret=float(m.total_return), mean_mo=ms["mean_mo"], median_mo=ms.get("median_mo", float("nan")),
        pct_pos=ms["pct_pos"], top3=ms["top3_share"], gates=bool(m.gates_pass),
        p2t=p2t, static=float(m.static_loss_from_initial), daily=float(m.max_daily_dd),
        n_months=ms["n_months"], trades=trades, overlap=overlap, n_legs=n_legs,
    )


def score_2024(row: EvalRow | None, bl: EvalRow | None = None) -> float:
    if row is None or not row.gates:
        return -1e9
    top = row.top3 if row.top3 == row.top3 else 0.7
    smooth = -0.08 * max(0.0, top - 0.60)
    p2t_pen = -0.5 if row.p2t > 0.085 else 0.0
    pos_bonus = 0.05 * max(0.0, row.pct_pos - 0.55)
    # smoothness vs baseline (IS only): penalize dropping +months or raising top3
    vs = 0.0
    if bl is not None:
        vs -= 0.15 * max(0.0, bl.pct_pos - row.pct_pos)
        vs -= 0.10 * max(0.0, row.top3 - bl.top3)
        if row.pct_pos < bl.pct_pos - 0.03:
            vs -= 0.4
        if row.p2t > 0.085:
            return -1e9
    return row.mean_mo * 10.0 + smooth + p2t_pen + pos_bonus + vs


def build_port(curves, poss, weights, max_k, recycle_cap, vt: float | None):
    eq, rets, occ = aligned_frames(curves, poss)
    if eq.empty:
        return pd.Series(dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    if max_k is None and recycle_cap <= 1.0 + 1e-12:
        port = combine_weighted(curves, w)
    else:
        port = causal_max_concurrent_recycle(
            rets, occ, w, max_k=max_k, recycle_cap=recycle_cap, priority=w, initial=INITIAL,
        )
    if vt:
        port = apply_vol_target(port, float(vt))
    return port


def main() -> None:
    src = data_source()
    print(f"data_source={src} RF={RF} risk_per={RISK_PER:.4f} warmup={WARMUP}", flush=True)
    cfg = load_config(ROOT / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    lock = locked_legs()
    df = pd.read_csv(ROOT / "reports" / "walk_forward_summary.csv")
    extra = os.environ.get("EXTRA_SUMMARY", "").strip()
    if extra and Path(extra).exists():
        df = pd.concat([df, pd.read_csv(extra)], ignore_index=True)
        df = df.drop_duplicates(["symbol", "timeframe", "strategy"], keep="last")

    dual = dual_confirm_symbols(df)
    cands = candidate_duals(df, dual)
    print(f"dual-confirm symbols={sorted(dual)}", flush=True)
    print(f"add-only candidates={len(cands)}: " + ", ".join(f"{c['symbol']}:{c['timeframe']}:{c['strategy']}" for c in cands), flush=True)

    path0 = ewc.resolve_csv(lock[0]["symbol"], lock[0]["timeframe"])
    end = load_ohlc_csv(path0, symbol=lock[0]["symbol"], timeframe=lock[0]["timeframe"]).index.max()
    wins = windows_for(end)
    s24 = pd.Timestamp("2024-01-01", tz="UTC")
    e24 = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")

    print("Building 2024 locked + candidate curves...", flush=True)
    lk_c, lk_p, lk_tr = pack_legs(lock, cfg, s24, e24)
    if not lk_c:
        raise SystemExit("locked 2024 curves missing")
    lock_w = np.array([l["weight"] for l in lock], dtype=float)
    lock_w = lock_w / lock_w.sum()
    base24 = combine_weighted(lk_c, lock_w)
    base24_vt = apply_vol_target(base24, VT)

    screen_rows = []
    passing = []
    for c in cands:
        eq, pos, nt = cached_leg(c, cfg, s24, e24)
        if eq.empty:
            print(f"  skip {c['symbol']} {c['timeframe']} {c['strategy']} (empty)", flush=True)
            continue
        row = eval_port(eq, "2024", "research/WF", nt, "cand", f"{c['symbol']}|{c['timeframe']}|{c['strategy']}", 1)
        corr = daily_return_corr(eq, base24_vt)
        c["corr_2024"] = corr
        c["_eq24"] = eq
        mean_mo = row.mean_mo if row else float("nan")
        gates = row.gates if row else False
        ok = bool(row and row.gates and row.mean_mo > 0 and nt >= 6)
        screen_rows.append(
            {
                "symbol": c["symbol"], "timeframe": c["timeframe"], "strategy": c["strategy"],
                "oos_sharpe": c["oos_sharpe"], "mean_mo_2024": mean_mo, "gates": gates,
                "trades": nt, "corr_vs_locked": corr, "is_pass": ok,
            }
        )
        print(
            f"  cand {c['symbol']:7s} {c['timeframe']} {c['strategy']:22s} "
            f"mo={mean_mo*100:5.2f}% gates={gates} n={nt:3d} corr={corr:5.2f} pass={ok}",
            flush=True,
        )
        if ok:
            passing.append(c)

    # Corr-gate grid (IS=2024). Greedy add by 2024 mean_mo * (1-corr), 1 per symbol already.
    corr_grid = [0.25, 0.40, 0.55]
    add_sets: dict[str, list[dict]] = {"none": []}
    for cm in corr_grid:
        gated = [c for c in passing if float(c.get("corr_2024", 1)) <= cm]
        gated = sorted(gated, key=lambda x: -(x["_eq24"].iloc[-1] / x["_eq24"].iloc[0] - 1) * (1.0 - min(max(x["corr_2024"], 0.0), 1.0)))
        for n_add in (1, 2, 3):
            if len(gated) < n_add:
                continue
            add_sets[f"corr{cm}_n{n_add}"] = gated[:n_add]

    print(f"2024-positive duals: {len(passing)}; add-set keys={list(add_sets)}", flush=True)

    # Tune basket construction on 2024: add-set × clock × max_k × recycle × vt
    h4_shares = [None, 0.55, 0.70, 0.85]  # None = oos_sharpe (no clock split)
    max_ks = [None, 2, 3, 4]
    recaps = [1.0, 1.5, 2.0]
    vt_opts: list[float | None] = [VT, None]

    best = (-1e9, None, None)  # score, spec, row
    tune_log = []
    # Evaluate baseline first
    row_bl = eval_port(base24_vt, "2024", "research/WF", lk_tr, "baseline_vt0025", "locked+vt", 5)
    print(f"baseline 2024 mo={row_bl.mean_mo*100:.2f}% pos={row_bl.pct_pos*100:.0f}% top3={row_bl.top3*100:.0f}% gates={row_bl.gates}", flush=True)

    print("=== Tune on 2024 (add/clock/recycle) ===", flush=True)
    for set_name, extras in add_sets.items():
        legs = lock + extras
        n = len(legs)
        tfs = [l["timeframe"] for l in legs]
        base_w = np.array([max(l["oos_sharpe"], 0.01) for l in legs], dtype=float)
        curves, poss, trn = pack_legs(legs, cfg, s24, e24)
        if not curves:
            continue
        for h4s in h4_shares:
            if h4s is None:
                w = base_w / base_w.sum()
                clock_lab = "oos_sharpe"
            else:
                if not (any(t == "D1" for t in tfs) and any(t == "H4" for t in tfs)):
                    continue
                w = clock_budget_weights(tfs, base_w, h4s)
                clock_lab = f"h4={h4s:.2f}"
            for mk in max_ks:
                if mk is not None and mk >= n:
                    continue
                for cap in recaps:
                    if mk is None and cap <= 1.0:
                        # skip duplicate of plain combine except once (handled as mk=None,cap=1)
                        pass
                    if mk is None and cap > 1.0:
                        # recycle without cap still uses occupancy; allowed
                        pass
                    for vt in vt_opts:
                        port = build_port(curves, poss, w, mk, cap, vt)
                        spec = {
                            "add_set": set_name,
                            "extras": [f"{e['symbol']}|{e['timeframe']}|{e['strategy']}" for e in extras],
                            "clock": clock_lab,
                            "h4_share": h4s,
                            "max_k": mk,
                            "recycle_cap": cap,
                            "vt": vt,
                            "n_legs": n,
                        }
                        lab = f"{set_name}|{clock_lab}|k={mk}|cap={cap}|vt={vt}"
                        row = eval_port(port, "2024", "research/WF", trn, "tune", lab, n)
                        sc = score_2024(row, row_bl)
                        if row:
                            tune_log.append({**spec, "mean_mo": row.mean_mo, "pct_pos": row.pct_pos,
                                             "top3": row.top3, "gates": row.gates, "p2t": row.p2t, "score": sc})
                        if sc > best[0]:
                            best = (sc, spec, row)
                            if row:
                                print(
                                    f"  best-so-far mo={row.mean_mo*100:.2f}% pos={row.pct_pos*100:.0f}% "
                                    f"top3={row.top3*100:.0f}% p2t={row.p2t*100:.1f}% | {lab} sc={sc:.3f}",
                                    flush=True,
                                )

    selected = best[1] or {
        "add_set": "none", "extras": [], "clock": "oos_sharpe", "h4_share": None,
        "max_k": None, "recycle_cap": 1.0, "vt": VT, "n_legs": 5,
    }
    print("SELECTED", json.dumps(selected, default=str), flush=True)

    def legs_from_spec(spec):
        extras = []
        want = set(spec.get("extras") or [])
        for c in cands:
            tag = f"{c['symbol']}|{c['timeframe']}|{c['strategy']}"
            if tag in want:
                extras.append(c)
        return lock + extras

    # Full-window evaluation: baseline, selected, and a few ablations
    specs_eval = {
        "baseline_vt0025": {
            "add_set": "none", "extras": [], "clock": "oos_sharpe", "h4_share": None,
            "max_k": None, "recycle_cap": 1.0, "vt": VT, "n_legs": 5,
        },
        "selected": selected,
    }
    # Ablations around selected
    abl = dict(selected)
    abl["max_k"] = None
    abl["recycle_cap"] = 1.0
    specs_eval["selected_no_recycle"] = abl
    abl2 = dict(selected)
    abl2["h4_share"] = None
    abl2["clock"] = "oos_sharpe"
    specs_eval["selected_no_clock"] = abl2
    if selected.get("extras"):
        only_add = dict(selected)
        only_add["max_k"] = None
        only_add["recycle_cap"] = 1.0
        only_add["h4_share"] = None
        only_add["clock"] = "oos_sharpe"
        specs_eval["addonly_plain"] = only_add
    # Best add-only (corr-gated extras, plain oos_sharpe + vt, no recycle) even if overlay won
    best_add = None
    best_add_sc = -1e9
    for rec in tune_log:
        if not rec.get("extras"):
            continue
        if rec.get("max_k") is not None:
            continue
        if float(rec.get("recycle_cap") or 1) > 1.0 + 1e-9:
            continue
        if rec.get("h4_share") is not None:
            continue
        if rec.get("vt") != VT:
            continue
        if rec.get("gates") and rec.get("score", -1e9) > best_add_sc:
            best_add_sc = rec["score"]
            best_add = rec
    if best_add:
        specs_eval["addonly_plain"] = {
            "add_set": best_add.get("add_set"),
            "extras": list(best_add.get("extras") or []),
            "clock": "oos_sharpe",
            "h4_share": None,
            "max_k": None,
            "recycle_cap": 1.0,
            "vt": VT,
            "n_legs": int(best_add.get("n_legs") or (5 + len(best_add.get("extras") or []))),
        }
    # clock+recycle on locked only (no add) — selected overlay params
    specs_eval["locked_clock_recycle"] = {
        "add_set": "none", "extras": [], "clock": selected.get("clock") or "oos_sharpe",
        "h4_share": selected.get("h4_share"), "max_k": selected.get("max_k"),
        "recycle_cap": selected.get("recycle_cap") or 1.0, "vt": selected.get("vt"), "n_legs": 5,
    }
    # Conservative locked overlays from the same 2024 grid (confirmation only)
    specs_eval["locked_h4_055_cap15"] = {
        "add_set": "none", "extras": [], "clock": "h4=0.55", "h4_share": 0.55,
        "max_k": None, "recycle_cap": 1.5, "vt": VT, "n_legs": 5,
    }
    specs_eval["locked_h4_055_k3_cap20"] = {
        "add_set": "none", "extras": [], "clock": "h4=0.55", "h4_share": 0.55,
        "max_k": 3, "recycle_cap": 2.0, "vt": VT, "n_legs": 5,
    }
    specs_eval["locked_oos_cap15"] = {
        "add_set": "none", "extras": [], "clock": "oos_sharpe", "h4_share": None,
        "max_k": None, "recycle_cap": 1.5, "vt": VT, "n_legs": 5,
    }
    # Single best 2024 dual (GBPJPY squeeze, corr -0.3) add-only, no recycle
    specs_eval["add_gbpjpy_plain"] = {
        "add_set": "corr0.25_n1", "extras": ["GBPJPY|H4|squeeze_breakout"],
        "clock": "oos_sharpe", "h4_share": None, "max_k": None, "recycle_cap": 1.0, "vt": VT, "n_legs": 6,
    }
    specs_eval["add_gbpjpy_cap15"] = {
        "add_set": "corr0.25_n1", "extras": ["GBPJPY|H4|squeeze_breakout"],
        "clock": "oos_sharpe", "h4_share": None, "max_k": None, "recycle_cap": 1.5, "vt": VT, "n_legs": 6,
    }

    results: list[EvalRow] = []
    print("\n=== Full window evaluation ===", flush=True)
    for label, s, e, ov in wins:
        print(f"\n-- {label} --", flush=True)
        for idea, spec in specs_eval.items():
            legs = legs_from_spec(spec)
            curves, poss, trn = pack_legs(legs, cfg, s, e)
            if not curves:
                print(f"  {idea}: skip missing curves", flush=True)
                continue
            base_w = np.array([max(l["oos_sharpe"], 0.01) for l in legs], dtype=float)
            tfs = [l["timeframe"] for l in legs]
            h4s = spec.get("h4_share")
            if h4s is None:
                w = base_w / base_w.sum()
            else:
                w = clock_budget_weights(tfs, base_w, float(h4s))
            port = build_port(curves, poss, w, spec.get("max_k"), float(spec.get("recycle_cap") or 1.0), spec.get("vt"))
            row = eval_port(port, label, ov, trn, idea, json.dumps({k: spec[k] for k in spec if k != "extras"}), len(legs))
            if row:
                results.append(row)
                print(
                    f"  {idea:24s} mo={row.mean_mo*100:5.2f}% pos={row.pct_pos*100:3.0f}% "
                    f"top3={row.top3*100:3.0f}% gates={'PASS' if row.gates else 'FAIL'} "
                    f"p2t={row.p2t*100:.1f}% n={row.n_legs} trades={row.trades}",
                    flush=True,
                )

    # Persist
    pd.DataFrame(screen_rows).to_csv(ROOT / "reports" / "quest_addonly_screen_2024.csv", index=False)
    pd.DataFrame(tune_log).to_csv(ROOT / "reports" / "quest_addonly_tune_2024.csv", index=False)
    pd.DataFrame([asdict(r) for r in results]).to_csv(ROOT / "reports" / "quest_addonly_clock_recycle.csv", index=False)

    # Markdown
    lines = [
        "# Quest wave — add-only duals / D1-H4 budgets / causal max-concurrent recycle",
        "",
        f"**data_source:** `{src}` — never golive without FTMO MT5 exports.",
        f"**risk_fraction:** {RF:.0%}  **per-leg (locked ref):** {RISK_PER:.2%}  **warmup:** {WARMUP}  **signal_lag=1**.",
        "**Selection:** 2024 only (IS-screen + corr gate + overlay grid). Holdout confirmation only.",
        "**Leverage:** recycle exposure clipped to 1; per-leg risk frozen at locked RF/5 — not raised.",
        "",
        "## Dual-confirm universe",
        "",
        f"Symbols with meaningful OOS+gates on **both** H4 and D1: `{sorted(dual)}`",
        "",
        "## 2024 IS screen (add-only unused symbols)",
        "",
        "| Symbol | TF | Strategy | OOS Sh | 2024 mo | Gates | n | corr vs locked | pass |",
        "|--------|----|----------|-------:|--------:|:-----:|--:|---------------:|:----:|",
    ]
    for r in screen_rows:
        lines.append(
            f"| {r['symbol']} | {r['timeframe']} | {r['strategy']} | {r['oos_sharpe']:.2f} | "
            f"{r['mean_mo_2024']*100:.2f}% | {'PASS' if r['gates'] else 'FAIL'} | {r['trades']} | "
            f"{r['corr_vs_locked']:.2f} | {'YES' if r['is_pass'] else 'NO'} |"
        )
    lines += [
        "",
        "## Selected (IS=2024)",
        "```json",
        json.dumps(selected, indent=2, default=str),
        "```",
        "",
        "| Idea | Window | Mean mo | Med mo | %pos | Top3 | Gates | P2T | Static | Daily | Legs | Trades | Overlap |",
        "|------|--------|--------:|-------:|-----:|-----:|:-----:|----:|-------:|------:|-----:|-------:|---------|",
    ]
    for r in results:
        lines.append(
            f"| {r.idea} | {r.window} | {r.mean_mo*100:.2f}% | {r.median_mo*100:.2f}% | {r.pct_pos*100:.0f}% | "
            f"{r.top3*100:.0f}% | {'PASS' if r.gates else 'FAIL'} | {r.p2t*100:.1f}% | {r.static*100:.1f}% | "
            f"{r.daily*100:.1f}% | {r.n_legs} | {r.trades} | {r.overlap} |"
        )

    def slice_idea(idea):
        sub = [r for r in results if r.idea == idea]
        get = lambda w: next((x for x in sub if x.window == w), None)
        return get("2024"), get("2025"), get("2026"), get("holdout_365d"), get("roll12_m6")

    lines += ["", "## Honest verdict", ""]
    for idea in specs_eval:
        y24, y25, y26, ho, r6 = slice_idea(idea)
        def fmt(r):
            return "n/a" if r is None else f"{r.mean_mo*100:.2f}%/{r.pct_pos*100:.0f}%pos/top3={r.top3*100:.0f}% gates={r.gates}"
        lines.append(f"- **{idea}**: 2024={fmt(y24)}; 2025={fmt(y25)}; 2026={fmt(y26)}; HO={fmt(ho)}")

    bl24, bl25, bl26, blho, _ = slice_idea("baseline_vt0025")
    s24r, s25r, s26r, sho, s6 = slice_idea("selected")
    ho_ok = bool(sho and sho.gates and sho.mean_mo >= 0.01 and sho.pct_pos >= 0.65)
    y24_ok = bool(s24r and s24r.gates and s24r.mean_mo >= 0.009)
    y25_ok = bool(s25r and s25r.gates and s25r.mean_mo >= 0.009)
    smoother = bool(sho and blho and sho.top3 < blho.top3 - 0.05)
    better_24 = bool(s24r and bl24 and s24r.mean_mo > bl24.mean_mo + 0.001)
    years_ok = y24_ok and y25_ok and ho_ok
    promote = bool(
        s24r and bl24 and sho and blho
        and s24r.gates and sho.gates and (s25r.gates if s25r else False)
        and s24r.mean_mo >= bl24.mean_mo - 0.0005
        and sho.mean_mo >= 0.009
        and (s25r is None or s25r.mean_mo >= 0.008)
        and (s24r.mean_mo > bl24.mean_mo + 0.001 or (smoother and sho.mean_mo >= blho.mean_mo - 0.002))
    )
    lines += [
        "",
        "### vs ≥1%/mo & ≥65–70% pos & smoother top3",
        "",
        f"- Baseline locked+vt: 2024 mo={((bl24.mean_mo*100) if bl24 else float('nan')):.2f}%; HO mo={((blho.mean_mo*100) if blho else float('nan')):.2f}% top3={((blho.top3*100) if blho else float('nan')):.0f}%",
        f"- Selected: 2024 mo={((s24r.mean_mo*100) if s24r else float('nan')):.2f}%; HO mo={((sho.mean_mo*100) if sho else float('nan')):.2f}% pos={((sho.pct_pos*100) if sho else float('nan')):.0f}% top3={((sho.top3*100) if sho else float('nan')):.0f}%",
        f"- HO ≥1% & ≥65% pos & gates: **{'YES' if ho_ok else 'NO'}**",
        f"- 2024 ~1%: **{'YES' if y24_ok else 'NO'}**",
        f"- 2025 ~1% (confirmation): **{'YES' if y25_ok else 'NO'}**",
        f"- Materially smoother vs baseline: **{'YES' if smoother else 'NO'}**; better 2024: **{'YES' if better_24 else 'NO'}**",
        f"- **Overall target met:** **{'YES' if years_ok else 'NO'}**",
        f"- **Promote over locked candidate:** **{'YES' if promote else 'NO'}**",
        "",
        "## Methodology",
        "- Per-window BT + 250-bar warmup; FTMO static 10% / daily 5% Europe/Prague.",
        "- Dual-confirm = meaningful OOS (trades≥10, gates, sharpe>0.3) on **both** H4 and D1.",
        "- Add-only: unused dual symbols, 2024 mean_mo>0 + gates; corr vs locked 2024 daily returns.",
        "- Clock budgets: within-H4 / within-D1 relative OOS-Sharpe, then H4/D1 share.",
        "- Max-concurrent + recycle: lagged occupancy, a-priori priority, exposure ≤ 1.",
        "- Yahoo ≠ FTMO.",
        "",
    ]
    (ROOT / "reports" / "quest_addonly_clock_recycle.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "configs" / "quest_addonly_selected.json").write_text(
        json.dumps({"data_source": src, "selected": selected, "risk_fraction": RF,
                    "risk_per_leg": RISK_PER, "locked": lock,
                    "extras": [ {k: v for k, v in e.items() if k != "_eq24"} for e in (legs_from_spec(selected)[len(lock):]) ]},
                   indent=2, default=str),
        encoding="utf-8",
    )
    print("Wrote reports/quest_addonly_clock_recycle.md", flush=True)
    print(f"TARGET_MET={'YES' if years_ok else 'NO'} PROMOTE={'YES' if promote else 'NO'}", flush=True)


if __name__ == "__main__":
    main()
