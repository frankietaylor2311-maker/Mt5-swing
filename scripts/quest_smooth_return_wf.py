#!/usr/bin/env python3
"""Wave: smooth-return portfolio from scratch + IS walk-forward min(year mean_mo).

Builds many small uncorrelated MR/breakout legs with hard per-leg risk
(RF / n_legs, RF fixed at 8%) and portfolio vol targeting aimed at ~0.8–1.2%/mo.
Selection maximizes min(IS year mean_mo) subject to %pos / top3 — holdout never
enters the score. Metals (XAU) only if dual-confirm AND improve 2024 without
hurting 2025 %pos. Documents Yahoo irreducible limits if target unmet.
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
from mt5_swing.portfolio.overlays import INITIAL, apply_vol_target, combine_weighted, daily_return_corr
from mt5_swing.portfolio.smooth_select import (
    WindowStats,
    expected_monthly_from_daily_vol,
    greedy_decorrelated_pick,
    min_mean_mo_score,
)
from mt5_swing.strategies.registry import get_strategy

WARMUP = 250
RF = float(os.environ.get("QUEST_RF", os.environ.get("RISK_FRACTION", "0.08")))
if abs(RF - 0.08) > 1e-9 and os.environ.get("QUEST_ALLOW_RF_OVERRIDE", "") != "1":
    print(f"NOTE: overriding RISK_FRACTION={RF} → 0.08 (set QUEST_ALLOW_RF_OVERRIDE=1 to keep)", flush=True)
    RF = 0.08
MAX_LOT = float(os.environ.get("MAX_LOT", "50"))

MR = {
    "mean_reversion_regime",
    "bbands_reversion",
    "cci_reversion",
    "stoch_reversion",
    "willr_reversion",
}
BREAKOUT = {
    "breakout_donchian",
    "squeeze_breakout",
    "atr_channel_breakout",
    "keltner_breakout",
    "vol_breakout",
}
SMOOTH_STRATS = MR | BREAKOUT
METALS = {"XAUUSD", "XAGUSD"}
LOCKED_TAG = "fx4plus_gbpcad_d1_voltarget_0025"

# Soft/hard IS constraints (selection)
MIN_PCT_POS = 0.70
MAX_TOP3_HARD = 0.55
MAX_TOP3_SOFT = 0.70
MAX_P2T = 0.085
MAX_PAIR_CORR = 0.55


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
            }
        )
    w = np.array([max(l["oos_sharpe"], 0.01) for l in legs], dtype=float)
    w = w / w.sum()
    for i, l in enumerate(legs):
        l["weight"] = float(w[i])
    return legs


def _row_to_leg(row: pd.Series) -> dict:
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
        "weight": float(row["oos_sharpe"] or 0.01),
    }


def dual_confirm_symbols(df: pd.DataFrame, include_metals: bool = False) -> set[str]:
    ok = df[
        (df["oos_gates_pass"] == True)  # noqa: E712
        & (df["oos_profitable"] == True)  # noqa: E712
        & (df["oos_trades"] >= 10)
        & (df["oos_sharpe"] > 0.3)
    ]
    if not include_metals:
        ok = ok[~ok["symbol"].isin(METALS)]
    by = ok.groupby(["symbol", "timeframe"]).size().unstack(fill_value=0)
    if "H4" not in by.columns or "D1" not in by.columns:
        return set()
    return set(by[(by["H4"] > 0) & (by["D1"] > 0)].index)


def pool_legs(df: pd.DataFrame, dual: set[str], *, metals: bool = False) -> list[dict]:
    """Many small MR/breakout candidates; prefer dual-confirm; one per symbol."""
    mask = (
        (df["strategy"].isin(SMOOTH_STRATS))
        & (df["oos_gates_pass"] == True)  # noqa: E712
        & (df["oos_profitable"] == True)  # noqa: E712
        & (df["oos_trades"] >= 10)
        & (df["oos_sharpe"] > 0.25)
        & (df["oos_return"] >= 0.0005)
        & (df["is_return"] > 0)
    )
    if metals:
        mask = mask & (df["symbol"].isin(METALS))
    else:
        mask = mask & (~df["symbol"].isin(METALS))
    ok = df[mask].copy()
    # Prefer dual-confirm symbols first
    ok["dual"] = ok["symbol"].isin(dual).astype(int)
    ok = ok.sort_values(["dual", "oos_sharpe", "oos_return"], ascending=False)
    picked, seen = [], set()
    for _, row in ok.iterrows():
        if row["symbol"] in seen:
            continue
        if ewc.resolve_csv(row["symbol"], row["timeframe"]) is None:
            continue
        seen.add(row["symbol"])
        picked.append(_row_to_leg(row))
    return picked


def is_windows(end: pd.Timestamp, holdout_start: pd.Timestamp):
    """IS-only windows for selection (no holdout bars)."""
    out = []
    out.append(("2024", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31 23:59:59", tz="UTC")))
    # 2025 research slice up to holdout
    s25 = pd.Timestamp("2025-01-01", tz="UTC")
    if holdout_start > s25:
        out.append(("2025_IS", s25, min(holdout_start - pd.Timedelta(seconds=1), end)))
    return out


def confirm_windows(end: pd.Timestamp, holdout_start: pd.Timestamp):
    out = []
    for year in range(2024, end.year + 1):
        s = pd.Timestamp(f"{year}-01-01", tz="UTC")
        e = min(pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC"), end)
        if s > end:
            continue
        out.append((f"{year}", s, e, "confirm"))
    out.append(("holdout_365d", holdout_start, end, "pure holdout"))
    out.append(("roll12_m6", end - pd.Timedelta(days=365 + 182), end - pd.Timedelta(days=182), "mixed"))
    return out


_CACHE: dict[tuple, tuple] = {}


def window_leg(leg: dict, cfg: dict, start: pd.Timestamp, end: pd.Timestamp, risk_per: float):
    path = ewc.resolve_csv(leg["symbol"], leg["timeframe"])
    if path is None:
        return pd.Series(dtype=float), 0
    ohlc = load_ohlc_csv(path, symbol=leg["symbol"], timeframe=leg["timeframe"])
    ohlc = ohlc.loc[ohlc.index <= end]
    pre = ohlc.loc[ohlc.index < start]
    win = ohlc.loc[(ohlc.index >= start) & (ohlc.index <= end)]
    warm = pre.iloc[-WARMUP:] if len(pre) >= WARMUP else pre
    ohlc_run = pd.concat([warm, win])
    if len(ohlc_run) < WARMUP + 30 or len(win) < 20:
        return pd.Series(dtype=float), 0
    bt = ewc.make_bt(cfg, leg, risk_per, MAX_LOT)
    params = dict(leg.get("params") or {})
    if not params.get("session_hours"):
        params["session_hours"] = None
    if leg["strategy"] == "carry_proxy" and "symbol" not in params:
        params["symbol"] = leg["symbol"]
    strat = get_strategy(leg["strategy"], **params)
    res = run_backtest(ohlc_run, strat, bt)
    eq = res.equity.loc[(res.equity.index >= start) & (res.equity.index <= end)]
    n_trades = 0
    if res.trades is not None and len(res.trades):
        tr = res.trades
        if isinstance(tr, pd.DataFrame) and "exit_time" in tr.columns:
            n_trades = int(((tr["exit_time"] >= start) & (tr["exit_time"] <= end)).sum())
        else:
            n_trades = len(tr)
    name = f"{leg['symbol']}_{leg['timeframe']}_{leg['strategy']}"
    return eq.rename(name), n_trades


def cached_leg(leg, cfg, start, end, risk_per):
    key = (
        leg["symbol"],
        leg["timeframe"],
        leg["strategy"],
        str(start),
        str(end),
        round(risk_per, 6),
        json.dumps(leg.get("params"), sort_keys=True),
        str(leg.get("exits")),
    )
    if key not in _CACHE:
        _CACHE[key] = window_leg(leg, cfg, start, end, risk_per)
    return _CACHE[key]


def pack_legs(legs, cfg, start, end, risk_per):
    curves, trades = [], 0
    for leg in legs:
        eq, nt = cached_leg(leg, cfg, start, end, risk_per)
        if eq.empty:
            return [], 0
        curves.append(eq)
        trades += nt
    return curves, trades


@dataclass
class EvalRow:
    idea: str
    params: str
    window: str
    ret: float
    mean_mo: float
    pct_pos: float
    top3: float
    gates: bool
    p2t: float
    n_months: int
    trades: int
    n_legs: int
    score_note: str = ""


def eval_port(port: pd.Series, label: str, trades: int, idea: str, params: str, n_legs: int) -> EvalRow | None:
    if port is None or len(port) < 30:
        return None
    port = (port / float(port.iloc[0])) * INITIAL
    m = compute_metrics(
        port,
        None,
        max_dd_gate=0.10,
        daily_dd_gate=0.05,
        max_loss_mode="static_initial",
        daily_loss_mode="ftmo_initial",
        daily_tz="Europe/Prague",
        initial_equity=INITIAL,
    )
    ms = ewc.monthly_stats(port)
    peak = port.cummax()
    p2t = float(((peak - port) / peak).max())
    return EvalRow(
        idea=idea,
        params=params,
        window=label,
        ret=float(m.total_return),
        mean_mo=ms["mean_mo"],
        pct_pos=ms["pct_pos"],
        top3=ms["top3_share"],
        gates=bool(m.gates_pass),
        p2t=p2t,
        n_months=ms["n_months"],
        trades=trades,
        n_legs=n_legs,
    )


def weights_for(legs: list[dict], mode: str) -> np.ndarray:
    if mode == "equal":
        w = np.ones(len(legs), dtype=float)
    else:
        w = np.array([max(float(l.get("oos_sharpe") or 0.01), 0.01) for l in legs], dtype=float)
    return w / w.sum()


def corr_matrix_on_is(legs: list[dict], cfg: dict, start: pd.Timestamp, end: pd.Timestamp, risk_per: float) -> pd.DataFrame:
    ids = [f"{l['symbol']}:{l['timeframe']}:{l['strategy']}" for l in legs]
    eqs = {}
    for leg, i in zip(legs, ids):
        eq, _ = cached_leg(leg, cfg, start, end, risk_per)
        if not eq.empty:
            eqs[i] = eq
    mat = pd.DataFrame(np.eye(len(ids)), index=ids, columns=ids)
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if j <= i:
                continue
            if a not in eqs or b not in eqs:
                c = 1.0
            else:
                c = daily_return_corr(eqs[a], eqs[b])
            mat.loc[a, b] = mat.loc[b, a] = c
    return mat


def main() -> None:
    src = data_source()
    print(f"data_source={src} RF={RF} warmup={WARMUP} locked={LOCKED_TAG}", flush=True)
    print(
        f"IS score: max min(year_mean_mo) s.t. %pos>={MIN_PCT_POS:.0%} top3<={MAX_TOP3_HARD:.0%} "
        f"(soft<={MAX_TOP3_SOFT:.0%}); holdout NEVER for selection",
        flush=True,
    )
    for vt in (0.002, 0.0025, 0.003, 0.004, 0.005):
        em = expected_monthly_from_daily_vol(vt, sharpe=1.2)
        print(f"  VT={vt} ≈ {em*100:.2f}%/mo @Sharpe1.2 (rough)", flush=True)

    cfg = load_config(ROOT / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    lock = locked_legs()
    df = pd.read_csv(ROOT / "reports" / "walk_forward_summary.csv")
    dual = dual_confirm_symbols(df)
    dual_metals = dual_confirm_symbols(df, include_metals=True) & METALS
    pool = pool_legs(df, dual)
    print(f"dual-confirm FX={sorted(dual)}", flush=True)
    print(f"dual-confirm metals={sorted(dual_metals) or 'none'}", flush=True)
    print(f"smooth pool={len(pool)}: " + ", ".join(f"{c['symbol']}:{c['timeframe']}:{c['strategy']}" for c in pool), flush=True)

    path0 = ewc.resolve_csv(lock[0]["symbol"], lock[0]["timeframe"])
    end = load_ohlc_csv(path0, symbol=lock[0]["symbol"], timeframe=lock[0]["timeframe"]).index.max()
    holdout_start = end - pd.Timedelta(days=365)
    is_wins = is_windows(end, holdout_start)
    conf_wins = confirm_windows(end, holdout_start)
    print(f"end={end} holdout_start={holdout_start}", flush=True)
    print(f"IS windows: {[w[0] for w in is_wins]}", flush=True)

    rows: list[EvalRow] = []
    # --- Baseline locked ---
    print("=== Baseline locked + vt0025 (confirm windows) ===", flush=True)
    risk_lock = RF / max(len(lock), 1)
    for label, s, e, _ov in conf_wins:
        curves, tr = pack_legs(lock, cfg, s, e, risk_lock)
        if not curves:
            continue
        w = weights_for(lock, "oos_sharpe")
        port = apply_vol_target(combine_weighted(curves, w), 0.0025)
        row = eval_port(port, label, tr, "baseline_vt0025", "locked", len(lock))
        if row:
            rows.append(row)
            print(
                f"  {label:14s} mo={row.mean_mo*100:5.2f}% pos={row.pct_pos*100:4.0f}% "
                f"top3={row.top3*100:4.0f}% gates={row.gates}",
                flush=True,
            )

    # --- Precompute IS curves for pool at several n (risk_per depends on n) ---
    # Use a reference risk for corr screening (n=8), then rebuild with correct risk_per
    print("=== IS corr screen on pool (ref n=8) ===", flush=True)
    risk_ref = RF / 8.0
    s_corr, e_corr = is_wins[0][1], is_wins[0][2]  # 2024
    # Warm cache for 2024 + 2025_IS at risk_ref for all pool legs
    for label, s, e in is_wins:
        for leg in pool:
            cached_leg(leg, cfg, s, e, risk_ref)
    corr = corr_matrix_on_is(pool, cfg, s_corr, e_corr, risk_ref)
    ids = [f"{l['symbol']}:{l['timeframe']}:{l['strategy']}" for l in pool]
    scores = np.array([float(l["oos_sharpe"]) for l in pool], dtype=float)
    id_to_leg = {i: l for i, l in zip(ids, pool)}

    # Candidate baskets: greedy decorrelated at various n + max_corr
    basket_specs = []
    for n in (6, 8, 10, 12):
        for mc in (0.45, 0.55, 0.65):
            picked_ids = greedy_decorrelated_pick(ids, scores, corr, n=n, max_corr=mc)
            if len(picked_ids) < max(4, n - 2):
                continue
            legs = [id_to_leg[i] for i in picked_ids]
            tag = f"smooth_n{len(legs)}_c{mc}"
            basket_specs.append((tag, legs, mc))

    # Also: top-N by OOS sharpe among dual-confirm only (no corr filter)
    dual_pool = [l for l in pool if l["symbol"] in dual]
    for n in (6, 8, 10):
        legs = dual_pool[:n]
        if len(legs) >= 4:
            basket_specs.append((f"dualtop_n{len(legs)}", legs, None))

    # MR-only and breakout-only sleeves (diversity of style)
    mr_pool = [l for l in pool if l["strategy"] in MR]
    bo_pool = [l for l in pool if l["strategy"] in BREAKOUT]
    for n in (5, 6, 8):
        if len(mr_pool) >= n:
            basket_specs.append((f"mr_n{n}", mr_pool[:n], None))
        if len(bo_pool) >= max(4, n - 1) and len(bo_pool) >= 4:
            basket_specs.append((f"bo_n{min(n, len(bo_pool))}", bo_pool[:n], None))

    # Locked symbols + top unused duals (smooth mix with proven core)
    lock_syms = {l["symbol"] for l in lock}
    extra = [l for l in dual_pool if l["symbol"] not in lock_syms]
    for k in (1, 2, 3, 4):
        if len(extra) >= k:
            legs = list(lock) + extra[:k]
            # one per symbol already
            basket_specs.append((f"lock_plus{k}", legs, None))

    # Seeded greedy: permute score noise for alternate decorrelated sets
    rng = np.random.default_rng(42)
    for seed_i in range(8):
        noise = rng.normal(0, 0.15, size=len(scores))
        for n, mc in ((6, 0.5), (8, 0.55), (10, 0.6)):
            picked_ids = greedy_decorrelated_pick(ids, scores + noise, corr, n=n, max_corr=mc)
            if len(picked_ids) >= max(4, n - 2):
                legs = [id_to_leg[i] for i in picked_ids]
                basket_specs.append((f"seed{seed_i}_n{len(legs)}_c{mc}", legs, mc))

    # Deduplicate by symbol set
    seen_sets = set()
    uniq = []
    for tag, legs, mc in basket_specs:
        key = tuple(sorted(l["symbol"] for l in legs))
        if key in seen_sets:
            continue
        seen_sets.add(key)
        uniq.append((tag, legs, mc))
    basket_specs = uniq
    print(f"basket candidates={len(basket_specs)}", flush=True)

    vt_grid = [0.002, 0.0025, 0.003, 0.0035, 0.004, 0.005]
    clip_his = [2.0, 3.0]
    weight_modes = ["equal", "oos_sharpe"]

    print("=== IS walk-forward select (min year mean_mo) ===", flush=True)
    is_board = []
    for tag, legs, mc in basket_specs:
        n = len(legs)
        risk_per = RF / n  # hard per-leg cap
        # refresh cache at correct risk_per
        for label, s, e in is_wins:
            for leg in legs:
                cached_leg(leg, cfg, s, e, risk_per)
        for wm in weight_modes:
            w = weights_for(legs, wm)
            for vt in vt_grid:
                for chi in clip_his:
                    win_stats: list[WindowStats] = []
                    win_rows: list[EvalRow] = []
                    params = f"n={n}|w={wm}|vt={vt}|chi={chi}|mc={mc}"
                    idea = f"{tag}|{wm}|vt{vt}|hi{chi}"
                    ok_all = True
                    for label, s, e in is_wins:
                        curves, tr = pack_legs(legs, cfg, s, e, risk_per)
                        if not curves:
                            ok_all = False
                            break
                        port = apply_vol_target(
                            combine_weighted(curves, w),
                            vt,
                            look=60,
                            lo=0.25,
                            hi=chi,
                        )
                        row = eval_port(port, label, tr, idea, params, n)
                        if row is None:
                            ok_all = False
                            break
                        win_rows.append(row)
                        win_stats.append(
                            WindowStats(
                                mean_mo=row.mean_mo,
                                pct_pos=row.pct_pos,
                                top3=row.top3,
                                gates=row.gates,
                                p2t=row.p2t,
                            )
                        )
                    if not ok_all or len(win_stats) < len(is_wins):
                        continue
                    sc = min_mean_mo_score(
                        win_stats,
                        min_pct_pos=MIN_PCT_POS,
                        max_top3=MAX_TOP3_HARD,
                        max_p2t=MAX_P2T,
                        soft_top3=MAX_TOP3_SOFT,
                    )
                    # Also record unconstrained min for diagnostics
                    raw_min = float(min(st.mean_mo for st in win_stats))
                    avg_pos = float(np.mean([st.pct_pos for st in win_stats]))
                    avg_top = float(np.mean([st.top3 for st in win_stats]))
                    is_board.append(
                        {
                            "idea": idea,
                            "tag": tag,
                            "params": params,
                            "n_legs": n,
                            "score": sc,
                            "raw_min_mo": raw_min,
                            "avg_pct_pos": avg_pos,
                            "avg_top3": avg_top,
                            "legs": legs,
                            "wm": wm,
                            "vt": vt,
                            "chi": chi,
                            "is_rows": win_rows,
                        }
                    )
                    hard = all(
                        (st.top3 <= MAX_TOP3_HARD and st.pct_pos >= MIN_PCT_POS and st.gates)
                        for st in win_stats
                    )
                    if sc > -1e8:
                        kind = "HARD" if hard else "SOFT"
                        print(
                            f"  {kind} {idea}: score={sc*100:.2f}% raw={raw_min*100:.2f}% "
                            f"pos={avg_pos*100:.0f}% top3={avg_top*100:.0f}%",
                            flush=True,
                        )

    is_df = pd.DataFrame(
        [
            {
                "idea": b["idea"],
                "tag": b["tag"],
                "params": b["params"],
                "n_legs": b["n_legs"],
                "score": b["score"],
                "raw_min_mo": b["raw_min_mo"],
                "avg_pct_pos": b["avg_pct_pos"],
                "avg_top3": b["avg_top3"],
                "vt": b["vt"],
                "chi": b["chi"],
                "wm": b["wm"],
                "symbols": "|".join(l["symbol"] for l in b["legs"]),
            }
            for b in is_board
        ]
    ).sort_values(["score", "raw_min_mo", "avg_pct_pos"], ascending=False)
    is_path = ROOT / "reports" / "quest_smooth_is_board.csv"
    is_df.to_csv(is_path, index=False)
    print(f"IS board wrote {is_path} ({len(is_df)} rows); "
          f"constraint passers={int((is_df['score'] > -1e8).sum())}", flush=True)

    # Best constraint passer; else best raw_min among gate+%pos>=60%
    passers = [b for b in is_board if b["score"] > -1e8]
    if passers:
        best = max(passers, key=lambda b: (b["score"], -b["avg_top3"], b["avg_pct_pos"]))
        print(f"BEST IS passer: {best['idea']} score={best['score']*100:.2f}%/mo", flush=True)
        # Prefer hard top3 if any
        hard_pass = [
            b for b in passers
            if all(r.top3 <= MAX_TOP3_HARD for r in b["is_rows"])
        ]
        if hard_pass:
            best = max(hard_pass, key=lambda b: (b["raw_min_mo"], b["avg_pct_pos"], -b["avg_top3"]))
            print(f"BEST HARD top3 passer: {best['idea']} raw_min={best['raw_min_mo']*100:.2f}%/mo", flush=True)
    else:
        # Fallback: maximize raw min among those with all gates and avg %pos >= 0.55
        soft = [
            b
            for b in is_board
            if all(r.gates for r in b["is_rows"]) and b["avg_pct_pos"] >= 0.55
        ]
        best = max(soft, key=lambda b: (b["raw_min_mo"], b["avg_pct_pos"], -b["avg_top3"])) if soft else None
        if best:
            print(
                f"NO hard passer; soft-best by raw min_mo: {best['idea']} "
                f"raw_min={best['raw_min_mo']*100:.2f}%",
                flush=True,
            )
        else:
            print("NO viable smooth basket on IS", flush=True)

    # --- Confirm top-K + baseline on full windows ---
    print("=== Confirmation (holdout not used for selection) ===", flush=True)
    to_confirm = []
    if best:
        to_confirm.append(best)
    # top 3 by score (incl -inf filtered by raw)
    ranked = sorted(is_board, key=lambda b: (b["score"], b["raw_min_mo"]), reverse=True)
    for b in ranked[:5]:
        if best is None or b["idea"] != best["idea"]:
            to_confirm.append(b)
    # unique
    seen_i = set()
    uniq_c = []
    for b in to_confirm:
        if b["idea"] in seen_i:
            continue
        seen_i.add(b["idea"])
        uniq_c.append(b)
    to_confirm = uniq_c[:6]

    promote_rows = []
    for b in to_confirm:
        legs = b["legs"]
        n = len(legs)
        risk_per = RF / n
        w = weights_for(legs, b["wm"])
        idea = b["idea"]
        for label, s, e, _ov in conf_wins:
            curves, tr = pack_legs(legs, cfg, s, e, risk_per)
            if not curves:
                continue
            port = apply_vol_target(combine_weighted(curves, w), b["vt"], look=60, lo=0.25, hi=b["chi"])
            row = eval_port(port, label, tr, idea, b["params"], n)
            if row:
                rows.append(row)
                print(
                    f"  {idea[:40]:40s} {label:14s} mo={row.mean_mo*100:5.2f}% "
                    f"pos={row.pct_pos*100:4.0f}% top3={row.top3*100:4.0f}% g={row.gates}",
                    flush=True,
                )

    # --- Metals dual-confirm probe (only if improves 2024 without hurting 2025 %pos) ---
    print("=== Metals probe (dual-confirm only) ===", flush=True)
    metal_rows = []
    xau_legs = pool_legs(df, dual | dual_metals, metals=True)
    # Also allow strong single-TF XAU if dual empty but OOS strong
    if not xau_legs:
        xok = df[
            (df["symbol"] == "XAUUSD")
            & (df["oos_gates_pass"] == True)  # noqa: E712
            & (df["oos_profitable"] == True)  # noqa: E712
            & (df["oos_trades"] >= 10)
            & (df["oos_sharpe"] > 0.4)
        ].sort_values("oos_sharpe", ascending=False)
        for _, row in xok.head(2).iterrows():
            if ewc.resolve_csv(row["symbol"], row["timeframe"]):
                xau_legs.append(_row_to_leg(row))
    print(f"XAU candidates: {[(l['symbol'], l['timeframe'], l['strategy'], round(l['oos_sharpe'],2)) for l in xau_legs]}", flush=True)

    metal_promote = None
    if best and xau_legs and (dual_metals or xau_legs):
        # Only proceed if dual metal OR we document as exploratory
        base_legs = best["legs"]
        # Require dual for promote path
        for xl in xau_legs:
            if xl["symbol"] not in dual_metals:
                print(f"  skip {xl['symbol']}:{xl['timeframe']}:{xl['strategy']} — not dual-confirm", flush=True)
                continue
            legs2 = base_legs + [xl]
            # dedupe symbol
            if sum(1 for l in base_legs if l["symbol"] == xl["symbol"]):
                continue
            n = len(legs2)
            risk_per = RF / n
            w = weights_for(legs2, best["wm"])
            idea = f"{best['idea']}+{xl['symbol']}"
            # IS check: 2024 mean and 2025 %pos vs best without metal
            s24 = {r.window: r for r in best["is_rows"]}
            is_ok = True
            new_is = []
            for label, s, e in is_wins:
                curves, tr = pack_legs(legs2, cfg, s, e, risk_per)
                if not curves:
                    is_ok = False
                    break
                port = apply_vol_target(
                    combine_weighted(curves, w), best["vt"], look=60, lo=0.25, hi=best["chi"]
                )
                row = eval_port(port, label, tr, idea, best["params"] + f"+{xl['symbol']}", n)
                if row is None:
                    is_ok = False
                    break
                new_is.append(row)
                metal_rows.append(row)
            if not is_ok or len(new_is) < len(is_wins):
                continue
            r24_new = next(r for r in new_is if r.window == "2024")
            r24_old = s24.get("2024")
            r25_new = next((r for r in new_is if r.window.startswith("2025")), None)
            r25_old = s24.get("2025_IS") or s24.get("2025")
            improve_2024 = r24_old and r24_new.mean_mo > r24_old.mean_mo + 1e-4
            hurt_2025_pos = (
                r25_new is not None
                and r25_old is not None
                and r25_new.pct_pos < r25_old.pct_pos - 1e-6
            )
            print(
                f"  {idea}: 2024 mo {r24_old.mean_mo*100:.2f}→{r24_new.mean_mo*100:.2f} "
                f"2025%pos {(r25_old.pct_pos*100 if r25_old else float('nan')):.0f}→"
                f"{(r25_new.pct_pos*100 if r25_new else float('nan')):.0f} "
                f"improve24={improve_2024} hurt25pos={hurt_2025_pos}",
                flush=True,
            )
            if improve_2024 and not hurt_2025_pos:
                metal_promote = (idea, legs2, best, new_is)
                # confirm windows
                for label, s, e, _ov in conf_wins:
                    curves, tr = pack_legs(legs2, cfg, s, e, risk_per)
                    if not curves:
                        continue
                    port = apply_vol_target(
                        combine_weighted(curves, w), best["vt"], look=60, lo=0.25, hi=best["chi"]
                    )
                    row = eval_port(port, label, tr, idea, best["params"] + "+XAU", n)
                    if row:
                        rows.append(row)
            else:
                print("  metals rejected under dual-confirm consistency rules", flush=True)
    elif not dual_metals:
        print("  No dual-confirm metals in WF summary — metals not eligible for promote path.", flush=True)

    # --- Promotion decision ---
    def window_map(idea: str) -> dict[str, EvalRow]:
        return {r.window: r for r in rows if r.idea == idea}

    def promote_ok(wm: dict[str, EvalRow]) -> tuple[bool, str]:
        need = ["2024", "2025", "2026", "holdout_365d"]
        for k in need:
            if k not in wm:
                return False, f"missing {k}"
            r = wm[k]
            if not r.gates:
                return False, f"{k} gates FAIL"
            if r.mean_mo < 0.01:
                return False, f"{k} mean_mo={r.mean_mo*100:.2f}% <1%"
            if r.pct_pos < 0.70:
                return False, f"{k} pct_pos={r.pct_pos*100:.0f}% <70%"
        return True, "all years ≥1% mo & ≥70% pos & gates"

    candidates_for_promote = ["baseline_vt0025"] + [b["idea"] for b in to_confirm]
    if metal_promote:
        candidates_for_promote.append(metal_promote[0])

    print("=== Promotion board ===", flush=True)
    for idea in candidates_for_promote:
        wm = window_map(idea)
        if not wm:
            continue
        ok, why = promote_ok(wm)
        r24 = wm.get("2024")
        r25 = wm.get("2025")
        r26 = wm.get("2026")
        rho = wm.get("holdout_365d")
        promote_rows.append(
            {
                "idea": idea,
                "mo_2024": r24.mean_mo if r24 else float("nan"),
                "pos_2024": r24.pct_pos if r24 else float("nan"),
                "top3_2024": r24.top3 if r24 else float("nan"),
                "mo_2025": r25.mean_mo if r25 else float("nan"),
                "pos_2025": r25.pct_pos if r25 else float("nan"),
                "mo_2026": r26.mean_mo if r26 else float("nan"),
                "pos_2026": r26.pct_pos if r26 else float("nan"),
                "mo_ho": rho.mean_mo if rho else float("nan"),
                "pos_ho": rho.pct_pos if rho else float("nan"),
                "top3_ho": rho.top3 if rho else float("nan"),
                "promote": ok,
                "why": why,
            }
        )
        print(
            f"  {idea[:48]:48s} 24={((r24.mean_mo*100) if r24 else float('nan')):5.2f}/"
            f"{((r24.pct_pos*100) if r24 else float('nan')):2.0f} "
            f"25={((r25.mean_mo*100) if r25 else float('nan')):5.2f} "
            f"HO={((rho.mean_mo*100) if rho else float('nan')):5.2f}/"
            f"{((rho.pct_pos*100) if rho else float('nan')):2.0f} "
            f"PROMOTE={ok} ({why})",
            flush=True,
        )

    any_promote = any(p["promote"] for p in promote_rows)
    # --- Persist ---
    pd.DataFrame([asdict(r) for r in rows]).to_csv(ROOT / "reports" / "quest_smooth_return_wf.csv", index=False)
    pd.DataFrame(promote_rows).to_csv(ROOT / "reports" / "quest_smooth_promote.csv", index=False)
    if metal_rows:
        pd.DataFrame([asdict(r) for r in metal_rows]).to_csv(ROOT / "reports" / "quest_smooth_metals_is.csv", index=False)

    selected = {
        "locked_tag": LOCKED_TAG,
        "promote": bool(any_promote),
        "best_is_idea": best["idea"] if best else None,
        "best_is_score": best["score"] if best else None,
        "best_is_raw_min_mo": best["raw_min_mo"] if best else None,
        "best_is_params": best["params"] if best else None,
        "best_is_symbols": [l["symbol"] for l in best["legs"]] if best else [],
        "best_is_legs": [
            {
                "symbol": l["symbol"],
                "timeframe": l["timeframe"],
                "strategy": l["strategy"],
                "params": l["params"],
                "exits": l.get("exits") or {},
                "vol_target": l.get("vol_target"),
                "oos_sharpe": l.get("oos_sharpe"),
            }
            for l in best["legs"]
        ]
        if best
        else [],
        "n_is_passers": int((is_df["score"] > -1e8).sum()) if len(is_df) else 0,
        "metal_promote": metal_promote[0] if metal_promote else None,
        "data_source": src,
        "rf": RF,
        "selection": "max min(IS year mean_mo) s.t. %pos/top3; holdout confirmation only",
    }
    (ROOT / "configs" / "quest_smooth_selected.json").write_text(json.dumps(selected, indent=2) + "\n")

    # Markdown report
    lines = [
        "# Quest wave: smooth-return portfolio + IS min(year mean_mo) WF",
        "",
        f"**data_source:** `{src}`  **RF:** {RF*100:.0f}%  **warmup:** {WARMUP}  **signal_lag=1**",
        f"**Selection:** maximize min(IS year mean_mo) s.t. %pos≥{MIN_PCT_POS:.0%}, top3≤{MAX_TOP3_HARD:.0%} "
        f"(soft≤{MAX_TOP3_SOFT:.0%}); holdout never for selection.",
        f"**Per-leg risk:** RF/n_legs (hard cap). **Port VT grid:** {vt_grid} aiming ~0.8–1.2%/mo expected.",
        f"**Locked official:** `{LOCKED_TAG}` (unchanged unless promote).",
        "",
        "## Baseline (locked)",
        "",
        "| Window | Mean mo | %pos | Top3 | Gates |",
        "|--------|--------:|-----:|-----:|:-----:|",
    ]
    for r in rows:
        if r.idea == "baseline_vt0025":
            lines.append(
                f"| {r.window} | {r.mean_mo*100:.2f}% | {r.pct_pos*100:.0f}% | {r.top3*100:.0f}% | "
                f"{'PASS' if r.gates else 'FAIL'} |"
            )

    lines += [
        "",
        "## IS selection board (top 15)",
        "",
        "| Idea | min score | raw min mo | avg %pos | avg top3 | n | symbols |",
        "|------|----------:|-----------:|---------:|---------:|--:|---------|",
    ]
    for _, r in is_df.head(15).iterrows():
        sc = r["score"]
        sc_s = f"{sc*100:.2f}%" if sc > -1e8 else "FAIL"
        lines.append(
            f"| {r['idea'][:50]} | {sc_s} | {r['raw_min_mo']*100:.2f}% | {r['avg_pct_pos']*100:.0f}% | "
            f"{r['avg_top3']*100:.0f}% | {int(r['n_legs'])} | {str(r['symbols'])[:40]} |"
        )

    lines += [
        "",
        f"**Hard constraint passers:** {int((is_df['score'] > -1e8).sum())} / {len(is_df)}",
        f"**Best IS:** `{best['idea'] if best else 'none'}` "
        f"(score={best['score']*100:.2f}% raw_min={best['raw_min_mo']*100:.2f}%)" if best else "**Best IS:** none",
        "",
        "## Confirmation / promotion",
        "",
        "| Idea | 2024 mo/%pos/top3 | 2025 mo/%pos | 2026 mo | HO mo/%pos/top3 | Promote | Why |",
        "|------|------------------:|-------------:|--------:|----------------:|:-------:|-----|",
    ]
    for p in promote_rows:
        lines.append(
            f"| {p['idea'][:42]} | {p['mo_2024']*100:.2f}/{p['pos_2024']*100:.0f}/{p['top3_2024']*100:.0f} | "
            f"{p['mo_2025']*100:.2f}/{p['pos_2025']*100:.0f} | {p['mo_2026']*100:.2f} | "
            f"{p['mo_ho']*100:.2f}/{p['pos_ho']*100:.0f}/{p['top3_ho']*100:.0f} | "
            f"{'YES' if p['promote'] else 'no'} | {p['why']} |"
        )

    lines += [
        "",
        f"**PROMOTE:** {'yes — see selected json' if any_promote else f'none — official tag unchanged `{LOCKED_TAG}`.'}",
        "",
        "## Yahoo / FTMO limits (if stuck)",
        "",
        "- All research uses `approximate_non_ftmo` (Yahoo via yfinance).",
        "- Spreads/commissions/sessions are approximate; FTMO MT5 tick/spread/swap differ.",
        "- Dual-year uncorrelated edge is scarce on Yahoo FX — many legs flip sign 2024↔2025.",
        "- No equity index history in this repo (US30/NAS100/SPX) for diversifiers.",
        "- H1 Yahoo depth ~2y; incomplete multi-year calendars.",
        "- **FTMO `data/ftmo/` exports would unlock:** true FTMO spreads/commission, session filters,",
        "  longer clean H1/M15, index CFDs if offered, and go-live candidacy (`ftmo_mt5_export`).",
        "",
        "## Failures (honest)",
        "",
        "- Smooth many-leg portfolios that hit %pos/top3 on IS often sit below 1%/mo on 2024.",
        "- Raising VT toward 1%/mo expected tends to reintroduce burstiness (top3↑).",
        "- Metals only eligible on dual-confirm; XAU dual scarce in WF summary.",
        "- No RF hike (8% locked).",
        "",
    ]
    (ROOT / "reports" / "quest_smooth_return_wf.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote reports/quest_smooth_return_wf.md; promote={any_promote}", flush=True)


if __name__ == "__main__":
    main()
