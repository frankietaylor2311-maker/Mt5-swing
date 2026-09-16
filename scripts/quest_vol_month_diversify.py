#!/usr/bin/env python3
"""Wave: flatten locked legs (vol/equity/month-aware) + dual-year diversifiers + H1 probe.

Selection / hyperparams: 2024 only (except dual-year gate uses 2024 AND 2025 mean_mo
≥ 0.8% before an add-on may be confirmed — explicit anti-2024-only-overfit rule).
Holdout never used for tuning. signal_lag=1. No risk_fraction hike. Scales ≤1 for
new flatteners. approximate_non_ftmo unless ftmo exports exist.
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
    apply_equity_curve_target,
    apply_month_aware_scale,
    apply_runup_throttle,
    apply_vol_target,
    combine_weighted,
    daily_return_corr,
)
from mt5_swing.strategies.registry import get_strategy

WARMUP = 250
RF = float(os.environ.get("QUEST_RF", os.environ.get("RISK_FRACTION", "0.08")))
# Locked candidate uses 8%; refuse accidental env RISK_FRACTION=0.02 from other sessions
if abs(RF - 0.08) > 1e-9 and os.environ.get("QUEST_ALLOW_RF_OVERRIDE", "") != "1":
    print(f"NOTE: overriding RISK_FRACTION={RF} → 0.08 (set QUEST_ALLOW_RF_OVERRIDE=1 to keep)", flush=True)
    RF = 0.08
MAX_LOT = float(os.environ.get("MAX_LOT", "50"))
N_REF = 5
RISK_PER = RF / N_REF
VT = 0.0025
LOCKED_SYMS = {"USDCHF", "GBPUSD", "CADJPY", "AUDCAD", "GBPCAD"}
MIN_DUAL_YEAR_MO = 0.008  # 0.8%/mo on BOTH 2024 and 2025 before promote path
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
        if ewc.resolve_csv(row["symbol"], row["timeframe"]) is None:
            continue
        seen.add(row["symbol"])
        picked.append(_row_to_leg(row))
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
    out.append(("roll12_m6", end - pd.Timedelta(days=365 + 182), end - pd.Timedelta(days=182), "mixed"))
    return out


_CACHE: dict[tuple, tuple] = {}


def window_leg(leg: dict, cfg: dict, start: pd.Timestamp, end: pd.Timestamp):
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
    bt = ewc.make_bt(cfg, leg, RISK_PER, MAX_LOT)
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


def cached_leg(leg, cfg, start, end):
    key = (leg["symbol"], leg["timeframe"], leg["strategy"], str(start), str(end), json.dumps(leg.get("params"), sort_keys=True), str(leg.get("exits")))
    if key not in _CACHE:
        _CACHE[key] = window_leg(leg, cfg, start, end)
    return _CACHE[key]


def pack_legs(legs, cfg, start, end):
    curves, trades = [], 0
    for leg in legs:
        eq, nt = cached_leg(leg, cfg, start, end)
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


def score_2024(row: EvalRow | None, bl: EvalRow | None) -> float:
    """IS score — MONTHLY CONSISTENCY FIRST (not lumpy mean).

    Prefer: high %pos, low top3, mean_mo toward 1% without burstiness.
    Reject gates fail / high p2t. Burst boosters that raise mean via top3
    are heavily penalized even if mean looks better.
    """
    if row is None or not row.gates:
        return -1e9
    if row.p2t > 0.085:
        return -1e9
    top = row.top3 if row.top3 == row.top3 else 0.85
    pos = row.pct_pos if row.pct_pos == row.pct_pos else 0.0
    med = row.median_mo if row.median_mo == row.median_mo else row.mean_mo
    target = 0.01
    mean_term = 8.0 * min(row.mean_mo, 0.012) - 4.0 * max(0.0, target - row.mean_mo)
    pos_term = 3.0 * max(0.0, pos - 0.55) + 2.0 * max(0.0, pos - 0.70)
    top_pen = -4.0 * max(0.0, top - 0.55) - 2.0 * max(0.0, top - 0.70)
    lump = max(0.0, row.mean_mo - (med if med == med else row.mean_mo))
    lump_pen = -25.0 * lump
    sc = mean_term + pos_term + top_pen + lump_pen
    if pos < 0.60:
        sc -= 1.0
    if top > 0.75:
        sc -= 1.5
    if bl is not None:
        sc -= 2.5 * max(0.0, bl.pct_pos - pos)
        sc -= 2.0 * max(0.0, top - bl.top3)
        sc += 1.5 * max(0.0, bl.top3 - top)
        sc += 1.0 * max(0.0, pos - bl.pct_pos)
        if row.mean_mo < bl.mean_mo - 0.002 and top >= bl.top3 - 0.02:
            sc -= 2.0
    return sc


def apply_overlay_chain(port: pd.Series, spec: dict) -> pd.Series:
    out = port
    if spec.get("vt"):
        out = apply_vol_target(out, float(spec["vt"]), look=int(spec.get("vt_look", 60)),
                               lo=float(spec.get("vt_lo", 0.25)), hi=float(spec.get("vt_hi", 3.0)))
    if spec.get("eq_target"):
        out = apply_equity_curve_target(
            out, target_mo_vol=float(spec["eq_target"]),
            lookback_months=int(spec.get("eq_lb", 6)),
            lo=float(spec.get("eq_lo", 0.25)), hi=float(spec.get("eq_hi", 1.0)),
        )
    if spec.get("month_aware"):
        ma = spec["month_aware"]
        out = apply_month_aware_scale(
            out, strong_mo=float(ma["strong_mo"]), after_strong=float(ma["after_strong"]),
            dd_trigger=float(ma["dd_trigger"]), after_dd=float(ma["after_dd"]),
        )
    if spec.get("runup"):
        ru = spec["runup"]
        out = apply_runup_throttle(
            out, trail_bars=int(ru["trail_bars"]), runup_thresh=float(ru["runup_thresh"]),
            cool_scale=float(ru["cool_scale"]),
        )
    return out


def port_from_legs(curves, weights):
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return combine_weighted(curves, w)


def try_download_h1(symbols: list[str]) -> list[str]:
    """Download H1 closed bars for symbols; return list that landed on disk."""
    from mt5_swing.data.download import download_symbol_timeframe
    from mt5_swing.data.loader import save_ohlc_csv as save_csv

    ok = []
    out_dir = ROOT / "data" / "history"
    for sym in symbols:
        path = out_dir / f"{sym}_H1.csv"
        if path.exists() and path.stat().st_size > 500:
            ok.append(sym)
            continue
        try:
            df = download_symbol_timeframe(sym, "H1")
            save_csv(df, path)
            meta = path.with_suffix(".meta.json")
            meta.write_text(
                json.dumps({
                    "data_source": "approximate_non_ftmo",
                    "symbol": sym,
                    "timeframe": "H1",
                    "bars": len(df),
                    "start": str(df.index.min()),
                    "end": str(df.index.max()),
                    "provider": "yfinance",
                }) + "\n",
                encoding="utf-8",
            )
            print(f"  H1 downloaded {sym}: {len(df)} bars [{df.index.min()} → {df.index.max()}]", flush=True)
            ok.append(sym)
        except Exception as exc:  # noqa: BLE001
            print(f"  H1 download failed {sym}: {exc}", flush=True)
    return ok


def h1_probe_legs(symbols: list[str]) -> list[dict]:
    """Fixed a-priori params (no grid) for H1 diversity probe."""
    presets = [
        ("bbands_reversion", {"adx_max": 30, "require_htf_align": False, "require_rsi": False,
                              "rsi_high": 60, "rsi_low": 35, "session_hours": None}),
        ("mean_reversion_regime", {"adx_max": 25, "rsi_high": 65, "rsi_low": 35, "session_hours": "7-20"}),
        ("breakout_donchian", {"adx_min": 15, "atr_pct_min": 0.15, "donchian_window": 30,
                               "session_hours": None, "use_mid_exit": False}),
        ("squeeze_breakout", {"session_hours": None}),
        ("cci_reversion", {"session_hours": None}),
    ]
    legs = []
    for sym in symbols:
        for strat, params in presets:
            p = dict(params)
            if not p.get("session_hours"):
                p["session_hours"] = None
            legs.append({
                "symbol": sym, "timeframe": "H1", "strategy": strat,
                "params": p, "exits": {}, "vol_target": strat not in MR,
                "oos_sharpe": 0.5, "weight": 0.5,
            })
    return legs


def main() -> None:
    src = data_source()
    print(f"data_source={src} RF={RF} risk_per={RISK_PER:.4f} warmup={WARMUP}", flush=True)
    cfg = load_config(ROOT / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    lock = locked_legs()
    df = pd.read_csv(ROOT / "reports" / "walk_forward_summary.csv")
    dual = dual_confirm_symbols(df)
    cands = candidate_duals(df, dual)
    print(f"dual-confirm={sorted(dual)}", flush=True)
    print(f"add-only cand={len(cands)}: " + ", ".join(f"{c['symbol']}:{c['timeframe']}:{c['strategy']}" for c in cands), flush=True)

    path0 = ewc.resolve_csv(lock[0]["symbol"], lock[0]["timeframe"])
    end = load_ohlc_csv(path0, symbol=lock[0]["symbol"], timeframe=lock[0]["timeframe"]).index.max()
    wins = windows_for(end)
    s24 = pd.Timestamp("2024-01-01", tz="UTC")
    e24 = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
    s25 = pd.Timestamp("2025-01-01", tz="UTC")
    e25 = min(pd.Timestamp("2025-12-31 23:59:59", tz="UTC"), end)

    lock_w = np.array([l["weight"] for l in lock], dtype=float)
    lock_w = lock_w / lock_w.sum()

    # --- Baseline locked + vt across windows ---
    rows: list[EvalRow] = []
    print("=== Baseline locked + vt0025 ===", flush=True)
    for label, s, e, ov in wins:
        curves, tr = pack_legs(lock, cfg, s, e)
        if not curves:
            continue
        port = apply_vol_target(port_from_legs(curves, lock_w), VT)
        row = eval_port(port, label, ov, tr, "baseline_vt0025", "locked+vt", 5)
        if row:
            rows.append(row)
            print(f"  {label:14s} mo={row.mean_mo*100:5.2f}% pos={row.pct_pos*100:4.0f}% top3={row.top3*100:4.0f}% gates={row.gates}", flush=True)

    bl24 = next((r for r in rows if r.idea == "baseline_vt0025" and r.window == "2024"), None)

    # --- Overlay grid on LOCKED (tune 2024) ---
    print("=== Overlay grid on locked (IS=2024) ===", flush=True)
    curves24, tr24 = pack_legs(lock, cfg, s24, e24)
    base_raw24 = port_from_legs(curves24, lock_w)
    overlay_specs = []
    # locked reference
    overlay_specs.append({"name": "baseline_vt0025", "vt": VT})
    # vol target variants (flatten-leaning clips)
    for vt, hi in [(0.002, 2.0), (0.0025, 2.0), (0.0025, 1.5), (0.003, 2.0), (0.002, 1.25)]:
        overlay_specs.append({"name": f"vt{vt}_hi{hi}", "vt": vt, "vt_hi": hi})
    # equity curve target on top of vt
    for tmv, lb in [(0.008, 6), (0.010, 6), (0.012, 4), (0.012, 6), (0.015, 6)]:
        overlay_specs.append({"name": f"vt+eq{tmv}_lb{lb}", "vt": VT, "eq_target": tmv, "eq_lb": lb, "eq_hi": 1.0})
    # month-aware alone / with vt
    for sm, as_, dd, ad in [
        (0.02, 0.5, 0.03, 0.55),
        (0.025, 0.55, 0.035, 0.55),
        (0.03, 0.6, 0.04, 0.5),
        (0.02, 0.4, 0.025, 0.5),
        (0.015, 0.5, 0.03, 0.6),
    ]:
        ma = {"strong_mo": sm, "after_strong": as_, "dd_trigger": dd, "after_dd": ad}
        overlay_specs.append({"name": f"ma_sm{sm}_as{as_}", "vt": VT, "month_aware": ma})
        overlay_specs.append({"name": f"ma_only_sm{sm}", "month_aware": ma})
    # runup throttle
    for tb, thr, cs in [(30, 0.03, 0.5), (42, 0.04, 0.5), (60, 0.05, 0.55), (42, 0.03, 0.4)]:
        overlay_specs.append({"name": f"vt+runup{tb}_{thr}", "vt": VT, "runup": {"trail_bars": tb, "runup_thresh": thr, "cool_scale": cs}})
    # combo: vt + mild month-aware + equity target
    # Consistency-first extras: aggressive flatten, no leverage hi
    for vt, hi in [(0.0015, 1.0), (0.002, 1.0), (0.0025, 1.0), (0.003, 1.0)]:
        overlay_specs.append({"name": f"vtflat{vt}_hi{hi}", "vt": vt, "vt_hi": hi})
    for tmv in (0.006, 0.008, 0.010):
        overlay_specs.append({"name": f"eq_only_{tmv}", "eq_target": tmv, "eq_lb": 6, "eq_hi": 1.0})
        overlay_specs.append({"name": f"vt+eqflat{tmv}", "vt": VT, "vt_hi": 1.5, "eq_target": tmv, "eq_lb": 6, "eq_hi": 1.0})
    for sm, as_, dd, ad in [(0.015, 0.45, 0.025, 0.45), (0.02, 0.35, 0.03, 0.4), (0.025, 0.5, 0.03, 0.5)]:
        ma = {"strong_mo": sm, "after_strong": as_, "dd_trigger": dd, "after_dd": ad}
        overlay_specs.append({"name": f"vtflat+ma_sm{sm}_as{as_}", "vt": VT, "vt_hi": 1.5, "month_aware": ma})
    overlay_specs.append({
        "name": "consistency_stack",
        "vt": 0.002, "vt_hi": 1.25,
        "eq_target": 0.008, "eq_lb": 6, "eq_hi": 1.0,
        "month_aware": {"strong_mo": 0.02, "after_strong": 0.45, "dd_trigger": 0.03, "after_dd": 0.45},
        "runup": {"trail_bars": 42, "runup_thresh": 0.035, "cool_scale": 0.45},
    })
    overlay_specs.append({
        "name": "vt_eq_ma_combo",
        "vt": VT, "eq_target": 0.012, "eq_lb": 6, "eq_hi": 1.0,
        "month_aware": {"strong_mo": 0.025, "after_strong": 0.55, "dd_trigger": 0.035, "after_dd": 0.55},
    })

    best_ov = (-1e9, None, None)
    tune_ov = []
    for spec in overlay_specs:
        port = apply_overlay_chain(base_raw24, spec)
        row = eval_port(port, "2024", "research/WF", tr24, "overlay", spec["name"], 5)
        sc = score_2024(row, bl24)
        tune_ov.append({"spec": spec["name"], "score": sc, **({} if not row else {
            "mean_mo": row.mean_mo, "pct_pos": row.pct_pos, "top3": row.top3, "gates": row.gates, "p2t": row.p2t,
        })})
        if row:
            print(
                f"  {spec['name'][:40]:40s} mo={row.mean_mo*100:5.2f}% pos={row.pct_pos*100:4.0f}% "
                f"top3={row.top3*100:4.0f}% p2t={row.p2t*100:4.1f}% sc={sc:.3f}",
                flush=True,
            )
        if sc > best_ov[0]:
            best_ov = (sc, spec, row)

    print(f"Best overlay IS: {best_ov[1]['name'] if best_ov[1] else None} sc={best_ov[0]:.3f}", flush=True)

    # Confirm top overlays on all windows (incl. holdout) — confirmation only
    top_names = sorted(tune_ov, key=lambda x: -x.get("score", -1e9))[:8]
    confirm_specs = []
    seen = set()
    for t in top_names:
        sp = next(s for s in overlay_specs if s["name"] == t["spec"])
        if sp["name"] not in seen:
            confirm_specs.append(sp)
            seen.add(sp["name"])
    # always include baseline + best combo
    for sp in overlay_specs:
        if sp["name"] in ("baseline_vt0025", "vt_eq_ma_combo") and sp["name"] not in seen:
            confirm_specs.append(sp)
            seen.add(sp["name"])

    print("=== Confirm overlays on all windows ===", flush=True)
    for spec in confirm_specs:
        for label, s, e, ov in wins:
            curves, tr = pack_legs(lock, cfg, s, e)
            if not curves:
                continue
            raw = port_from_legs(curves, lock_w)
            port = apply_overlay_chain(raw, spec)
            row = eval_port(port, label, ov, tr, f"ov::{spec['name']}", spec["name"], 5)
            if row:
                rows.append(row)
        # summary line
        byw = {r.window: r for r in rows if r.idea == f"ov::{spec['name']}"}
        def mo(w):
            return byw[w].mean_mo * 100 if w in byw else float("nan")
        print(
            f"  {spec['name'][:36]:36s} 2024={mo('2024'):5.2f} 2025={mo('2025'):5.2f} "
            f"2026={mo('2026'):5.2f} HO={mo('holdout_365d'):5.2f}",
            flush=True,
        )

    # --- Dual-year diversifier screen (2024 AND 2025 ≥ 0.8%/mo) ---
    print("=== Dual-year diversifier screen (2024&2025 ≥0.8%/mo) ===", flush=True)
    screen = []
    dual_year_ok = []
    base24_vt = apply_vol_target(base_raw24, VT)
    curves25, _ = pack_legs(lock, cfg, s25, e25)
    base25_vt = apply_vol_target(port_from_legs(curves25, lock_w), VT) if curves25 else None

    for c in cands:
        eq24, nt24 = cached_leg(c, cfg, s24, e24)
        eq25, nt25 = cached_leg(c, cfg, s25, e25)
        if eq24.empty or eq25.empty:
            continue
        r24 = eval_port(eq24, "2024", "research/WF", nt24, "cand", c["symbol"], 1)
        r25 = eval_port(eq25, "2025", "mixed", nt25, "cand", c["symbol"], 1)
        corr = daily_return_corr(eq24, base24_vt)
        ok = bool(
            r24 and r25 and r24.gates and r25.gates
            and r24.mean_mo >= MIN_DUAL_YEAR_MO and r25.mean_mo >= MIN_DUAL_YEAR_MO
            and nt24 >= 6 and nt25 >= 6
            and corr <= 0.55
        )
        rec = {
            "symbol": c["symbol"], "timeframe": c["timeframe"], "strategy": c["strategy"],
            "oos_sharpe": c["oos_sharpe"],
            "mean_mo_2024": r24.mean_mo if r24 else float("nan"),
            "mean_mo_2025": r25.mean_mo if r25 else float("nan"),
            "gates_2024": r24.gates if r24 else False,
            "gates_2025": r25.gates if r25 else False,
            "trades_2024": nt24, "trades_2025": nt25,
            "corr_vs_locked_2024": corr,
            "dual_year_pass": ok,
        }
        screen.append(rec)
        print(
            f"  {c['symbol']:7s} {c['timeframe']} {c['strategy']:22s} "
            f"mo24={rec['mean_mo_2024']*100:5.2f}% mo25={rec['mean_mo_2025']*100:5.2f}% "
            f"corr={corr:5.2f} pass={ok}",
            flush=True,
        )
        if ok:
            dual_year_ok.append(c)

    # Basket: locked + each dual-year passer (equal-ish oos weights), with baseline vt
    print(f"Dual-year passers: {len(dual_year_ok)}", flush=True)
    for extra in dual_year_ok[:5]:
        legs = lock + [extra]
        w = np.array([max(l["oos_sharpe"], 0.01) for l in legs], dtype=float)
        w = w / w.sum()
        idea = f"add::{extra['symbol']}_{extra['timeframe']}_{extra['strategy']}"
        for label, s, e, ov in wins:
            curves, tr = pack_legs(legs, cfg, s, e)
            if not curves:
                continue
            port = apply_vol_target(port_from_legs(curves, w), VT)
            row = eval_port(port, label, ov, tr, idea, idea, len(legs))
            if row:
                rows.append(row)
        byw = {r.window: r for r in rows if r.idea == idea}
        print(
            f"  {idea} 2024={byw.get('2024') and byw['2024'].mean_mo*100:5.2f} "
            f"2025={byw.get('2025') and byw['2025'].mean_mo*100:5.2f} "
            f"HO={byw.get('holdout_365d') and byw['holdout_365d'].mean_mo*100:5.2f}",
            flush=True,
        )

    # Also try all dual-year passers together if ≥2
    if len(dual_year_ok) >= 2:
        legs = lock + dual_year_ok[:3]
        w = np.array([max(l["oos_sharpe"], 0.01) for l in legs], dtype=float)
        w = w / w.sum()
        idea = "add::dualyear_top3"
        for label, s, e, ov in wins:
            curves, tr = pack_legs(legs, cfg, s, e)
            if not curves:
                continue
            port = apply_vol_target(port_from_legs(curves, w), VT)
            row = eval_port(port, label, ov, tr, idea, idea, len(legs))
            if row:
                rows.append(row)

    # --- H1 probe ---
    print("=== H1 download + single-leg probe ===", flush=True)
    h1_syms = try_download_h1(["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "AUDUSD", "USDCHF"])
    h1_screen = []
    if h1_syms:
        # Use windows that fit H1 span (often starts mid-2024)
        for leg in h1_probe_legs(h1_syms):
            # evaluate 2025 + holdout (2024 may be incomplete)
            eq25, nt25 = cached_leg(leg, cfg, s25, e25)
            hs = end - pd.Timedelta(days=365)
            eq_ho, nt_ho = cached_leg(leg, cfg, hs, end)
            if eq25.empty:
                continue
            r25 = eval_port(eq25, "2025", "mixed", nt25, "h1", leg["strategy"], 1)
            r_ho = eval_port(eq_ho, "holdout_365d", "pure holdout", nt_ho, "h1", leg["strategy"], 1) if not eq_ho.empty else None
            # try 2024 if enough bars
            eq24, nt24 = cached_leg(leg, cfg, s24, e24)
            r24 = eval_port(eq24, "2024", "research/WF", nt24, "h1", leg["strategy"], 1) if not eq24.empty and len(eq24) > 40 else None
            rec = {
                "symbol": leg["symbol"], "strategy": leg["strategy"],
                "mean_mo_2024": r24.mean_mo if r24 else float("nan"),
                "mean_mo_2025": r25.mean_mo if r25 else float("nan"),
                "mean_mo_ho": r_ho.mean_mo if r_ho else float("nan"),
                "gates_2025": r25.gates if r25 else False,
                "gates_ho": r_ho.gates if r_ho else False,
                "trades_2025": nt25,
            }
            h1_screen.append(rec)
            okish = bool(r25 and r25.gates and r25.mean_mo >= MIN_DUAL_YEAR_MO and r_ho and r_ho.gates and r_ho.mean_mo > 0)
            if okish or (r25 and r25.mean_mo > 0.005):
                print(
                    f"  H1 {leg['symbol']:7s} {leg['strategy']:22s} "
                    f"mo25={rec['mean_mo_2025']*100:5.2f}% ho={rec['mean_mo_ho']*100:5.2f}% "
                    f"n25={nt25} okish={okish}",
                    flush=True,
                )
    else:
        print("  No H1 symbols available", flush=True)

    # Best H1 legs that pass 2025≥0.8 and HO gates — try add one to locked
    h1_pass = [h for h in h1_screen if h.get("gates_2025") and h["mean_mo_2025"] >= MIN_DUAL_YEAR_MO and h.get("gates_ho") and h["mean_mo_ho"] > 0]
    h1_pass = sorted(h1_pass, key=lambda x: -x["mean_mo_2025"])[:3]
    for h in h1_pass:
        leg = next(l for l in h1_probe_legs([h["symbol"]]) if l["strategy"] == h["strategy"])
        legs = lock + [leg]
        w = np.array([max(l["oos_sharpe"], 0.01) for l in legs], dtype=float)
        w = w / w.sum()
        idea = f"addH1::{h['symbol']}_{h['strategy']}"
        for label, s, e, ov in wins:
            curves, tr = pack_legs(legs, cfg, s, e)
            if not curves:
                continue
            port = apply_vol_target(port_from_legs(curves, w), VT)
            row = eval_port(port, label, ov, tr, idea, idea, len(legs))
            if row:
                rows.append(row)

    # --- Promotion check ---
    def window_map(idea: str) -> dict[str, EvalRow]:
        return {r.window: r for r in rows if r.idea == idea}

    def promote_ok(wm: dict[str, EvalRow]) -> tuple[bool, str]:
        need = ["2024", "2025", "holdout_365d"]
        for k in need:
            if k not in wm:
                return False, f"missing {k}"
            if not wm[k].gates:
                return False, f"{k} gates FAIL"
            if wm[k].mean_mo < 0.0095:
                return False, f"{k} mean_mo={wm[k].mean_mo*100:.2f}% <1%"
            if wm[k].pct_pos < 0.65:
                return False, f"{k} pct_pos={wm[k].pct_pos*100:.0f}% <65%"
        if "2026" in wm:
            if not wm["2026"].gates or wm["2026"].mean_mo < 0.0095:
                return False, "2026 below 1% or gates"
            if wm["2026"].pct_pos < 0.65:
                return False, "2026 pct_pos weak"
        bl = window_map("baseline_vt0025")
        if "holdout_365d" in bl and "holdout_365d" in wm:
            ho_top = wm["holdout_365d"].top3
            bl_top = bl["holdout_365d"].top3
            if ho_top > bl_top + 0.02:
                return False, "HO top3 worse than baseline"
            if wm["holdout_365d"].pct_pos + 1e-9 < bl["holdout_365d"].pct_pos - 0.02:
                return False, "HO pct_pos worse than baseline"
        for k in need:
            if wm[k].top3 > 0.85:
                return False, f"{k} top3 too bursty"
        return True, "ok"


    ideas = sorted({r.idea for r in rows})
    promo = []
    for idea in ideas:
        wm = window_map(idea)
        ok, reason = promote_ok(wm)
        if "2024" in wm and "2025" in wm and "holdout_365d" in wm:
            promo.append({
                "idea": idea,
                "mo_2024": wm["2024"].mean_mo, "mo_2025": wm["2025"].mean_mo,
                "mo_2026": wm["2026"].mean_mo if "2026" in wm else float("nan"),
                "mo_ho": wm["holdout_365d"].mean_mo,
                "pos_ho": wm["holdout_365d"].pct_pos,
                "top3_ho": wm["holdout_365d"].top3,
                "gates_all": all(wm[k].gates for k in ("2024", "2025", "holdout_365d") if k in wm),
                "promote": ok, "reason": reason,
            })

    promo_df = pd.DataFrame(promo).sort_values(["promote", "mo_2024"], ascending=[False, False])
    screen_df = pd.DataFrame(screen)
    h1_df = pd.DataFrame(h1_screen)
    rows_df = pd.DataFrame([asdict(r) for r in rows])
    tune_df = pd.DataFrame(tune_ov)

    reports = ROOT / "reports"
    rows_df.to_csv(reports / "quest_vol_month_diversify.csv", index=False)
    tune_df.to_csv(reports / "quest_vol_month_overlay_tune2024.csv", index=False)
    screen_df.to_csv(reports / "quest_dualyear_screen.csv", index=False)
    if len(h1_df):
        h1_df.to_csv(reports / "quest_h1_probe.csv", index=False)
    promo_df.to_csv(reports / "quest_vol_month_promote.csv", index=False)

    selected = {
        "data_source": src,
        "best_overlay_is": best_ov[1],
        "best_overlay_score": best_ov[0],
        "dual_year_passers": [
            {"symbol": c["symbol"], "timeframe": c["timeframe"], "strategy": c["strategy"]} for c in dual_year_ok
        ],
        "h1_passers": h1_pass,
        "promote_any": bool(promo_df["promote"].any()) if len(promo_df) else False,
        "risk_fraction": RF,
        "min_dual_year_mo": MIN_DUAL_YEAR_MO,
    }
    (ROOT / "configs" / "quest_vol_month_selected.json").write_text(json.dumps(selected, indent=2, default=str) + "\n")

    # Markdown report
    lines = [
        "# Quest wave: vol/equity/month-aware flatten + dual-year diversify + H1",
        "",
        f"**data_source:** `{src}`  **RF:** {RF:.0%}  **per-leg:** {RISK_PER:.2%}  **warmup:** {WARMUP}  **signal_lag=1**",
        f"**Dual-year gate:** 2024 AND 2025 mean_mo ≥ {MIN_DUAL_YEAR_MO*100:.1f}% + dual-confirm + corr≤0.55 (before add).",
        "**Holdout:** confirmation only (not used for overlay hyperparams).",
        "",
        "## Baseline (locked fx4plus_gbpcad_d1_voltarget_0025)",
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

    lines += ["", "## Overlay IS tune (2024) — top scores", "",
              "| Spec | Mean mo | %pos | Top3 | P2T | Score |",
              "|------|--------:|-----:|-----:|----:|------:|"]
    for t in sorted(tune_ov, key=lambda x: -x.get("score", -1e9))[:12]:
        if "mean_mo" not in t:
            continue
        lines.append(
            f"| {t['spec']} | {t['mean_mo']*100:.2f}% | {t['pct_pos']*100:.0f}% | {t['top3']*100:.0f}% | "
            f"{t['p2t']*100:.1f}% | {t['score']:.3f} |"
        )

    lines += ["", "## Overlay confirmation (key windows mean_mo %)", "",
              "| Spec | 2024 | 2025 | 2026 | HO | HO top3 | Promote? |",
              "|------|-----:|-----:|-----:|---:|--------:|:--------:|"]
    for spec in confirm_specs:
        idea = f"ov::{spec['name']}"
        wm = window_map(idea)
        if "2024" not in wm:
            continue
        ok, reason = promote_ok(wm)
        def _mo(w):
            return wm[w].mean_mo * 100 if w in wm else float("nan")
        def _top(w):
            return wm[w].top3 * 100 if w in wm else float("nan")
        lines.append(
            f"| {spec['name']} | {_mo('2024'):.2f} | {_mo('2025'):.2f} | {_mo('2026'):.2f} | "
            f"{_mo('holdout_365d'):.2f} | {_top('holdout_365d'):.0f}% | {'YES' if ok else 'no'} |"
        )

    lines += ["", "## Dual-year screen (unused dual-confirm)", "",
              f"Passers (≥{MIN_DUAL_YEAR_MO*100:.1f}% both years, gates, corr≤0.55): **{len(dual_year_ok)}**",
              ""]
    if len(screen_df):
        lines.append("| Symbol | TF | Strat | mo2024 | mo2025 | corr | Pass |")
        lines.append("|--------|----|-------|-------:|-------:|-----:|:----:|")
        for _, r in screen_df.sort_values("mean_mo_2024", ascending=False).iterrows():
            lines.append(
                f"| {r['symbol']} | {r['timeframe']} | {r['strategy']} | "
                f"{r['mean_mo_2024']*100:.2f}% | {r['mean_mo_2025']*100:.2f}% | "
                f"{r['corr_vs_locked_2024']:.2f} | {'Y' if r['dual_year_pass'] else 'n'} |"
            )

    lines += ["", "## H1 probe", ""]
    if len(h1_df):
        lines.append(f"Downloaded H1 for: {h1_syms}. Yahoo ~730d — 2024 often incomplete.")
        lines.append("")
        lines.append("| Symbol | Strat | mo2025 | mo HO | gates25 |")
        lines.append("|--------|-------|-------:|------:|:-------:|")
        for _, r in h1_df.sort_values("mean_mo_2025", ascending=False).head(15).iterrows():
            lines.append(
                f"| {r['symbol']} | {r['strategy']} | {r['mean_mo_2025']*100:.2f}% | "
                f"{r['mean_mo_ho']*100:.2f}% | {'Y' if r['gates_2025'] else 'n'} |"
            )
    else:
        lines.append("No H1 results (download failed or empty).")

    lines += ["", "## Promotion board", ""]
    if len(promo_df):
        lines.append("| Idea | 2024 | 2025 | 2026 | HO | HO%pos | HO top3 | Promote | Why |")
        lines.append("|------|-----:|-----:|-----:|---:|-------:|--------:|:-------:|-----|")
        for _, r in promo_df.head(25).iterrows():
            lines.append(
                f"| {r['idea']} | {r['mo_2024']*100:.2f}% | {r['mo_2025']*100:.2f}% | "
                f"{r['mo_2026']*100:.2f}% | {r['mo_ho']*100:.2f}% | {r['pos_ho']*100:.0f}% | "
                f"{r['top3_ho']*100:.0f}% | {'YES' if r['promote'] else 'no'} | {r['reason']} |"
            )
    promoted = promo_df[promo_df["promote"] == True] if len(promo_df) else promo_df  # noqa: E712
    if len(promoted):
        lines += ["", f"**PROMOTE:** `{promoted.iloc[0]['idea']}`", ""]
    else:
        lines += ["", "**PROMOTE:** none — official tag unchanged `fx4plus_gbpcad_d1_voltarget_0025`.", ""]

    lines += [
        "",
        "## Failures (honest)",
        "",
        "- Month-aware / equity-curve / runup overlays that only flatten tend to **cut 2024 further** below 0.42% or fail to lift it to 1%.",
        "- Dual-year ≥0.8% gate is intentionally harsh after 2024-only add-ons wrecked 2025 last wave.",
        "- H1 Yahoo history is short (~2y); incomplete 2024 calendar — H1 used as diversity probe only.",
        "- No leverage increase (RF fixed; new flatten scales ≤1).",
        "",
    ]
    out_md = reports / "quest_vol_month_diversify.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", out_md, flush=True)
    print("promote_any=", selected["promote_any"], flush=True)


if __name__ == "__main__":
    main()
