#!/usr/bin/env python3
"""Wave: regime sleeves / corr-throttle / hot-streak / roll-IS (IS=2024 only).

Per-window backtests with warmup (avoids kill-switch contamination from full history).
Holdout never for selection. signal_lag=1. approximate_non_ftmo unless ftmo exports.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass, asdict
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
from mt5_swing.features.indicators import apply_feature_pipeline
from mt5_swing.strategies.registry import get_strategy

INITIAL = 100_000.0
WARMUP = 250
RF = float(os.environ.get("RISK_FRACTION", "0.08"))
MAX_LOT = float(os.environ.get("MAX_LOT", "50"))


def load_summary_leg(symbol: str, tf: str, strategy: str, exits: dict | None = None, vol_target: bool = True) -> dict:
    df = pd.read_csv(ROOT / "reports" / "walk_forward_summary.csv")
    hit = df[(df["symbol"] == symbol) & (df["timeframe"] == tf) & (df["strategy"] == strategy)]
    if hit.empty:
        raise SystemExit(f"Missing {symbol} {tf} {strategy}")
    r = hit.iloc[0]
    params = json.loads(r["params"]) if isinstance(r["params"], str) else (r["params"] or {})
    if not params.get("session_hours"):
        params["session_hours"] = None
    return {
        "symbol": symbol, "timeframe": tf, "strategy": strategy,
        "params": params, "exits": exits or {}, "vol_target": vol_target,
        "oos_sharpe": float(r["oos_sharpe"]), "oos_return": float(r["oos_return"]),
        "oos_trades": int(r["oos_trades"]), "weight": float(r["oos_sharpe"]),
    }


def locked_fx4_plus_gbpcad_d1() -> list[dict]:
    locked = yaml.safe_load((ROOT / "configs" / "best_interim_approximate.yaml").read_text())
    legs = []
    for c in locked["candidates"]:
        params = dict(c.get("params") or {})
        if not params.get("session_hours"):
            params["session_hours"] = None
        legs.append({
            "symbol": c["symbol"], "timeframe": c["timeframe"], "strategy": c["strategy"],
            "params": params, "exits": dict(c.get("exits") or {}),
            "vol_target": bool(c.get("vol_target", False)),
            "oos_sharpe": float(c.get("oos_sharpe") or 0.01),
            "weight": float(c.get("weight") or c.get("oos_sharpe") or 0.01),
        })
    legs.append(load_summary_leg("GBPCAD", "D1", "bbands_reversion", exits={}, vol_target=True))
    w = np.array([max(l["oos_sharpe"], 0.01) for l in legs], dtype=float)
    w = w / w.sum()
    for i, l in enumerate(legs):
        l["weight"] = float(w[i])
    return legs


def regime_sleeves() -> tuple[list[dict], list[dict]]:
    locked = yaml.safe_load((ROOT / "configs" / "best_interim_approximate.yaml").read_text())
    by_key = {(c["symbol"], c["timeframe"], c["strategy"]): c for c in locked["candidates"]}

    def from_locked(sym, tf, st):
        c = by_key[(sym, tf, st)]
        params = dict(c.get("params") or {})
        if not params.get("session_hours"):
            params["session_hours"] = None
        return {
            "symbol": sym, "timeframe": tf, "strategy": st,
            "params": params, "exits": dict(c.get("exits") or {}),
            "vol_target": bool(c.get("vol_target", False)),
            "oos_sharpe": float(c.get("oos_sharpe") or 0.01),
            "weight": float(c.get("oos_sharpe") or 0.01),
        }

    mr = [
        from_locked("USDCHF", "H4", "bbands_reversion"),
        from_locked("CADJPY", "H4", "mean_reversion_regime"),
        from_locked("AUDCAD", "H4", "mean_reversion_regime"),
        load_summary_leg("EURCHF", "H4", "cci_reversion", vol_target=True),
    ]
    tr = [
        from_locked("GBPUSD", "H4", "breakout_donchian"),
        load_summary_leg("GBPJPY", "H4", "squeeze_breakout", vol_target=True),
        load_summary_leg("EURJPY", "H4", "breakout_donchian", vol_target=True),
        load_summary_leg("GBPCAD", "H4", "atr_channel_breakout", vol_target=True),
    ]
    for sleeve in (mr, tr):
        w = np.array([max(l["oos_sharpe"], 0.01) for l in sleeve], dtype=float)
        w = w / w.sum()
        for i, l in enumerate(sleeve):
            l["weight"] = float(w[i])
    return mr, tr


def window_leg_equity(leg: dict, cfg: dict, risk_per: float, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, int]:
    path = ewc.resolve_csv(leg["symbol"], leg["timeframe"])
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
    strat = get_strategy(leg["strategy"], **params)
    res = run_backtest(ohlc_run, strat, bt)
    eq = res.equity.loc[res.equity.index >= start]
    eq = eq.loc[eq.index <= end]
    n_trades = 0
    if res.trades is not None and len(res.trades):
        tr = res.trades
        # trades may be DataFrame or list
        if isinstance(tr, pd.DataFrame):
            if "exit_time" in tr.columns:
                n_trades = int(((tr["exit_time"] >= start) & (tr["exit_time"] <= end)).sum())
            else:
                n_trades = len(tr)
        else:
            n_trades = len(tr)
    return eq.rename(f"{leg['symbol']}_{leg['strategy']}"), n_trades


def combine_weighted(curves: list[pd.Series], weights: np.ndarray) -> pd.Series:
    eq = pd.concat(curves, axis=1, sort=True).sort_index().ffill().dropna(how="any")
    if eq.empty:
        return eq
    norms = eq / eq.iloc[0]
    return (norms * weights).sum(axis=1) * INITIAL


def apply_vol_target(port: pd.Series, target: float, look: int = 60, lo: float = 0.25, hi: float = 3.0) -> pd.Series:
    r = port.pct_change()
    trail = r.shift(1).rolling(look, min_periods=max(20, look // 3)).std()
    scale = (target / trail.replace(0, np.nan)).clip(lo, hi).fillna(1.0)
    return (1.0 + r.fillna(0) * scale).cumprod() * float(port.iloc[0])


def eurusd_adx_slice(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    path = ewc.resolve_csv("EURUSD", "H4")
    ohlc = load_ohlc_csv(path, symbol="EURUSD", timeframe="H4")
    # include warmup for ADX stability
    pre = ohlc.loc[ohlc.index < start]
    warm = pre.iloc[-WARMUP:] if len(pre) >= WARMUP else pre
    chunk = pd.concat([warm, ohlc.loc[(ohlc.index >= start) & (ohlc.index <= end)]])
    feat = apply_feature_pipeline(chunk, signal_lag=1)
    return feat["adx"].loc[(feat.index >= start) & (feat.index <= end)]


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


def eval_port(port: pd.Series, label: str, overlap: str, trades: int, idea: str, params: str) -> EvalRow | None:
    if port is None or len(port) < 30:
        return None
    # rebase
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
        n_months=ms["n_months"], trades=trades, overlap=overlap,
    )


def score_2024(row: EvalRow | None) -> float:
    if row is None or not row.gates:
        return -1e9
    smooth = -0.05 * max(0.0, (row.top3 if row.top3 == row.top3 else 0.7) - 0.60)
    trade_pen = -0.2 if row.trades < 25 else 0.0
    p2t_pen = -0.5 if row.p2t > 0.085 else 0.0
    return row.mean_mo * 10.0 + smooth + trade_pen + p2t_pen


def build_regime(mr_c, tr_c, mr_w, tr_w, adx, thresh, w_hi, w_lo) -> pd.Series:
    mr = combine_weighted(mr_c, mr_w)
    tr = combine_weighted(tr_c, tr_w)
    both = pd.concat([mr.rename("mr"), tr.rename("tr")], axis=1).dropna()
    if both.empty:
        return pd.Series(dtype=float)
    a = adx.reindex(both.index).ffill()
    w_tr = pd.Series(np.where(a.values >= thresh, w_hi, w_lo), index=both.index).shift(1).fillna(0.5)
    r = w_tr * both["tr"].pct_change().fillna(0) + (1 - w_tr) * both["mr"].pct_change().fillna(0)
    return (1.0 + r).cumprod() * INITIAL


def build_corr(curves, base_w, look, corr0, k) -> pd.Series:
    eq = pd.concat(curves, axis=1, sort=True).sort_index().ffill().dropna(how="any")
    if eq.empty:
        return pd.Series(dtype=float)
    rets = eq.pct_change()
    scales = []
    for i in range(rets.shape[1]):
        others = rets.drop(rets.columns[i], axis=1).mean(axis=1)
        c = rets.iloc[:, i].rolling(look, min_periods=max(15, look // 3)).corr(others).shift(1)
        excess = (c - corr0).clip(lower=0)
        scales.append((1.0 / (1.0 + k * excess)).clip(0.25, 1.0).fillna(1.0))
    S = pd.concat(scales, axis=1)
    w = base_w.reshape(1, -1) * S.values
    w = np.where(np.isfinite(w), w, 0.0)
    rs = w.sum(axis=1, keepdims=True)
    rs = np.where(rs > 0, rs, 1.0)
    w = w / rs
    r = (rets.fillna(0).values * w).sum(axis=1)
    return pd.Series((1.0 + r).cumprod() * INITIAL, index=eq.index)


def build_hotstreak(port, look, hot, cool, cold, heat) -> pd.Series:
    r = port.pct_change()
    trail = r.rolling(look, min_periods=max(5, look // 3)).sum().shift(1)
    scale = pd.Series(1.0, index=port.index)
    scale = scale.where(~(trail > hot), cool)
    scale = scale.where(~(trail < cold), heat)
    scale = scale.fillna(1.0).clip(0.25, 2.0)
    return (1.0 + r.fillna(0) * scale).cumprod() * float(port.iloc[0])


def build_roll_sharpe(curves, look) -> pd.Series:
    eq = pd.concat(curves, axis=1, sort=True).sort_index().ffill().dropna(how="any")
    rets = eq.pct_change()
    mu = rets.rolling(look, min_periods=max(20, look // 3)).mean()
    sd = rets.rolling(look, min_periods=max(20, look // 3)).std().replace(0, np.nan)
    sh = (mu / sd).clip(lower=0).fillna(0).shift(1)
    wsum = sh.sum(axis=1).replace(0, np.nan)
    w = sh.div(wsum, axis=0).fillna(1.0 / sh.shape[1])
    r = (rets.fillna(0).values * w.values).sum(axis=1)
    return pd.Series((1.0 + r).cumprod() * INITIAL, index=eq.index)


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


def curves_for_legs(legs, cfg, risk_per, start, end):
    curves, trades = [], 0
    for leg in legs:
        eq, nt = window_leg_equity(leg, cfg, risk_per, start, end)
        if eq.empty:
            return [], 0
        curves.append(eq)
        trades += nt
    return curves, trades


def main() -> None:
    cfg = load_config(ROOT / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    has_ftmo = any((ROOT / "data" / "ftmo").glob("*.csv"))
    data_src = "ftmo_mt5_export" if has_ftmo else "approximate_non_ftmo"
    print(f"data_source={data_src} RF={RF} warmup={WARMUP}", flush=True)

    mr_legs, tr_legs = regime_sleeves()
    locked_legs = locked_fx4_plus_gbpcad_d1()
    mr_w = np.array([l["weight"] for l in mr_legs], dtype=float)
    tr_w = np.array([l["weight"] for l in tr_legs], dtype=float)
    lock_w = np.array([l["weight"] for l in locked_legs], dtype=float)
    lock_w = lock_w / lock_w.sum()

    path0 = ewc.resolve_csv(locked_legs[0]["symbol"], locked_legs[0]["timeframe"])
    end = load_ohlc_csv(path0, symbol=locked_legs[0]["symbol"], timeframe=locked_legs[0]["timeframe"]).index.max()
    wins = windows_for(end)

    # Narrow grids (resource-aware)
    grid_a = [(t, wh, wl) for t in (18.0, 22.0, 25.0) for wh in (0.7, 0.85) for wl in (0.15, 0.3)]
    grid_b = [(look, c0, k) for look in (40, 60, 90) for c0 in (0.25, 0.45) for k in (1.5, 3.0)]
    grid_c = [(look, hot, cool) for look in (12, 20, 40) for hot in (0.02, 0.035) for cool in (0.55, 0.75)]
    grid_d = [60, 90, 120]

    results: list[EvalRow] = []
    selected = {}

    # ---- tune on 2024 only ----
    s24 = pd.Timestamp("2024-01-01", tz="UTC")
    e24 = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")

    print("Building 2024 curves...", flush=True)
    risk_mr = RF / len(mr_legs)
    risk_tr = RF / len(tr_legs)
    risk_lock = RF / len(locked_legs)

    mr24, tr_mr = curves_for_legs(mr_legs, cfg, risk_mr, s24, e24)
    tr24, tr_tr = curves_for_legs(tr_legs, cfg, risk_tr, s24, e24)
    lk24, tr_lk = curves_for_legs(locked_legs, cfg, risk_lock, s24, e24)
    adx24 = eurusd_adx_slice(s24, e24)
    print(f"  trades MR={tr_mr} TR={tr_tr} LOCK={tr_lk}", flush=True)

    best_a = (-1e9, None, None)
    print("\n=== Idea A tune 2024 ===", flush=True)
    for thresh, wh, wl in grid_a:
        port = build_regime(mr24, tr24, mr_w, tr_w, adx24, thresh, wh, wl)
        row = eval_port(port, "2024", "research/WF", tr_mr + tr_tr, "regime_sleeves",
                        f"adx>={thresh}|wh={wh}|wl={wl}")
        sc = score_2024(row)
        if row:
            print(f"  A mo={row.mean_mo*100:.2f}% pos={row.pct_pos*100:.0f}% top3={row.top3*100:.0f}% "
                  f"gates={row.gates} p2t={row.p2t*100:.1f}% | {row.params} sc={sc:.3f}", flush=True)
        if sc > best_a[0]:
            best_a = (sc, (thresh, wh, wl), row)
    if best_a[1]:
        selected["regime_sleeves"] = {"thresh": best_a[1][0], "w_tr_hi": best_a[1][1], "w_tr_lo": best_a[1][2]}
        print("SELECTED A", selected["regime_sleeves"], f"2024 mo={best_a[2].mean_mo*100:.2f}%", flush=True)

    base24 = combine_weighted(lk24, lock_w)
    base24_vt = apply_vol_target(base24, 0.0025)

    best_b = (-1e9, None, None)
    print("\n=== Idea B tune 2024 ===", flush=True)
    for look, c0, k in grid_b:
        port = build_corr(lk24, lock_w, look, c0, k)
        # also try with vt after corr
        for name, ptry in (("raw", port), ("vt0025", apply_vol_target(port, 0.0025))):
            row = eval_port(ptry, "2024", "research/WF", tr_lk, "corr_throttle",
                            f"look={look}|c0={c0}|k={k}|{name}")
            sc = score_2024(row)
            if row:
                print(f"  B mo={row.mean_mo*100:.2f}% pos={row.pct_pos*100:.0f}% top3={row.top3*100:.0f}% "
                      f"gates={row.gates} | {row.params} sc={sc:.3f}", flush=True)
            if sc > best_b[0]:
                best_b = (sc, (look, c0, k, name), row)
    if best_b[1]:
        selected["corr_throttle"] = {"look": best_b[1][0], "corr0": best_b[1][1], "k": best_b[1][2], "overlay": best_b[1][3]}
        print("SELECTED B", selected["corr_throttle"], f"2024 mo={best_b[2].mean_mo*100:.2f}%", flush=True)

    best_c = (-1e9, None, None)
    print("\n=== Idea C tune 2024 ===", flush=True)
    for look, hot, cool in grid_c:
        cold, heat = -hot, min(1.0 / cool, 1.45)
        for bname, bport in (("raw", base24), ("vt0025", base24_vt)):
            port = build_hotstreak(bport, look, hot, cool, cold, heat)
            row = eval_port(port, "2024", "research/WF", tr_lk, "hotstreak",
                            f"{bname}|look={look}|hot={hot}|cool={cool}")
            sc = score_2024(row)
            if row and sc > best_c[0]:
                best_c = (sc, (bname, look, hot, cool, heat), row)
                print(f"  C best-so-far mo={row.mean_mo*100:.2f}% | {row.params} sc={sc:.3f}", flush=True)
    if best_c[1]:
        selected["hotstreak"] = {
            "base": best_c[1][0], "look": best_c[1][1], "hot": best_c[1][2],
            "cool": best_c[1][3], "heat": best_c[1][4],
        }
        print("SELECTED C", selected["hotstreak"], f"2024 mo={best_c[2].mean_mo*100:.2f}%", flush=True)

    best_d = (-1e9, None, None)
    print("\n=== Idea D tune 2024 ===", flush=True)
    for look in grid_d:
        port = build_roll_sharpe(lk24, look)
        for name, ptry in (("raw", port), ("vt0025", apply_vol_target(port, 0.0025))):
            row = eval_port(ptry, "2024", "research/WF", tr_lk, "roll_is_sharpe", f"look={look}|{name}")
            sc = score_2024(row)
            if row:
                print(f"  D mo={row.mean_mo*100:.2f}% pos={row.pct_pos*100:.0f}% gates={row.gates} | {row.params} sc={sc:.3f}", flush=True)
            if sc > best_d[0]:
                best_d = (sc, (look, name), row)
    if best_d[1]:
        selected["roll_is_sharpe"] = {"look": best_d[1][0], "overlay": best_d[1][1]}
        print("SELECTED D", selected["roll_is_sharpe"], f"2024 mo={best_d[2].mean_mo*100:.2f}%", flush=True)

    # ---- evaluate selected on all windows ----
    print("\n=== Full window evaluation ===", flush=True)
    for label, s, e, ov in wins:
        print(f"\n-- {label} --", flush=True)
        mr_c, tmr = curves_for_legs(mr_legs, cfg, risk_mr, s, e)
        tr_c, ttr = curves_for_legs(tr_legs, cfg, risk_tr, s, e)
        lk_c, tlk = curves_for_legs(locked_legs, cfg, risk_lock, s, e)
        if not mr_c or not tr_c or not lk_c:
            print("  skip (missing curves)", flush=True)
            continue
        adx = eurusd_adx_slice(s, e)
        base = combine_weighted(lk_c, lock_w)
        base_vt = apply_vol_target(base, 0.0025)

        # baseline
        row = eval_port(base_vt, label, ov, tlk, "baseline_vt0025", "vt=0.0025")
        if row:
            results.append(row)
            print(f"  BL mo={row.mean_mo*100:5.2f}% pos={row.pct_pos*100:3.0f}% top3={row.top3*100:3.0f}% "
                  f"gates={'PASS' if row.gates else 'FAIL'} p2t={row.p2t*100:.1f}% trades={row.trades}", flush=True)

        if "regime_sleeves" in selected:
            p = selected["regime_sleeves"]
            port = build_regime(mr_c, tr_c, mr_w, tr_w, adx, p["thresh"], p["w_tr_hi"], p["w_tr_lo"])
            row = eval_port(port, label, ov, tmr + ttr, "regime_sleeves",
                            f"adx>={p['thresh']}|wh={p['w_tr_hi']}|wl={p['w_tr_lo']}")
            if row:
                results.append(row)
                print(f"  A  mo={row.mean_mo*100:5.2f}% pos={row.pct_pos*100:3.0f}% top3={row.top3*100:3.0f}% "
                      f"gates={'PASS' if row.gates else 'FAIL'} p2t={row.p2t*100:.1f}% trades={row.trades}", flush=True)

        if "corr_throttle" in selected:
            p = selected["corr_throttle"]
            port = build_corr(lk_c, lock_w, p["look"], p["corr0"], p["k"])
            if p["overlay"] == "vt0025":
                port = apply_vol_target(port, 0.0025)
            row = eval_port(port, label, ov, tlk, "corr_throttle",
                            f"look={p['look']}|c0={p['corr0']}|k={p['k']}|{p['overlay']}")
            if row:
                results.append(row)
                print(f"  B  mo={row.mean_mo*100:5.2f}% pos={row.pct_pos*100:3.0f}% top3={row.top3*100:3.0f}% "
                      f"gates={'PASS' if row.gates else 'FAIL'} p2t={row.p2t*100:.1f}% trades={row.trades}", flush=True)

        if "hotstreak" in selected:
            p = selected["hotstreak"]
            bport = base_vt if p["base"] == "vt0025" else base
            port = build_hotstreak(bport, p["look"], p["hot"], p["cool"], -p["hot"], p["heat"])
            row = eval_port(port, label, ov, tlk, "hotstreak",
                            f"{p['base']}|look={p['look']}|hot={p['hot']}|cool={p['cool']}")
            if row:
                results.append(row)
                print(f"  C  mo={row.mean_mo*100:5.2f}% pos={row.pct_pos*100:3.0f}% top3={row.top3*100:3.0f}% "
                      f"gates={'PASS' if row.gates else 'FAIL'} p2t={row.p2t*100:.1f}% trades={row.trades}", flush=True)

        if "roll_is_sharpe" in selected:
            p = selected["roll_is_sharpe"]
            port = build_roll_sharpe(lk_c, p["look"])
            if p["overlay"] == "vt0025":
                port = apply_vol_target(port, 0.0025)
            row = eval_port(port, label, ov, tlk, "roll_is_sharpe", f"look={p['look']}|{p['overlay']}")
            if row:
                results.append(row)
                print(f"  D  mo={row.mean_mo*100:5.2f}% pos={row.pct_pos*100:3.0f}% top3={row.top3*100:3.0f}% "
                      f"gates={'PASS' if row.gates else 'FAIL'} p2t={row.p2t*100:.1f}% trades={row.trades}", flush=True)

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(ROOT / "reports" / "quest_regime_diversify.csv", index=False)

    lines = [
        "# Quest wave — regime / diversification overlays",
        "",
        f"**data_source:** `{data_src}` — never golive without FTMO MT5 exports.",
        f"**risk_fraction:** {RF:.0%}  **warmup:** {WARMUP}  **signal_lag=1**  **per-window BT** (no full-history kill contamination).",
        "**Selection:** overlay hyperparams on **2024 only**; holdout confirmation only.",
        "",
        "## Ideas",
        "1. **regime_sleeves** — MR (USDCHF/CADJPY/AUDCAD/EURCHF) vs Trend (GBPUSD/GBPJPY/EURJPY/GBPCAD-H4); EURUSD ADX gate.",
        "2. **corr_throttle** — scale locked FX4+GBPCAD-D1 legs when trailing peer-corr high; optional vt.",
        "3. **hotstreak** — anti-chase cool/heat after trailing portfolio returns.",
        "4. **roll_is_sharpe** — causal rolling IS Sharpe weights; optional vt.",
        "",
        "## Selected (IS=2024)",
        "```json",
        json.dumps(selected, indent=2),
        "```",
        "",
        "| Idea | Window | Mean mo | Med mo | %pos | Top3 | Gates | P2T | Static | Daily | Trades | Overlap |",
        "|------|--------|--------:|-------:|-----:|-----:|:-----:|----:|-------:|------:|-------:|---------|",
    ]
    for r in results:
        lines.append(
            f"| {r.idea} | {r.window} | {r.mean_mo*100:.2f}% | {r.median_mo*100:.2f}% | {r.pct_pos*100:.0f}% | "
            f"{r.top3*100:.0f}% | {'PASS' if r.gates else 'FAIL'} | {r.p2t*100:.1f}% | {r.static*100:.1f}% | "
            f"{r.daily*100:.1f}% | {r.trades} | {r.overlap} |"
        )
    lines += ["", "## Honest verdict", ""]
    for idea in ("regime_sleeves", "corr_throttle", "hotstreak", "roll_is_sharpe", "baseline_vt0025"):
        sub = [r for r in results if r.idea == idea]
        if not sub:
            continue
        ho = next((r for r in sub if r.window == "holdout_365d"), None)
        y24 = next((r for r in sub if r.window == "2024"), None)
        y25 = next((r for r in sub if r.window == "2025"), None)
        y26 = next((r for r in sub if r.window == "2026"), None)
        lines.append(
            f"- **{idea}**: 2024={((y24.mean_mo*100) if y24 else float('nan')):.2f}%/"
            f"{((y24.pct_pos*100) if y24 else float('nan')):.0f}%pos; "
            f"2025={((y25.mean_mo*100) if y25 else float('nan')):.2f}%; "
            f"2026={((y26.mean_mo*100) if y26 else float('nan')):.2f}%; "
            f"HO={((ho.mean_mo*100) if ho else float('nan')):.2f}%/"
            f"{((ho.pct_pos*100) if ho else float('nan')):.0f}%pos/"
            f"top3={((ho.top3*100) if ho else float('nan')):.0f}% "
            f"gates={ho.gates if ho else None}"
        )

    lines += ["", "### vs ≥1%/mo & ≥65–70% pos & smoother top3", ""]
    best = None
    for idea in ("regime_sleeves", "corr_throttle", "hotstreak", "roll_is_sharpe"):
        ho = next((r for r in results if r.idea == idea and r.window == "holdout_365d"), None)
        y24 = next((r for r in results if r.idea == idea and r.window == "2024"), None)
        bl = next((r for r in results if r.idea == "baseline_vt0025" and r.window == "holdout_365d"), None)
        if not ho or not ho.gates:
            continue
        # material improvement: better 2024 OR (similar HO and lower top3)
        y24_ok = y24 and y24.mean_mo >= 0.009
        ho_ok = ho.mean_mo >= 0.01 and ho.pct_pos >= 0.65
        smoother = bl and ho.top3 < bl.top3 - 0.05
        better_24 = y24 and bl and y24.mean_mo > (next((r.mean_mo for r in results if r.idea=='baseline_vt0025' and r.window=='2024'), 0)) + 0.001
        if best is None or (ho.mean_mo, -(ho.top3 or 1)) > (best[1].mean_mo, -(best[1].top3 or 1)):
            best = (idea, ho, y24, ho_ok, y24_ok, smoother, better_24)
    if best:
        idea, ho, y24, ho_ok, y24_ok, smoother, better_24 = best
        lines.append(f"- Best new idea by HO mean_mo: **{idea}**")
        lines.append(f"- HO ≥1% & ≥65% pos: **{'YES' if ho_ok else 'NO'}** (mo={ho.mean_mo*100:.2f}%, pos={ho.pct_pos*100:.0f}%, top3={ho.top3*100:.0f}%)")
        lines.append(f"- 2024 ~1%: **{'YES' if y24_ok else 'NO'}** (mo={((y24.mean_mo*100) if y24 else float('nan')):.2f}%)")
        lines.append(f"- Materially smoother vs baseline: **{'YES' if smoother else 'NO'}**; better 2024: **{'YES' if better_24 else 'NO'}**")
        lines.append(f"- **Overall target met:** **{'YES' if (ho_ok and y24_ok) else 'NO'}**")
        # promote only if improves vs baseline on IS (2024) without wrecking HO
        bl24 = next((r for r in results if r.idea == "baseline_vt0025" and r.window == "2024"), None)
        blho = next((r for r in results if r.idea == "baseline_vt0025" and r.window == "holdout_365d"), None)
        promote = (
            y24 and bl24 and ho and blho
            and y24.mean_mo >= bl24.mean_mo - 0.0005
            and ho.mean_mo >= 0.009
            and ho.gates and y24.gates
            and (y24.mean_mo > bl24.mean_mo + 0.001 or (ho.top3 < blho.top3 - 0.05 and ho.mean_mo >= blho.mean_mo - 0.002))
        )
        lines.append(f"- **Promote over locked candidate:** **{'YES' if promote else 'NO'}** (needs IS 2024 lift or smoother HO without giving up ~1%/mo)")
    else:
        lines.append("- No gate-passing new idea on holdout.")
    lines += [
        "",
        "## Methodology",
        "- Per-window backtest + 250-bar warmup; FTMO static/daily Europe/Prague.",
        "- Overlays causal (`.shift(1)`); ADX via `apply_feature_pipeline(..., signal_lag=1)`.",
        "- Yahoo ≠ FTMO.",
        "",
    ]
    (ROOT / "reports" / "quest_regime_diversify.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "configs" / "quest_regime_selected.json").write_text(
        json.dumps({"data_source": data_src, "selected": selected, "risk_fraction": RF,
                    "mr_legs": mr_legs, "tr_legs": tr_legs, "locked_legs": locked_legs}, indent=2, default=str),
        encoding="utf-8",
    )
    print("Wrote reports/quest_regime_diversify.md", flush=True)


if __name__ == "__main__":
    main()
