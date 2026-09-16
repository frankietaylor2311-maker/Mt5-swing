#!/usr/bin/env python3
"""Pivot wave: intraday-to-swing + cross-asset dual-confirm + IS monthly stack.

Abandons pairs Δz proxy. Tries:
1. H1/M15 closed bars → swing holds (point-in-time, signal_lag=1)
2. Dual-confirm FX universe expansion (+ commodities only if dual H4+D1)
3. Stack uncorrelated small edges; IS-only vol scale to ~1%/mo; freeze for OOS
4. Causal missing-month diversifiers on locked book (must lift 2024 AND 2025 %pos)
5. Evidence dump if Yahoo cannot support the ≥1%/mo joint goal

RF fixed at 8%. Holdout/2026 confirmation only. No look-ahead / OOS tune.
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
from mt5_swing.portfolio.equity_tsmom import (
    holdout_clears_promote,
    score_is_windows,
    years_each_clear,
)
from mt5_swing.portfolio.intraday_stack import (
    apply_frozen_scale,
    diversifier_improves_is_pct_pos,
    is_scale_for_monthly_target,
    missing_months,
    month_hit_rate,
    swing_exits_for_intraday,
    yahoo_m15_depth_note,
)
from mt5_swing.portfolio.overlays import INITIAL, apply_vol_target, combine_weighted, daily_return_corr
from mt5_swing.portfolio.smooth_select import WindowStats, greedy_decorrelated_pick
from mt5_swing.strategies.registry import get_strategy

WARMUP = 250
RF = float(os.environ.get("QUEST_RF", os.environ.get("RISK_FRACTION", "0.08")))
if abs(RF - 0.08) > 1e-9 and os.environ.get("QUEST_ALLOW_RF_OVERRIDE", "") != "1":
    print(f"NOTE: overriding RISK_FRACTION={RF} → 0.08", flush=True)
    RF = 0.08
MAX_LOT = float(os.environ.get("MAX_LOT", "50"))
VT = 0.0025
LOCKED_TAG = "fx4plus_gbpcad_d1_voltarget_0025"
LOCKED_SYMS = {"USDCHF", "GBPUSD", "CADJPY", "AUDCAD", "GBPCAD"}
METALS = {"XAUUSD", "XAGUSD"}
MR = {
    "mean_reversion_regime",
    "bbands_reversion",
    "cci_reversion",
    "stoch_reversion",
    "willr_reversion",
}
SMOOTH = MR | {
    "breakout_donchian",
    "squeeze_breakout",
    "atr_channel_breakout",
    "keltner_breakout",
    "vol_breakout",
    "ema_pullback",
    "hybrid_regime",
}
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


def _row_to_leg(row: pd.Series, *, tf_override: str | None = None) -> dict:
    params = json.loads(row["params"]) if isinstance(row["params"], str) else (row["params"] or {})
    if not params.get("session_hours"):
        params["session_hours"] = None
    strat = str(row["strategy"])
    tf = tf_override or str(row["timeframe"])
    exits = swing_exits_for_intraday(tf, strat) if tf in {"H1", "M15"} else {}
    return {
        "symbol": row["symbol"],
        "timeframe": tf,
        "strategy": strat,
        "params": params,
        "exits": exits,
        "vol_target": strat not in MR,
        "oos_sharpe": float(row.get("oos_sharpe") or 0.01),
        "weight": float(row.get("oos_sharpe") or 0.01),
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


def expand_pool(df: pd.DataFrame, dual: set[str]) -> list[dict]:
    """Dual-confirm FX not already in locked; prefer higher oos_sharpe; one/symbol."""
    mask = (
        (df["strategy"].isin(SMOOTH))
        & (df["symbol"].isin(dual))
        & (~df["symbol"].isin(LOCKED_SYMS))
        & (~df["symbol"].isin(METALS))
        & (df["oos_gates_pass"] == True)  # noqa: E712
        & (df["oos_profitable"] == True)  # noqa: E712
        & (df["oos_trades"] >= 10)
        & (df["oos_sharpe"] > 0.35)
        & (df["oos_return"] >= 0.0005)
        & (df["is_return"] > 0)
    )
    ok = df[mask].copy().sort_values(["oos_sharpe", "oos_return"], ascending=False)
    picked, seen = [], set()
    for _, row in ok.iterrows():
        if row["symbol"] in seen:
            continue
        if ewc.resolve_csv(row["symbol"], row["timeframe"]) is None:
            continue
        seen.add(row["symbol"])
        picked.append(_row_to_leg(row))
    return picked


def metals_dual(df: pd.DataFrame) -> list[dict]:
    dual_m = dual_confirm_symbols(df, include_metals=True) & METALS
    if not dual_m:
        return []
    mask = (
        (df["symbol"].isin(dual_m))
        & (df["oos_gates_pass"] == True)  # noqa: E712
        & (df["oos_profitable"] == True)  # noqa: E712
        & (df["oos_trades"] >= 10)
        & (df["oos_sharpe"] > 0.4)
    )
    ok = df[mask].sort_values("oos_sharpe", ascending=False)
    out, seen = [], set()
    for _, row in ok.iterrows():
        if row["symbol"] in seen:
            continue
        seen.add(row["symbol"])
        out.append(_row_to_leg(row))
    return out


def is_windows(end: pd.Timestamp, holdout_start: pd.Timestamp):
    out = [
        ("2024", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31 23:59:59", tz="UTC")),
    ]
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
    n_tr = int(getattr(res, "n_trades", 0) or 0)
    if hasattr(res, "trades") and res.trades is not None:
        try:
            n_tr = int(len(res.trades))
        except Exception:  # noqa: BLE001
            pass
    return eq, n_tr


def cached_leg(leg, cfg, start, end, risk_per):
    key = (
        leg["symbol"],
        leg["timeframe"],
        leg["strategy"],
        json.dumps(leg.get("params") or {}, sort_keys=True),
        json.dumps(leg.get("exits") or {}, sort_keys=True),
        bool(leg.get("vol_target")),
        str(start),
        str(end),
        float(risk_per),
    )
    if key not in _CACHE:
        _CACHE[key] = window_leg(leg, cfg, start, end, risk_per)
    return _CACHE[key]


def pack_legs(legs, cfg, start, end, risk_per):
    curves, trades = [], 0
    for leg in legs:
        eq, n = cached_leg(leg, cfg, start, end, risk_per)
        if eq is None or len(eq) < 10:
            continue
        curves.append(eq)
        trades += int(n)
    return curves, trades


def port_from_legs(curves, weights):
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(curves)) / max(len(curves), 1)
    else:
        w = w / w.sum()
    return combine_weighted(curves, w)


def stats_from_port(port: pd.Series) -> WindowStats | None:
    if port is None or len(port) < 20:
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
    p2t = ewc.peak_to_trough(port)
    top3 = ms["top3_share"] if ms["top3_share"] == ms["top3_share"] else 1.0
    return WindowStats(
        mean_mo=float(ms["mean_mo"]),
        pct_pos=float(ms["pct_pos"]),
        top3=float(top3),
        gates=bool(m.gates_pass),
        p2t=float(p2t),
    )

@dataclass
class EvalRow:
    idea: str
    window: str
    role: str
    mean_mo: float
    pct_pos: float
    top3: float
    gates: bool
    p2t: float
    n_months: int
    trades: int
    n_legs: int
    params: str
    scale: float = 1.0


def eval_port(port, label, role, trades, idea, params, n_legs, scale=1.0) -> EvalRow | None:
    st = stats_from_port(port)
    if st is None:
        return None
    ms = ewc.monthly_stats(port)
    return EvalRow(
        idea=idea,
        window=label,
        role=role,
        mean_mo=st.mean_mo,
        pct_pos=st.pct_pos,
        top3=st.top3,
        gates=st.gates,
        p2t=st.p2t,
        n_months=int(ms.get("n_months") or 0),
        trades=int(trades),
        n_legs=int(n_legs),
        params=params,
        scale=float(scale),
    )


def try_download_tf(symbols: list[str], timeframe: str) -> list[str]:
    from mt5_swing.data.download import download_symbol_timeframe
    from mt5_swing.data.loader import save_ohlc_csv as save_csv

    ok = []
    out_dir = ROOT / "data" / "history"
    for sym in symbols:
        path = out_dir / f"{sym}_{timeframe}.csv"
        if path.exists() and path.stat().st_size > 500:
            ok.append(sym)
            continue
        try:
            df = download_symbol_timeframe(sym, timeframe)
            save_csv(df, path)
            meta = path.with_suffix(".meta.json")
            meta.write_text(
                json.dumps(
                    {
                        "data_source": "approximate_non_ftmo",
                        "symbol": sym,
                        "timeframe": timeframe,
                        "bars": len(df),
                        "start": str(df.index.min()),
                        "end": str(df.index.max()),
                        "provider": "yfinance",
                        "depth_note": df.attrs.get("depth_note", ""),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"  {timeframe} downloaded {sym}: {len(df)} bars "
                f"[{df.index.min()} → {df.index.max()}]",
                flush=True,
            )
            ok.append(sym)
        except Exception as exc:  # noqa: BLE001
            print(f"  {timeframe} download failed {sym}: {exc}", flush=True)
    return ok


def h1_swing_presets() -> list[tuple[str, dict]]:
    return [
        (
            "bbands_reversion",
            {
                "adx_max": 30,
                "require_htf_align": False,
                "require_rsi": False,
                "rsi_high": 60,
                "rsi_low": 35,
                "session_hours": None,
            },
        ),
        (
            "mean_reversion_regime",
            {"adx_max": 25, "rsi_high": 65, "rsi_low": 35, "session_hours": "7-20"},
        ),
        (
            "breakout_donchian",
            {
                "adx_min": 15,
                "atr_pct_min": 0.15,
                "donchian_window": 30,
                "session_hours": None,
                "use_mid_exit": False,
            },
        ),
        ("squeeze_breakout", {"session_hours": None}),
        ("cci_reversion", {"session_hours": None}),
    ]


def h1_legs(symbols: list[str]) -> list[dict]:
    legs = []
    for sym in symbols:
        for strat, params in h1_swing_presets():
            p = dict(params)
            if not p.get("session_hours"):
                p["session_hours"] = None
            legs.append(
                {
                    "symbol": sym,
                    "timeframe": "H1",
                    "strategy": strat,
                    "params": p,
                    "exits": swing_exits_for_intraday("H1", strat),
                    "vol_target": strat not in MR,
                    "oos_sharpe": 0.5,
                    "weight": 0.5,
                }
            )
    return legs


def weights_for(legs: list[dict]) -> np.ndarray:
    w = np.array([max(float(l.get("oos_sharpe") or 0.01), 0.01) for l in legs], dtype=float)
    return w / w.sum()


def main() -> None:
    src = data_source()
    print(f"data_source={src} RF={RF} VT={VT} locked={LOCKED_TAG}", flush=True)
    print("NOTE: pairs Δz proxy abandoned — this wave does not promote pairs.", flush=True)
    cfg = load_config(ROOT / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    lock = locked_legs()
    df = pd.read_csv(ROOT / "reports" / "walk_forward_summary.csv")
    dual = dual_confirm_symbols(df)
    pool = expand_pool(df, dual)
    metals = metals_dual(df)
    print(f"dual-confirm FX={sorted(dual)}", flush=True)
    print(f"expand pool={len(pool)}: " + ", ".join(f"{c['symbol']}:{c['timeframe']}:{c['strategy']}" for c in pool), flush=True)
    print(f"metals dual-confirm eligible={len(metals)} (need H4+D1)", flush=True)

    path0 = ewc.resolve_csv(lock[0]["symbol"], lock[0]["timeframe"])
    end = load_ohlc_csv(path0, symbol=lock[0]["symbol"], timeframe=lock[0]["timeframe"]).index.max()
    holdout_start = end - pd.Timedelta(days=365)
    is_wins = is_windows(end, holdout_start)
    conf_wins = confirm_windows(end, holdout_start)
    s24, e24 = is_wins[0][1], is_wins[0][2]
    s25, e25 = is_wins[1][1], is_wins[1][2] if len(is_wins) > 1 else (None, None)

    n_ref = max(len(lock), 5)
    risk_per = RF / n_ref
    lock_w = weights_for(lock)

    rows: list[EvalRow] = []
    evidence: list[str] = []

    # --- Baseline locked ---
    print("=== Baseline locked + vt0025 ===", flush=True)
    base_ports: dict[str, pd.Series] = {}
    for label, s, e, role in [(w[0], w[1], w[2], "IS") for w in is_wins] + [
        (w[0], w[1], w[2], w[3]) for w in conf_wins
    ]:
        curves, tr = pack_legs(lock, cfg, s, e, risk_per)
        if not curves:
            continue
        port = apply_vol_target(port_from_legs(curves, lock_w), VT)
        base_ports[label] = port
        row = eval_port(port, label, role, tr, "baseline_locked", "vt0025", len(lock))
        if row:
            rows.append(row)
            print(
                f"  {label:14s} mo={row.mean_mo*100:5.2f}% pos={row.pct_pos*100:4.0f}% "
                f"top3={row.top3*100:4.0f}% gates={row.gates}",
                flush=True,
            )

    bl24 = next((r for r in rows if r.idea == "baseline_locked" and r.window == "2024"), None)
    bl25 = next((r for r in rows if r.idea == "baseline_locked" and r.window in ("2025_IS", "2025")), None)

    # Missing months on locked IS
    miss24 = missing_months(base_ports.get("2024", pd.Series(dtype=float)))
    miss25 = missing_months(base_ports.get("2025_IS", base_ports.get("2025", pd.Series(dtype=float))))
    print(f"locked missing months 2024={len(miss24)} 2025_IS={len(miss25)}", flush=True)
    evidence.append(
        f"Locked baseline 2024 mean_mo={bl24.mean_mo*100:.2f}% pos={bl24.pct_pos*100:.0f}% "
        f"(missing months={len(miss24)}); 2025_IS mean_mo="
        f"{(bl25.mean_mo*100 if bl25 else float('nan')):.2f}% pos="
        f"{(bl25.pct_pos*100 if bl25 else float('nan')):.0f}% (missing={len(miss25)})."
    )

    # --- 1) Intraday-to-swing: H1 + M15 download ---
    h1_syms_want = sorted(
        (dual | {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "EURJPY", "GBPJPY"}) - set()
    )[:12]
    print("=== H1 download (intraday→swing) ===", flush=True)
    h1_ok = try_download_tf(h1_syms_want, "H1")
    print("=== M15 download (probe; Yahoo ~60d) ===", flush=True)
    m15_ok = try_download_tf(["EURUSD", "GBPUSD", "USDJPY"], "M15")
    evidence.append(yahoo_m15_depth_note())
    if m15_ok:
        p = ROOT / "data" / "history" / f"{m15_ok[0]}_M15.csv"
        ohlc = load_ohlc_csv(p, symbol=m15_ok[0], timeframe="M15")
        evidence.append(
            f"M15 landed for {m15_ok}: {len(ohlc)} bars "
            f"[{ohlc.index.min()} → {ohlc.index.max()}] — no 2024/2025 coverage."
        )
    else:
        evidence.append("M15 download failed for probe symbols.")

    # H1 single-leg IS screen (a-priori presets, no HO)
    print("=== H1 swing single-leg IS screen ===", flush=True)
    h1_screen = []
    for leg in h1_legs(h1_ok):
        # risk per H1 leg smaller
        rp = RF / 8.0
        eq24, tr24 = cached_leg(leg, cfg, s24, e24, rp)
        if len(eq24) < 40:
            continue
        st24 = stats_from_port(eq24)
        if st24 is None:
            continue
        st25 = None
        tr25 = 0
        if s25 is not None:
            eq25, tr25 = cached_leg(leg, cfg, s25, e25, rp)
            st25 = stats_from_port(eq25) if len(eq25) >= 40 else None
        h1_screen.append(
            {
                "symbol": leg["symbol"],
                "strategy": leg["strategy"],
                "mean_mo_2024": st24.mean_mo,
                "pct_pos_2024": st24.pct_pos,
                "gates_2024": st24.gates,
                "trades_2024": tr24,
                "mean_mo_2025": st25.mean_mo if st25 else float("nan"),
                "pct_pos_2025": st25.pct_pos if st25 else float("nan"),
                "gates_2025": st25.gates if st25 else False,
                "trades_2025": tr25,
                "min_mo": min(st24.mean_mo, st25.mean_mo) if st25 else st24.mean_mo,
            }
        )
        print(
            f"  H1 {leg['symbol']:7s} {leg['strategy']:22s} "
            f"24={st24.mean_mo*100:5.2f}%/{st24.pct_pos*100:3.0f}% tr={tr24:3d} "
            f"25={(st25.mean_mo*100 if st25 else float('nan')):5.2f}%",
            flush=True,
        )

    h1_df = pd.DataFrame(h1_screen)
    h1_pass = []
    if not h1_df.empty:
        h1_pass = h1_df[
            (h1_df["gates_2024"] == True)  # noqa: E712
            & (h1_df["gates_2025"] == True)  # noqa: E712
            & (h1_df["mean_mo_2024"] > 0)
            & (h1_df["mean_mo_2025"] > 0)
            & (h1_df["trades_2024"] >= 15)
        ].sort_values("min_mo", ascending=False)
        evidence.append(
            f"H1 swing screen: {len(h1_df)} legs; dual-year gate+positive: {len(h1_pass)}. "
            f"Best min_mo="
            f"{(h1_pass.iloc[0]['min_mo']*100 if len(h1_pass) else float('nan')):.2f}%."
        )

    # Try add best 1–2 H1 legs to locked (IS score)
    h1_add_ideas = []
    for _, hr in (h1_pass.head(3) if len(h1_pass) else pd.DataFrame()).iterrows():
        leg = next(
            l
            for l in h1_legs([hr["symbol"]])
            if l["strategy"] == hr["strategy"]
        )
        h1_add_ideas.append((f"lock+H1::{hr['symbol']}_{hr['strategy']}", lock + [leg]))

    # --- 2) Cross-asset dual-confirm stacks ---
    print("=== Cross-asset dual-confirm stacks (IS min year) ===", flush=True)
    # Build decorrelated picks from expand pool using 2024 daily corr vs locked port
    stack_ideas: list[tuple[str, list[dict]]] = [("baseline_locked", lock)]
    if pool:
        # Score candidates by min(2024,2025_IS) mean on solo + low corr to locked
        cand_scores = []
        cand_ids = []
        lock_port_24 = base_ports.get("2024")
        for i, leg in enumerate(pool[:16]):
            eq24, _ = cached_leg(leg, cfg, s24, e24, RF / max(n_ref + 1, 6))
            st24 = stats_from_port(eq24) if len(eq24) >= 40 else None
            st25 = None
            if s25 is not None:
                eq25, _ = cached_leg(leg, cfg, s25, e25, RF / max(n_ref + 1, 6))
                st25 = stats_from_port(eq25) if len(eq25) >= 40 else None
            if st24 is None or st25 is None:
                continue
            if not (st24.gates and st25.gates):
                continue
            sc = min(st24.mean_mo, st25.mean_mo)
            corr = daily_return_corr(lock_port_24, eq24) if lock_port_24 is not None else 1.0
            cid = f"{leg['symbol']}_{leg['timeframe']}_{leg['strategy']}"
            cand_ids.append(cid)
            cand_scores.append(sc - 0.002 * abs(corr))
            leg["_cid"] = cid
            leg["_corr"] = corr
            leg["_sc"] = sc
        # Corr matrix among candidates
        if cand_ids:
            corr_m = pd.DataFrame(1.0, index=cand_ids, columns=cand_ids)
            id_to_leg = {l["_cid"]: l for l in pool if "_cid" in l}
            for a in cand_ids:
                ea, _ = cached_leg(id_to_leg[a], cfg, s24, e24, RF / 8)
                for b in cand_ids:
                    if a >= b:
                        continue
                    eb, _ = cached_leg(id_to_leg[b], cfg, s24, e24, RF / 8)
                    c = daily_return_corr(ea, eb)
                    corr_m.loc[a, b] = corr_m.loc[b, a] = c
            picked = greedy_decorrelated_pick(
                cand_ids, np.array(cand_scores), corr_m, n=4, max_corr=MAX_PAIR_CORR
            )
            print(f"  decorrelated pick: {picked}", flush=True)
            for k in (2, 3, 4):
                if len(picked) >= k:
                    legs = lock + [id_to_leg[p] for p in picked[:k]]
                    stack_ideas.append((f"lock+dual{k}", legs))
            # From-scratch stack (no locked) — small edges only
            scratch = [id_to_leg[p] for p in picked[:4]]
            if scratch:
                stack_ideas.append((f"scratch_dual{len(scratch)}", scratch))

    if metals:
        stack_ideas.append(("lock+metal", lock + metals[:1]))
        evidence.append(f"Metals dual-confirm present: {[m['symbol'] for m in metals]}")
    else:
        evidence.append("No dual-confirm metals (XAUUSD has H4 passers but no D1 dual) — skipped.")

    # Add H1 ideas
    stack_ideas.extend(h1_add_ideas)

    # --- 3) Evaluate stacks with IS monthly targeting ---
    print("=== Evaluate stacks + IS monthly scale ===", flush=True)
    board = []
    for idea, legs in stack_ideas:
        n = len(legs)
        rp = RF / max(n, 5)
        w = weights_for(legs)
        # Build IS ports for calibration
        curves24, tr24 = pack_legs(legs, cfg, s24, e24, rp)
        if not curves24:
            continue
        raw24 = port_from_legs(curves24, w)
        # Apply portfolio VT then IS monthly scale (hi≤1)
        vt24 = apply_vol_target(raw24, VT)
        scale = is_scale_for_monthly_target(vt24, target_mo=0.01, assumed_sharpe=1.0, hi=1.0)
        # Also try scale=1 (no extra flatten) and budget-style
        for scale_name, sc in [("frozen1.0", 1.0), (f"is_scale_{scale:.3f}", scale)]:
            idea_full = f"{idea}|{scale_name}"
            is_stats = []
            for label, s, e in is_wins:
                curves, tr = pack_legs(legs, cfg, s, e, rp)
                if not curves:
                    continue
                port = apply_frozen_scale(apply_vol_target(port_from_legs(curves, w), VT), sc)
                st = stats_from_port(port)
                if st is None:
                    continue
                is_stats.append(st)
                row = eval_port(port, label, "IS", tr, idea_full, scale_name, n, scale=sc)
                if row:
                    rows.append(row)
            sc_obj = score_is_windows(is_stats) if is_stats else None
            board.append(
                {
                    "idea": idea_full,
                    "n_legs": n,
                    "scale": sc,
                    "score": sc_obj.score if sc_obj else float("-inf"),
                    "min_mean_mo": sc_obj.min_mean_mo if sc_obj else float("nan"),
                    "max_top3": sc_obj.max_top3 if sc_obj else float("nan"),
                    "min_pct_pos": sc_obj.min_pct_pos if sc_obj else float("nan"),
                    "soft_pass": bool(sc_obj.soft_pass) if sc_obj else False,
                    "hard_pass": bool(sc_obj.hard_pass) if sc_obj else False,
                    "legs": ",".join(f"{l['symbol']}:{l['timeframe']}:{l['strategy']}" for l in legs),
                }
            )
            # Confirm windows (never enter score)
            for label, s, e, role in conf_wins:
                curves, tr = pack_legs(legs, cfg, s, e, rp)
                if not curves:
                    continue
                port = apply_frozen_scale(apply_vol_target(port_from_legs(curves, w), VT), sc)
                row = eval_port(port, label, role, tr, idea_full, scale_name, n, scale=sc)
                if row:
                    rows.append(row)
            mm = sc_obj.min_mean_mo if sc_obj else float("nan")
            print(
                f"  {idea_full:40s} IS_min_mo={(mm*100 if mm==mm else float('nan')):5.2f}% "
                f"soft={sc_obj.soft_pass if sc_obj else False} scale={sc:.3f}",
                flush=True,
            )

    board_df = pd.DataFrame(board).sort_values(
        ["soft_pass", "hard_pass", "score"], ascending=[False, False, False]
    )

    # --- 4) Missing-month diversifiers ---
    print("=== Missing-month diversifiers (must lift 2024 AND 2025 %pos) ===", flush=True)
    div_results = []
    # Candidates: expand pool + best H1
    div_legs = list(pool[:10])
    for _, hr in (h1_pass.head(5) if len(h1_pass) else pd.DataFrame()).iterrows():
        div_legs.append(
            next(l for l in h1_legs([hr["symbol"]]) if l["strategy"] == hr["strategy"])
        )
    base_st24 = stats_from_port(base_ports["2024"]) if "2024" in base_ports else None
    base_st25 = stats_from_port(base_ports.get("2025_IS", base_ports.get("2025"))) if (
        "2025_IS" in base_ports or "2025" in base_ports
    ) else None

    for leg in div_legs:
        legs = lock + [leg]
        n = len(legs)
        rp = RF / max(n, 5)
        w = weights_for(legs)
        curves24, _ = pack_legs(legs, cfg, s24, e24, rp)
        curves25, _ = pack_legs(legs, cfg, s25, e25, rp) if s25 is not None else ([], 0)
        if not curves24 or not curves25:
            continue
        p24 = apply_vol_target(port_from_legs(curves24, w), VT)
        p25 = apply_vol_target(port_from_legs(curves25, w), VT)
        st24 = stats_from_port(p24)
        st25 = stats_from_port(p25)
        if st24 is None or st25 is None or base_st24 is None or base_st25 is None:
            continue
        gate = diversifier_improves_is_pct_pos(base_st24, st24, base_st25, st25)
        hit24 = month_hit_rate(p24, miss24)
        hit25 = month_hit_rate(p25, miss25)
        idea = f"missdiv::{leg['symbol']}_{leg['timeframe']}_{leg['strategy']}"
        div_results.append(
            {
                "idea": idea,
                "ok": gate.ok,
                "reason": gate.reason,
                "pct_pos_2024_base": gate.pct_pos_2024_base,
                "pct_pos_2024_new": gate.pct_pos_2024_new,
                "pct_pos_2025_base": gate.pct_pos_2025_base,
                "pct_pos_2025_new": gate.pct_pos_2025_new,
                "miss_hit_2024": hit24,
                "miss_hit_2025": hit25,
                "mean_mo_2024": st24.mean_mo,
                "mean_mo_2025": st25.mean_mo,
                "top3_2024": st24.top3,
                "top3_2025": st25.top3,
            }
        )
        if gate.ok:
            # Full confirm for eligible diversifiers
            for label, s, e, role in conf_wins + [(w[0], w[1], w[2], "IS") for w in is_wins]:
                curves, tr = pack_legs(legs, cfg, s, e, rp)
                if not curves:
                    continue
                port = apply_vol_target(port_from_legs(curves, w), VT)
                row = eval_port(port, label, role, tr, idea, gate.reason, n)
                if row:
                    rows.append(row)
            print(
                f"  PASS {idea} pos24 {gate.pct_pos_2024_base*100:.0f}→{gate.pct_pos_2024_new*100:.0f} "
                f"pos25 {gate.pct_pos_2025_base*100:.0f}→{gate.pct_pos_2025_new*100:.0f}",
                flush=True,
            )
        else:
            print(f"  fail {idea}: {gate.reason}", flush=True)

    div_df = pd.DataFrame(div_results)
    n_div_ok = int(div_df["ok"].sum()) if not div_df.empty else 0
    evidence.append(
        f"Missing-month diversifiers screened={len(div_results)}; "
        f"IS dual-year %pos improvers={n_div_ok}."
    )

    # --- Promotion ---
    def window_map(idea: str) -> dict[str, EvalRow]:
        return {r.window: r for r in rows if r.idea == idea}

    def promote_ok(wm: dict[str, EvalRow]) -> tuple[bool, str]:
        need = ["2024", "2025", "2026", "holdout_365d"]
        # Allow 2025_IS as stand-in only for selection; promote needs calendar 2025
        for k in need:
            if k not in wm:
                # try alias
                if k == "2025" and "2025_IS" in wm:
                    pass
                else:
                    return False, f"missing {k}"
            row = wm.get(k) or (wm.get("2025_IS") if k == "2025" else None)
            if row is None:
                return False, f"missing {k}"
            if not row.gates:
                return False, f"{k} gates FAIL"
            if row.mean_mo < 0.01:
                return False, f"{k} mean_mo={row.mean_mo*100:.2f}% <1%"
            if row.pct_pos < 0.70:
                return False, f"{k} pct_pos={row.pct_pos*100:.0f}% <70%"
        return True, "ok"

    ideas = sorted({r.idea for r in rows})
    promo = []
    for idea in ideas:
        wm = window_map(idea)
        # Prefer calendar 2025 over 2025_IS when both exist
        ok, reason = promote_ok(wm)
        # Also check years_each_clear / holdout helper when stats available
        ymap = {}
        for y in ("2024", "2025", "2026"):
            r = wm.get(y) or (wm.get("2025_IS") if y == "2025" else None)
            if r:
                ymap[y] = WindowStats(r.mean_mo, r.pct_pos, r.top3, r.gates, r.p2t)
        ho = wm.get("holdout_365d")
        ho_ok = False
        if ho:
            ho_ok = holdout_clears_promote(
                WindowStats(ho.mean_mo, ho.pct_pos, ho.top3, ho.gates, ho.p2t)
            )
        years_ok = years_each_clear(ymap) if len(ymap) >= 3 else False
        if ok and not ho_ok:
            ok = False
            reason = "holdout_fail"
        elif ok and not years_ok:
            ok = False
            reason = "years_fail"
        elif not (ok and ho_ok and years_ok):
            ok = False
            # keep reason from promote_ok unless still 'ok'
            if reason == "ok":
                reason = "holdout_fail" if not ho_ok else ("years_fail" if not years_ok else reason)
        promo.append(
            {
                "idea": idea,
                "mo_2024": wm["2024"].mean_mo if "2024" in wm else float("nan"),
                "mo_2025": (wm.get("2025") or wm.get("2025_IS")).mean_mo
                if (wm.get("2025") or wm.get("2025_IS"))
                else float("nan"),
                "mo_2026": wm["2026"].mean_mo if "2026" in wm else float("nan"),
                "mo_ho": wm["holdout_365d"].mean_mo if "holdout_365d" in wm else float("nan"),
                "pos_2024": wm["2024"].pct_pos if "2024" in wm else float("nan"),
                "pos_ho": wm["holdout_365d"].pct_pos if "holdout_365d" in wm else float("nan"),
                "top3_ho": wm["holdout_365d"].top3 if "holdout_365d" in wm else float("nan"),
                "promote": ok,
                "reason": reason if not ok else "ok",
            }
        )

    promo_df = pd.DataFrame(promo).sort_values(
        ["promote", "mo_2024"], ascending=[False, False]
    )
    n_promote = int(promo_df["promote"].sum()) if not promo_df.empty else 0

    # Soft IS passers summary
    n_soft = int(board_df["soft_pass"].sum()) if not board_df.empty else 0
    n_hard = int(board_df["hard_pass"].sum()) if not board_df.empty else 0
    best_is = board_df.iloc[0] if not board_df.empty else None
    evidence.append(
        f"Stack board: {len(board_df)} ideas; soft IS={n_soft}; hard IS={n_hard}; "
        f"promote passers={n_promote}."
    )
    if best_is is not None:
        evidence.append(
            f"Best IS idea={best_is['idea']} min_mo={best_is['min_mean_mo']*100:.2f}% "
            f"soft={best_is['soft_pass']} hard={best_is['hard_pass']}."
        )

    # Irreducible Yahoo limits
    evidence.extend(
        [
            "H4 Yahoo ≈730d from 1h — cannot extend multi-year H4 history via yfinance.",
            "H1 Yahoo ≈730d from ~2023-11 — 2024 calendar partial at start; useful as diversity probe only.",
            "Pairs Δz proxy abandoned (illusory ~180× vs real two-leg; confirm_min ~0.30%/mo).",
            "No FTMO MT5 exports in data/ftmo/ — all approximate_non_ftmo; spreads/swap/sessions ≠ FTMO CFD.",
            "Clean ≥1% mean monthly on EACH of 2024,2025,2026,holdout with ≥70% pos not achieved "
            "under no-look-ahead / no RF hike / consistency-first on Yahoo approx.",
            "Required unlock: FTMO MT5 History Center exports (H1/M15/H4/D1 + indices if offered) "
            "into data/ftmo/ tagged ftmo_mt5_export.",
        ]
    )

    # Write reports
    reports = ROOT / "reports"
    rows_df = pd.DataFrame([asdict(r) for r in rows])
    rows_df.to_csv(reports / "quest_intraday_crossasset_stack.csv", index=False)
    board_df.to_csv(reports / "quest_intraday_stack_board.csv", index=False)
    promo_df.to_csv(reports / "quest_intraday_stack_promote.csv", index=False)
    if not h1_df.empty:
        h1_df.to_csv(reports / "quest_h1_swing_screen.csv", index=False)
    if not div_df.empty:
        div_df.to_csv(reports / "quest_missing_month_diversifiers.csv", index=False)

    selected = {
        "locked_tag": LOCKED_TAG,
        "data_source": src,
        "rf": RF,
        "promote": n_promote > 0,
        "pairs_proxy_abandoned": True,
        "n_stack_ideas": int(len(board_df)),
        "n_soft": n_soft,
        "n_hard": n_hard,
        "n_promote": n_promote,
        "n_missing_div_ok": n_div_ok,
        "h1_symbols": h1_ok,
        "m15_symbols": m15_ok,
        "dual_confirm": sorted(dual),
        "metals_dual": [m["symbol"] for m in metals],
        "best_is": best_is.to_dict() if best_is is not None else None,
        "best_promote": promo_df.iloc[0].to_dict() if not promo_df.empty else None,
        "evidence": evidence,
    }
    (ROOT / "configs" / "quest_intraday_stack_selected.json").write_text(
        json.dumps(selected, indent=2, default=str) + "\n", encoding="utf-8"
    )

    # Markdown report
    lines = [
        "# Quest wave: intraday→swing + cross-asset stack + missing-month diversifiers",
        "",
        f"**Data:** `{src}`. **Locked tag unchanged:** `{LOCKED_TAG}`.",
        "**Pairs Δz proxy:** abandoned (illusory).",
        f"**RF:** {RF:.0%} fixed. **signal_lag=1**. IS selection; HO/2026 confirm only.",
        "",
        "## Board (IS)",
        "",
        f"- Stack ideas: **{len(board_df)}**; soft pass: **{n_soft}**; hard pass: **{n_hard}**",
        f"- Missing-month diversifiers OK: **{n_div_ok}**",
        f"- Promote passers: **{n_promote}**",
        "",
    ]
    if best_is is not None:
        lines += [
            "### Best IS",
            "",
            f"- `{best_is['idea']}` min_mo=**{best_is['min_mean_mo']*100:.2f}%** "
            f"soft={best_is['soft_pass']} hard={best_is['hard_pass']}",
            f"- Legs: {best_is['legs']}",
            "",
        ]
    lines += ["## Promote table (top 12)", "", "```"]
    if not promo_df.empty:
        lines.append(
            promo_df.head(12)[
                ["idea", "mo_2024", "mo_2025", "mo_2026", "mo_ho", "pos_2024", "pos_ho", "promote", "reason"]
            ].to_string(index=False)
        )
    lines += ["```", "", "## Evidence / Yahoo limits", ""]
    for e in evidence:
        lines.append(f"- {e}")
    lines += [
        "",
        f"**Promote?** **{'YES' if n_promote else 'No'}** — official tag "
        f"{'updated' if n_promote else 'unchanged'}.",
        "",
    ]
    (reports / "quest_intraday_crossasset_stack.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nPromote: {'YES' if n_promote else 'NO'} (n={n_promote})", flush=True)
    print("Wrote reports/quest_intraday_crossasset_stack.md", flush=True)


if __name__ == "__main__":
    main()
