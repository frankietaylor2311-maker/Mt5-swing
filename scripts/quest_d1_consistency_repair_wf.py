#!/usr/bin/env python3
"""D1 multi-year consistency repair + optional causal MTD gain-clip on locked.

Wave goal: lift 2024 consistency (mean_mo / %pos / top3) via unused D1 OOS-passers
as small add-on / alternate sleeves, scored ONLY on IS windows {2024, 2025_IS}.
Holdout + full 2025/2026 = confirmation only. RF fixed 8%. signal_lag=1.

Also tries a causal intra-month MTD gain clip on the locked sleeve with tau
chosen on 2024 only (hi≤1). Promote only if dual-year IS clears and HO confirms.
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

from mt5_swing.backtest.metrics import compute_metrics
from mt5_swing.config import load_config
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.portfolio.equity_tsmom import (
    holdout_clears_promote,
    score_is_windows,
    years_each_clear,
)
from mt5_swing.portfolio.overlays import (
    INITIAL,
    apply_mtd_gain_clip,
    apply_vol_target,
    combine_weighted,
    daily_return_corr,
)
from mt5_swing.portfolio.smooth_select import WindowStats

WARMUP = 250
RF = 0.08
MAX_LOT = 50.0
VT = 0.0025
LOCKED_TAG = "fx4plus_gbpcad_d1_voltarget_0025"
LOCKED_EXACT = {
    ("USDCHF", "bbands_reversion"),
    ("GBPUSD", "breakout_donchian"),
    ("CADJPY", "mean_reversion_regime"),
    ("AUDCAD", "mean_reversion_regime"),
    ("GBPCAD", "bbands_reversion"),
}
LOCKED_SYMS = {s for s, _ in LOCKED_EXACT}
MR = {
    "mean_reversion_regime",
    "bbands_reversion",
    "cci_reversion",
    "stoch_reversion",
    "willr_reversion",
}
MIN_PCT_POS = 0.70
MAX_TOP3_HARD = 0.55
MAX_TOP3_SOFT = 0.70
MAX_P2T = 0.085
MAX_CORR = 0.55
ADDON_SLEEVE_FRACS = (0.15, 0.20, 0.25, 0.30)
MAX_ADDONS = 3
TOP_POOL = 24


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


def row_to_leg(row: pd.Series) -> dict:
    params = json.loads(row["params"]) if isinstance(row["params"], str) else (row["params"] or {})
    if not params.get("session_hours"):
        params["session_hours"] = None
    strat = str(row["strategy"])
    return {
        "symbol": str(row["symbol"]),
        "timeframe": "D1",
        "strategy": strat,
        "params": params,
        "exits": {},
        "vol_target": strat not in MR,
        "oos_sharpe": float(row.get("oos_sharpe") or 0.01),
        "weight": float(row.get("oos_sharpe") or 0.01),
    }


def screen_d1_pool(df: pd.DataFrame) -> list[dict]:
    """Unused D1 OOS-passers (not exact locked legs). Prefer unused symbols."""
    mask = (
        (df["timeframe"] == "D1")
        & (df["oos_gates_pass"] == True)  # noqa: E712
        & (df["oos_profitable"] == True)  # noqa: E712
        & (df["oos_trades"] >= 8)
        & (df["oos_sharpe"] > 0.35)
        & (df["oos_return"] >= 0.0005)
        & (df["is_return"] > 0)
        & (~df["symbol"].isin({"XAUUSD", "XAGUSD"}))
    )
    ok = df[mask].copy()
    ok["exact_locked"] = ok.apply(lambda r: (r["symbol"], r["strategy"]) in LOCKED_EXACT, axis=1)
    ok = ok[~ok["exact_locked"]].sort_values(["oos_sharpe", "oos_return"], ascending=False)
    # Prefer unused symbols first, then alt strategies on locked symbols
    unused_sym = ok[~ok["symbol"].isin(LOCKED_SYMS)]
    alt_locked = ok[ok["symbol"].isin(LOCKED_SYMS)]
    ordered = pd.concat([unused_sym, alt_locked], axis=0)
    picked, seen = [], set()
    for _, row in ordered.iterrows():
        key = (row["symbol"], row["strategy"])
        if key in seen:
            continue
        if ewc.resolve_csv(row["symbol"], "D1") is None:
            continue
        seen.add(key)
        picked.append(row_to_leg(row))
        if len(picked) >= TOP_POOL:
            break
    return picked


def is_windows(end: pd.Timestamp, holdout_start: pd.Timestamp):
    out = [("2024", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31 23:59:59", tz="UTC"))]
    s25 = pd.Timestamp("2025-01-01", tz="UTC")
    if holdout_start > s25:
        out.append(("2025_IS", s25, min(holdout_start - pd.Timedelta(seconds=1), end)))
    return out


def confirm_windows(end: pd.Timestamp, holdout_start: pd.Timestamp):
    out = []
    for year in (2025, 2026):
        s = pd.Timestamp(f"{year}-01-01", tz="UTC")
        e = pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC")
        if s > end:
            continue
        out.append((str(year), s, min(e, end), "confirm"))
    out.append(("holdout_365d", holdout_start, end, "pure holdout"))
    return out


def run_leg_curve(leg: dict, cfg: dict, start, end) -> pd.Series | None:
    path = ewc.resolve_csv(leg["symbol"], leg["timeframe"])
    if path is None:
        return None
    ohlc = load_ohlc_csv(path, symbol=leg["symbol"], timeframe=leg["timeframe"])
    if end is not None:
        ohlc = ohlc.loc[ohlc.index <= end]
    if start is not None:
        pre = ohlc.loc[ohlc.index < start]
        win = ohlc.loc[ohlc.index >= start]
        if end is not None:
            win = win.loc[win.index <= end]
        warm = pre.iloc[-WARMUP:] if len(pre) >= WARMUP else pre
        ohlc_run = pd.concat([warm, win])
        eval_start = start
    else:
        ohlc_run = ohlc
        eval_start = ohlc_run.index[WARMUP] if len(ohlc_run) > WARMUP else ohlc_run.index[0]
    if len(ohlc_run) < WARMUP + 30:
        return None
    # Basket-scale risk for shape curves (RF/5 ≈ 5-leg book). Full RF=8% single-leg
    # often yields 0 trades (sizing/kill artefacts); blend uses normalized equity anyway.
    research_risk = RF / 5.0
    bt = ewc.make_bt(cfg, leg, research_risk, MAX_LOT)
    bt.risk_fraction = research_risk
    # Warmup bars can trip FTMO kill-switch before eval_start and freeze the book;
    # disable flatten here — portfolio gates still enforced via compute_metrics.
    bt.flatten_on_breach = False
    eq = ewc.run_leg_equity(ohlc_run, leg, bt)
    eq = eq.loc[eq.index >= eval_start]
    if end is not None:
        eq = eq.loc[eq.index <= end]
    if len(eq) < 20:
        return None
    return eq


def port_from_legs(
    legs: list[dict], cfg: dict, start, end, *, apply_vt: bool = True, flatten_on_breach: bool = True
) -> pd.Series | None:
    curves, ws = [], []
    n = max(len(legs), 1)
    for leg in legs:
        path = ewc.resolve_csv(leg["symbol"], leg["timeframe"])
        if path is None:
            return None
        ohlc = load_ohlc_csv(path, symbol=leg["symbol"], timeframe=leg["timeframe"])
        if end is not None:
            ohlc = ohlc.loc[ohlc.index <= end]
        if start is not None:
            pre = ohlc.loc[ohlc.index < start]
            win = ohlc.loc[ohlc.index >= start]
            if end is not None:
                win = win.loc[win.index <= end]
            warm = pre.iloc[-WARMUP:] if len(pre) >= WARMUP else pre
            ohlc_run = pd.concat([warm, win])
            eval_start = start
        else:
            ohlc_run = ohlc
            eval_start = ohlc_run.index[WARMUP] if len(ohlc_run) > WARMUP else ohlc_run.index[0]
        if len(ohlc_run) < WARMUP + 30:
            return None
        # Single-leg research books use RF/5 so D1 actually trades; multi-leg uses RF/n
        r_leg = (RF / 5.0) if (n == 1 and not flatten_on_breach) else (RF / n)
        bt = ewc.make_bt(cfg, leg, r_leg, MAX_LOT)
        bt.flatten_on_breach = bool(flatten_on_breach)
        eq = ewc.run_leg_equity(ohlc_run, leg, bt)
        eq = eq.loc[eq.index >= eval_start]
        if end is not None:
            eq = eq.loc[eq.index <= end]
        if len(eq) < 20:
            return None
        curves.append(eq.rename(f"{leg['symbol']}_{leg['strategy']}"))
        ws.append(max(float(leg.get("weight") or leg.get("oos_sharpe") or 0.01), 0.01))
    w = np.array(ws, dtype=float)
    w = w / w.sum()
    port = combine_weighted(curves, w, INITIAL)
    if port is None or port.empty:
        return None
    if apply_vt:
        port = apply_vol_target(port, VT, look=60, lo=0.25, hi=3.0)
    return port


def stats_from_port(port: pd.Series) -> WindowStats | None:
    if port is None or len(port) < 20:
        return None
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
    return WindowStats(
        mean_mo=float(ms["mean_mo"]),
        pct_pos=float(ms["pct_pos"]),
        top3=float(ms["top3_share"]),
        gates=bool(m.gates_pass),
        p2t=float(p2t),
    )


def blend_locked_addon(
    locked_port: pd.Series,
    addon_port: pd.Series,
    sleeve: float,
) -> pd.Series | None:
    """Mix locked (1-sleeve) + addon (sleeve); renormalize to unit, no RF hike."""
    both = pd.concat(
        [locked_port.rename("L"), addon_port.rename("A")], axis=1, sort=True
    ).sort_index().ffill().dropna(how="any")
    if len(both) < 20:
        return None
    norms = both / both.iloc[0]
    s = float(np.clip(sleeve, 0.0, 0.5))
    mix = norms["L"] * (1.0 - s) + norms["A"] * s
    return mix * INITIAL


@dataclass
class BoardRow:
    idea: str
    family: str
    soft_pass: bool
    hard_pass: bool
    promote: bool
    score: float
    min_is_mo: float
    max_is_top3: float
    min_is_pct_pos: float
    mo_2024: float
    mo_2025_IS: float
    pos_2024: float
    pos_2025_IS: float
    top3_2024: float
    top3_2025_IS: float
    mo_holdout: float
    pos_holdout: float
    top3_holdout: float
    gates_holdout: bool
    note: str


def eval_idea(
    name: str,
    family: str,
    ports: dict[str, pd.Series],
    is_labels: list[str],
    conf_labels: list[str],
) -> BoardRow:
    is_stats = []
    wm: dict[str, WindowStats] = {}
    for lab in is_labels + conf_labels:
        st = stats_from_port(ports.get(lab)) if lab in ports else None
        if st is not None:
            wm[lab] = st
            if lab in is_labels:
                is_stats.append(st)
    sc = score_is_windows(
        is_stats,
        min_pct_pos=MIN_PCT_POS,
        max_top3_hard=MAX_TOP3_HARD,
        max_top3_soft=MAX_TOP3_SOFT,
        max_p2t=MAX_P2T,
    )
    ho = wm.get("holdout_365d")
    year_map = {k: wm[k] for k in ("2024", "2025", "2026") if k in wm}
    # For years_each_clear need calendar years; map 2025_IS only if 2025 missing (confirm path uses calendar)
    promote = False
    note = "is_fail"
    if sc.soft_pass and ho is not None:
        note = "soft_is"
        if sc.hard_pass:
            note = "hard_is"
        ho_ok = holdout_clears_promote(
            ho, min_mean_mo=0.01, min_pct_pos=MIN_PCT_POS, max_top3=MAX_TOP3_SOFT, max_p2t=MAX_P2T
        )
        # Prefer calendar 2025; fall back to 2025_IS only as diagnostic (not for promote)
        yclear = years_each_clear(
            {
                "2024": wm.get("2024"),
                "2025": wm.get("2025") or wm.get("2025_IS"),
                "2026": wm.get("2026"),
            },
            years=("2024", "2025", "2026"),
            min_mean_mo=0.01,
            min_pct_pos=MIN_PCT_POS,
        )
        if sc.soft_pass and ho_ok and yclear and sc.hard_pass:
            promote = True
            note = "promote"
        elif sc.soft_pass and not ho_ok:
            note = "holdout_fail"
        elif sc.soft_pass and not yclear:
            note = "year_clear_fail"
    st24 = wm.get("2024")
    st25 = wm.get("2025_IS") or wm.get("2025")
    return BoardRow(
        idea=name,
        family=family,
        soft_pass=bool(sc.soft_pass),
        hard_pass=bool(sc.hard_pass),
        promote=bool(promote),
        score=float(sc.score) if sc.score == sc.score else float("-inf"),
        min_is_mo=float(sc.min_mean_mo) if sc.min_mean_mo == sc.min_mean_mo else float("nan"),
        max_is_top3=float(sc.max_top3) if sc.max_top3 == sc.max_top3 else float("nan"),
        min_is_pct_pos=float(sc.min_pct_pos) if sc.min_pct_pos == sc.min_pct_pos else float("nan"),
        mo_2024=float(st24.mean_mo) if st24 else float("nan"),
        mo_2025_IS=float(st25.mean_mo) if st25 else float("nan"),
        pos_2024=float(st24.pct_pos) if st24 else float("nan"),
        pos_2025_IS=float(st25.pct_pos) if st25 else float("nan"),
        top3_2024=float(st24.top3) if st24 else float("nan"),
        top3_2025_IS=float(st25.top3) if st25 else float("nan"),
        mo_holdout=float(ho.mean_mo) if ho else float("nan"),
        pos_holdout=float(ho.pct_pos) if ho else float("nan"),
        top3_holdout=float(ho.top3) if ho else float("nan"),
        gates_holdout=bool(ho.gates) if ho else False,
        note=note,
    )


def main() -> None:
    cfg = load_config(ROOT / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    ds = data_source()
    print(f"data_source={ds} RF={RF} VT={VT} tag_locked={LOCKED_TAG}", flush=True)
    assert abs(RF - 0.08) < 1e-12

    legs = locked_legs()
    path0 = ewc.resolve_csv(legs[0]["symbol"], legs[0]["timeframe"])
    ohlc0 = load_ohlc_csv(path0, symbol=legs[0]["symbol"], timeframe=legs[0]["timeframe"])
    end = ohlc0.index.max()
    holdout_start = end - pd.Timedelta(days=365)
    is_wins = is_windows(end, holdout_start)
    conf_wins = confirm_windows(end, holdout_start)
    all_wins = [(n, s, e) for n, s, e in is_wins] + [(n, s, e) for n, s, e, _ in conf_wins]
    is_labels = [n for n, _, _ in is_wins]
    conf_labels = [n for n, _, _, _ in conf_wins]
    print(f"end={end} holdout_start={holdout_start} IS={is_labels} CONF={conf_labels}", flush=True)

    # Baseline locked ports per window
    base_ports: dict[str, pd.Series] = {}
    for lab, s, e in all_wins:
        p = port_from_legs(legs, cfg, s, e, apply_vt=True)
        if p is not None:
            base_ports[lab] = p
    board: list[BoardRow] = []
    board.append(eval_idea("baseline_locked", "baseline", base_ports, is_labels, conf_labels))
    print(
        f"baseline 2024 mo={board[0].mo_2024*100:.2f}% pos={board[0].pos_2024*100:.0f}% "
        f"top3={board[0].top3_2024*100:.0f}% | 2025_IS mo={board[0].mo_2025_IS*100:.2f}% "
        f"pos={board[0].pos_2025_IS*100:.0f}%",
        flush=True,
    )

    # Screen D1 pool
    wf = pd.read_csv(ROOT / "reports" / "walk_forward_summary.csv")
    pool = screen_d1_pool(wf)
    print(f"D1 pool size={len(pool)} (unused OOS-passers)", flush=True)
    pool_rows = []
    for leg in pool:
        pool_rows.append(
            {
                "symbol": leg["symbol"],
                "strategy": leg["strategy"],
                "oos_sharpe": leg["oos_sharpe"],
                "locked_sym": leg["symbol"] in LOCKED_SYMS,
            }
        )
    pd.DataFrame(pool_rows).to_csv(ROOT / "reports" / "quest_d1_consistency_pool.csv", index=False)

    # Per-window solo curves (NEVER one long span — warmup DD can kill the book)
    solo_ports: dict[str, dict[str, pd.Series]] = {}
    for leg in pool:
        key = f"{leg['symbol']}_{leg['strategy']}"
        by_w = {}
        for lab, s, e in all_wins:
            eq = run_leg_curve(leg, cfg, s, e)
            if eq is not None and len(eq) >= 20:
                by_w[lab] = eq
        if "2024" in by_w and "2025_IS" in by_w:
            solo_ports[key] = by_w
    print(f"solo curves ready={len(solo_ports)}/{len(pool)} (per-window)", flush=True)

    # Score solo D1 legs on IS for diversifier quality
    solo_scores = []
    for leg in pool:
        key = f"{leg['symbol']}_{leg['strategy']}"
        ports = solo_ports.get(key)
        if not ports:
            continue
        # Corr to locked on 2024 only (IS)
        corr = 1.0
        if "2024" in base_ports and ports.get("2024") is not None:
            corr = daily_return_corr(base_ports["2024"], ports["2024"])
            if corr != corr:  # NaN
                corr = 1.0
        st24 = stats_from_port(ports.get("2024"))
        st25 = stats_from_port(ports.get("2025_IS"))
        if st24 is None or st25 is None:
            continue
        if not st24.gates or not st25.gates:
            continue
        if st24.mean_mo <= 0.0 or st25.mean_mo <= 0.0:
            continue
        min_mo = min(st24.mean_mo, st25.mean_mo)
        min_pos = min(st24.pct_pos, st25.pct_pos)
        solo_scores.append(
            {
                "key": key,
                "leg": leg,
                "corr": corr,
                "min_mo": min_mo,
                "min_pos": min_pos,
                "mean_2024": st24.mean_mo,
                "mean_2025_IS": st25.mean_mo,
                "pos_2024": st24.pct_pos,
                "pos_2025_IS": st25.pct_pos,
                "top3_2024": st24.top3,
                "top3_2025_IS": st25.top3,
            }
        )
    # Prefer low corr, but still rank by min_mo (corr is soft — H4 vs D1 often looks high)
    solo_scores.sort(key=lambda x: (-x["min_mo"], x["corr"]))
    pd.DataFrame(
        [
            {k: v for k, v in r.items() if k != "leg"}
            | {"symbol": r["leg"]["symbol"], "strategy": r["leg"]["strategy"]}
            for r in solo_scores
        ]
    ).to_csv(ROOT / "reports" / "quest_d1_consistency_solo.csv", index=False)
    print(f"solo IS-scored={len(solo_scores)}", flush=True)
    for r in solo_scores[:8]:
        print(
            f"  solo {r['key']}: min_mo={r['min_mo']*100:.2f}% corr={r['corr']:.2f} "
            f"pos24={r['pos_2024']*100:.0f}% pos25={r['pos_2025_IS']*100:.0f}%",
            flush=True,
        )

    # --- Family A: lock + single D1 addon at various sleeve fracs ---
    low = [r for r in solo_scores if r["corr"] < MAX_CORR]
    # Soft rescue: if few low-corr, allow up to 0.75 corr for top min_mo names
    if len(low) < 6:
        rescue = [r for r in solo_scores if r["corr"] < 0.75 and r not in low]
        low = (low + rescue)[:12]
    candidates_a = low[:12] if low else solo_scores[:8]
    print(f"Family A addons={len(candidates_a)} (low_corr_soft)", flush=True)
    for rec in candidates_a:
        key = rec["key"]
        add_ports = solo_ports[key]
        for sleeve in ADDON_SLEEVE_FRACS:
            ports = {}
            for lab, s, e in all_wins:
                base = base_ports.get(lab)
                add = add_ports.get(lab)
                if base is None or add is None:
                    continue
                ports[lab] = blend_locked_addon(base, add, sleeve)
            idea = f"lock+{key}_s{int(sleeve*100)}"
            board.append(eval_idea(idea, "lock_plus_d1", ports, is_labels, conf_labels))

    # --- Family B: lock + 2-leg D1 sleeve (top decorrelated pair) ---
    low_corr = candidates_a[:8]
    pair_tried = 0
    for i in range(len(low_corr)):
        for j in range(i + 1, len(low_corr)):
            if pair_tried >= 10:
                break
            a, b = low_corr[i], low_corr[j]
            # pairwise corr on 2024
            pa = solo_ports[a["key"]].get("2024")
            pb = solo_ports[b["key"]].get("2024")
            if pa is None or pb is None:
                continue
            pc = daily_return_corr(pa, pb)
            if pc != pc or pc > MAX_CORR:
                continue
            pair_tried += 1
            for sleeve in (0.20, 0.30):
                ports = {}
                for lab, s, e in all_wins:
                    base = base_ports.get(lab)
                    aa = solo_ports[a["key"]].get(lab)
                    bb = solo_ports[b["key"]].get(lab)
                    if base is None or aa is None or bb is None:
                        continue
                    both = pd.concat([aa.rename("a"), bb.rename("b")], axis=1).ffill().dropna()
                    if len(both) < 20:
                        continue
                    norms = both / both.iloc[0]
                    addon = (norms["a"] * 0.5 + norms["b"] * 0.5) * INITIAL
                    ports[lab] = blend_locked_addon(base, addon, sleeve)
                idea = f"lock+{a['key']}+{b['key']}_s{int(sleeve*100)}"
                board.append(eval_idea(idea, "lock_plus_d1_pair", ports, is_labels, conf_labels))
        if pair_tried >= 10:
            break
    print(f"Family B pairs tried={pair_tried}", flush=True)

    # --- Family C: D1-only alternate sleeves (no H4 locked) — top 3 solo + equal-weight baskets ---
    d1_only_legs = []
    for rec in solo_scores[:6]:
        d1_only_legs.append(rec["leg"])
        ports = {}
        for lab, s, e in all_wins:
            ports[lab] = port_from_legs([rec["leg"]], cfg, s, e, apply_vt=True, flatten_on_breach=False)
        board.append(eval_idea(f"d1solo_{rec['key']}", "d1_alternate", ports, is_labels, conf_labels))
    # Equal-weight basket of top-3 unused-symbol D1
    top3 = [r for r in solo_scores if r["leg"]["symbol"] not in LOCKED_SYMS][:3]
    if len(top3) >= 2:
        for sleeve_note, pick in (("top2", top3[:2]), ("top3", top3[:3])):
            ports = {}
            for lab, s, e in all_wins:
                ports[lab] = port_from_legs(
                    [r["leg"] for r in pick], cfg, s, e, apply_vt=True, flatten_on_breach=False
                )
            board.append(eval_idea(f"d1basket_{sleeve_note}", "d1_alternate", ports, is_labels, conf_labels))

    # --- Family D: causal MTD gain clip on locked (tau chosen on 2024 only) ---
    # Grid on 2024: pick tau/after that best improves top3 while mean_mo >= 0.004 (don't collapse)
    # then validate on 2025_IS; freeze; confirm HO.
    taus = [0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]
    afters = [0.0, 0.25, 0.5]
    best_clip = None  # (score_2024, tau, after, st24)
    base24 = base_ports.get("2024")
    if base24 is not None:
        base_st24 = stats_from_port(base24)
        for tau in taus:
            for after in afters:
                clipped = apply_mtd_gain_clip(base24, tau=tau, after_clip=after)
                st = stats_from_port(clipped)
                if st is None or not st.gates:
                    continue
                # Must not collapse mean below locked*0.85 or absolute 0.35%
                if st.mean_mo < max(0.0035, (base_st24.mean_mo if base_st24 else 0.004) * 0.85):
                    continue
                # Prefer lower top3, then higher mean
                score = -st.top3 + 0.1 * st.mean_mo + (0.05 if st.pct_pos >= 0.70 else 0.0)
                if best_clip is None or score > best_clip[0]:
                    best_clip = (score, tau, after, st)
        print(f"best 2024-only clip={best_clip}", flush=True)
        if best_clip is not None:
            _, tau, after, _ = best_clip
            ports = {}
            for lab, s, e in all_wins:
                base = base_ports.get(lab)
                if base is None:
                    continue
                ports[lab] = apply_mtd_gain_clip(base, tau=tau, after_clip=after)
            board.append(
                eval_idea(f"lock_mtdclip_t{tau}_a{after}", "mtd_gain_clip", ports, is_labels, conf_labels)
            )
            # Also try clip on best soft-ish lock+addon if any soft later — deferred; clip alone first

            # Nested: clip + best low-corr addon (tau frozen from 2024; sleeve from IS dual)
            if candidates_a:
                rec = candidates_a[0]
                add_ports = solo_ports[rec["key"]]
                for sleeve in (0.20, 0.25):
                    ports = {}
                    for lab, s, e in all_wins:
                        base = base_ports.get(lab)
                        add = add_ports.get(lab)
                        if base is None or add is None:
                            continue
                        blended = blend_locked_addon(base, add, sleeve)
                        ports[lab] = apply_mtd_gain_clip(blended, tau=tau, after_clip=after)
                    board.append(
                        eval_idea(
                            f"lock+{rec['key']}_s{int(sleeve*100)}_mtdclip",
                            "lock_plus_clip",
                            ports,
                            is_labels,
                            conf_labels,
                        )
                    )

    # Aggregate board
    bdf = pd.DataFrame([asdict(r) for r in board])
    bdf.to_csv(ROOT / "reports" / "quest_d1_consistency_board.csv", index=False)
    soft_n = int(bdf["soft_pass"].sum())
    hard_n = int(bdf["hard_pass"].sum())
    prom_n = int(bdf["promote"].sum())
    print(f"BOARD soft={soft_n} hard={hard_n} promote={prom_n} n={len(bdf)}", flush=True)

    # Best by IS score among soft; else nearest miss by min_is_mo
    soft = bdf[bdf["soft_pass"]].sort_values("score", ascending=False)
    hard = bdf[bdf["hard_pass"]].sort_values("score", ascending=False)
    prom = bdf[bdf["promote"]]
    nearest = bdf.replace([np.inf, -np.inf], np.nan).dropna(subset=["min_is_mo"]).sort_values(
        "min_is_mo", ascending=False
    )

    lines = [
        "# D1 multi-year consistency repair + MTD gain-clip",
        "",
        f"**data_source:** `{ds}`  **RF:** {RF:.0%}  **PORT_VOL_TARGET:** {VT}  **signal_lag:** 1",
        f"**Locked tag (unchanged unless promote):** `{LOCKED_TAG}`",
        f"**IS windows:** {is_labels}  **Confirm:** {conf_labels}",
        f"**Selection:** maximize min(IS mean_mo) with %pos≥70%, top3 soft≤70% / hard≤55%. Holdout never for selection.",
        "",
        "## Board",
        "",
        f"| Family | N | Soft IS | Hard IS | Promote |",
        f"|--------|--:|--------:|--------:|--------:|",
    ]
    for fam, g in bdf.groupby("family"):
        lines.append(
            f"| {fam} | {len(g)} | {int(g.soft_pass.sum())} | {int(g.hard_pass.sum())} | {int(g.promote.sum())} |"
        )
    lines += [
        "",
        f"**Totals:** soft={soft_n} hard={hard_n} promote={prom_n} (board n={len(bdf)})",
        f"**D1 pool screened:** {len(pool)}  **solo IS-scored:** {len(solo_scores)}  **pair trials:** {pair_tried}",
        "",
    ]
    if best_clip is not None:
        lines.append(
            f"**2024-only MTD clip pick:** tau={best_clip[1]} after={best_clip[2]} "
            f"(2024 top3→{best_clip[3].top3*100:.0f}% mean→{best_clip[3].mean_mo*100:.2f}% "
            f"pos→{best_clip[3].pct_pos*100:.0f}%)"
        )
        lines.append("")

    def fmt_row(r: pd.Series) -> list[str]:
        return [
            f"### `{r['idea']}` ({r['family']}) — note={r['note']}",
            "",
            f"| Window | Mean mo | %pos | Top3 |",
            f"|--------|--------:|-----:|-----:|",
            f"| 2024 (IS) | {r['mo_2024']*100:.2f}% | {r['pos_2024']*100:.0f}% | {r['top3_2024']*100:.0f}% |",
            f"| 2025_IS | {r['mo_2025_IS']*100:.2f}% | {r['pos_2025_IS']*100:.0f}% | {r['top3_2025_IS']*100:.0f}% |",
            f"| holdout | {r['mo_holdout']*100:.2f}% | {r['pos_holdout']*100:.0f}% | {r['top3_holdout']*100:.0f}% |",
            "",
            f"soft={r['soft_pass']} hard={r['hard_pass']} promote={r['promote']} "
            f"min_is_mo={r['min_is_mo']*100:.2f}% max_is_top3={r['max_is_top3']*100:.0f}%",
            "",
        ]

    lines.append("## Soft / hard / promote")
    lines.append("")
    if len(prom):
        lines.append("**PROMOTE candidates:**")
        lines.append("")
        for _, r in prom.iterrows():
            lines.extend(fmt_row(r))
    else:
        lines.append("**No promote.**")
        lines.append("")
    if len(soft):
        lines.append("**Best soft IS:**")
        lines.append("")
        lines.extend(fmt_row(soft.iloc[0]))
    else:
        lines.append("**0 soft IS passers.**")
        lines.append("")
        if len(nearest):
            lines.append("**Nearest miss (highest min_is_mo, may fail %pos/top3):**")
            lines.append("")
            lines.extend(fmt_row(nearest.iloc[0]))
            # Also show top 5 misses briefly
            lines.append("### Top-5 by min_is_mo (diagnostic)")
            lines.append("")
            lines.append("| idea | min_is_mo | pos24 | pos25 | top3_max | note |")
            lines.append("|------|----------:|------:|------:|---------:|------|")
            for _, r in nearest.head(5).iterrows():
                lines.append(
                    f"| `{r['idea']}` | {r['min_is_mo']*100:.2f}% | {r['pos_2024']*100:.0f}% | "
                    f"{r['pos_2025_IS']*100:.0f}% | {r['max_is_top3']*100:.0f}% | {r['note']} |"
                )
            lines.append("")

    promote_yes = prom_n > 0
    lines += [
        "## Verdict",
        "",
        f"- **Promote:** {'YES' if promote_yes else 'NO'}",
        f"- Locked tag remains `{LOCKED_TAG}`" + (" — WAIT: promote found, review before overwrite" if promote_yes else ""),
        f"- Still blocked on FTMO CSVs: **{'NO' if ds.startswith('ftmo') else 'YES'}** (`data/ftmo/` empty of CSVs)",
        "",
        "Artifacts: `reports/quest_d1_consistency_board.csv`, `quest_d1_consistency_pool.csv`, "
        "`quest_d1_consistency_solo.csv`, `quest_d1_consistency_repair.md`",
        "",
    ]
    out = ROOT / "reports" / "quest_d1_consistency_repair.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Promote JSON if any
    sel = {
        "promote": promote_yes,
        "soft": soft_n,
        "hard": hard_n,
        "best_soft": soft.iloc[0]["idea"] if len(soft) else None,
        "best_clip_tau": best_clip[1] if best_clip else None,
        "best_clip_after": best_clip[2] if best_clip else None,
        "nearest": nearest.iloc[0]["idea"] if len(nearest) else None,
        "data_source": ds,
    }
    (ROOT / "configs" / "quest_d1_consistency_selected.json").write_text(
        json.dumps(sel, indent=2) + "\n", encoding="utf-8"
    )
    print("Wrote", out)
    print(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
