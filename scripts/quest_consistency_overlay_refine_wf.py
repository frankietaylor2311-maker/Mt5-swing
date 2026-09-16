#!/usr/bin/env python3
"""Consistency overlay refine on locked basket (no D1 add-ons).

Wave goal: causal overlays + IS-only weight rebalance of locked 5 legs to lift
multi-year consistency (~1% mean_mo, high %pos, low top3) under FTMO gates.
RF fixed 8%. Overlay hi<=1. signal_lag=1. Holdout NEVER for selection.

Nested selection: pick overlay/weight params on 2024 only (or max min(2024,2025_IS)
when both available without peeking HO), freeze, then confirm on 2025/2026/HO.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
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
    apply_equity_curve_target,
    apply_month_aware_scale,
    apply_mtd_gain_clip,
    apply_runup_throttle,
    apply_vol_target,
    combine_weighted,
)
from mt5_swing.portfolio.smooth_select import WindowStats

WARMUP = 250
RF = 0.08
MAX_LOT = 50.0
VT = 0.0025
LOCKED_TAG = "fx4plus_gbpcad_d1_voltarget_0025"
MIN_PCT_POS = 0.70
MAX_TOP3_HARD = 0.55
MAX_TOP3_SOFT = 0.70
MAX_P2T = 0.085


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
    research_risk = RF / 5.0
    bt = ewc.make_bt(cfg, leg, research_risk, MAX_LOT)
    bt.flatten_on_breach = True
    eq = ewc.run_leg_equity(ohlc_run, leg, bt)
    eq = eq.loc[eq.index >= eval_start]
    if end is not None:
        eq = eq.loc[eq.index <= end]
    if len(eq) < 20:
        return None
    return eq


def port_from_leg_curves(
    curves: list, weights: np.ndarray, *, apply_vt: bool = True
):
    if not curves or any(c is None or len(c) < 20 for c in curves):
        return None
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 0.0)
    if w.sum() <= 0:
        return None
    w = w / w.sum()
    port = combine_weighted(curves, w, INITIAL)
    if port is None or port.empty:
        return None
    if apply_vt:
        port = apply_vol_target(port, VT, look=60, lo=0.25, hi=3.0)
    return port


def stats_from_port(port: pd.Series):
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
    mo_2025: float
    mo_2026: float
    pos_2025: float
    pos_2026: float
    top3_2025: float
    top3_2026: float
    mo_holdout: float
    pos_holdout: float
    top3_holdout: float
    gates_holdout: bool
    note: str


def eval_idea(name, family, ports, is_labels, conf_labels) -> BoardRow:
    is_stats = []
    wm = {}
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
    promote = False
    note = "is_fail"
    if sc.soft_pass and ho is not None:
        note = "soft_is"
        if sc.hard_pass:
            note = "hard_is"
        ho_ok = holdout_clears_promote(
            ho, min_mean_mo=0.01, min_pct_pos=MIN_PCT_POS, max_top3=MAX_TOP3_SOFT, max_p2t=MAX_P2T
        )
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
        elif sc.soft_pass and not sc.hard_pass:
            note = "soft_is_hard_fail"
    st24 = wm.get("2024")
    st25is = wm.get("2025_IS")
    st25 = wm.get("2025")
    st26 = wm.get("2026")
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
        mo_2025_IS=float(st25is.mean_mo) if st25is else float("nan"),
        pos_2024=float(st24.pct_pos) if st24 else float("nan"),
        pos_2025_IS=float(st25is.pct_pos) if st25is else float("nan"),
        top3_2024=float(st24.top3) if st24 else float("nan"),
        top3_2025_IS=float(st25is.top3) if st25is else float("nan"),
        mo_2025=float(st25.mean_mo) if st25 else float("nan"),
        mo_2026=float(st26.mean_mo) if st26 else float("nan"),
        pos_2025=float(st25.pct_pos) if st25 else float("nan"),
        pos_2026=float(st26.pct_pos) if st26 else float("nan"),
        top3_2025=float(st25.top3) if st25 else float("nan"),
        top3_2026=float(st26.top3) if st26 else float("nan"),
        mo_holdout=float(ho.mean_mo) if ho else float("nan"),
        pos_holdout=float(ho.pct_pos) if ho else float("nan"),
        top3_holdout=float(ho.top3) if ho else float("nan"),
        gates_holdout=bool(ho.gates) if ho else False,
        note=note,
    )


def score_2024_pick(st, base_st) -> float:
    if st is None or not st.gates:
        return float("-inf")
    floor = 0.0035
    if base_st is not None:
        floor = max(floor, base_st.mean_mo * 0.80)
    if st.mean_mo < floor:
        return float("-inf")
    bonus_pos = 0.08 if st.pct_pos >= 0.70 else (0.03 if st.pct_pos >= 0.60 else 0.0)
    return -st.top3 + 0.15 * st.mean_mo + bonus_pos + (0.02 if st.pct_pos >= 0.75 else 0.0)


def apply_named_overlay(port, name, params):
    if name == "mtd":
        return apply_mtd_gain_clip(port, tau=params["tau"], after_clip=params["after"])
    if name == "month":
        return apply_month_aware_scale(
            port,
            strong_mo=params["strong_mo"],
            after_strong=params["after_strong"],
            dd_trigger=params["dd_trigger"],
            after_dd=params["after_dd"],
            lo=0.25,
        )
    if name == "runup":
        return apply_runup_throttle(
            port,
            trail_bars=params["trail_bars"],
            runup_thresh=params["runup_thresh"],
            cool_scale=params["cool_scale"],
            lo=0.25,
        )
    if name == "eqtarget":
        return apply_equity_curve_target(
            port,
            target_mo_vol=params["target_mo_vol"],
            lookback_months=params["lookback_months"],
            lo=0.25,
            hi=1.0,
        )
    raise ValueError(name)


def tag_overlay(name, params) -> str:
    if name == "mtd":
        return f"mtd_t{params['tau']}_a{params['after']}"
    if name == "month":
        return (
            f"mo_s{params['strong_mo']}_as{params['after_strong']}"
            f"_dd{params['dd_trigger']}_ad{params['after_dd']}"
        )
    if name == "runup":
        return f"ru_tb{params['trail_bars']}_th{params['runup_thresh']}_cs{params['cool_scale']}"
    if name == "eqtarget":
        return f"eq_tv{params['target_mo_vol']}_lb{params['lookback_months']}"
    return name


def ports_with_overlay(base_ports, name, params):
    out = {}
    for lab, p in base_ports.items():
        if p is None:
            continue
        out[lab] = apply_named_overlay(p, name, params)
    return out


def ports_with_two(base_ports, n1, p1, n2, p2):
    out = {}
    for lab, p in base_ports.items():
        if p is None:
            continue
        mid = apply_named_overlay(p, n1, p1)
        out[lab] = apply_named_overlay(mid, n2, p2)
    return out


def simplex_perturbations(base_w, scales, n_dirichlet=24):
    n = len(base_w)
    cands = [base_w.copy()]
    for i in range(n):
        for s in scales:
            w = base_w.copy()
            w[i] *= s
            w = np.maximum(w, 1e-6)
            w = w / w.sum()
            cands.append(w)
    for i, j in itertools.combinations(range(n), 2):
        for s in (0.75, 1.25):
            w = base_w.copy()
            w[i] *= s
            w[j] *= 2.0 - s
            w = np.maximum(w, 1e-6)
            w = w / w.sum()
            cands.append(w)
    rng = np.random.default_rng(42)
    for conc in (40.0, 80.0, 120.0):
        alpha = np.maximum(base_w * conc, 0.5)
        for _ in range(n_dirichlet // 3):
            w = rng.dirichlet(alpha)
            cands.append(w)
    uniq = []
    seen = set()
    for w in cands:
        key = tuple(np.round(w, 4))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(w)
    return uniq


def main() -> None:
    cfg = load_config(ROOT / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    ds = data_source()
    print(f"data_source={ds} RF={RF} VT={VT} tag_locked={LOCKED_TAG}", flush=True)
    assert abs(RF - 0.08) < 1e-12

    legs = locked_legs()
    base_w = np.array([l["weight"] for l in legs], dtype=float)
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

    leg_curves = {}
    for lab, s, e in all_wins:
        curves = []
        ok = True
        for leg in legs:
            eq = run_leg_curve(leg, cfg, s, e)
            if eq is None:
                ok = False
                break
            curves.append(eq.rename(f"{leg['symbol']}_{leg['strategy']}"))
        if ok:
            leg_curves[lab] = curves
            print(f"  legs ready {lab}: {[len(c) for c in curves]}", flush=True)

    base_ports = {}
    for lab, curves in leg_curves.items():
        p = port_from_leg_curves(curves, base_w, apply_vt=True)
        if p is not None:
            base_ports[lab] = p

    board = []
    board.append(eval_idea("baseline_locked", "baseline", base_ports, is_labels, conf_labels))
    print(
        f"baseline 2024 mo={board[0].mo_2024*100:.2f}% pos={board[0].pos_2024*100:.0f}% "
        f"top3={board[0].top3_2024*100:.0f}% | 2025_IS mo={board[0].mo_2025_IS*100:.2f}% "
        f"pos={board[0].pos_2025_IS*100:.0f}%",
        flush=True,
    )
    base24 = base_ports.get("2024")
    base_st24 = stats_from_port(base24) if base24 is not None else None

    # MTD finer grid
    mtd_taus = [0.008, 0.010, 0.012, 0.014, 0.015, 0.016, 0.018, 0.020, 0.025]
    mtd_afters = [0.0, 0.15, 0.25, 0.35, 0.5]
    best_mtd = None
    if base24 is not None:
        for tau, after in itertools.product(mtd_taus, mtd_afters):
            st = stats_from_port(apply_mtd_gain_clip(base24, tau=tau, after_clip=after))
            if st is None:
                continue
            sc = score_2024_pick(st, base_st24)
            if best_mtd is None or sc > best_mtd[0]:
                best_mtd = (sc, {"tau": tau, "after": after}, st)
        print(f"best 2024 MTD={best_mtd[1] if best_mtd else None}", flush=True)
        ranked = []
        for tau, after in itertools.product(mtd_taus, mtd_afters):
            st = stats_from_port(apply_mtd_gain_clip(base24, tau=tau, after_clip=after))
            if st is None:
                continue
            sc = score_2024_pick(st, base_st24)
            if sc > float("-inf"):
                ranked.append((sc, {"tau": tau, "after": after}))
        ranked.sort(key=lambda x: -x[0])
        for sc, params in ranked[:4]:
            ports = ports_with_overlay(base_ports, "mtd", params)
            board.append(
                eval_idea(f"lock_{tag_overlay('mtd', params)}", "mtd_gain_clip", ports, is_labels, conf_labels)
            )

    # month_aware
    month_grid = list(itertools.product([0.015, 0.02, 0.025, 0.03], [0.4, 0.55, 0.7], [0.03, 0.04, 0.05], [0.4, 0.55, 0.7]))
    best_mo = None
    if base24 is not None:
        for strong_mo, after_strong, dd_trigger, after_dd in month_grid:
            params = {
                "strong_mo": strong_mo,
                "after_strong": after_strong,
                "dd_trigger": dd_trigger,
                "after_dd": after_dd,
            }
            st = stats_from_port(apply_named_overlay(base24, "month", params))
            if st is None:
                continue
            sc = score_2024_pick(st, base_st24)
            if best_mo is None or sc > best_mo[0]:
                best_mo = (sc, params, st)
        print(f"best 2024 month={best_mo[1] if best_mo else None}", flush=True)
        if best_mo is not None:
            ports = ports_with_overlay(base_ports, "month", best_mo[1])
            board.append(
                eval_idea(f"lock_{tag_overlay('month', best_mo[1])}", "month_aware", ports, is_labels, conf_labels)
            )

    # runup
    runup_grid = list(itertools.product([21, 42, 63], [0.03, 0.04, 0.05, 0.06], [0.35, 0.5, 0.65]))
    best_ru = None
    if base24 is not None:
        for trail_bars, runup_thresh, cool_scale in runup_grid:
            params = {"trail_bars": trail_bars, "runup_thresh": runup_thresh, "cool_scale": cool_scale}
            st = stats_from_port(apply_named_overlay(base24, "runup", params))
            if st is None:
                continue
            sc = score_2024_pick(st, base_st24)
            if best_ru is None or sc > best_ru[0]:
                best_ru = (sc, params, st)
        print(f"best 2024 runup={best_ru[1] if best_ru else None}", flush=True)
        if best_ru is not None:
            ports = ports_with_overlay(base_ports, "runup", best_ru[1])
            board.append(
                eval_idea(f"lock_{tag_overlay('runup', best_ru[1])}", "runup_throttle", ports, is_labels, conf_labels)
            )

    # equity curve target
    eq_grid = list(itertools.product([0.008, 0.010, 0.012, 0.015], [4, 6, 9]))
    best_eq = None
    if base24 is not None:
        for target_mo_vol, lookback_months in eq_grid:
            params = {"target_mo_vol": target_mo_vol, "lookback_months": lookback_months}
            st = stats_from_port(apply_named_overlay(base24, "eqtarget", params))
            if st is None:
                continue
            sc = score_2024_pick(st, base_st24)
            if best_eq is None or sc > best_eq[0]:
                best_eq = (sc, params, st)
        print(f"best 2024 eqtarget={best_eq[1] if best_eq else None}", flush=True)
        if best_eq is not None:
            ports = ports_with_overlay(base_ports, "eqtarget", best_eq[1])
            board.append(
                eval_idea(
                    f"lock_{tag_overlay('eqtarget', best_eq[1])}",
                    "equity_curve_target",
                    ports,
                    is_labels,
                    conf_labels,
                )
            )

    singles = []
    if best_mtd is not None:
        singles.append(("mtd", best_mtd[1]))
    if best_mo is not None:
        singles.append(("month", best_mo[1]))
    if best_ru is not None:
        singles.append(("runup", best_ru[1]))
    if best_eq is not None:
        singles.append(("eqtarget", best_eq[1]))

    mild_mtd = {"tau": 0.02, "after": 0.25}
    combo_pairs = list(itertools.combinations(singles, 2))
    if best_mo is not None:
        combo_pairs.append((("mtd", mild_mtd), ("month", best_mo[1])))
    if best_ru is not None:
        combo_pairs.append((("mtd", mild_mtd), ("runup", best_ru[1])))
    if best_mtd is not None and best_eq is not None:
        combo_pairs.append((("mtd", best_mtd[1]), ("eqtarget", best_eq[1])))

    seen_combo = set()
    for (n1, p1), (n2, p2) in combo_pairs:
        if base24 is None:
            continue
        a = apply_named_overlay(apply_named_overlay(base24, n1, p1), n2, p2)
        b = apply_named_overlay(apply_named_overlay(base24, n2, p2), n1, p1)
        sa, sb = stats_from_port(a), stats_from_port(b)
        sca = score_2024_pick(sa, base_st24) if sa else float("-inf")
        scb = score_2024_pick(sb, base_st24) if sb else float("-inf")
        if sca >= scb and sca > float("-inf"):
            tag = f"lock_{tag_overlay(n1, p1)}+{tag_overlay(n2, p2)}"
            if tag in seen_combo:
                continue
            seen_combo.add(tag)
            ports = ports_with_two(base_ports, n1, p1, n2, p2)
            board.append(eval_idea(tag, "combo_two", ports, is_labels, conf_labels))
        elif scb > float("-inf"):
            tag = f"lock_{tag_overlay(n2, p2)}+{tag_overlay(n1, p1)}"
            if tag in seen_combo:
                continue
            seen_combo.add(tag)
            ports = ports_with_two(base_ports, n2, p2, n1, p1)
            board.append(eval_idea(tag, "combo_two", ports, is_labels, conf_labels))
    print(f"combo_two adds={len(seen_combo)}", flush=True)

    # IS-only weight rebalance
    weight_cands = simplex_perturbations(base_w, scales=[0.6, 0.75, 0.9, 1.1, 1.25, 1.5], n_dirichlet=24)
    print(f"weight candidates={len(weight_cands)}", flush=True)
    best_w = None
    curves24 = leg_curves.get("2024")
    curves25 = leg_curves.get("2025_IS")
    if curves24 is not None:
        for w in weight_cands:
            p24 = port_from_leg_curves(curves24, w, apply_vt=True)
            st24 = stats_from_port(p24)
            if st24 is None or not st24.gates:
                continue
            st25 = None
            if curves25 is not None:
                p25 = port_from_leg_curves(curves25, w, apply_vt=True)
                st25 = stats_from_port(p25)
            if st25 is not None and st25.gates:
                min_mo = min(st24.mean_mo, st25.mean_mo)
                min_pos = min(st24.pct_pos, st25.pct_pos)
                max_t3 = max(st24.top3, st25.top3)
            else:
                min_mo = st24.mean_mo
                min_pos = st24.pct_pos
                max_t3 = st24.top3
            if min_pos < MIN_PCT_POS or max_t3 > MAX_TOP3_SOFT:
                continue
            sc = min_mo - 0.02 * max(0.0, max_t3 - MAX_TOP3_HARD)
            if best_w is None or sc > best_w[0]:
                best_w = (sc, w.copy(), st24, st25)
        print(
            f"best weight sc={best_w[0] if best_w else None} "
            f"w={np.round(best_w[1], 4).tolist() if best_w else None}",
            flush=True,
        )
        if best_w is not None:
            w = best_w[1]
            ports = {lab: port_from_leg_curves(curves, w, apply_vt=True) for lab, curves in leg_curves.items()}
            wtag = "wrebal_" + "_".join(f"{x:.3f}" for x in w)
            board.append(eval_idea(wtag, "weight_rebalance", ports, is_labels, conf_labels))
            if "2024" in ports:
                best_wm = None
                for tau, after in itertools.product([0.012, 0.015, 0.018, 0.02, 0.025], [0.0, 0.25, 0.5]):
                    st = stats_from_port(apply_mtd_gain_clip(ports["2024"], tau=tau, after_clip=after))
                    if st is None:
                        continue
                    sc = score_2024_pick(st, stats_from_port(ports["2024"]))
                    if best_wm is None or sc > best_wm[0]:
                        best_wm = (sc, {"tau": tau, "after": after})
                if best_wm is not None:
                    params = best_wm[1]
                    cports = ports_with_overlay(ports, "mtd", params)
                    board.append(
                        eval_idea(
                            f"{wtag}+{tag_overlay('mtd', params)}",
                            "weight_plus_mtd",
                            cports,
                            is_labels,
                            conf_labels,
                        )
                    )
            if best_mo is not None:
                cports = ports_with_overlay(ports, "month", best_mo[1])
                board.append(
                    eval_idea(
                        f"{wtag}+{tag_overlay('month', best_mo[1])}",
                        "weight_plus_month",
                        cports,
                        is_labels,
                        conf_labels,
                    )
                )

    eq_w = np.ones(len(legs)) / len(legs)
    ports_eq = {lab: port_from_leg_curves(curves, eq_w, apply_vt=True) for lab, curves in leg_curves.items()}
    board.append(eval_idea("equal_weight_vt", "weight_rebalance", ports_eq, is_labels, conf_labels))

    bdf = pd.DataFrame([asdict(r) for r in board])
    bdf.to_csv(ROOT / "reports" / "quest_consistency_overlay_board.csv", index=False)
    soft_n = int(bdf["soft_pass"].sum())
    hard_n = int(bdf["hard_pass"].sum())
    prom_n = int(bdf["promote"].sum())
    print(f"BOARD soft={soft_n} hard={hard_n} promote={prom_n} n={len(bdf)}", flush=True)

    soft = bdf[bdf["soft_pass"]].sort_values("score", ascending=False)
    hard = bdf[bdf["hard_pass"]].sort_values("score", ascending=False)
    prom = bdf[bdf["promote"]]
    nearest = bdf.replace([np.inf, -np.inf], np.nan).dropna(subset=["min_is_mo"]).sort_values(
        "min_is_mo", ascending=False
    )

    lines = [
        "# Consistency overlay refine (locked sleeve only)",
        "",
        f"**data_source:** `{ds}`  **RF:** {RF:.0%}  **PORT_VOL_TARGET:** {VT}  **signal_lag:** 1",
        f"**Locked tag (unchanged unless promote):** `{LOCKED_TAG}`",
        f"**IS windows:** {is_labels}  **Confirm:** {conf_labels}",
        "**Selection:** nested on 2024 (or max min(2024,2025_IS) for weights); HO never for selection.",
        "**Overlays:** mtd / month_aware / runup / equity_curve_target / combos<=2 / IS weight rebalance. hi<=1.",
        "",
        "## Board",
        "",
        "| Family | N | Soft IS | Hard IS | Promote |",
        "|--------|--:|--------:|--------:|--------:|",
    ]
    for fam, g in bdf.groupby("family"):
        lines.append(
            f"| {fam} | {len(g)} | {int(g.soft_pass.sum())} | {int(g.hard_pass.sum())} | {int(g.promote.sum())} |"
        )
    lines += [
        "",
        f"**Totals:** soft={soft_n} hard={hard_n} promote={prom_n} (board n={len(bdf)})",
        "",
    ]
    if best_mtd:
        lines.append(
            f"**2024-only MTD pick:** {best_mtd[1]} -> mean={best_mtd[2].mean_mo*100:.2f}% "
            f"pos={best_mtd[2].pct_pos*100:.0f}% top3={best_mtd[2].top3*100:.0f}%"
        )
    if best_mo:
        lines.append(f"**2024-only month_aware pick:** {best_mo[1]}")
    if best_ru:
        lines.append(f"**2024-only runup pick:** {best_ru[1]}")
    if best_eq:
        lines.append(f"**2024-only eqtarget pick:** {best_eq[1]}")
    if best_w:
        lines.append(f"**IS weight rebalance:** {np.round(best_w[1], 4).tolist()} score={best_w[0]:.4f}")
    lines.append("")

    def fmt_row(r):
        return [
            f"### `{r['idea']}` ({r['family']}) — note={r['note']}",
            "",
            "| Window | Mean mo | %pos | Top3 |",
            "|--------|--------:|-----:|-----:|",
            f"| 2024 (IS) | {r['mo_2024']*100:.2f}% | {r['pos_2024']*100:.0f}% | {r['top3_2024']*100:.0f}% |",
            f"| 2025_IS | {r['mo_2025_IS']*100:.2f}% | {r['pos_2025_IS']*100:.0f}% | {r['top3_2025_IS']*100:.0f}% |",
            f"| 2025 (cal) | {r['mo_2025']*100:.2f}% | {r['pos_2025']*100:.0f}% | {r['top3_2025']*100:.0f}% |",
            f"| 2026 | {r['mo_2026']*100:.2f}% | {r['pos_2026']*100:.0f}% | {r['top3_2026']*100:.0f}% |",
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
        lines.append("### Top-8 by min_is_mo (diagnostic)")
        lines.append("")
        lines.append("| idea | family | min_is_mo | pos24 | pos25is | top3_max | note |")
        lines.append("|------|--------|----------:|------:|--------:|---------:|------|")
        for _, r in nearest.head(8).iterrows():
            lines.append(
                f"| `{r['idea']}` | {r['family']} | {r['min_is_mo']*100:.2f}% | "
                f"{r['pos_2024']*100:.0f}% | {r['pos_2025_IS']*100:.0f}% | "
                f"{r['max_is_top3']*100:.0f}% | {r['note']} |"
            )
        lines.append("")

    promote_yes = prom_n > 0
    best_soft_tag = soft.iloc[0]["idea"] if len(soft) else None
    nearest_tag = nearest.iloc[0]["idea"] if len(nearest) else None
    lines += [
        "## Verdict",
        "",
        f"- **Promote:** {'YES' if promote_yes else 'NO'}",
        f"- Locked tag remains `{LOCKED_TAG}`"
        + (" — WAIT: promote found, review before overwrite" if promote_yes else ""),
        f"- Still blocked on FTMO CSVs: **{'NO' if ds.startswith('ftmo') else 'YES'}** (`data/ftmo/` empty of CSVs)",
        "",
        "Artifacts: `reports/quest_consistency_overlay_board.csv`, "
        "`reports/quest_consistency_overlay_refine.md`, "
        "`configs/quest_consistency_overlay_selected.json`",
        "",
    ]
    out = ROOT / "reports" / "quest_consistency_overlay_refine.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sel = {
        "promote": promote_yes,
        "soft": soft_n,
        "hard": hard_n,
        "board_n": int(len(bdf)),
        "best_soft": best_soft_tag,
        "best_hard": hard.iloc[0]["idea"] if len(hard) else None,
        "nearest": nearest_tag,
        "best_mtd": best_mtd[1] if best_mtd else None,
        "best_month": best_mo[1] if best_mo else None,
        "best_runup": best_ru[1] if best_ru else None,
        "best_eqtarget": best_eq[1] if best_eq else None,
        "best_weights": np.round(best_w[1], 6).tolist() if best_w else None,
        "locked_tag": LOCKED_TAG,
        "locked_changed": False,
        "data_source": ds,
        "rf": RF,
        "port_vol_target": VT,
    }
    if len(soft):
        r = soft.iloc[0]
        sel["best_soft_metrics"] = {
            "mo_2024": float(r["mo_2024"]),
            "pos_2024": float(r["pos_2024"]),
            "top3_2024": float(r["top3_2024"]),
            "mo_2025_IS": float(r["mo_2025_IS"]),
            "pos_2025_IS": float(r["pos_2025_IS"]),
            "top3_2025_IS": float(r["top3_2025_IS"]),
            "mo_2025": float(r["mo_2025"]),
            "mo_2026": float(r["mo_2026"]),
            "mo_holdout": float(r["mo_holdout"]),
            "pos_holdout": float(r["pos_holdout"]),
            "top3_holdout": float(r["top3_holdout"]),
            "note": str(r["note"]),
        }
    (ROOT / "configs" / "quest_consistency_overlay_selected.json").write_text(
        json.dumps(sel, indent=2) + "\n", encoding="utf-8"
    )
    print("Wrote", out)
    print(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
