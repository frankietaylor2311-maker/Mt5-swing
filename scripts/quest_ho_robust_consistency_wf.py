#!/usr/bin/env python3
"""HO-robust consistency wave on locked sleeve (RF=0.08, VT=0.0025).

Families: baseline, monthly/daily budget VT, equity_tsmom, dense IS weight
rebalance, wrebal+MTD / +budget, optional mild combos. Holdout NEVER for
selection. Overlay hi<=1. Promote only soft+hard+year_clear+holdout.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location(
    "cor", ROOT / "scripts" / "quest_consistency_overlay_refine_wf.py"
)
cor = importlib.util.module_from_spec(spec)
sys.modules["cor"] = cor
assert spec.loader is not None
spec.loader.exec_module(cor)

from mt5_swing.portfolio.equity_tsmom import (
    apply_daily_budget_vt,
    apply_equity_tsmom,
    apply_monthly_budget_vt,
)
from mt5_swing.portfolio.overlays import apply_mtd_gain_clip

WARMUP = cor.WARMUP
RF = cor.RF
VT = cor.VT
LOCKED_TAG = cor.LOCKED_TAG
MIN_PCT_POS = cor.MIN_PCT_POS
MAX_TOP3_HARD = cor.MAX_TOP3_HARD
MAX_TOP3_SOFT = cor.MAX_TOP3_SOFT
MAX_P2T = cor.MAX_P2T


def dense_simplex(base_w: np.ndarray, n_target: int = 100) -> list[np.ndarray]:
    """≥80 unique weight vectors: equal, corners, pairwise, Dirichlet."""
    n = len(base_w)
    cands = [base_w.copy(), np.ones(n) / n]
    # Corner-ish tilts
    for i in range(n):
        for s in (0.5, 0.65, 0.8, 1.2, 1.4, 1.7, 2.0):
            w = base_w.copy()
            w[i] *= s
            w = np.maximum(w, 1e-6)
            cands.append(w / w.sum())
        # near-corner: boost one, shrink rest
        w = np.full(n, 0.08)
        w[i] = 1.0 - 0.08 * (n - 1)
        cands.append(w / w.sum())
    for i, j in itertools.combinations(range(n), 2):
        for s in (0.6, 0.75, 0.9, 1.1, 1.25, 1.5):
            w = base_w.copy()
            w[i] *= s
            w[j] *= 2.0 - min(s, 1.5)
            w = np.maximum(w, 1e-6)
            cands.append(w / w.sum())
    rng = np.random.default_rng(7)
    for conc in (20.0, 40.0, 60.0, 100.0, 160.0):
        alpha = np.maximum(base_w * conc, 0.4)
        for _ in range(18):
            cands.append(rng.dirichlet(alpha))
    # flat Dirichlet
    for _ in range(12):
        cands.append(rng.dirichlet(np.ones(n) * 2.0))
    uniq = []
    seen = set()
    for w in cands:
        key = tuple(np.round(w, 4))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(w)
        if len(uniq) >= max(n_target, 80):
            break
    # ensure we hit 80 even if rounding collided early
    while len(uniq) < 80:
        w = rng.dirichlet(np.maximum(base_w * 50.0, 0.5))
        key = tuple(np.round(w, 4))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(w)
    return uniq


def score_2024_budget(st, base_st) -> float:
    """2024-only pick: prefer consistency then mean near 1%."""
    if st is None or not st.gates:
        return float("-inf")
    floor = 0.0035
    if base_st is not None:
        floor = max(floor, base_st.mean_mo * 0.80)
    if st.mean_mo < floor:
        return float("-inf")
    # Prefer ~1% mean, high %pos, low top3
    mean_pen = abs(st.mean_mo - 0.01) * 0.5
    bonus_pos = 0.10 if st.pct_pos >= 0.70 else (0.04 if st.pct_pos >= 0.60 else 0.0)
    return -st.top3 + 0.12 * st.mean_mo + bonus_pos - mean_pen


def score_2024_mtd_prefer_mild(st, base_st, after: float) -> float:
    sc = cor.score_2024_pick(st, base_st)
    if sc == float("-inf"):
        return sc
    # Prefer higher after (less aggressive) when scores close — small tie-break
    return sc + 0.003 * float(after)


def apply_ports(fn, base_ports, **kw):
    out = {}
    for lab, p in base_ports.items():
        if p is None:
            continue
        out[lab] = fn(p, **kw)
    return out


def pick_monthly_budget(port24, base_st24):
    grid = list(
        itertools.product([0.008, 0.01, 0.012, 0.015], [4, 6, 9], [0.8, 1.0, 1.2])
    )
    best = None
    for tm, lb, sh in grid:
        st = cor.stats_from_port(
            apply_monthly_budget_vt(
                port24, target_mo=tm, lookback_months=lb, assumed_sharpe=sh, hi=1.0
            )
        )
        sc = score_2024_budget(st, base_st24)
        if best is None or sc > best[0]:
            best = (sc, {"target_mo": tm, "lookback_months": lb, "assumed_sharpe": sh}, st)
    return best


def pick_daily_budget(port24, base_st24):
    grid = list(
        itertools.product([0.008, 0.01, 0.012, 0.015], [0.8, 1.0, 1.2], [42, 60, 90])
    )
    best = None
    for tm, sh, look in grid:
        st = cor.stats_from_port(
            apply_daily_budget_vt(
                port24, target_mo=tm, assumed_sharpe=sh, look=look, hi=1.0
            )
        )
        sc = score_2024_budget(st, base_st24)
        if best is None or sc > best[0]:
            best = (sc, {"target_mo": tm, "assumed_sharpe": sh, "look": look}, st)
    return best


def pick_equity_tsmom(port24, base_st24):
    # Mild params only; hi=1
    grid = []
    for lb in (42, 60, 90, 120):
        for neg in (0.20, 0.25, 0.35, 0.5):
            for pos in (0.85, 1.0):
                for band in (0.0, 0.01):
                    grid.append(
                        {
                            "lookback": lb,
                            "neg_scale": neg,
                            "pos_scale": pos,
                            "flat_band": band,
                        }
                    )
    best = None
    for params in grid:
        st = cor.stats_from_port(
            apply_equity_tsmom(port24, hi=1.0, lo=0.0, **params)
        )
        sc = score_2024_budget(st, base_st24)
        if best is None or sc > best[0]:
            best = (sc, params, st)
    return best


def tag_mb(p):
    return f"mb_tm{p['target_mo']}_lb{p['lookback_months']}_sh{p['assumed_sharpe']}"


def tag_db(p):
    return f"db_tm{p['target_mo']}_sh{p['assumed_sharpe']}_lk{p['look']}"


def tag_ts(p):
    return (
        f"ts_lb{p['lookback']}_neg{p['neg_scale']}_pos{p['pos_scale']}"
        f"_fb{p['flat_band']}"
    )


def main() -> None:
    from mt5_swing.config import load_config
    from mt5_swing.data.loader import load_ohlc_csv

    cfg = load_config(ROOT / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    ds = cor.data_source()
    print(f"data_source={ds} RF={RF} VT={VT} tag_locked={LOCKED_TAG}", flush=True)
    assert abs(RF - 0.08) < 1e-12

    legs = cor.locked_legs()
    base_w = np.array([l["weight"] for l in legs], dtype=float)
    path0 = cor.ewc.resolve_csv(legs[0]["symbol"], legs[0]["timeframe"])
    ohlc0 = load_ohlc_csv(path0, symbol=legs[0]["symbol"], timeframe=legs[0]["timeframe"])
    end = ohlc0.index.max()
    holdout_start = end - pd.Timedelta(days=365)
    is_wins = cor.is_windows(end, holdout_start)
    conf_wins = cor.confirm_windows(end, holdout_start)
    all_wins = [(n, s, e) for n, s, e in is_wins] + [(n, s, e) for n, s, e, _ in conf_wins]
    is_labels = [n for n, _, _ in is_wins]
    conf_labels = [n for n, _, _, _ in conf_wins]
    print(f"end={end} holdout_start={holdout_start} IS={is_labels} CONF={conf_labels}", flush=True)

    leg_curves = {}
    for lab, s, e in all_wins:
        curves = []
        ok = True
        for leg in legs:
            eq = cor.run_leg_curve(leg, cfg, s, e)
            if eq is None:
                ok = False
                break
            curves.append(eq.rename(f"{leg['symbol']}_{leg['strategy']}"))
        if ok:
            leg_curves[lab] = curves
            print(f"  legs ready {lab}: {[len(c) for c in curves]}", flush=True)

    base_ports = {
        lab: cor.port_from_leg_curves(curves, base_w, apply_vt=True)
        for lab, curves in leg_curves.items()
    }
    base_ports = {k: v for k, v in base_ports.items() if v is not None}

    board = []
    board.append(cor.eval_idea("baseline_locked", "baseline", base_ports, is_labels, conf_labels))
    print(
        f"baseline 2024 mo={board[0].mo_2024*100:.2f}% pos={board[0].pos_2024*100:.0f}% "
        f"top3={board[0].top3_2024*100:.0f}%",
        flush=True,
    )
    base24 = base_ports.get("2024")
    base_st24 = cor.stats_from_port(base24) if base24 is not None else None

    # --- monthly_budget_vt ---
    best_mb = pick_monthly_budget(base24, base_st24) if base24 is not None else None
    if best_mb is not None:
        print(f"best 2024 monthly_budget={best_mb[1]}", flush=True)
        ports = apply_ports(apply_monthly_budget_vt, base_ports, hi=1.0, **best_mb[1])
        board.append(
            cor.eval_idea(f"lock_{tag_mb(best_mb[1])}", "monthly_budget_vt", ports, is_labels, conf_labels)
        )

    # --- daily_budget_vt ---
    best_db = pick_daily_budget(base24, base_st24) if base24 is not None else None
    if best_db is not None:
        print(f"best 2024 daily_budget={best_db[1]}", flush=True)
        ports = apply_ports(apply_daily_budget_vt, base_ports, hi=1.0, **best_db[1])
        board.append(
            cor.eval_idea(f"lock_{tag_db(best_db[1])}", "daily_budget_vt", ports, is_labels, conf_labels)
        )

    # --- equity_tsmom ---
    best_ts = pick_equity_tsmom(base24, base_st24) if base24 is not None else None
    if best_ts is not None:
        print(f"best 2024 equity_tsmom={best_ts[1]}", flush=True)
        ports = apply_ports(apply_equity_tsmom, base_ports, hi=1.0, lo=0.0, **best_ts[1])
        board.append(
            cor.eval_idea(f"lock_{tag_ts(best_ts[1])}", "equity_tsmom", ports, is_labels, conf_labels)
        )

    # --- dense IS weight rebalance (merge prior simplex + dense + GBPUSD-shrink) ---
    weight_cands = cor.simplex_perturbations(
        base_w, scales=[0.6, 0.75, 0.9, 1.1, 1.25, 1.5], n_dirichlet=36
    )
    weight_cands.extend(dense_simplex(base_w, n_target=100))
    # Targeted tilts: shrink GBPUSD (idx 1), boost others — prior soft region
    rng = np.random.default_rng(11)
    for gbpusd_w in (0.02, 0.03, 0.04, 0.05, 0.06):
        for _ in range(8):
            rest = rng.dirichlet(np.array([3.0, 4.0, 3.5, 4.5]))  # USDCHF CADJPY AUDCAD GBPCAD
            w = np.zeros(5)
            w[1] = gbpusd_w
            w[[0, 2, 3, 4]] = rest * (1.0 - gbpusd_w)
            weight_cands.append(w)
    # Dedup
    uniq, seen = [], set()
    for w in weight_cands:
        key = tuple(np.round(w, 4))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(w)
    weight_cands = uniq
    print(f"weight candidates={len(weight_cands)}", flush=True)
    assert len(weight_cands) >= 80, len(weight_cands)
    best_w = None
    curves24 = leg_curves.get("2024")
    curves25 = leg_curves.get("2025_IS")
    if curves24 is not None:
        for w in weight_cands:
            p24 = cor.port_from_leg_curves(curves24, w, apply_vt=True)
            st24 = cor.stats_from_port(p24)
            if st24 is None or not st24.gates:
                continue
            if curves25 is None:
                continue
            p25 = cor.port_from_leg_curves(curves25, w, apply_vt=True)
            st25 = cor.stats_from_port(p25)
            if st25 is None or not st25.gates:
                continue
            # Require soft on BOTH IS windows
            if st24.pct_pos < MIN_PCT_POS or st25.pct_pos < MIN_PCT_POS:
                continue
            if st24.top3 > MAX_TOP3_SOFT or st25.top3 > MAX_TOP3_SOFT:
                continue
            if st24.p2t > MAX_P2T or st25.p2t > MAX_P2T:
                continue
            min_mo = min(st24.mean_mo, st25.mean_mo)
            min_pos = min(st24.pct_pos, st25.pct_pos)
            max_t3 = max(st24.top3, st25.top3)
            # Primary: max min(mean_mo); secondary: higher min(%pos); tertiary: lower top3
            sc = (min_mo, min_pos, -max_t3)
            if best_w is None or sc > best_w[0]:
                best_w = (sc, w.copy(), st24, st25)
        print(
            f"best weight sc={best_w[0] if best_w else None} "
            f"w={np.round(best_w[1], 4).tolist() if best_w else None}",
            flush=True,
        )

    wrebal_ports = None
    wtag = None
    if best_w is not None:
        w = best_w[1]
        wrebal_ports = {
            lab: cor.port_from_leg_curves(curves, w, apply_vt=True)
            for lab, curves in leg_curves.items()
        }
        wrebal_ports = {k: v for k, v in wrebal_ports.items() if v is not None}
        wtag = "wrebal_" + "_".join(f"{x:.3f}" for x in w)
        board.append(cor.eval_idea(wtag, "weight_rebalance", wrebal_ports, is_labels, conf_labels))

    # --- wrebal + mild MTD (2024 nested; prefer higher after) ---
    best_wm = None
    if wrebal_ports is not None and "2024" in wrebal_ports:
        st_base_w = cor.stats_from_port(wrebal_ports["2024"])
        for tau, after in itertools.product(
            [0.012, 0.015, 0.018, 0.02, 0.025], [0.0, 0.25, 0.5, 0.7]
        ):
            st = cor.stats_from_port(
                apply_mtd_gain_clip(wrebal_ports["2024"], tau=tau, after_clip=after)
            )
            if st is None:
                continue
            sc = score_2024_mtd_prefer_mild(st, st_base_w, after)
            if best_wm is None or sc > best_wm[0]:
                best_wm = (sc, {"tau": tau, "after": after}, st)
        if best_wm is not None:
            print(f"best wrebal+MTD={best_wm[1]}", flush=True)
            cports = cor.ports_with_overlay(wrebal_ports, "mtd", best_wm[1])
            board.append(
                cor.eval_idea(
                    f"{wtag}+{cor.tag_overlay('mtd', best_wm[1])}",
                    "wrebal_plus_mtd",
                    cports,
                    is_labels,
                    conf_labels,
                )
            )

    # --- wrebal + monthly_budget_vt ---
    best_wmb = None
    if wrebal_ports is not None and "2024" in wrebal_ports:
        st_base_w = cor.stats_from_port(wrebal_ports["2024"])
        best_wmb = pick_monthly_budget(wrebal_ports["2024"], st_base_w)
        if best_wmb is not None:
            print(f"best wrebal+monthly_budget={best_wmb[1]}", flush=True)
            cports = apply_ports(
                apply_monthly_budget_vt, wrebal_ports, hi=1.0, **best_wmb[1]
            )
            board.append(
                cor.eval_idea(
                    f"{wtag}+{tag_mb(best_wmb[1])}",
                    "wrebal_plus_monthly_budget",
                    cports,
                    is_labels,
                    conf_labels,
                )
            )

    # --- wrebal + daily_budget_vt ---
    best_wdb = None
    if wrebal_ports is not None and "2024" in wrebal_ports:
        st_base_w = cor.stats_from_port(wrebal_ports["2024"])
        best_wdb = pick_daily_budget(wrebal_ports["2024"], st_base_w)
        if best_wdb is not None:
            print(f"best wrebal+daily_budget={best_wdb[1]}", flush=True)
            cports = apply_ports(
                apply_daily_budget_vt, wrebal_ports, hi=1.0, **best_wdb[1]
            )
            board.append(
                cor.eval_idea(
                    f"{wtag}+{tag_db(best_wdb[1])}",
                    "wrebal_plus_daily_budget",
                    cports,
                    is_labels,
                    conf_labels,
                )
            )

    # --- optional combo ≤2 of (mtd, monthly_budget, equity_tsmom) on best wrebal ---
    if wrebal_ports is not None and "2024" in wrebal_ports:
        pieces = []
        if best_wm is not None:
            pieces.append(
                (
                    "mtd",
                    best_wm[1],
                    lambda p, kw=best_wm[1]: apply_mtd_gain_clip(
                        p, tau=kw["tau"], after_clip=kw["after"]
                    ),
                    cor.tag_overlay("mtd", best_wm[1]),
                )
            )
        if best_wmb is not None:
            pieces.append(
                (
                    "mb",
                    best_wmb[1],
                    lambda p, kw=best_wmb[1]: apply_monthly_budget_vt(p, hi=1.0, **kw),
                    tag_mb(best_wmb[1]),
                )
            )
        if best_ts is not None:
            pieces.append(
                (
                    "ts",
                    best_ts[1],
                    lambda p, kw=best_ts[1]: apply_equity_tsmom(p, hi=1.0, lo=0.0, **kw),
                    tag_ts(best_ts[1]),
                )
            )
        st_base_w = cor.stats_from_port(wrebal_ports["2024"])
        for (n1, p1, f1, t1), (n2, p2, f2, t2) in itertools.combinations(pieces, 2):
            # order by 2024 score
            a = f2(f1(wrebal_ports["2024"]))
            b = f1(f2(wrebal_ports["2024"]))
            sa, sb = cor.stats_from_port(a), cor.stats_from_port(b)
            sca = score_2024_budget(sa, st_base_w) if sa else float("-inf")
            scb = score_2024_budget(sb, st_base_w) if sb else float("-inf")
            if sca >= scb and sca > float("-inf"):
                order = [(f1, t1), (f2, t2)]
            elif scb > float("-inf"):
                order = [(f2, t2), (f1, t1)]
            else:
                continue
            cports = {}
            for lab, p in wrebal_ports.items():
                mid = order[0][0](p)
                cports[lab] = order[1][0](mid)
            tag = f"{wtag}+{order[0][1]}+{order[1][1]}"
            board.append(
                cor.eval_idea(tag, "wrebal_combo_two", cports, is_labels, conf_labels)
            )
        print(f"combo adds done (pieces={len(pieces)})", flush=True)

    bdf = pd.DataFrame([asdict(r) for r in board])
    # Tighten soft to promote Soft-IS rule: mean_mo >= ~1% on BOTH IS windows
    MEAN_FLOOR = 0.0095
    for i, r in bdf.iterrows():
        mean_ok = (
            (r["mo_2024"] == r["mo_2024"] and r["mo_2024"] >= MEAN_FLOOR)
            and (r["mo_2025_IS"] == r["mo_2025_IS"] and r["mo_2025_IS"] >= MEAN_FLOOR)
        )
        if r["soft_pass"] and not mean_ok:
            bdf.at[i, "soft_pass"] = False
            bdf.at[i, "hard_pass"] = False
            bdf.at[i, "promote"] = False
            if r["note"] in ("soft_is", "hard_is", "holdout_fail", "year_clear_fail", "soft_is_hard_fail", "promote"):
                bdf.at[i, "note"] = "soft_mean_fail"
            bdf.at[i, "score"] = float("-inf")
    bdf.to_csv(ROOT / "reports" / "quest_ho_robust_board.csv", index=False)
    soft_n = int(bdf["soft_pass"].sum())
    hard_n = int(bdf["hard_pass"].sum())
    prom_n = int(bdf["promote"].sum())
    print(f"BOARD soft={soft_n} hard={hard_n} promote={prom_n} n={len(bdf)}", flush=True)

    soft = bdf[bdf["soft_pass"]].sort_values(
        ["score", "min_is_pct_pos"], ascending=False
    )
    hard = bdf[bdf["hard_pass"]].sort_values("score", ascending=False)
    prom = bdf[bdf["promote"]]
    nearest = (
        bdf.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["min_is_mo"])
        .sort_values("min_is_mo", ascending=False)
    )

    lines = [
        "# HO-robust consistency (locked sleeve)",
        "",
        f"**data_source:** `{ds}`  **RF:** {RF:.0%}  **PORT_VOL_TARGET:** {VT}  **signal_lag:** 1",
        f"**Locked tag (unchanged unless promote):** `{LOCKED_TAG}`",
        f"**IS windows:** {is_labels}  **Confirm:** {conf_labels}",
        "**Selection:** nested on 2024 (weights: max min(2024,2025_IS)); HO never for selection.",
        "**Families:** baseline / monthly_budget_vt / daily_budget_vt / equity_tsmom / "
        "dense wrebal / wrebal+mtd / wrebal+budget / optional combos. hi≤1.",
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
    if best_mb:
        lines.append(f"**2024 monthly_budget pick:** {best_mb[1]}")
    if best_db:
        lines.append(f"**2024 daily_budget pick:** {best_db[1]}")
    if best_ts:
        lines.append(f"**2024 equity_tsmom pick:** {best_ts[1]}")
    if best_w:
        lines.append(
            f"**IS weight rebalance:** {np.round(best_w[1], 4).tolist()} "
            f"score={best_w[0]}"
        )
    if best_wm:
        lines.append(f"**wrebal+MTD pick:** {best_wm[1]}")
    if best_wmb:
        lines.append(f"**wrebal+monthly_budget pick:** {best_wmb[1]}")
    if best_wdb:
        lines.append(f"**wrebal+daily_budget pick:** {best_wdb[1]}")
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
    if len(hard):
        lines.append("**Best hard IS (not necessarily promote):**")
        lines.append("")
        lines.extend(fmt_row(hard.iloc[0]))
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
        "Artifacts: `reports/quest_ho_robust_board.csv`, "
        "`reports/quest_ho_robust_consistency.md`, "
        "`configs/quest_ho_robust_selected.json`",
        "",
    ]
    out = ROOT / "reports" / "quest_ho_robust_consistency.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sel = {
        "promote": promote_yes,
        "soft": soft_n,
        "hard": hard_n,
        "board_n": int(len(bdf)),
        "best_soft": best_soft_tag,
        "best_hard": hard.iloc[0]["idea"] if len(hard) else None,
        "nearest": nearest_tag,
        "best_monthly_budget": best_mb[1] if best_mb else None,
        "best_daily_budget": best_db[1] if best_db else None,
        "best_equity_tsmom": best_ts[1] if best_ts else None,
        "best_weights": np.round(best_w[1], 6).tolist() if best_w else None,
        "best_wrebal_mtd": best_wm[1] if best_wm else None,
        "best_wrebal_monthly_budget": best_wmb[1] if best_wmb else None,
        "best_wrebal_daily_budget": best_wdb[1] if best_wdb else None,
        "locked_tag": LOCKED_TAG,
        "locked_changed": False,
        "data_source": ds,
        "rf": RF,
        "port_vol_target": VT,
        "weight_candidates": len(weight_cands),
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
            "pos_2025": float(r["pos_2025"]),
            "top3_2025": float(r["top3_2025"]),
            "mo_2026": float(r["mo_2026"]),
            "pos_2026": float(r["pos_2026"]),
            "top3_2026": float(r["top3_2026"]),
            "mo_holdout": float(r["mo_holdout"]),
            "pos_holdout": float(r["pos_holdout"]),
            "top3_holdout": float(r["top3_holdout"]),
            "note": str(r["note"]),
        }
    if promote_yes:
        # Update locked config carefully documenting overlay
        r = prom.iloc[0]
        locked_path = ROOT / "configs" / "quest_one_pct_candidate.yaml"
        import yaml

        locked = yaml.safe_load(locked_path.read_text())
        locked["methodology"] = (
            (locked.get("methodology") or "")
            + f" HO-robust promote `{r['idea']}`: overlay documented in "
            f"configs/quest_ho_robust_selected.json. RF=0.08 VT=0.0025 hi<=1."
        )
        if best_w is not None and "wrebal" in str(r["idea"]):
            for i, c in enumerate(locked["candidates"]):
                c["weight"] = float(best_w[1][i])
        locked_path.write_text(yaml.safe_dump(locked, sort_keys=False), encoding="utf-8")
        sel["locked_changed"] = True
        lines_note = f"\n**PROMOTED** — updated `{locked_path.name}` with idea `{r['idea']}`.\n"
        out.write_text(out.read_text(encoding="utf-8") + lines_note, encoding="utf-8")

    (ROOT / "configs" / "quest_ho_robust_selected.json").write_text(
        json.dumps(sel, indent=2) + "\n", encoding="utf-8"
    )
    print("Wrote", out)
    print(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
