#!/usr/bin/env python3
"""Causal MTD loss-halt + after-loss throttle consistency wave on locked sleeve.

RF=0.08, VT=0.0025, overlay hi<=1. Nested picks on 2024 only; HO never for
selection. Soft mean floor ~1% on BOTH 2024 and 2025_IS (same as HO-robust).
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

from mt5_swing.portfolio.overlays import (
    apply_after_loss_throttle,
    apply_mtd_gain_clip,
    apply_mtd_loss_halt,
)

WARMUP = cor.WARMUP
RF = cor.RF
VT = cor.VT
LOCKED_TAG = cor.LOCKED_TAG
MEAN_FLOOR = 0.0095

# Frozen prior near-miss (lock_mtdclip_t0.015_a0.0) — partner without HO retune
PRIOR_MTD_NEAR = {"tau": 0.015, "after": 0.0}


def apply_ports(fn, base_ports, **kw):
    out = {}
    for lab, p in base_ports.items():
        if p is None:
            continue
        out[lab] = fn(p, **kw)
    return out


def tag_lh(params) -> str:
    return f"lh_t{params['tau']}_ah{params['after_halt']}"


def tag_al(params) -> str:
    return f"al_{params['after_loss']}"


def tag_mtd(params) -> str:
    return f"mtd_t{params['tau']}_a{params['after']}"


def pick_loss_halt_2024(port24, base_st24, top_k: int = 3):
    """Return ranked list of (score, params, st) on 2024 only."""
    grid = list(
        itertools.product(
            [0.008, 0.010, 0.012, 0.015, 0.020, 0.025, 0.030],
            [0.0, 0.25],
        )
    )
    ranked = []
    for tau, ah in grid:
        st = cor.stats_from_port(
            apply_mtd_loss_halt(port24, tau=tau, after_halt=ah, lo=0.0)
        )
        sc = cor.score_2024_pick(st, base_st24)
        if sc == float("-inf"):
            continue
        ranked.append((sc, {"tau": float(tau), "after_halt": float(ah)}, st))
    ranked.sort(key=lambda x: x[0], reverse=True)
    # Distinct by rounded params
    uniq = []
    seen = set()
    for item in ranked:
        key = (round(item[1]["tau"], 4), round(item[1]["after_halt"], 4))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
        if len(uniq) >= top_k:
            break
    return uniq


def pick_after_loss_2024(port24, base_st24):
    best = None
    for al in (0.25, 0.5, 0.75):
        st = cor.stats_from_port(
            apply_after_loss_throttle(port24, after_loss=al, lo=0.25)
        )
        sc = cor.score_2024_pick(st, base_st24)
        if best is None or sc > best[0]:
            best = (sc, {"after_loss": float(al)}, st)
    return best


def pick_mtd_on_transformed(port24_xf, base_st_xf):
    """Nested MTD gain-clip on already-transformed 2024 curve."""
    best = None
    for tau, after in itertools.product(
        [0.012, 0.015, 0.018, 0.020], [0.0, 0.25, 0.5]
    ):
        st = cor.stats_from_port(
            apply_mtd_gain_clip(port24_xf, tau=tau, after_clip=after)
        )
        sc = cor.score_2024_pick(st, base_st_xf)
        if best is None or sc > best[0]:
            best = (sc, {"tau": float(tau), "after": float(after)}, st)
    return best


def pick_loss_halt_on_transformed(port24_xf, base_st_xf):
    best = None
    for tau, ah in itertools.product(
        [0.008, 0.010, 0.012, 0.015, 0.020, 0.025, 0.030], [0.0, 0.25]
    ):
        st = cor.stats_from_port(
            apply_mtd_loss_halt(port24_xf, tau=tau, after_halt=ah, lo=0.0)
        )
        sc = cor.score_2024_pick(st, base_st_xf)
        if best is None or sc > best[0]:
            best = (sc, {"tau": float(tau), "after_halt": float(ah)}, st)
    return best


def tighten_soft_mean(bdf: pd.DataFrame) -> pd.DataFrame:
    for i, r in bdf.iterrows():
        mean_ok = (
            (r["mo_2024"] == r["mo_2024"] and r["mo_2024"] >= MEAN_FLOOR)
            and (r["mo_2025_IS"] == r["mo_2025_IS"] and r["mo_2025_IS"] >= MEAN_FLOOR)
        )
        if r["soft_pass"] and not mean_ok:
            bdf.at[i, "soft_pass"] = False
            bdf.at[i, "hard_pass"] = False
            bdf.at[i, "promote"] = False
            if r["note"] in (
                "soft_is",
                "hard_is",
                "holdout_fail",
                "year_clear_fail",
                "soft_is_hard_fail",
                "promote",
            ):
                bdf.at[i, "note"] = "soft_mean_fail"
            bdf.at[i, "score"] = float("-inf")
    return bdf


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
    all_wins = [(n, s, e) for n, s, e in is_wins] + [
        (n, s, e) for n, s, e, _ in conf_wins
    ]
    is_labels = [n for n, _, _ in is_wins]
    conf_labels = [n for n, _, _, _ in conf_wins]
    print(
        f"end={end} holdout_start={holdout_start} IS={is_labels} CONF={conf_labels}",
        flush=True,
    )

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
    board.append(
        cor.eval_idea("baseline_locked", "baseline", base_ports, is_labels, conf_labels)
    )
    print(
        f"baseline 2024 mo={board[0].mo_2024*100:.2f}% pos={board[0].pos_2024*100:.0f}% "
        f"top3={board[0].top3_2024*100:.0f}%",
        flush=True,
    )
    base24 = base_ports.get("2024")
    base_st24 = cor.stats_from_port(base24) if base24 is not None else None

    # --- mtd_loss_halt: top-3 distinct 2024 picks ---
    lh_ranked = pick_loss_halt_2024(base24, base_st24, top_k=3) if base24 is not None else []
    best_lh = lh_ranked[0] if lh_ranked else None
    for rank, item in enumerate(lh_ranked):
        params = item[1]
        print(f"loss_halt 2024 rank{rank+1}={params} sc={item[0]:.4f}", flush=True)
        ports = apply_ports(
            apply_mtd_loss_halt,
            base_ports,
            tau=params["tau"],
            after_halt=params["after_halt"],
            lo=0.0,
        )
        idea = f"lock_{tag_lh(params)}"
        if rank > 0:
            idea = f"lock_{tag_lh(params)}_r{rank+1}"
        board.append(
            cor.eval_idea(idea, "mtd_loss_halt", ports, is_labels, conf_labels)
        )

    # --- after_loss_throttle ---
    best_al = pick_after_loss_2024(base24, base_st24) if base24 is not None else None
    al_ports = None
    if best_al is not None:
        print(f"best after_loss={best_al[1]}", flush=True)
        al_ports = apply_ports(
            apply_after_loss_throttle,
            base_ports,
            after_loss=best_al[1]["after_loss"],
            lo=0.25,
        )
        board.append(
            cor.eval_idea(
                f"lock_{tag_al(best_al[1])}",
                "after_loss_throttle",
                al_ports,
                is_labels,
                conf_labels,
            )
        )

    lh_ports = None
    if best_lh is not None:
        lh_ports = apply_ports(
            apply_mtd_loss_halt,
            base_ports,
            tau=best_lh[1]["tau"],
            after_halt=best_lh[1]["after_halt"],
            lo=0.0,
        )

    # --- combo: loss_halt + after_loss ---
    if best_lh is not None and best_al is not None and base24 is not None:
        # order by 2024 score
        a = apply_after_loss_throttle(
            apply_mtd_loss_halt(
                base24,
                tau=best_lh[1]["tau"],
                after_halt=best_lh[1]["after_halt"],
            ),
            after_loss=best_al[1]["after_loss"],
            lo=0.25,
        )
        b = apply_mtd_loss_halt(
            apply_after_loss_throttle(
                base24, after_loss=best_al[1]["after_loss"], lo=0.25
            ),
            tau=best_lh[1]["tau"],
            after_halt=best_lh[1]["after_halt"],
        )
        sa, sb = cor.stats_from_port(a), cor.stats_from_port(b)
        sca = cor.score_2024_pick(sa, base_st24) if sa else float("-inf")
        scb = cor.score_2024_pick(sb, base_st24) if sb else float("-inf")
        if sca >= scb:
            order = "lh_then_al"

            def _combo_fn(p):
                mid = apply_mtd_loss_halt(
                    p,
                    tau=best_lh[1]["tau"],
                    after_halt=best_lh[1]["after_halt"],
                )
                return apply_after_loss_throttle(
                    mid, after_loss=best_al[1]["after_loss"], lo=0.25
                )

        else:
            order = "al_then_lh"

            def _combo_fn(p):
                mid = apply_after_loss_throttle(
                    p, after_loss=best_al[1]["after_loss"], lo=0.25
                )
                return apply_mtd_loss_halt(
                    mid,
                    tau=best_lh[1]["tau"],
                    after_halt=best_lh[1]["after_halt"],
                )

        cports = {lab: _combo_fn(p) for lab, p in base_ports.items() if p is not None}
        board.append(
            cor.eval_idea(
                f"lock_{tag_lh(best_lh[1])}+{tag_al(best_al[1])}_{order}",
                "combo_lh_al",
                cports,
                is_labels,
                conf_labels,
            )
        )
        print(f"combo lh+al order={order}", flush=True)

    # --- combo: loss_halt + MTD gain-clip (nested on lh-transformed 2024) ---
    best_lh_mtd = None
    if lh_ports is not None and "2024" in lh_ports:
        st_lh24 = cor.stats_from_port(lh_ports["2024"])
        best_lh_mtd = pick_mtd_on_transformed(lh_ports["2024"], st_lh24)
        if best_lh_mtd is not None:
            print(f"best lh+mtd nested={best_lh_mtd[1]}", flush=True)
            cports = {}
            for lab, p in lh_ports.items():
                cports[lab] = apply_mtd_gain_clip(
                    p, tau=best_lh_mtd[1]["tau"], after_clip=best_lh_mtd[1]["after"]
                )
            board.append(
                cor.eval_idea(
                    f"lock_{tag_lh(best_lh[1])}+{tag_mtd(best_lh_mtd[1])}",
                    "combo_lh_mtd",
                    cports,
                    is_labels,
                    conf_labels,
                )
            )
        # Also frozen prior near-miss as partner (no retune)
        cports_nm = {}
        for lab, p in lh_ports.items():
            cports_nm[lab] = apply_mtd_gain_clip(
                p, tau=PRIOR_MTD_NEAR["tau"], after_clip=PRIOR_MTD_NEAR["after"]
            )
        board.append(
            cor.eval_idea(
                f"lock_{tag_lh(best_lh[1])}+{tag_mtd(PRIOR_MTD_NEAR)}_prior",
                "combo_lh_mtd_prior",
                cports_nm,
                is_labels,
                conf_labels,
            )
        )

    # --- combo: after_loss + MTD gain-clip ---
    best_al_mtd = None
    if al_ports is not None and "2024" in al_ports:
        st_al24 = cor.stats_from_port(al_ports["2024"])
        best_al_mtd = pick_mtd_on_transformed(al_ports["2024"], st_al24)
        if best_al_mtd is not None:
            print(f"best al+mtd nested={best_al_mtd[1]}", flush=True)
            cports = {}
            for lab, p in al_ports.items():
                cports[lab] = apply_mtd_gain_clip(
                    p, tau=best_al_mtd[1]["tau"], after_clip=best_al_mtd[1]["after"]
                )
            board.append(
                cor.eval_idea(
                    f"lock_{tag_al(best_al[1])}+{tag_mtd(best_al_mtd[1])}",
                    "combo_al_mtd",
                    cports,
                    is_labels,
                    conf_labels,
                )
            )
        cports_nm = {}
        for lab, p in al_ports.items():
            cports_nm[lab] = apply_mtd_gain_clip(
                p, tau=PRIOR_MTD_NEAR["tau"], after_clip=PRIOR_MTD_NEAR["after"]
            )
        board.append(
            cor.eval_idea(
                f"lock_{tag_al(best_al[1])}+{tag_mtd(PRIOR_MTD_NEAR)}_prior",
                "combo_al_mtd_prior",
                cports_nm,
                is_labels,
                conf_labels,
            )
        )

    # --- optional: gain-clip t0.015 a0.0 first, then loss_halt nested on 2024 ---
    prior_mtd_ports = cor.ports_with_overlay(base_ports, "mtd", PRIOR_MTD_NEAR)
    board.append(
        cor.eval_idea(
            f"lock_{tag_mtd(PRIOR_MTD_NEAR)}_prior_alone",
            "mtd_gain_clip_prior",
            prior_mtd_ports,
            is_labels,
            conf_labels,
        )
    )
    best_prior_lh = None
    if "2024" in prior_mtd_ports:
        st_pm = cor.stats_from_port(prior_mtd_ports["2024"])
        best_prior_lh = pick_loss_halt_on_transformed(prior_mtd_ports["2024"], st_pm)
        if best_prior_lh is not None:
            print(f"best prior_mtd+lh nested={best_prior_lh[1]}", flush=True)
            cports = {}
            for lab, p in prior_mtd_ports.items():
                cports[lab] = apply_mtd_loss_halt(
                    p,
                    tau=best_prior_lh[1]["tau"],
                    after_halt=best_prior_lh[1]["after_halt"],
                    lo=0.0,
                )
            board.append(
                cor.eval_idea(
                    f"lock_{tag_mtd(PRIOR_MTD_NEAR)}+{tag_lh(best_prior_lh[1])}",
                    "combo_prior_mtd_lh",
                    cports,
                    is_labels,
                    conf_labels,
                )
            )

    bdf = pd.DataFrame([asdict(r) for r in board])
    bdf = tighten_soft_mean(bdf)
    board_path = ROOT / "reports" / "quest_loss_halt_board.csv"
    bdf.to_csv(board_path, index=False)
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
        "# Loss-halt / after-loss consistency (locked sleeve)",
        "",
        f"**data_source:** `{ds}`  **RF:** {RF:.0%}  **PORT_VOL_TARGET:** {VT}  **signal_lag:** 1",
        f"**Locked tag (unchanged unless promote):** `{LOCKED_TAG}`",
        f"**IS windows:** {is_labels}  **Confirm:** {conf_labels}",
        "**Selection:** nested on 2024 only; HO never for selection.",
        "**Families:** baseline / mtd_loss_halt / after_loss_throttle / "
        "combos (lh+al, lh+mtd, al+mtd) / prior mtd±lh. hi≤1.",
        f"**Soft mean floor:** {MEAN_FLOOR} on BOTH 2024 and 2025_IS.",
        "",
        "## Board",
        "",
        "| Family | N | Soft IS | Hard IS | Promote |",
        "|--------|--:|--------:|--------:|--------:|",
    ]
    for fam, g in bdf.groupby("family"):
        lines.append(
            f"| {fam} | {len(g)} | {int(g.soft_pass.sum())} | "
            f"{int(g.hard_pass.sum())} | {int(g.promote.sum())} |"
        )
    lines += [
        "",
        f"**Totals:** soft={soft_n} hard={hard_n} promote={prom_n} (board n={len(bdf)})",
        "",
    ]
    if best_lh:
        lines.append(f"**2024 loss_halt pick:** {best_lh[1]} (sc={best_lh[0]:.4f})")
    if best_al:
        lines.append(f"**2024 after_loss pick:** {best_al[1]}")
    if best_lh_mtd:
        lines.append(f"**lh+mtd nested pick:** {best_lh_mtd[1]}")
    if best_al_mtd:
        lines.append(f"**al+mtd nested pick:** {best_al_mtd[1]}")
    if best_prior_lh:
        lines.append(f"**prior_mtd+lh nested pick:** {best_prior_lh[1]}")
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
        lines.append(
            "| idea | family | min_is_mo | pos24 | pos25is | top3_max | note |"
        )
        lines.append(
            "|------|--------|----------:|------:|--------:|---------:|------|"
        )
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
        + (
            " — WAIT: promote found; prefer NOT overwrite unless every bar clears"
            if promote_yes
            else ""
        ),
        f"- Still blocked on FTMO CSVs: "
        f"**{'NO' if ds.startswith('ftmo') else 'YES'}** (`data/ftmo/` empty of CSVs)",
        "",
        "Artifacts: `reports/quest_loss_halt_board.csv`, "
        "`reports/quest_loss_halt_consistency.md`, "
        "`configs/quest_loss_halt_selected.json`",
        "",
    ]
    out = ROOT / "reports" / "quest_loss_halt_consistency.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sel = {
        "promote": promote_yes,
        "soft": soft_n,
        "hard": hard_n,
        "board_n": int(len(bdf)),
        "best_soft": best_soft_tag,
        "best_hard": hard.iloc[0]["idea"] if len(hard) else None,
        "nearest": nearest_tag,
        "best_loss_halt": best_lh[1] if best_lh else None,
        "best_after_loss": best_al[1] if best_al else None,
        "best_lh_mtd": best_lh_mtd[1] if best_lh_mtd else None,
        "best_al_mtd": best_al_mtd[1] if best_al_mtd else None,
        "best_prior_mtd_lh": best_prior_lh[1] if best_prior_lh else None,
        "prior_mtd_near": PRIOR_MTD_NEAR,
        "locked_tag": LOCKED_TAG,
        "locked_changed": False,
        "data_source": ds,
        "rf": RF,
        "port_vol_target": VT,
        "mean_floor": MEAN_FLOOR,
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
    # Prefer NOT overwriting locked yaml unless every bar clearly clears
    if promote_yes:
        r = prom.iloc[0]
        clears_all = (
            r["mo_2024"] >= 0.01
            and r["pos_2024"] >= 0.70
            and r["mo_2025"] >= 0.01
            and r["pos_2025"] >= 0.70
            and r["mo_2026"] >= 0.01
            and r["pos_2026"] >= 0.70
            and r["mo_holdout"] >= 0.01
            and r["pos_holdout"] >= 0.70
            and r["top3_2024"] <= 0.55
            and r["top3_2025_IS"] <= 0.55
            and r["top3_holdout"] <= 0.70
        )
        sel["promote_clears_every_bar"] = bool(clears_all)
        if clears_all:
            import yaml

            locked_path = ROOT / "configs" / "quest_one_pct_candidate.yaml"
            locked = yaml.safe_load(locked_path.read_text())
            locked["methodology"] = (
                (locked.get("methodology") or "")
                + f" Loss-halt promote `{r['idea']}`: see "
                f"configs/quest_loss_halt_selected.json. RF=0.08 VT=0.0025 hi<=1."
            )
            locked_path.write_text(
                yaml.safe_dump(locked, sort_keys=False), encoding="utf-8"
            )
            sel["locked_changed"] = True
            out.write_text(
                out.read_text(encoding="utf-8")
                + f"\n**PROMOTED** — updated `{locked_path.name}` with idea `{r['idea']}`.\n",
                encoding="utf-8",
            )
        else:
            out.write_text(
                out.read_text(encoding="utf-8")
                + "\n**Promote flagged but locked YAML NOT overwritten "
                "(prefer clear every-bar pass).**\n",
                encoding="utf-8",
            )

    (ROOT / "configs" / "quest_loss_halt_selected.json").write_text(
        json.dumps(sel, indent=2) + "\n", encoding="utf-8"
    )
    print("Wrote", out)
    print(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
