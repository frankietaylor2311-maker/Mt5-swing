#!/usr/bin/env python3
"""Trailing gain-concentration dampen consistency wave on locked sleeve.

RF=0.08, VT=0.0025, overlay hi<=1. Nested picks on 2024 only; HO never for
selection. Soft mean floor 0.0095 on BOTH 2024 and 2025_IS.
Optionally freezes HO-robust wrebal weights from quest_ho_robust_selected.json.
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
    apply_mtd_gain_clip,
    apply_trailing_gain_concentration_dampen,
)

WARMUP = cor.WARMUP
RF = cor.RF
VT = cor.VT
LOCKED_TAG = cor.LOCKED_TAG
MEAN_FLOOR = 0.0095

# Frozen HO-robust best weights (same leg order as locked yaml)
FROZEN_WREBAL = np.array(
    [0.328198, 0.029051, 0.2413, 0.177605, 0.223846], dtype=float
)
PRIOR_MTD = {"tau": 0.015, "after": 0.0}
WREBAL_MTD_NEARS = [
    {"tau": 0.015, "after": 0.5},
    {"tau": 0.015, "after": 0.65},
]


def apply_ports(fn, base_ports, **kw):
    out = {}
    for lab, p in base_ports.items():
        if p is None:
            continue
        out[lab] = fn(p, **kw)
    return out


def tag_conc(params) -> str:
    return (
        f"conc_lb{params['lookback_months']}_th{params['thresh']}"
        f"_cs{params['cool_scale']}_k{params.get('k', 3)}"
    )


def tag_mtd(params) -> str:
    return f"mtd_t{params['tau']}_a{params['after']}"


def tag_w(w: np.ndarray) -> str:
    return "wrebal_" + "_".join(f"{x:.3f}" for x in w)


def pick_conc_2024(port24, base_st24, top_k: int = 1):
    """Nested pick on 2024 only over modest conc_dampen grid."""
    # Primary grid + mild extended thresh: 2024 sleeve top3-of-pos sits ~0.88–1.0,
    # so thresh<=0.70 always-fires; 0.80/0.85/0.90 allow selective dampen.
    grid = list(
        itertools.product(
            [4, 6, 8],
            [0.55, 0.60, 0.65, 0.70, 0.80, 0.85, 0.90],
            [0.35, 0.5, 0.65],
        )
    )
    ranked = []
    for lb, th, cs in grid:
        params = {
            "lookback_months": int(lb),
            "thresh": float(th),
            "cool_scale": float(cs),
            "k": 3,
        }
        st = cor.stats_from_port(
            apply_trailing_gain_concentration_dampen(
                port24,
                lookback_months=params["lookback_months"],
                thresh=params["thresh"],
                cool_scale=params["cool_scale"],
                k=params["k"],
                lo=0.25,
            )
        )
        sc = cor.score_2024_pick(st, base_st24)
        if sc == float("-inf"):
            continue
        ranked.append((sc, params, st))
    ranked.sort(key=lambda x: x[0], reverse=True)
    uniq = []
    seen = set()
    for item in ranked:
        key = (
            item[1]["lookback_months"],
            round(item[1]["thresh"], 4),
            round(item[1]["cool_scale"], 4),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
        if len(uniq) >= top_k:
            break
    return uniq


def pick_conc_on_transformed(port24_xf, base_st_xf):
    """Single best conc_dampen on already-transformed 2024 curve."""
    best = None
    for lb, th, cs in itertools.product(
        [4, 6, 8], [0.55, 0.60, 0.65, 0.70, 0.80, 0.85, 0.90], [0.35, 0.5, 0.65]
    ):
        params = {
            "lookback_months": int(lb),
            "thresh": float(th),
            "cool_scale": float(cs),
            "k": 3,
        }
        st = cor.stats_from_port(
            apply_trailing_gain_concentration_dampen(
                port24_xf,
                lookback_months=params["lookback_months"],
                thresh=params["thresh"],
                cool_scale=params["cool_scale"],
                k=params["k"],
                lo=0.25,
            )
        )
        sc = cor.score_2024_pick(st, base_st_xf)
        if best is None or sc > best[0]:
            best = (sc, params, st)
    return best


def pick_mild_mtd_on_transformed(port24_xf, base_st_xf):
    """Mild MTD only (for wrebal+mild_mtd+conc nesting)."""
    best = None
    for tau, after in itertools.product([0.015, 0.018], [0.5, 0.65]):
        params = {"tau": float(tau), "after": float(after)}
        st = cor.stats_from_port(
            apply_mtd_gain_clip(port24_xf, tau=params["tau"], after_clip=params["after"])
        )
        sc = cor.score_2024_pick(st, base_st_xf)
        if best is None or sc > best[0]:
            best = (sc, params, st)
    return best


def tighten_soft_mean(bdf: pd.DataFrame) -> pd.DataFrame:
    for i, r in bdf.iterrows():
        mean_ok = (r["mo_2024"] == r["mo_2024"] and r["mo_2024"] >= MEAN_FLOOR) and (
            r["mo_2025_IS"] == r["mo_2025_IS"] and r["mo_2025_IS"] >= MEAN_FLOOR
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
    # Prefer JSON frozen weights if present (sanity)
    ho_json = ROOT / "configs" / "quest_ho_robust_selected.json"
    frozen_w = FROZEN_WREBAL.copy()
    if ho_json.exists():
        try:
            jw = json.loads(ho_json.read_text()).get("best_weights")
            if jw and len(jw) == len(legs):
                frozen_w = np.asarray(jw, dtype=float)
                frozen_w = frozen_w / frozen_w.sum()
        except Exception:
            pass
    assert len(frozen_w) == len(legs)

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

    wrebal_ports = {
        lab: cor.port_from_leg_curves(curves, frozen_w, apply_vt=True)
        for lab, curves in leg_curves.items()
    }
    wrebal_ports = {k: v for k, v in wrebal_ports.items() if v is not None}

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

    # --- conc_dampen alone: top-3 2024 picks ---
    conc_ranked = pick_conc_2024(base24, base_st24, top_k=3) if base24 is not None else []
    best_conc = conc_ranked[0] if conc_ranked else None
    if not conc_ranked and base24 is not None:
        # Diagnostic: all primary/extended picks failed score_2024 floor — still
        # surface best gated candidate by raw consistency proxy for the board.
        fallback = []
        for lb, th, cs in itertools.product(
            [4, 6, 8], [0.55, 0.60, 0.65, 0.70, 0.80, 0.85, 0.90], [0.35, 0.5, 0.65]
        ):
            params = {
                "lookback_months": int(lb),
                "thresh": float(th),
                "cool_scale": float(cs),
                "k": 3,
            }
            st = cor.stats_from_port(
                apply_trailing_gain_concentration_dampen(
                    base24,
                    lookback_months=params["lookback_months"],
                    thresh=params["thresh"],
                    cool_scale=params["cool_scale"],
                    k=3,
                    lo=0.25,
                )
            )
            if st is None or not st.gates:
                continue
            proxy = -st.top3 + 0.15 * st.mean_mo + (0.08 if st.pct_pos >= 0.70 else 0.0)
            fallback.append((proxy, params, st))
        fallback.sort(key=lambda x: x[0], reverse=True)
        if fallback:
            conc_ranked = fallback[:1]
            best_conc = conc_ranked[0]
            print(f"conc fallback (score_2024 all -inf): {best_conc[1]}", flush=True)
    for rank, item in enumerate(conc_ranked):
        params = item[1]
        print(f"conc 2024 rank{rank+1}={params} sc={item[0]:.4f}", flush=True)
        ports = apply_ports(
            apply_trailing_gain_concentration_dampen,
            base_ports,
            lookback_months=params["lookback_months"],
            thresh=params["thresh"],
            cool_scale=params["cool_scale"],
            k=params["k"],
            lo=0.25,
        )
        idea = f"lock_{tag_conc(params)}"
        if rank > 0:
            idea = f"{idea}_r{rank+1}"
        board.append(
            cor.eval_idea(idea, "conc_dampen", ports, is_labels, conf_labels)
        )

    # --- frozen wrebal alone ---
    wtag = tag_w(frozen_w)
    board.append(
        cor.eval_idea(f"{wtag}_alone", "frozen_wrebal", wrebal_ports, is_labels, conf_labels)
    )
    print(f"frozen wrebal={frozen_w.tolist()}", flush=True)
    w24 = wrebal_ports.get("2024")
    w_st24 = cor.stats_from_port(w24) if w24 is not None else None

    # --- frozen_wrebal + conc_dampen (2024 nested) ---
    best_w_conc = None
    if w24 is not None and w_st24 is not None:
        best_w_conc = pick_conc_on_transformed(w24, w_st24)
        if best_w_conc is not None and best_w_conc[0] != float("-inf"):
            print(f"best wrebal+conc={best_w_conc[1]} sc={best_w_conc[0]:.4f}", flush=True)
            ports = apply_ports(
                apply_trailing_gain_concentration_dampen,
                wrebal_ports,
                lookback_months=best_w_conc[1]["lookback_months"],
                thresh=best_w_conc[1]["thresh"],
                cool_scale=best_w_conc[1]["cool_scale"],
                k=best_w_conc[1]["k"],
                lo=0.25,
            )
            board.append(
                cor.eval_idea(
                    f"{wtag}+{tag_conc(best_w_conc[1])}",
                    "frozen_wrebal_conc",
                    ports,
                    is_labels,
                    conf_labels,
                )
            )

    # --- frozen_wrebal + prior near-miss MTD τ=0.015 after∈{0.5,0.65} ---
    for mtd_p in WREBAL_MTD_NEARS:
        ports = apply_ports(
            apply_mtd_gain_clip,
            wrebal_ports,
            tau=mtd_p["tau"],
            after_clip=mtd_p["after"],
        )
        board.append(
            cor.eval_idea(
                f"{wtag}+{tag_mtd(mtd_p)}_frozen",
                "frozen_wrebal_mtd",
                ports,
                is_labels,
                conf_labels,
            )
        )

    # --- frozen_wrebal + mild MTD + conc_dampen (≤2 overlays after wrebal) ---
    best_w_mtd_conc = None
    if w24 is not None and w_st24 is not None:
        # Nest: pick mild MTD on wrebal 2024, then conc on that transform
        mild = pick_mild_mtd_on_transformed(w24, w_st24)
        if mild is not None:
            print(f"wrebal mild_mtd pick={mild[1]}", flush=True)
            mtd_ports = apply_ports(
                apply_mtd_gain_clip,
                wrebal_ports,
                tau=mild[1]["tau"],
                after_clip=mild[1]["after"],
            )
            st_m = cor.stats_from_port(mtd_ports["2024"]) if "2024" in mtd_ports else None
            if st_m is not None:
                best_w_mtd_conc = pick_conc_on_transformed(mtd_ports["2024"], st_m)
                if best_w_mtd_conc is not None:
                    print(
                        f"wrebal+mtd+conc nested={best_w_mtd_conc[1]} "
                        f"sc={best_w_mtd_conc[0]:.4f}",
                        flush=True,
                    )
                    cports = apply_ports(
                        apply_trailing_gain_concentration_dampen,
                        mtd_ports,
                        lookback_months=best_w_mtd_conc[1]["lookback_months"],
                        thresh=best_w_mtd_conc[1]["thresh"],
                        cool_scale=best_w_mtd_conc[1]["cool_scale"],
                        k=best_w_mtd_conc[1]["k"],
                        lo=0.25,
                    )
                    board.append(
                        cor.eval_idea(
                            f"{wtag}+{tag_mtd(mild[1])}+{tag_conc(best_w_mtd_conc[1])}",
                            "frozen_wrebal_mtd_conc",
                            cports,
                            is_labels,
                            conf_labels,
                        )
                    )
            # Also keep mild MTD alone on wrebal (if not duplicate of frozen nears)
            if mild[1] not in WREBAL_MTD_NEARS:
                board.append(
                    cor.eval_idea(
                        f"{wtag}+{tag_mtd(mild[1])}_mild",
                        "frozen_wrebal_mtd_mild",
                        mtd_ports,
                        is_labels,
                        conf_labels,
                    )
                )

    # --- optional: baseline + prior MTD τ=0.015 a=0.0 alone ---
    prior_ports = cor.ports_with_overlay(base_ports, "mtd", PRIOR_MTD)
    board.append(
        cor.eval_idea(
            f"lock_{tag_mtd(PRIOR_MTD)}_prior_alone",
            "mtd_gain_clip_prior",
            prior_ports,
            is_labels,
            conf_labels,
        )
    )

    # Cap board if somehow huge (should be ~10–12)
    if len(board) > 25:
        board = board[:25]

    bdf = pd.DataFrame([asdict(r) for r in board])
    bdf = tighten_soft_mean(bdf)
    board_path = ROOT / "reports" / "quest_conc_dampen_board.csv"
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
        "# Trailing gain-concentration dampen consistency (locked sleeve)",
        "",
        f"**data_source:** `{ds}`  **RF:** {RF:.0%}  **PORT_VOL_TARGET:** {VT}  **signal_lag:** 1",
        f"**Locked tag (unchanged unless promote):** `{LOCKED_TAG}`",
        f"**IS windows:** {is_labels}  **Confirm:** {conf_labels}",
        "**Selection:** nested on 2024 only; HO never for selection.",
        "**Families:** baseline / conc_dampen / frozen_wrebal / wrebal+conc / "
        "wrebal+mtd / wrebal+mtd+conc / prior mtd. hi≤1. K=3.",
        f"**Soft mean floor:** {MEAN_FLOOR} on BOTH 2024 and 2025_IS.",
        f"**Frozen wrebal weights:** `{frozen_w.tolist()}`",
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
    if best_conc:
        lines.append(f"**2024 conc pick:** {best_conc[1]} (sc={best_conc[0]:.4f})")
    if best_w_conc:
        lines.append(f"**wrebal+conc nested:** {best_w_conc[1]}")
    if best_w_mtd_conc:
        lines.append(f"**wrebal+mtd+conc nested:** {best_w_mtd_conc[1]}")
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
        "Artifacts: `reports/quest_conc_dampen_board.csv`, "
        "`reports/quest_conc_dampen_consistency.md`, "
        "`configs/quest_conc_dampen_selected.json`",
        "",
    ]
    out = ROOT / "reports" / "quest_conc_dampen_consistency.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sel = {
        "promote": promote_yes,
        "soft": soft_n,
        "hard": hard_n,
        "board_n": int(len(bdf)),
        "best_soft": best_soft_tag,
        "best_hard": hard.iloc[0]["idea"] if len(hard) else None,
        "nearest": nearest_tag,
        "best_conc": best_conc[1] if best_conc else None,
        "best_wrebal_conc": best_w_conc[1] if best_w_conc else None,
        "best_wrebal_mtd_conc": best_w_mtd_conc[1] if best_w_mtd_conc else None,
        "frozen_wrebal": frozen_w.tolist(),
        "prior_mtd": PRIOR_MTD,
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
                + f" Conc-dampen promote `{r['idea']}`: see "
                f"configs/quest_conc_dampen_selected.json. RF=0.08 VT=0.0025 hi<=1."
            )
            locked_path.write_text(
                yaml.safe_dump(locked, sort_keys=False), encoding="utf-8"
            )
            sel["locked_changed"] = True

    (ROOT / "configs" / "quest_conc_dampen_selected.json").write_text(
        json.dumps(sel, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out}", flush=True)
    print(f"wrote {board_path}", flush=True)
    print(json.dumps({k: sel[k] for k in ('promote', 'soft', 'hard', 'board_n', 'best_soft', 'locked_changed')}, indent=2))


if __name__ == "__main__":
    main()
