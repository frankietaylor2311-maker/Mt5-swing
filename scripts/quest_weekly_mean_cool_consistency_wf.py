#!/usr/bin/env python3
"""Causal rolling weekly mean cool consistency wave on locked sleeve.

Distinct from apply_trailing_month_mean_cool (completed calendar months),
apply_prior_month_win_throttle (binary last-month win),
apply_equity_curve_target (trailing monthly VOL), and apply_mtd_gain_clip /
pace / month-end surplus / burst (intra-month MTD).

Uses ONLY completed weeks: lag-1 trailing mean of last `lookback` week
returns; if mean >= mean_thresh → cool_scale else 1.0. Clip [lo, 1], hi<=1.

RF=0.08, VT=0.0025, overlay hi<=1. Nested picks on 2024 only; HO never for
selection. Soft mean floor 0.0095 on BOTH 2024 and 2025_IS.
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
    apply_rolling_weekly_mean_cool,
)

WARMUP = cor.WARMUP
RF = cor.RF
VT = cor.VT
LOCKED_TAG = cor.LOCKED_TAG
MEAN_FLOOR = 0.0095

FROZEN_WREBAL = np.array(
    [0.328198, 0.029051, 0.2413, 0.177605, 0.223846], dtype=float
)
PRIOR_MTD = [
    {"tau": 0.015, "after": 0.5},
    {"tau": 0.015, "after": 0.65},
]
LOOKBACKS = [4, 6, 8, 12]
MEAN_THRESH = [0.002, 0.003, 0.004, 0.005]
COOL_SCALES = [0.35, 0.5, 0.65]
BLEND_ALPHAS = [0.25, 0.4, 0.55]


def blend_ports(a: pd.Series, b: pd.Series, alpha: float) -> pd.Series:
    if a is None or b is None:
        return a if a is not None else b
    ra = a.astype(float).pct_change().fillna(0.0)
    rb = b.astype(float).pct_change().reindex(ra.index).fillna(0.0)
    idx = ra.index.intersection(rb.index)
    if len(idx) < 20:
        return a
    ra = ra.loc[idx]
    rb = rb.loc[idx]
    r = (1.0 - float(alpha)) * ra + float(alpha) * rb
    start = float(a.astype(float).reindex(idx).dropna().iloc[0])
    return (1.0 + r).cumprod() * start


def apply_ports(fn, base_ports, **kw):
    out = {}
    for lab, p in base_ports.items():
        if p is None:
            continue
        out[lab] = fn(p, **kw)
    return out


def tag_cool(lb: int, thr: float, cool: float) -> str:
    return f"wmc_lb{lb}_t{thr}_c{cool}"


def tag_mtd(params) -> str:
    return f"mtd_t{params['tau']}_a{params['after']}"


def tag_w(w: np.ndarray) -> str:
    return "wrebal_" + "_".join(f"{x:.3f}" for x in w)


def _cool_grid():
    return itertools.product(LOOKBACKS, MEAN_THRESH, COOL_SCALES)


def pick_cool_2024(port24, base_st24, top_k: int = 3):
    ranked = []
    for lb, thr, cool in _cool_grid():
        params = {
            "lookback": int(lb),
            "mean_thresh": float(thr),
            "cool_scale": float(cool),
        }
        st = cor.stats_from_port(
            apply_rolling_weekly_mean_cool(port24, **params, lo=0.25)
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
            item[1]["lookback"],
            round(item[1]["mean_thresh"], 4),
            round(item[1]["cool_scale"], 4),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
        if len(uniq) >= top_k:
            break
    return uniq


def pick_cool_on_transformed(port24_xf, base_st_xf):
    best = None
    for lb, thr, cool in _cool_grid():
        params = {
            "lookback": int(lb),
            "mean_thresh": float(thr),
            "cool_scale": float(cool),
        }
        st = cor.stats_from_port(
            apply_rolling_weekly_mean_cool(port24_xf, **params, lo=0.25)
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
    frozen_w = frozen_w / frozen_w.sum()
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

    cool_soft_ports = {}
    # Sparse board: lookbacks x mid thresh x cool extremes
    sparse_cool = list(itertools.product(LOOKBACKS, [0.003], COOL_SCALES))
    for lb, thr, cool in sparse_cool:
        params = {
            "lookback": int(lb),
            "mean_thresh": float(thr),
            "cool_scale": float(cool),
        }
        ports = apply_ports(apply_rolling_weekly_mean_cool, base_ports, **params, lo=0.25)
        name = f"lock+{tag_cool(lb, thr, cool)}"
        row = cor.eval_idea(name, "weekly_mean_cool", ports, is_labels, conf_labels)
        board.append(row)
        print(
            f"weekly_mean_cool: {name} soft={row.soft_pass} 2024={row.mo_2024*100:.2f}% "
            f"HO={row.mo_holdout*100:.2f}%/{row.pos_holdout*100:.0f}% "
            f"top3HO={row.top3_holdout*100:.0f}% note={row.note}",
            flush=True,
        )
        if row.soft_pass:
            cool_soft_ports[name] = ports

    # Nest top-k mean-cool alone on locked (2024-only pick → board)
    if base24 is not None and base_st24 is not None:
        for sc, params, _ in pick_cool_2024(base24, base_st24, top_k=3):
            ports = apply_ports(
                apply_rolling_weekly_mean_cool, base_ports, **params, lo=0.25
            )
            name = (
                f"lock+{tag_cool(params['lookback'], params['mean_thresh'], params['cool_scale'])}"
                f"_nested"
            )
            row = cor.eval_idea(
                name, "weekly_mean_cool_nested", ports, is_labels, conf_labels
            )
            board.append(row)
            print(
                f"weekly_mean_cool_nested: {name} soft={row.soft_pass} 2024={row.mo_2024*100:.2f}% "
                f"HO={row.mo_holdout*100:.2f}%/{row.pos_holdout*100:.0f}% note={row.note}",
                flush=True,
            )
            if row.soft_pass:
                cool_soft_ports[name] = ports

    row = cor.eval_idea(
        f"{tag_w(frozen_w)}_frozen", "frozen_wrebal", wrebal_ports, is_labels, conf_labels
    )
    board.append(row)

    wrebal_mtd_ports = {}
    for mtd in PRIOR_MTD:
        ports = apply_ports(
            apply_mtd_gain_clip, wrebal_ports, tau=mtd["tau"], after_clip=mtd["after"]
        )
        name = f"{tag_w(frozen_w)}+{tag_mtd(mtd)}_frozen"
        row = cor.eval_idea(name, "frozen_wrebal_mtd", ports, is_labels, conf_labels)
        board.append(row)
        wrebal_mtd_ports[name] = (mtd, ports)
        print(
            f"wrebal+mtd: {name} soft={row.soft_pass} 2024={row.mo_2024*100:.2f}% "
            f"HO={row.mo_holdout*100:.2f}%/{row.pos_holdout*100:.0f}% note={row.note}",
            flush=True,
        )

    soft_structures = {}
    w24 = wrebal_ports.get("2024")
    if w24 is not None:
        wst = cor.stats_from_port(w24)
        for sc, params, _ in pick_cool_2024(w24, wst, top_k=2):
            ports = apply_ports(
                apply_rolling_weekly_mean_cool, wrebal_ports, **params, lo=0.25
            )
            name = (
                f"{tag_w(frozen_w)}+"
                f"{tag_cool(params['lookback'], params['mean_thresh'], params['cool_scale'])}_frozen"
            )
            row = cor.eval_idea(
                name, "frozen_wrebal_weekly_mean_cool", ports, is_labels, conf_labels
            )
            board.append(row)
            print(
                f"wrebal+weekly_mean_cool: {name} soft={row.soft_pass} note={row.note}",
                flush=True,
            )
            if row.soft_pass:
                soft_structures[name] = ports

    for name, (mtd, ports) in list(wrebal_mtd_ports.items()):
        p24 = ports.get("2024")
        if p24 is None:
            continue
        st24 = cor.stats_from_port(p24)
        picked = pick_cool_on_transformed(p24, st24)
        if picked is None or picked[0] == float("-inf"):
            continue
        _, cparams, _ = picked
        lports = apply_ports(
            apply_rolling_weekly_mean_cool, ports, **cparams, lo=0.25
        )
        iname = (
            f"{name}+"
            f"{tag_cool(cparams['lookback'], cparams['mean_thresh'], cparams['cool_scale'])}"
        )
        row = cor.eval_idea(
            iname, "frozen_wrebal_mtd_weekly_mean_cool", lports, is_labels, conf_labels
        )
        board.append(row)
        print(
            f"wrebal+mtd+weekly_mean_cool: {iname} soft={row.soft_pass} note={row.note} "
            f"HO_pos={row.pos_holdout*100:.0f}% top3={row.top3_holdout*100:.0f}%",
            flush=True,
        )
        if row.soft_pass:
            soft_structures[iname] = lports
        soft_structures[name] = ports

    for n, p in list(cool_soft_ports.items())[:3]:
        soft_structures[n] = p

    blend_targets = []
    for n, p in soft_structures.items():
        if "wrebal" in n and ("wmc_" in n or "mtd_" in n):
            blend_targets.append((n, p))
    if not blend_targets:
        blend_targets = list(soft_structures.items())[:3]
    blend_targets = blend_targets[:3]

    for tname, tports in blend_targets:
        p24_l = base_ports.get("2024")
        p24_t = tports.get("2024")
        if p24_l is None or p24_t is None:
            continue
        base_st = cor.stats_from_port(p24_l)
        best_a = None
        for a in BLEND_ALPHAS:
            blended24 = blend_ports(p24_l, p24_t, a)
            st = cor.stats_from_port(blended24)
            sc = cor.score_2024_pick(st, base_st)
            if best_a is None or sc > best_a[0]:
                best_a = (sc, a, st)
        if best_a is None:
            continue
        _, alpha, _ = best_a
        alphas_board = sorted(set([alpha, 0.4, 0.55]))
        for a in alphas_board:
            bports = {}
            for lab in set(list(base_ports.keys()) + list(tports.keys())):
                if lab in base_ports and lab in tports:
                    bports[lab] = blend_ports(base_ports[lab], tports[lab], a)
            iname = f"blend_a{a:.2f}_lock_vs_{tname}"
            row = cor.eval_idea(iname, "is_blend", bports, is_labels, conf_labels)
            board.append(row)
            print(
                f"blend a={a:.2f} vs {tname}: soft={row.soft_pass} "
                f"2024={row.mo_2024*100:.2f}% HO={row.mo_holdout*100:.2f}%/"
                f"{row.pos_holdout*100:.0f}% note={row.note}",
                flush=True,
            )

    bdf = pd.DataFrame([asdict(r) for r in board])
    bdf = tighten_soft_mean(bdf)
    soft_n = int(bdf["soft_pass"].sum())
    hard_n = int(bdf["hard_pass"].sum())
    promote_n = int(bdf["promote"].sum())
    print(
        f"BOARD n={len(bdf)} soft={soft_n} hard={hard_n} promote={promote_n}",
        flush=True,
    )

    out_csv = ROOT / "reports" / "quest_weekly_mean_cool_board.csv"
    out_md = ROOT / "reports" / "quest_weekly_mean_cool_consistency.md"
    out_json = ROOT / "configs" / "quest_weekly_mean_cool_selected.json"
    bdf.to_csv(out_csv, index=False)

    ranked = bdf.sort_values(
        ["promote", "hard_pass", "soft_pass", "score"],
        ascending=[False, False, False, False],
    )
    best = ranked.iloc[0].to_dict() if len(ranked) else {}
    sel = {
        "wave": "weekly_mean_cool",
        "locked_tag": LOCKED_TAG,
        "data_source": ds,
        "board_n": int(len(bdf)),
        "soft_n": soft_n,
        "hard_n": hard_n,
        "promote_n": promote_n,
        "best": {
            k: (None if (isinstance(v, float) and (v != v)) else v)
            for k, v in best.items()
        },
        "mean_floor": MEAN_FLOOR,
        "note": "HO never used for selection; RF=0.08; overlay hi<=1",
    }
    out_json.write_text(json.dumps(sel, indent=2, default=str) + "\n", encoding="utf-8")

    lines = [
        "# Quest wave — rolling weekly mean cool",
        "",
        f"**data_source:** {ds}",
        f"**locked:** `{LOCKED_TAG}` RF={RF} VT={VT}",
        f"**board:** n={len(bdf)} soft={soft_n} hard={hard_n} promote={promote_n}",
        f"**soft mean floor:** {MEAN_FLOOR:.2%} on 2024 AND 2025_IS",
        "",
        "## Best row",
        "",
        f"- idea: `{best.get('idea')}`",
        f"- soft/hard/promote: {best.get('soft_pass')} / {best.get('hard_pass')} / {best.get('promote')}",
        f"- note: {best.get('note')}",
        f"- 2024 mo/pos/top3: {float(best.get('mo_2024') or 0)*100:.2f}% / "
        f"{float(best.get('pos_2024') or 0)*100:.0f}% / {float(best.get('top3_2024') or 0)*100:.0f}%",
        f"- holdout mo/pos/top3: {float(best.get('mo_holdout') or 0)*100:.2f}% / "
        f"{float(best.get('pos_holdout') or 0)*100:.0f}% / {float(best.get('top3_holdout') or 0)*100:.0f}%",
        f"- gates_holdout: {best.get('gates_holdout')}",
        "",
        "Locked tag unchanged unless promote=True.",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_csv}", flush=True)
    print(f"Wrote {out_md}", flush=True)
    print(f"Wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
