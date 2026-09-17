#!/usr/bin/env python3
"""Causal peak-proximity cool consistency wave on locked sleeve.

Distinct from apply_runup_throttle (trailing return) and win-streak / hit-rate /
loss-streak / rolling Sharpe cools. Cools when lag-1 equity is within eps of the
rolling N-bar high. RF=0.08, VT=0.0025, overlay hi<=1. Nested picks on 2024 only;
HO never for selection. Soft mean floor 0.0095 on BOTH 2024 and 2025_IS.
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
    apply_peak_proximity_cool,
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
LOOKBACKS = [21, 42, 63]
EPSS = [0.005, 0.01, 0.02]
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


def tag_pp(lb: int, eps: float, cool: float) -> str:
    return f"peakprox_lb{lb}_e{eps}_c{cool}"


def tag_mtd(params) -> str:
    return f"mtd_t{params['tau']}_a{params['after']}"


def tag_w(w: np.ndarray) -> str:
    return "wrebal_" + "_".join(f"{x:.3f}" for x in w)


def pick_peakprox_2024(port24, base_st24, top_k: int = 3):
    ranked = []
    for lb, eps, cool in itertools.product(LOOKBACKS, EPSS, COOL_SCALES):
        params = {
            "lookback_bars": int(lb),
            "eps": float(eps),
            "cool_scale": float(cool),
        }
        st = cor.stats_from_port(
            apply_peak_proximity_cool(
                port24,
                lookback_bars=params["lookback_bars"],
                eps=params["eps"],
                cool_scale=params["cool_scale"],
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
            item[1]["lookback_bars"],
            round(item[1]["eps"], 6),
            round(item[1]["cool_scale"], 4),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
        if len(uniq) >= top_k:
            break
    return uniq


def pick_peakprox_on_transformed(port24_xf, base_st_xf):
    best = None
    for lb, eps, cool in itertools.product(LOOKBACKS, EPSS, COOL_SCALES):
        params = {
            "lookback_bars": int(lb),
            "eps": float(eps),
            "cool_scale": float(cool),
        }
        st = cor.stats_from_port(
            apply_peak_proximity_cool(
                port24_xf,
                lookback_bars=params["lookback_bars"],
                eps=params["eps"],
                cool_scale=params["cool_scale"],
                lo=0.25,
            )
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

    # --- peakprox alone: full grid on locked ---
    peakprox_soft_ports = {}
    for lb, eps, cool in itertools.product(LOOKBACKS, EPSS, COOL_SCALES):
        params = {
            "lookback_bars": int(lb),
            "eps": float(eps),
            "cool_scale": float(cool),
        }
        ports = apply_ports(
            apply_peak_proximity_cool,
            base_ports,
            lookback_bars=params["lookback_bars"],
            eps=params["eps"],
            cool_scale=params["cool_scale"],
            lo=0.25,
        )
        name = f"lock+{tag_pp(lb, eps, cool)}"
        row = cor.eval_idea(name, "peakprox", ports, is_labels, conf_labels)
        board.append(row)
        print(
            f"peakprox: {name} soft={row.soft_pass} 2024={row.mo_2024*100:.2f}% "
            f"HO={row.mo_holdout*100:.2f}%/{row.pos_holdout*100:.0f}% note={row.note}",
            flush=True,
        )
        if row.soft_pass:
            peakprox_soft_ports[name] = ports

    # --- frozen wrebal ---
    row = cor.eval_idea(
        f"{tag_w(frozen_w)}_frozen", "frozen_wrebal", wrebal_ports, is_labels, conf_labels
    )
    board.append(row)

    # --- wrebal + mtd ---
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

    # --- wrebal + nested peakprox on 2024 ---
    soft_structures = {}
    w24 = wrebal_ports.get("2024")
    if w24 is not None:
        wst = cor.stats_from_port(w24)
        picked = pick_peakprox_2024(w24, wst, top_k=2)
        for sc, params, _ in picked:
            ports = apply_ports(
                apply_peak_proximity_cool,
                wrebal_ports,
                lookback_bars=params["lookback_bars"],
                eps=params["eps"],
                cool_scale=params["cool_scale"],
                lo=0.25,
            )
            name = (
                f"{tag_w(frozen_w)}+"
                f"{tag_pp(params['lookback_bars'], params['eps'], params['cool_scale'])}_frozen"
            )
            row = cor.eval_idea(name, "frozen_wrebal_peakprox", ports, is_labels, conf_labels)
            board.append(row)
            print(
                f"wrebal+peakprox: {name} soft={row.soft_pass} note={row.note}",
                flush=True,
            )
            if row.soft_pass:
                soft_structures[name] = ports

    # --- wrebal + mtd + nested peakprox ---
    for name, (mtd, ports) in list(wrebal_mtd_ports.items()):
        p24 = ports.get("2024")
        if p24 is None:
            continue
        st24 = cor.stats_from_port(p24)
        picked = pick_peakprox_on_transformed(p24, st24)
        if picked is None or picked[0] == float("-inf"):
            continue
        _, pparams, _ = picked
        lports = apply_ports(
            apply_peak_proximity_cool,
            ports,
            lookback_bars=pparams["lookback_bars"],
            eps=pparams["eps"],
            cool_scale=pparams["cool_scale"],
            lo=0.25,
        )
        iname = (
            f"{name}+"
            f"{tag_pp(pparams['lookback_bars'], pparams['eps'], pparams['cool_scale'])}"
        )
        row = cor.eval_idea(
            iname, "frozen_wrebal_mtd_peakprox", lports, is_labels, conf_labels
        )
        board.append(row)
        print(
            f"wrebal+mtd+peakprox: {iname} soft={row.soft_pass} note={row.note} "
            f"HO_pos={row.pos_holdout*100:.0f}%",
            flush=True,
        )
        if row.soft_pass:
            soft_structures[iname] = lports
        soft_structures[name] = ports

    for n, p in list(peakprox_soft_ports.items())[:2]:
        soft_structures[n] = p

    blend_targets = []
    for n, p in soft_structures.items():
        if "wrebal" in n and "mtd_" in n:
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
    print(f"BOARD raw n={len(bdf)}", flush=True)

    soft_n = int(bdf["soft_pass"].sum())
    hard_n = int(bdf["hard_pass"].sum())
    promote_n = int(bdf["promote"].sum())
    print(
        f"BOARD n={len(bdf)} soft={soft_n} hard={hard_n} promote={promote_n}",
        flush=True,
    )

    out_csv = ROOT / "reports" / "quest_peak_proximity_board.csv"
    out_md = ROOT / "reports" / "quest_peak_proximity_consistency.md"
    out_json = ROOT / "configs" / "quest_peak_proximity_selected.json"
    bdf.to_csv(out_csv, index=False)

    soft_df = bdf[bdf["soft_pass"]].sort_values("score", ascending=False)
    best_soft = soft_df.iloc[0].to_dict() if len(soft_df) else None
    best_any = bdf.sort_values("score", ascending=False).iloc[0].to_dict()

    selected = {
        "wave": "causal_peak_proximity_cool",
        "data_source": ds,
        "locked_tag": LOCKED_TAG,
        "promote": bool(promote_n > 0),
        "board_n": int(len(bdf)),
        "soft_n": soft_n,
        "hard_n": hard_n,
        "promote_n": promote_n,
        "best_soft": best_soft,
        "best_any": best_any,
        "frozen_wrebal": frozen_w.tolist(),
        "peakprox_grid": {
            "lookbacks": LOOKBACKS,
            "epss": EPSS,
            "cool_scales": COOL_SCALES,
            "lo": 0.25,
            "rule": "lag-1 equity within eps of rolling N-bar high → cool",
        },
    }
    out_json.write_text(json.dumps(selected, indent=2, default=str))

    lines = [
        "# Peak-proximity cool consistency wave",
        "",
        f"**Data:** `{ds}`  **RF:** {RF}  **VT:** {VT}  **Locked:** `{LOCKED_TAG}`",
        f"**Board:** n={len(bdf)} soft=**{soft_n}** hard=**{hard_n}** promote=**{promote_n}**",
        "",
        "Peak-proximity = lag-1 equity within eps of rolling N-bar high → cool; hi≤1.",
        "Distinct from runup throttle (trailing return) and win-streak / hit-rate / Sharpe cools.",
        "",
        "## Board (top by score)",
        "",
        "| Idea | Soft | Hard | Promote | 2024 mo/pos/top3 | 2025_IS | HO | Note |",
        "|------|:----:|:----:|:-------:|------------------|---------|----|------|",
    ]
    show = bdf.sort_values("score", ascending=False).head(15)
    for _, r in show.iterrows():
        lines.append(
            f"| `{r['idea'][:60]}` | {r['soft_pass']} | {r['hard_pass']} | {r['promote']} | "
            f"{r['mo_2024']*100:.2f}%/{r['pos_2024']*100:.0f}%/{r['top3_2024']*100:.0f}% | "
            f"{r['mo_2025_IS']*100:.2f}%/{r['pos_2025_IS']*100:.0f}%/{r['top3_2025_IS']*100:.0f}% | "
            f"{r['mo_holdout']*100:.2f}%/{r['pos_holdout']*100:.0f}%/{r['top3_holdout']*100:.0f}% | "
            f"{r['note']} |"
        )
    lines.append("")
    if best_soft:
        lines.append("## Best soft (not auto-promote unless promote=True)")
        lines.append("")
        lines.append(f"- **idea:** `{best_soft['idea']}`")
        lines.append(
            f"- 2024 **{best_soft['mo_2024']*100:.2f}%** / {best_soft['pos_2024']*100:.0f}% / {best_soft['top3_2024']*100:.0f}%"
        )
        lines.append(
            f"- 2025_IS **{best_soft['mo_2025_IS']*100:.2f}%** / {best_soft['pos_2025_IS']*100:.0f}% / {best_soft['top3_2025_IS']*100:.0f}%"
        )
        lines.append(
            f"- HO **{best_soft['mo_holdout']*100:.2f}%** / {best_soft['pos_holdout']*100:.0f}% / {best_soft['top3_holdout']*100:.0f}% note={best_soft['note']}"
        )
    else:
        lines.append("## No soft IS passer this wave.")
    lines.append("")
    lines.append(
        f"**Promote: {'YES' if promote_n else 'NO'}. Locked tag unchanged unless promote.**"
    )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_csv}", flush=True)
    print(f"Wrote {out_md}", flush=True)
    print(f"Wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
