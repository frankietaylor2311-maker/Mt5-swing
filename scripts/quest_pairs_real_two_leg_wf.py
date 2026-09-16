#!/usr/bin/env python3
"""Real two-leg pairs MR wave: replace Δz proxy with both-leg fills.

Nested IS (no holdout peek during search):
  - Fit hedge beta on 2024 only
  - Optimize knobs on 2024, validate on 2025_IS
  - Score = min(2024, 2025_IS mean_mo) with %pos / top3 constraints
  - Freeze winners; confirm on holdout + 2026 only after freeze

RF fixed at 8%. No look-ahead (signal_lag=1). FTMO gates on confirm.
If real two-leg cannot approach Δz proxy, document proxy as illusory.
"""
from __future__ import annotations

import importlib.util
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
from mt5_swing.portfolio.overlays import INITIAL
from mt5_swing.portfolio.pairs_residual import (
    DEFAULT_PAIRS,
    backtest_residual_equity,
    backtest_two_leg_spread,
    combine_sleeve_curves,
    diagnose_proxy_vs_residual,
    engle_granger_adf_stat,
    hedge_ratio_ols,
    per_leg_risk,
    residual_log,
    rolling_zscore,
)
from mt5_swing.portfolio.smooth_select import WindowStats

RF = 0.08
LOCKED_TAG = "fx4plus_gbpcad_d1_voltarget_0025"
MIN_PCT_POS = 0.70
MAX_TOP3_HARD = 0.55
MAX_TOP3_SOFT = 0.70
MAX_P2T = 0.085
TARGET_MO = 0.01
WARMUP = 120

HIST = ROOT / "data" / "history"
REPORTS = ROOT / "reports"

# Approximate FTMO-like spreads (pips) for pairs symbols
SPREADS = {
    "EURUSD": 1.2,
    "GBPUSD": 1.4,
    "USDCHF": 1.5,
    "AUDUSD": 1.4,
    "NZDUSD": 1.6,
    "AUDCAD": 2.0,
    "NZDCAD": 2.2,
    "EURJPY": 1.6,
    "GBPJPY": 2.0,
    "EURCHF": 1.8,
    "USDCAD": 1.5,
    "EURAUD": 2.2,
    "GBPCAD": 2.4,
}


def data_source() -> str:
    if any((ROOT / "data" / "ftmo").glob("*.csv")):
        return "ftmo_mt5_export"
    return "approximate_non_ftmo"


def load_close(sym: str, tf: str = "D1") -> pd.Series | None:
    path = ewc.resolve_csv(sym, tf)
    if path is None:
        return None
    df = load_ohlc_csv(path, symbol=sym, timeframe=tf)
    return df["close"].astype(float)


@dataclass
class EvalRow:
    idea: str
    params: str
    window: str
    ret: float
    mean_mo: float
    pct_pos: float
    top3: float
    gates: bool
    p2t: float
    n_months: int
    role: str
    sizing: str = ""


def eval_port(port: pd.Series, label: str, idea: str, params: str, role: str, sizing: str = "") -> EvalRow | None:
    if port is None or len(port) < 30:
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
    peak = port.cummax()
    p2t = float(((peak - port) / peak).max())
    return EvalRow(
        idea=idea,
        params=params,
        window=label,
        ret=float(m.total_return),
        mean_mo=ms["mean_mo"],
        pct_pos=ms["pct_pos"],
        top3=ms["top3_share"],
        gates=bool(m.gates_pass),
        p2t=p2t,
        n_months=ms["n_months"],
        role=role,
        sizing=sizing,
    )


def to_stats(row: EvalRow) -> WindowStats:
    return WindowStats(
        mean_mo=row.mean_mo,
        pct_pos=row.pct_pos,
        top3=row.top3 if row.top3 == row.top3 else 1.0,
        gates=row.gates,
        p2t=row.p2t,
    )


def fit_beta_on(a: str, b: str, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    ca, cb = load_close(a), load_close(b)
    if ca is None or cb is None:
        return None
    both = pd.concat([ca.rename("a"), cb.rename("b")], axis=1, sort=True).dropna()
    both = both.loc[(both.index >= start) & (both.index <= end)]
    if len(both) < 80:
        return None
    return hedge_ratio_ols(both["a"], both["b"])


def pair_curve(
    a: str,
    b: str,
    beta: float,
    win: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    entry: float,
    exit_z: float,
    risk_frac: float,
    sizing: str,
    mode: str,  # real | proxy
) -> tuple[pd.Series, dict | None]:
    ca, cb = load_close(a), load_close(b)
    if ca is None or cb is None:
        return pd.Series(dtype=float), None
    both = pd.concat([ca.rename("a"), cb.rename("b")], axis=1, sort=True).dropna()
    pre = both.loc[both.index < start].iloc[-max(WARMUP, win + 10) :]
    win_df = both.loc[(both.index >= start) & (both.index <= end)]
    full = pd.concat([pre, win_df])
    if len(full) < win + 10 or len(win_df) < 20:
        return pd.Series(dtype=float), None
    res = residual_log(full["a"], full["b"], beta)
    z = rolling_zscore(res, win)
    diag = None
    if mode == "proxy":
        eq = backtest_residual_equity(z, entry=entry, exit_z=exit_z, risk_frac=risk_frac)
    else:
        eq = backtest_two_leg_spread(
            full["a"],
            full["b"],
            z,
            beta,
            symbol_a=a,
            symbol_b=b,
            entry=entry,
            exit_z=exit_z,
            risk_frac=risk_frac,
            spread_pips_a=SPREADS.get(a, 1.5),
            spread_pips_b=SPREADS.get(b, 1.5),
            commission_per_lot=7.0,
            slippage_pips=0.5,
            sizing=sizing,
            residual_for_sd=res,
            sd_win=win,
        )
        # diagnosis on window slice
        diag = diagnose_proxy_vs_residual(z.loc[win_df.index.min() :], res.loc[win_df.index.min() :])
    eq = eq.loc[(eq.index >= start) & (eq.index <= end)]
    return eq, diag


def sleeve_port(chosen, betas, start, end, *, entry, exit_z, risk_frac, sizing, mode, sleeve_budget):
    curves = []
    diags = []
    for c in chosen:
        key = f"{c['a']}/{c['b']}"
        eq, diag = pair_curve(
            c["a"],
            c["b"],
            betas[key],
            c["win"],
            start,
            end,
            entry=entry,
            exit_z=exit_z,
            risk_frac=risk_frac,
            sizing=sizing,
            mode=mode,
        )
        if eq.empty:
            return pd.Series(dtype=float), []
        curves.append(eq)
        if diag:
            diags.append({"pair": key, **diag})
    return combine_sleeve_curves(curves, max_gross=float(sleeve_budget)), diags


def main() -> None:
    src = data_source()
    REPORTS.mkdir(exist_ok=True)
    print(f"data_source={src} RF={RF} locked={LOCKED_TAG}", flush=True)
    print(
        "Nested IS: optimize 2024 → validate 2025_IS; HO/2026 confirmation only. Real two-leg.",
        flush=True,
    )

    # Calendar anchors from any D1
    sample = load_close("EURUSD")
    assert sample is not None
    end = sample.index.max()
    holdout_start = end - pd.Timedelta(days=365)
    w2024_s = pd.Timestamp("2024-01-01", tz="UTC")
    w2024_e = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
    w2025_s = pd.Timestamp("2025-01-01", tz="UTC")
    w2025_e = min(holdout_start - pd.Timedelta(seconds=1), end)
    conf = [
        ("2024", w2024_s, w2024_e),
        ("2025", w2025_s, min(pd.Timestamp("2025-12-31 23:59:59", tz="UTC"), end)),
        ("2026", pd.Timestamp("2026-01-01", tz="UTC"), end),
        ("holdout_365d", holdout_start, end),
    ]
    print(f"end={end} holdout_start={holdout_start} 2025_IS→{w2025_e.date()}", flush=True)

    # Rank pairs by ADF on 2024-only fit (nested — no 2025 in fit for ranking seed)
    ranked = []
    for ps in DEFAULT_PAIRS:
        beta = fit_beta_on(ps.a, ps.b, w2024_s, w2024_e)
        if beta is None:
            continue
        ca, cb = load_close(ps.a), load_close(ps.b)
        both = pd.concat([ca.rename("a"), cb.rename("b")], axis=1, sort=True).dropna()
        both = both.loc[(both.index >= w2024_s) & (both.index <= w2024_e)]
        res = residual_log(both["a"], both["b"], beta)
        adf = engle_granger_adf_stat(res)
        for win in (40, 60, 90):
            ranked.append({"a": ps.a, "b": ps.b, "win": win, "beta2024": beta, "adf": adf})
        print(f"  fit2024 {ps.a}/{ps.b} beta={beta:.3f} adf={adf:.2f}", flush=True)

    # Unique pairs by best ADF, keep win variants for grid
    by_pair = {}
    for r in sorted(ranked, key=lambda x: x["adf"]):
        key = (r["a"], r["b"])
        if key not in by_pair:
            by_pair[key] = r
    pair_rank = list(by_pair.values())
    print(f"unique pairs by ADF: {len(pair_rank)}", flush=True)

    # --- Diagnosis: proxy vs unit_residual vs z_vol real on top pair ---
    print("=== Diagnosis: Δz proxy vs real two-leg ===", flush=True)
    diag_rows = []
    top = pair_rank[0]
    beta = top["beta2024"]
    for sizing, mode in [("z_vol", "proxy"), ("z_vol", "real"), ("unit_residual", "real")]:
        port, diags = sleeve_port(
            [{"a": top["a"], "b": top["b"], "win": 40}],
            {f"{top['a']}/{top['b']}": beta},
            w2024_s,
            w2024_e,
            entry=2.0,
            exit_z=0.30,
            risk_frac=0.01,
            sizing=sizing,
            mode=mode,
            sleeve_budget=1.0,
        )
        row = eval_port(port, "2024", f"diag_{mode}_{sizing}", "{}", "diag", sizing=sizing)
        amp = (diags[0]["amplification"] if diags else float("nan"))
        if row:
            print(
                f"  {mode:6s} sizing={sizing:14s} 2024 mo={row.mean_mo*100:.3f}% "
                f"pos={row.pct_pos*100:.0f}% amp≈{amp}",
                flush=True,
            )
            diag_rows.append(
                {
                    "mode": mode,
                    "sizing": sizing,
                    "mean_mo": row.mean_mo,
                    "pct_pos": row.pct_pos,
                    "amplification": amp,
                    "pair": f"{top['a']}/{top['b']}",
                }
            )

    # --- Nested IS grid (2024 optimize → 2025 validate); NEVER use HO/2026 ---
    print("=== Nested IS grid (real two-leg, z_vol + unit_residual) ===", flush=True)
    entry_grid = [1.5, 1.75, 2.0, 2.25, 2.5]
    exit_grid = [0.20, 0.30, 0.50]
    risk_grid = [0.005, 0.0075, 0.01]  # capped ≤ per_leg RF/n
    sleeve_grid = [0.55, 1.0]
    win_grid = [40, 60]
    board = []
    rows: list[EvalRow] = []

    for sizing in ("z_vol", "unit_residual"):
        for n_pairs in (2, 3, 4):
            chosen_base = pair_rank[:n_pairs]
            if len(chosen_base) < n_pairs:
                continue
            risk_cap = min(per_leg_risk(RF, n_pairs, legs_per_pair=1), 0.01)
            for win in win_grid:
                chosen = [{**c, "win": win} for c in chosen_base]
                # betas: fit on 2024 only (nested)
                betas = {}
                ok_beta = True
                for c in chosen:
                    b = fit_beta_on(c["a"], c["b"], w2024_s, w2024_e)
                    if b is None:
                        ok_beta = False
                        break
                    betas[f"{c['a']}/{c['b']}"] = b
                if not ok_beta:
                    continue
                for entry in entry_grid:
                    for exit_z in exit_grid:
                        for risk_frac in risk_grid:
                            if risk_frac > risk_cap + 1e-12:
                                continue
                            for sleeve_budget in sleeve_grid:
                                idea = f"real_n{n_pairs}_{sizing}"
                                params = json.dumps(
                                    {
                                        "pairs": [f"{c['a']}/{c['b']}:w{win}" for c in chosen],
                                        "entry": entry,
                                        "exit_z": exit_z,
                                        "risk_frac": risk_frac,
                                        "sleeve_budget": sleeve_budget,
                                        "sizing": sizing,
                                        "betas_2024": betas,
                                        "nested": "opt2024_val2025",
                                    },
                                    sort_keys=True,
                                )
                                # 2024
                                p24, _ = sleeve_port(
                                    chosen, betas, w2024_s, w2024_e,
                                    entry=entry, exit_z=exit_z, risk_frac=risk_frac,
                                    sizing=sizing, mode="real", sleeve_budget=sleeve_budget,
                                )
                                r24 = eval_port(p24, "2024", idea, params, "is_opt", sizing)
                                if r24 is None:
                                    continue
                                # 2025 validate (same frozen betas/knobs)
                                p25, _ = sleeve_port(
                                    chosen, betas, w2025_s, w2025_e,
                                    entry=entry, exit_z=exit_z, risk_frac=risk_frac,
                                    sizing=sizing, mode="real", sleeve_budget=sleeve_budget,
                                )
                                r25 = eval_port(p25, "2025_IS", idea, params, "is_val", sizing)
                                if r25 is None:
                                    continue
                                rows.extend([r24, r25])
                                sc = score_is_windows(
                                    [to_stats(r24), to_stats(r25)],
                                    min_pct_pos=MIN_PCT_POS,
                                    max_top3_hard=MAX_TOP3_HARD,
                                    max_top3_soft=MAX_TOP3_SOFT,
                                    max_p2t=MAX_P2T,
                                )
                                board.append(
                                    {
                                        "idea": idea,
                                        "params": params,
                                        "score": sc.score,
                                        "min_mean_mo": sc.min_mean_mo,
                                        "max_top3": sc.max_top3,
                                        "min_pct_pos": sc.min_pct_pos,
                                        "hard_pass": sc.hard_pass,
                                        "soft_pass": sc.soft_pass,
                                        "chosen": chosen,
                                        "betas": betas,
                                        "entry": entry,
                                        "exit_z": exit_z,
                                        "risk_frac": risk_frac,
                                        "sleeve_budget": sleeve_budget,
                                        "sizing": sizing,
                                        "y2024_mo": r24.mean_mo,
                                        "y2025_mo": r25.mean_mo,
                                        "y2024_pos": r24.pct_pos,
                                        "y2025_pos": r25.pct_pos,
                                    }
                                )

    soft = [c for c in board if c["soft_pass"]]
    hard = [c for c in board if c["hard_pass"]]
    by_score = sorted(board, key=lambda x: (-(x["score"] if x["score"] == x["score"] else -1), x["max_top3"]))
    by_min = sorted(
        [c for c in board if c["min_mean_mo"] == c["min_mean_mo"]],
        key=lambda x: (-x["min_mean_mo"], x["max_top3"]),
    )
    print(
        f"board={len(board)} soft={len(soft)} hard={len(hard)} "
        f"best_min_mo={by_min[0]['min_mean_mo']*100:.3f}% ({by_min[0]['idea']})" if by_min else
        f"board={len(board)} soft={len(soft)} hard={len(hard)}",
        flush=True,
    )
    for c in by_score[:10]:
        print(
            f"  {c['idea']:28s} e={c['entry']} x={c['exit_z']} rf={c['risk_frac']} "
            f"min_mo={c['min_mean_mo']*100:.3f}% pos={c['min_pct_pos']*100:.0f}% "
            f"t3={c['max_top3']*100:.0f}% soft={c['soft_pass']}",
            flush=True,
        )

    # --- Confirm top candidates (HO/2026 only now) ---
    to_confirm = []
    seen = set()
    for c in hard[:3] + soft[:5] + by_min[:5] + by_score[:5]:
        key = (c["idea"], c["params"])
        if key in seen:
            continue
        seen.add(key)
        to_confirm.append(c)
        if len(to_confirm) >= 12:
            break

    print(f"=== Confirm {len(to_confirm)} frozen candidates ===", flush=True)
    promote_results = []
    for c in to_confirm:
        year_map = {}
        ho_stat = None
        conf_rows = []
        for label, s, e in conf:
            port, _ = sleeve_port(
                c["chosen"], c["betas"], s, e,
                entry=c["entry"], exit_z=c["exit_z"], risk_frac=c["risk_frac"],
                sizing=c["sizing"], mode="real", sleeve_budget=c["sleeve_budget"],
            )
            row = eval_port(port, label, c["idea"], c["params"], "confirm" if label != "holdout_365d" else "holdout", c["sizing"])
            if row:
                conf_rows.append(row)
                rows.append(row)
                st = to_stats(row)
                if label in {"2024", "2025", "2026"}:
                    year_map[label] = st
                if label == "holdout_365d":
                    ho_stat = st
                print(
                    f"  {c['idea'][:32]:32s} {label:14s} mo={row.mean_mo*100:6.3f}% "
                    f"pos={row.pct_pos*100:4.0f}% top3={row.top3*100:4.0f}% gates={row.gates}",
                    flush=True,
                )
        reason = []
        promote = False
        if ho_stat is None:
            reason.append("no_holdout")
        else:
            ho_ok = holdout_clears_promote(
                ho_stat, min_mean_mo=TARGET_MO, min_pct_pos=MIN_PCT_POS,
                max_top3=MAX_TOP3_SOFT, max_p2t=MAX_P2T,
            )
            yrs_ok = years_each_clear(
                year_map, ("2024", "2025", "2026"), min_mean_mo=TARGET_MO, min_pct_pos=MIN_PCT_POS
            )
            if not c.get("soft_pass"):
                reason.append("is_soft_fail")
            if not ho_ok:
                reason.append("holdout_fail")
            if not yrs_ok:
                reason.append("years_fail")
            promote = bool(c.get("soft_pass")) and ho_ok and yrs_ok
        promote_results.append(
            {
                "cand": c,
                "conf_rows": conf_rows,
                "year_map": {k: asdict(v) for k, v in year_map.items()},
                "holdout": asdict(ho_stat) if ho_stat else None,
                "promote": promote,
                "reason": reason,
            }
        )

    any_promote = any(r["promote"] for r in promote_results)

    # Best REAL by confirm consistency (min of 2024/2025/2026/HO mean_mo among confirmed)
    def confirm_min_mo(r):
        vals = []
        for k in ("2024", "2025", "2026"):
            if k in r["year_map"]:
                vals.append(r["year_map"][k]["mean_mo"])
        if r["holdout"]:
            vals.append(r["holdout"]["mean_mo"])
        return min(vals) if vals else float("-inf")

    best_real = max(promote_results, key=confirm_min_mo) if promote_results else None

    # Proxy reference on best-IS pair set for documentation
    print("=== Proxy reference (not promote-eligible) on best REAL pair set ===", flush=True)
    proxy_ref = None
    if best_real is not None:
        c = best_real["cand"]
        # rebuild proxy sleeve on confirm windows
        proxy_ref = {"idea": "proxy_ref", "windows": {}}
        for label, s, e in conf:
            port, _ = sleeve_port(
                c["chosen"], c["betas"], s, e,
                entry=c["entry"], exit_z=c["exit_z"], risk_frac=c["risk_frac"],
                sizing="z_vol", mode="proxy", sleeve_budget=c["sleeve_budget"],
            )
            row = eval_port(port, label, "proxy_ref", c["params"], "proxy_ref", "proxy")
            if row:
                proxy_ref["windows"][label] = asdict(row)
                print(
                    f"  PROXY {label:14s} mo={row.mean_mo*100:6.3f}% pos={row.pct_pos*100:4.0f}%",
                    flush=True,
                )

    # Write artifacts
    board_df = pd.DataFrame(
        [
            {
                "idea": c["idea"],
                "params": c["params"],
                "score": c["score"],
                "min_mean_mo": c["min_mean_mo"],
                "max_top3": c["max_top3"],
                "min_pct_pos": c["min_pct_pos"],
                "soft_pass": c["soft_pass"],
                "hard_pass": c["hard_pass"],
                "sizing": c["sizing"],
                "y2024_mo": c["y2024_mo"],
                "y2025_mo": c["y2025_mo"],
            }
            for c in board
        ]
    ).sort_values(["soft_pass", "hard_pass", "score"], ascending=[False, False, False])
    board_df.to_csv(REPORTS / "quest_pairs_real_board.csv", index=False)

    promo_rows = []
    for r in promote_results:
        c = r["cand"]
        ho = r["holdout"] or {}
        promo_rows.append(
            {
                "idea": c["idea"],
                "params": c["params"],
                "sizing": c["sizing"],
                "is_score": c["score"],
                "is_min_mo": c["min_mean_mo"],
                "is_soft": c["soft_pass"],
                "is_hard": c["hard_pass"],
                "ho_mean_mo": ho.get("mean_mo"),
                "ho_pct_pos": ho.get("pct_pos"),
                "ho_top3": ho.get("top3"),
                "ho_gates": ho.get("gates"),
                "y2024_mo": (r["year_map"].get("2024") or {}).get("mean_mo"),
                "y2025_mo": (r["year_map"].get("2025") or {}).get("mean_mo"),
                "y2026_mo": (r["year_map"].get("2026") or {}).get("mean_mo"),
                "y2024_pos": (r["year_map"].get("2024") or {}).get("pct_pos"),
                "y2025_pos": (r["year_map"].get("2025") or {}).get("pct_pos"),
                "y2026_pos": (r["year_map"].get("2026") or {}).get("pct_pos"),
                "promote": r["promote"],
                "reason": "|".join(r["reason"]) if r["reason"] else "ok",
            }
        )
    promo_df = pd.DataFrame(promo_rows)
    promo_df.to_csv(REPORTS / "quest_pairs_real_promote.csv", index=False)
    pd.DataFrame([asdict(r) for r in rows]).to_csv(REPORTS / "quest_pairs_real_wf.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(REPORTS / "quest_pairs_proxy_diagnosis.csv", index=False)

    # Proxy illusory?
    proxy_mo = next((d["mean_mo"] for d in diag_rows if d["mode"] == "proxy"), float("nan"))
    unit_mo = next((d["mean_mo"] for d in diag_rows if d["sizing"] == "unit_residual"), float("nan"))
    zvol_mo = next((d["mean_mo"] for d in diag_rows if d["mode"] == "real" and d["sizing"] == "z_vol"), float("nan"))
    amp = next((d["amplification"] for d in diag_rows if d.get("amplification") == d.get("amplification")), float("nan"))
    best_confirm_mo = confirm_min_mo(best_real) if best_real else float("nan")
    proxy_illusory = (
        (amp == amp and amp > 20)
        and (unit_mo == unit_mo and unit_mo < 0.002)
        and (best_confirm_mo != best_confirm_mo or best_confirm_mo < 0.005)
    ) or (
        best_real is not None
        and best_confirm_mo < 0.005
        and (proxy_mo == proxy_mo and proxy_mo > 0.008)
    )

    selected = {
        "locked_tag": LOCKED_TAG,
        "data_source": src,
        "rf": RF,
        "promote": any_promote,
        "proxy_illusory": bool(proxy_illusory),
        "diagnosis": diag_rows,
        "n_candidates": len(board),
        "n_soft": len(soft),
        "n_hard": len(hard),
        "best_is": None
        if not by_score
        else {
            "idea": by_score[0]["idea"],
            "params": by_score[0]["params"],
            "score": by_score[0]["score"],
            "min_mean_mo": by_score[0]["min_mean_mo"],
            "soft_pass": by_score[0]["soft_pass"],
            "sizing": by_score[0]["sizing"],
        },
        "best_real_confirm": None
        if best_real is None
        else {
            "idea": best_real["cand"]["idea"],
            "params": best_real["cand"]["params"],
            "sizing": best_real["cand"]["sizing"],
            "promote": best_real["promote"],
            "reason": best_real["reason"],
            "year_map": best_real["year_map"],
            "holdout": best_real["holdout"],
            "confirm_min_mo": best_confirm_mo,
        },
        "proxy_ref": proxy_ref,
        "promote_results": [
            {
                "idea": r["cand"]["idea"],
                "sizing": r["cand"]["sizing"],
                "promote": r["promote"],
                "reason": r["reason"],
                "year_map": r["year_map"],
                "holdout": r["holdout"],
                "is_min_mo": r["cand"]["min_mean_mo"],
                "is_soft": r["cand"]["soft_pass"],
            }
            for r in promote_results
        ],
    }
    (ROOT / "configs" / "quest_pairs_real_selected.json").write_text(json.dumps(selected, indent=2) + "\n")

    # Markdown report
    lines = [
        "# Real two-leg pairs quest",
        "",
        f"**Data:** `{src}`. **RF={RF}**. Locked `{LOCKED_TAG}` unchanged unless promote.",
        "",
        "## Diagnosis: Δz proxy vs real",
        "",
    ]
    for d in diag_rows:
        lines.append(
            f"- `{d['mode']}` / `{d['sizing']}` on {d['pair']}: "
            f"2024 mean_mo={d['mean_mo']*100:.3f}%, pos={d['pct_pos']*100:.0f}%"
            + (f", amp≈{d['amplification']:.1f}x" if d.get("amplification") == d.get("amplification") else "")
        )
    lines += [
        "",
        f"**Proxy illusory?** **{'YES' if proxy_illusory else 'NO'}** — "
        "unit_residual (honest notional) and/or costed z_vol cannot clear ≥1% on confirmation.",
        "",
        f"Board: {len(board)} candidates, soft={len(soft)}, hard={len(hard)}.",
        f"**Promote?** **{'YES' if any_promote else 'NO'}**",
        "",
        "## Best REAL candidate (confirm table)",
        "",
    ]
    if best_real:
        c = best_real["cand"]
        lines.append(f"Idea: `{c['idea']}` sizing=`{c['sizing']}`")
        lines.append("")
        lines.append("| Window | Mean mo | %pos | Top3 | Gates |")
        lines.append("|--------|--------:|-----:|-----:|:-----:|")
        for label in ("2024", "2025", "2026", "holdout_365d"):
            if label == "holdout_365d":
                st = best_real["holdout"]
            else:
                st = best_real["year_map"].get(label)
            if not st:
                continue
            lines.append(
                f"| {label} | {st['mean_mo']*100:.3f}% | {st['pct_pos']*100:.0f}% | "
                f"{st['top3']*100:.0f}% | {st['gates']} |"
            )
        lines.append("")
        lines.append(f"confirm_min_mo={best_confirm_mo*100:.3f}%; promote={best_real['promote']}; reason={best_real['reason']}")
    else:
        lines.append("_No REAL candidate produced confirm stats._")

    (REPORTS / "quest_pairs_real_two_leg.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"promote": any_promote, "proxy_illusory": proxy_illusory, "best_confirm_min_mo": best_confirm_mo}, indent=2))
    print("Wrote reports/quest_pairs_real_*.csv/md and configs/quest_pairs_real_selected.json", flush=True)


if __name__ == "__main__":
    main()
