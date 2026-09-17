#!/usr/bin/env python3
"""Literature-backed macro / geopolitics FX quest (NOT overlay-knob grids).

Ideas (scholar-grounded, a-priori thresholds — selection on IS years only):
  A. fred_carry basket — Lustig–Verdelhan / Menkhoff rate-diff carry (FRED)
  B. locked + VIX risk gate — Menkhoff et al. (2012) global vol risk
  C. locked + GPR risk gate — Caldara–Iacoviello (2022); Liu–Zhang (2024)
  D. locked + VIX+GPR stack
  E. fred_carry + VIX+GPR stack

Holdout / 2026 confirmation only. RF=0.08 fixed. signal_lag=1.
approximate_non_ftmo — never golive without FTMO MT5 exports.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
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
from mt5_swing.portfolio.equity_tsmom import (
    holdout_clears_promote,
    score_is_windows,
    years_each_clear,
)
from mt5_swing.portfolio.macro_regimes import (
    apply_gpr_risk_gate,
    apply_vix_gpr_stack,
    apply_vix_risk_gate,
)
from mt5_swing.portfolio.smooth_select import WindowStats

INITIAL = 100_000.0
WARMUP = 250
RF = 0.08
MAX_LOT = 50.0
TARGET_MO = 0.01
MIN_PCT_POS = 0.70
MAX_TOP3_HARD = 0.55
MAX_TOP3_SOFT = 0.70
MAX_P2T = 0.085
SOFT_MEAN = 0.0095


def locked_legs() -> list[dict]:
    path = ROOT / "configs" / "quest_one_pct_candidate.yaml"
    locked = yaml.safe_load(path.read_text())
    legs = []
    for c in locked["candidates"]:
        params = dict(c.get("params") or {})
        if not params.get("session_hours"):
            params["session_hours"] = None
        legs.append({
            "symbol": c["symbol"],
            "timeframe": c["timeframe"],
            "strategy": c["strategy"],
            "params": params,
            "exits": dict(c.get("exits") or {}),
            "vol_target": bool(c.get("vol_target", False)),
            "oos_sharpe": float(c.get("oos_sharpe") or 0.01),
            "weight": float(c.get("weight") or c.get("oos_sharpe") or 0.01),
        })
    w = np.array([max(l["weight"], 0.01) for l in legs], dtype=float)
    w = w / w.sum()
    for i, l in enumerate(legs):
        l["weight"] = float(w[i])
    return legs


def fred_carry_legs() -> list[dict]:
    """A priori G10 carry basket (literature sign; not HO-tuned)."""
    # Prefer D1 for multi-year coverage on Yahoo approximate_non_ftmo
    symbols = [
        "AUDUSD", "USDJPY", "EURUSD", "GBPUSD", "AUDJPY", "USDCHF", "USDCAD", "EURJPY",
    ]
    legs = []
    for sym in symbols:
        path = ewc.resolve_csv(sym, "D1")
        if path is None:
            continue
        legs.append({
            "symbol": sym,
            "timeframe": "D1",
            "strategy": "fred_carry",
            "params": {
                "symbol": sym,
                "min_diff": 0.25,
                "monthly_lag": 1,
                "require_trend_agree": True,
                "session_hours": None,
            },
            "exits": {"atr_stop_mult": 2.0, "atr_target_mult": 4.0, "max_hold_bars": 60},
            "vol_target": True,
            "oos_sharpe": 1.0,
            "weight": 1.0,
        })
    n = max(len(legs), 1)
    for l in legs:
        l["weight"] = 1.0 / n
    return legs


def build_port(
    legs: list[dict],
    cfg: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series | None:
    risk_per = RF / max(len(legs), 1)
    curves, weights = [], []
    for leg in legs:
        path = ewc.resolve_csv(leg["symbol"], leg["timeframe"])
        if path is None:
            return None
        ohlc = ewc.load_ohlc_csv(path, symbol=leg["symbol"], timeframe=leg["timeframe"]) if False else __import__(
            "mt5_swing.data.loader", fromlist=["load_ohlc_csv"]
        ).load_ohlc_csv(path, symbol=leg["symbol"], timeframe=leg["timeframe"])
        ohlc = ohlc.loc[ohlc.index <= end]
        pre = ohlc.loc[ohlc.index < start]
        win = ohlc.loc[(ohlc.index >= start) & (ohlc.index <= end)]
        warm = pre.iloc[-WARMUP:] if len(pre) >= WARMUP else pre
        ohlc_run = pd.concat([warm, win])
        if len(ohlc_run) < WARMUP + 30 or len(win) < 20:
            return None
        bt = ewc.make_bt(cfg, leg, risk_per, MAX_LOT)
        eq = ewc.run_leg_equity(ohlc_run, leg, bt)
        eq = eq.loc[(eq.index >= start) & (eq.index <= end)]
        if len(eq) < 20:
            return None
        curves.append(eq.rename(f"{leg['symbol']}_{leg['strategy']}"))
        weights.append(float(leg.get("weight") or 0.01))
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    port = ewc.combine_portfolio(curves, w, INITIAL)
    # Optional causal portfolio VT (fixed a priori 0.0025 — locked methodology, not a grid)
    vt = os.environ.get("PORT_VOL_TARGET", "0.0025").strip()
    if vt:
        target = float(vt)
        look = 60
        r = port.pct_change()
        trail = r.shift(1).rolling(look, min_periods=20).std()
        scale = (target / trail.replace(0, np.nan)).clip(0.25, 3.0).fillna(1.0)
        # hi clip 3 on VT is existing locked methodology; RF unchanged
        port = (1.0 + r.fillna(0) * scale).cumprod() * INITIAL
    return port


def stats_from_port(port: pd.Series) -> WindowStats:
    m = compute_metrics(
        port, None, max_dd_gate=0.10, daily_dd_gate=0.05,
        max_loss_mode="static_initial", daily_loss_mode="ftmo_initial",
        daily_tz="Europe/Prague", initial_equity=INITIAL,
    )
    ms = ewc.monthly_stats(port)
    return WindowStats(
        mean_mo=float(ms["mean_mo"]),
        pct_pos=float(ms["pct_pos"]),
        top3=float(ms["top3_share"]),
        gates=bool(m.gates_pass),
        p2t=float(ewc.peak_to_trough(port)),
    )


def eval_idea(name: str, legs: list[dict], cfg: dict, windows: dict, gate: str = "none") -> dict:
    year_stats: dict[str, WindowStats] = {}
    rows = []
    for label, (start, end, role) in windows.items():
        port = build_port(legs, cfg, start, end)
        if port is None or port.empty:
            continue
        if gate == "vix":
            port = apply_vix_risk_gate(port)
        elif gate == "gpr":
            port = apply_gpr_risk_gate(port)
        elif gate == "vix_gpr":
            port = apply_vix_gpr_stack(port)
        st = stats_from_port(port)
        year_stats[label] = st
        rows.append({
            "idea": name, "window": label, "role": role,
            "mean_mo": st.mean_mo, "pct_pos": st.pct_pos, "top3": st.top3,
            "gates": st.gates, "p2t": st.p2t,
        })
        print(
            f"  {name:28s} {label:14s} mo={st.mean_mo*100:5.2f}% pos={st.pct_pos*100:4.0f}% "
            f"top3={st.top3*100:4.0f}% gates={st.gates} p2t={st.p2t*100:4.1f}%",
            flush=True,
        )
    # IS selection: 2024 + 2025_IS (2025 through holdout start)
    is_keys = [k for k in ("2024", "2025_IS") if k in year_stats]
    is_windows = [year_stats[k] for k in is_keys]
    # Soft mean floor
    soft_mean_ok = all(
        year_stats[k].mean_mo == year_stats[k].mean_mo and year_stats[k].mean_mo >= SOFT_MEAN
        for k in is_keys
    ) if is_keys else False
    sc = score_is_windows(
        is_windows, min_pct_pos=MIN_PCT_POS,
        max_top3_hard=MAX_TOP3_HARD, max_top3_soft=MAX_TOP3_SOFT, max_p2t=MAX_P2T,
    )
    if not soft_mean_ok:
        sc = type(sc)(float("-inf"), sc.min_mean_mo, sc.max_top3, sc.min_pct_pos, False, False)

    ho = year_stats.get("holdout_365d")
    yrs = {k: year_stats[k] for k in ("2024", "2025", "2026") if k in year_stats}
    ho_ok = bool(ho and holdout_clears_promote(ho, min_mean_mo=TARGET_MO, min_pct_pos=MIN_PCT_POS, max_top3=MAX_TOP3_SOFT, max_p2t=MAX_P2T))
    yrs_ok = years_each_clear(yrs, ("2024", "2025", "2026"), min_mean_mo=TARGET_MO, min_pct_pos=MIN_PCT_POS) if len(yrs) == 3 else False
    promote = bool(sc.soft_pass) and ho_ok and yrs_ok
    reason = "promote" if promote else (
        "holdout_fail" if sc.soft_pass and not ho_ok else (
            "years_fail" if sc.soft_pass and not yrs_ok else (
                "soft_mean_fail" if not soft_mean_ok else (
                    "is_fail" if not sc.soft_pass else "unknown"
                )
            )
        )
    )
    return {
        "idea": name,
        "gate": gate,
        "score": sc.score,
        "min_mean_mo": sc.min_mean_mo,
        "max_top3": sc.max_top3,
        "min_pct_pos": sc.min_pct_pos,
        "soft_pass": sc.soft_pass,
        "hard_pass": sc.hard_pass,
        "promote": promote,
        "reason": reason,
        "rows": rows,
        "year_stats": {k: {
            "mean_mo": v.mean_mo, "pct_pos": v.pct_pos, "top3": v.top3, "gates": v.gates, "p2t": v.p2t,
        } for k, v in year_stats.items()},
    }


def main() -> None:
    print("=== Macro / geopolitics literature quest (approximate_non_ftmo) ===", flush=True)
    print("Abort: technical overlay/indicator number-hunt. RF=0.08. Holdout confirm only.", flush=True)
    cfg = load_config(ROOT / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    # Span from locked first leg
    legs0 = locked_legs()
    path0 = ewc.resolve_csv(legs0[0]["symbol"], legs0[0]["timeframe"])
    from mt5_swing.data.loader import load_ohlc_csv
    ohlc0 = load_ohlc_csv(path0, symbol=legs0[0]["symbol"], timeframe=legs0[0]["timeframe"])
    end = ohlc0.index.max()
    holdout_days = 365
    holdout_start = end - pd.Timedelta(days=holdout_days)

    windows: dict[str, tuple] = {}
    for year in (2024, 2025, 2026):
        s = pd.Timestamp(f"{year}-01-01", tz="UTC")
        e = min(pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC"), end)
        if s > end:
            continue
        windows[str(year)] = (s, e, "calendar")
    # 2025_IS: 2025 until holdout
    s25 = pd.Timestamp("2025-01-01", tz="UTC")
    e25is = min(holdout_start - pd.Timedelta(days=1), end)
    if e25is > s25:
        windows["2025_IS"] = (s25, e25is, "IS")
    windows["holdout_365d"] = (holdout_start, end, "holdout")

    carry = fred_carry_legs()
    locked = locked_legs()
    ideas = [
        ("baseline_locked_vt", locked, "none"),
        ("fred_carry_d1", carry, "none"),
        ("locked+vix_gate", locked, "vix"),
        ("locked+gpr_gate", locked, "gpr"),
        ("locked+vix_gpr", locked, "vix_gpr"),
        ("fred_carry+vix_gpr", carry, "vix_gpr"),
    ]

    board = []
    all_rows = []
    for name, legs, gate in ideas:
        print(f"\n-- idea={name} legs={len(legs)} gate={gate}", flush=True)
        if not legs:
            print("  SKIP empty legs", flush=True)
            continue
        res = eval_idea(name, legs, cfg, windows, gate=gate)
        board.append({k: v for k, v in res.items() if k not in ("rows", "year_stats")})
        board[-1]["year_stats"] = res["year_stats"]
        all_rows.extend(res["rows"])

    soft_n = sum(1 for b in board if b["soft_pass"])
    hard_n = sum(1 for b in board if b["hard_pass"])
    promo = [b for b in board if b["promote"]]
    print(f"\nBoard n={len(board)} soft={soft_n} hard={hard_n} promote={len(promo)}", flush=True)

    out_csv = ROOT / "reports" / "quest_macro_factor_literature_board.csv"
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    sel = {
        "data_source": "approximate_non_ftmo",
        "locked_tag": "fx4plus_gbpcad_d1_voltarget_0025",
        "promote": bool(promo),
        "board": board,
        "literature": [
            "Lustig, Roussanov, Verdelhan (2011) RFS — currency risk factors / carry",
            "Menkhoff et al. (2012) JF — carry & global FX volatility",
            "Menkhoff et al. (2012) JFE — currency momentum",
            "Dahlquist & Hasseltoft (2020) JFE — economic momentum (needs multi-country CPI/IP)",
            "Caldara & Iacoviello (2022) AER — GPR index",
            "Liu & Zhang (2024) JBF — geopolitical risk and currency returns",
        ],
    }
    (ROOT / "configs" / "quest_macro_factor_literature_selected.json").write_text(json.dumps(sel, indent=2, default=str))

    lines = [
        "# Quest wave: literature-backed macro / geopolitics factors",
        "",
        "**Data:** `approximate_non_ftmo`. **RF:** 0.08 (no hike). **Holdout:** confirm only.",
        "**NOT** an overlay-knob grid on locked equity — new research line.",
        f"**Board:** n={len(board)} soft={soft_n} hard={hard_n} promote={len(promo)}",
        "",
        "## Idea vs tuning",
        "",
        "| Label | Kind | Notes |",
        "|-------|------|-------|",
        "| fred_carry_d1 | **idea** | FRED rate-diff carry (Lustig–Verdelhan sign); fixed min_diff=0.25, month lag=1 |",
        "| locked+vix_gate | **idea** | Menkhoff vol-risk gate; fixed z≥1.0 → scale 0.35 |",
        "| locked+gpr_gate | **idea** | Caldara–Iacoviello / Liu–Zhang GPR risk-off; fixed z≥1.5 → scale 0.35 |",
        "| locked+vix_gpr / fred_carry+vix_gpr | **idea** | Sequential literature gates (not a cool-knob search) |",
        "| baseline_locked_vt | baseline | Official locked tag reference |",
        "",
        "## Results",
        "",
        "| Idea | soft | hard | promote | reason | IS min_mo | HO mo/%pos |",
        "|------|:----:|:----:|:-------:|--------|----------:|-----------|",
    ]
    for b in board:
        ys = b.get("year_stats") or {}
        ho = ys.get("holdout_365d") or {}
        ho_s = (
            f"{ho.get('mean_mo', float('nan'))*100:.2f}%/{ho.get('pct_pos', float('nan'))*100:.0f}%"
            if ho else "n/a"
        )
        lines.append(
            f"| `{b['idea']}` | {b['soft_pass']} | {b['hard_pass']} | {b['promote']} | {b['reason']} | "
            f"{(b['min_mean_mo'] if b['min_mean_mo']==b['min_mean_mo'] else float('nan'))*100:.2f}% | {ho_s} |"
        )
    lines += [
        "",
        "## Datasets used vs required",
        "",
        "See `reports/MACRO_FACTOR_LITERATURE.md`.",
        "",
        f"**Promote?** **{'YES' if promo else 'NO'}** — locked tag unchanged." if not promo else f"**Promote?** YES — {promo[0]['idea']}",
        "",
    ]
    (ROOT / "reports" / "quest_macro_factor_literature.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-8:]), flush=True)


if __name__ == "__main__":
    main()
