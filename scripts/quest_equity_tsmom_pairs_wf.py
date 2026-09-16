#!/usr/bin/env python3
"""Wave: equity-curve TSMOM + monthly-budget VT + pairs residual sleeve.

NEW STRUCTURE (not another same-board basket grid):
1. Causal TSMOM / trend overlay on equity of the diversified locked sleeve,
   then vol-target to a ~1%/mo return budget (hi≤1 ⇒ no RF hike).
2. Pairs/spread residual MR across cointegrated FX with hard risk caps;
   monthly distribution focus (mean_mo / %pos / top3).
3. Optional yfinance D1 refresh for longer history (H4 Yahoo still ~730d).
4. Score by min(2024, 2025_IS mean_mo), max(top3), %pos — holdout confirmation
   only. Promote iff holdout ALSO ≥1% mean and ≥70% pos AND each of 2024/2025/2026
   clears the same (confirm path, not used in selection score).

Locked fx4plus_gbpcad_d1_voltarget_0025 stays official unless promote fires.
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
from mt5_swing.data.download import download_symbol_timeframe
from mt5_swing.data.loader import load_ohlc_csv, save_ohlc_csv
from mt5_swing.portfolio.equity_tsmom import (
    apply_daily_budget_vt,
    apply_equity_tsmom,
    apply_monthly_budget_vt,
    daily_vol_for_monthly_budget,
    holdout_clears_promote,
    score_is_windows,
    years_each_clear,
)
from mt5_swing.portfolio.overlays import INITIAL, apply_vol_target, combine_weighted
from mt5_swing.portfolio.pairs_residual import (
    DEFAULT_PAIRS,
    backtest_residual_equity,
    combine_sleeve_curves,
    engle_granger_adf_stat,
    hedge_ratio_ols,
    per_leg_risk,
    residual_log,
    rolling_zscore,
)
from mt5_swing.portfolio.smooth_select import WindowStats
from mt5_swing.strategies.registry import get_strategy

WARMUP = 250
RF = 0.08
MAX_LOT = float(os.environ.get("MAX_LOT", "50"))
LOCKED_TAG = "fx4plus_gbpcad_d1_voltarget_0025"
MIN_PCT_POS = 0.70
MAX_TOP3_HARD = 0.55
MAX_TOP3_SOFT = 0.70
MAX_P2T = 0.085
TARGET_MO = 0.01

HIST = ROOT / "data" / "history"
REPORTS = ROOT / "reports"


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
    out = []
    out.append(("2024", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31 23:59:59", tz="UTC")))
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
    out.append(("roll12_m6", end - pd.Timedelta(days=365 + 182), end - pd.Timedelta(days=182), "mixed"))
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
    strat = get_strategy(leg["strategy"], **params)
    res = run_backtest(ohlc_run, strat, bt)
    eq = res.equity.loc[(res.equity.index >= start) & (res.equity.index <= end)]
    n_trades = 0
    if res.trades is not None and len(res.trades):
        tr = res.trades
        if isinstance(tr, pd.DataFrame) and "exit_time" in tr.columns:
            n_trades = int(((tr["exit_time"] >= start) & (tr["exit_time"] <= end)).sum())
        else:
            n_trades = len(tr)
    name = f"{leg['symbol']}_{leg['timeframe']}_{leg['strategy']}"
    return eq.rename(name), n_trades


def cached_leg(leg, cfg, start, end, risk_per):
    key = (
        leg["symbol"],
        leg["timeframe"],
        leg["strategy"],
        str(start),
        str(end),
        round(risk_per, 6),
        json.dumps(leg.get("params"), sort_keys=True),
        str(leg.get("exits")),
    )
    if key not in _CACHE:
        _CACHE[key] = window_leg(leg, cfg, start, end, risk_per)
    return _CACHE[key]


def pack_legs(legs, cfg, start, end, risk_per):
    curves, trades = [], 0
    for leg in legs:
        eq, nt = cached_leg(leg, cfg, start, end, risk_per)
        if eq.empty:
            return [], 0
        curves.append(eq)
        trades += nt
    return curves, trades


def weights_for(legs: list[dict], mode: str = "oos_sharpe") -> np.ndarray:
    if mode == "equal":
        w = np.ones(len(legs), dtype=float)
    else:
        w = np.array([max(float(l.get("oos_sharpe") or 0.01), 0.01) for l in legs], dtype=float)
    return w / w.sum()


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
    trades: int
    role: str = ""  # is_select | confirm | holdout


def eval_port(port: pd.Series, label: str, trades: int, idea: str, params: str, role: str) -> EvalRow | None:
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
        trades=trades,
        role=role,
    )


def to_stats(row: EvalRow) -> WindowStats:
    return WindowStats(
        mean_mo=row.mean_mo,
        pct_pos=row.pct_pos,
        top3=row.top3 if row.top3 == row.top3 else 1.0,
        gates=row.gates,
        p2t=row.p2t,
    )


def locked_base_port(legs, cfg, start, end, *, vt: float | None = 0.0025) -> tuple[pd.Series, int]:
    risk_per = RF / max(len(legs), 1)
    curves, tr = pack_legs(legs, cfg, start, end, risk_per)
    if not curves:
        return pd.Series(dtype=float), 0
    w = weights_for(legs, "oos_sharpe")
    port = combine_weighted(curves, w)
    if vt is not None:
        port = apply_vol_target(port, float(vt))
    return port, tr


def apply_overlay_stack(port: pd.Series, stack: str, **kw) -> pd.Series:
    """Named overlay stacks (all causal, hi≤1 unless stack says otherwise)."""
    if port is None or port.empty:
        return port if port is not None else pd.Series(dtype=float)
    p = port
    if stack == "none":
        return p
    if stack == "tsmom":
        return apply_equity_tsmom(
            p,
            lookback=int(kw.get("lookback", 60)),
            neg_scale=float(kw.get("neg_scale", 0.25)),
            pos_scale=float(kw.get("pos_scale", 1.0)),
            flat_band=float(kw.get("flat_band", 0.0)),
            hi=1.0,
        )
    if stack == "budget_mo":
        return apply_monthly_budget_vt(
            p,
            target_mo=float(kw.get("target_mo", TARGET_MO)),
            lookback_months=int(kw.get("lookback_months", 6)),
            assumed_sharpe=float(kw.get("assumed_sharpe", 1.0)),
            hi=1.0,
        )
    if stack == "budget_daily":
        return apply_daily_budget_vt(
            p,
            target_mo=float(kw.get("target_mo", TARGET_MO)),
            assumed_sharpe=float(kw.get("assumed_sharpe", 1.2)),
            look=int(kw.get("look", 60)),
            hi=1.0,
        )
    if stack == "tsmom_then_budget_mo":
        p = apply_equity_tsmom(
            p,
            lookback=int(kw.get("lookback", 60)),
            neg_scale=float(kw.get("neg_scale", 0.25)),
            pos_scale=1.0,
            flat_band=float(kw.get("flat_band", 0.0)),
            hi=1.0,
        )
        return apply_monthly_budget_vt(
            p,
            target_mo=float(kw.get("target_mo", TARGET_MO)),
            lookback_months=int(kw.get("lookback_months", 6)),
            assumed_sharpe=float(kw.get("assumed_sharpe", 1.0)),
            hi=1.0,
        )
    if stack == "tsmom_then_budget_daily":
        p = apply_equity_tsmom(
            p,
            lookback=int(kw.get("lookback", 60)),
            neg_scale=float(kw.get("neg_scale", 0.25)),
            pos_scale=1.0,
            flat_band=float(kw.get("flat_band", 0.0)),
            hi=1.0,
        )
        return apply_daily_budget_vt(
            p,
            target_mo=float(kw.get("target_mo", TARGET_MO)),
            assumed_sharpe=float(kw.get("assumed_sharpe", 1.2)),
            look=int(kw.get("look", 60)),
            hi=1.0,
        )
    if stack == "budget_mo_then_tsmom":
        p = apply_monthly_budget_vt(
            p,
            target_mo=float(kw.get("target_mo", TARGET_MO)),
            lookback_months=int(kw.get("lookback_months", 6)),
            assumed_sharpe=float(kw.get("assumed_sharpe", 1.0)),
            hi=1.0,
        )
        return apply_equity_tsmom(
            p,
            lookback=int(kw.get("lookback", 60)),
            neg_scale=float(kw.get("neg_scale", 0.25)),
            pos_scale=1.0,
            hi=1.0,
        )
    raise ValueError(f"unknown stack {stack}")


def expand_d1_history(symbols: list[str], *, years_d1: int = 15, min_bars: int = 3500) -> list[str]:
    """Refresh/extend D1 via yfinance where possible. Returns notes."""
    notes = []
    for sym in symbols:
        existing = HIST / f"{sym}_D1.csv"
        if existing.exists():
            try:
                n = sum(1 for _ in open(existing)) - 1
                if n >= min_bars:
                    notes.append(f"{sym}_D1:keep:{n}")
                    print(f"  keep {sym}_D1 ({n} bars >= {min_bars})", flush=True)
                    continue
            except OSError:
                pass
        try:
            df = download_symbol_timeframe(sym, "D1", years_d1=years_d1)
            path = HIST / f"{sym}_D1.csv"
            save_ohlc_csv(df, path)
            meta = path.with_suffix(".meta.json")
            meta.write_text(
                json.dumps(
                    {
                        "data_source": "approximate_non_ftmo",
                        "symbol": sym,
                        "timeframe": "D1",
                        "bars": len(df),
                        "start": str(df.index.min()),
                        "end": str(df.index.max()),
                        "provider": "yfinance",
                        "years_d1": years_d1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            notes.append(f"{sym}_D1:{len(df)} [{df.index.min().date()}→{df.index.max().date()}]")
            print(f"  refreshed {notes[-1]}", flush=True)
        except Exception as exc:  # noqa: BLE001 — research continuity
            notes.append(f"{sym}_D1:FAIL:{exc}")
            print(f"  FAIL {sym}_D1: {exc}", flush=True)
    return notes


def load_close(sym: str, tf: str) -> pd.Series | None:
    path = ewc.resolve_csv(sym, tf)
    if path is None:
        return None
    df = load_ohlc_csv(path, symbol=sym, timeframe=tf)
    return df["close"].astype(float)


def fit_pair_is(
    a: str,
    b: str,
    tf: str,
    is_start: pd.Timestamp,
    is_end: pd.Timestamp,
    *,
    win: int,
) -> dict | None:
    ca, cb = load_close(a, tf), load_close(b, tf)
    if ca is None or cb is None:
        return None
    both = pd.concat([ca.rename("a"), cb.rename("b")], axis=1, sort=True).dropna()
    both = both.loc[(both.index >= is_start) & (both.index <= is_end)]
    if len(both) < max(80, win + 20):
        return None
    beta = hedge_ratio_ols(both["a"], both["b"])
    res = residual_log(both["a"], both["b"], beta)
    adf = engle_granger_adf_stat(res)
    return {"a": a, "b": b, "tf": tf, "beta": beta, "adf": adf, "win": win, "n_is": len(both)}


def pair_equity_window(
    fit: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    entry: float,
    risk_frac: float,
    warmup_bars: int = 120,
) -> pd.Series:
    ca, cb = load_close(fit["a"], fit["tf"]), load_close(fit["b"], fit["tf"])
    if ca is None or cb is None:
        return pd.Series(dtype=float)
    both = pd.concat([ca.rename("a"), cb.rename("b")], axis=1, sort=True).dropna()
    # Warmup before start for rolling z
    pre = both.loc[both.index < start].iloc[-warmup_bars:]
    win = both.loc[(both.index >= start) & (both.index <= end)]
    full = pd.concat([pre, win])
    if len(full) < fit["win"] + 10 or len(win) < 20:
        return pd.Series(dtype=float)
    res = residual_log(full["a"], full["b"], fit["beta"])
    z = rolling_zscore(res, fit["win"])
    eq = backtest_residual_equity(z, entry=entry, risk_frac=risk_frac)
    return eq.loc[(eq.index >= start) & (eq.index <= end)]


def main() -> None:
    src = data_source()
    REPORTS.mkdir(exist_ok=True)
    print(f"data_source={src} RF={RF} locked={LOCKED_TAG}", flush=True)
    print(
        "Score: max min(IS year mean_mo) s.t. %pos≥70% top3 soft≤70%; "
        "promote only if HO≥1% & ≥70% pos (confirm only). No RF hike.",
        flush=True,
    )
    for tm in (0.008, 0.01, 0.012):
        d = daily_vol_for_monthly_budget(tm, assumed_sharpe=1.2)
        print(f"  budget {tm*100:.1f}%/mo → daily VT≈{d:.5f} @Sharpe1.2", flush=True)

    # --- 3. Expand D1 history for pairs symbols (never overwrite locked D1 legs) ---
    lock_preview = locked_legs()
    locked_d1 = {l["symbol"] for l in lock_preview if str(l["timeframe"]).upper() == "D1"}
    pair_syms = sorted(({p.a for p in DEFAULT_PAIRS} | {p.b for p in DEFAULT_PAIRS}) - locked_d1)
    print(f"=== Expand D1 via yfinance (pairs symbols; skip locked D1={sorted(locked_d1)}) ===", flush=True)
    hist_notes = expand_d1_history(pair_syms, years_d1=15)
    if locked_d1:
        hist_notes.append(f"skipped_locked_d1:{','.join(sorted(locked_d1))}")
    # H4 Yahoo hard limit
    print(
        "NOTE: H4 via Yahoo 1h remains ~730d (≈ late-2023 start); cannot extend "
        "intraday beyond provider cap. D1 expanded where fetch succeeded.",
        flush=True,
    )

    cfg = load_config(ROOT / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    lock = locked_legs()
    path0 = ewc.resolve_csv(lock[0]["symbol"], lock[0]["timeframe"])
    end = load_ohlc_csv(path0, symbol=lock[0]["symbol"], timeframe=lock[0]["timeframe"]).index.max()
    holdout_start = end - pd.Timedelta(days=365)
    is_wins = is_windows(end, holdout_start)
    conf_wins = confirm_windows(end, holdout_start)
    # IS span for hedge fit = union of IS windows
    is_fit_start = is_wins[0][1]
    is_fit_end = is_wins[-1][2]
    print(f"end={end} holdout_start={holdout_start}", flush=True)
    print(f"IS windows: {[w[0] for w in is_wins]} fit=[{is_fit_start.date()}→{is_fit_end.date()}]", flush=True)

    rows: list[EvalRow] = []

    # --- Baseline locked ---
    print("=== Baseline locked vt0025 (confirm) ===", flush=True)
    for label, s, e, _ov in conf_wins:
        port, tr = locked_base_port(lock, cfg, s, e, vt=0.0025)
        row = eval_port(port, label, tr, "baseline_vt0025", "locked", "confirm")
        if row:
            rows.append(row)
            print(
                f"  {label:14s} mo={row.mean_mo*100:5.2f}% pos={row.pct_pos*100:4.0f}% "
                f"top3={row.top3*100:4.0f}% gates={row.gates}",
                flush=True,
            )

    # --- 1. Equity TSMOM + monthly budget grid on locked sleeve ---
    print("=== Equity TSMOM + monthly-budget VT grid (IS select) ===", flush=True)
    # Base without port VT so overlays own the budget (still RF=8% per-leg)
    stacks = [
        ("tsmom", {"lookback": 42, "neg_scale": 0.35}),
        ("tsmom", {"lookback": 60, "neg_scale": 0.25}),
        ("tsmom", {"lookback": 90, "neg_scale": 0.20}),
        ("tsmom", {"lookback": 120, "neg_scale": 0.25, "flat_band": 0.01}),
        ("budget_mo", {"target_mo": 0.01, "lookback_months": 4, "assumed_sharpe": 1.0}),
        ("budget_mo", {"target_mo": 0.01, "lookback_months": 6, "assumed_sharpe": 1.0}),
        ("budget_mo", {"target_mo": 0.012, "lookback_months": 6, "assumed_sharpe": 1.2}),
        ("budget_daily", {"target_mo": 0.01, "assumed_sharpe": 1.2}),
        ("budget_daily", {"target_mo": 0.012, "assumed_sharpe": 1.0}),
        ("tsmom_then_budget_mo", {"lookback": 60, "neg_scale": 0.25, "target_mo": 0.01, "lookback_months": 6}),
        ("tsmom_then_budget_mo", {"lookback": 90, "neg_scale": 0.30, "target_mo": 0.01, "lookback_months": 4}),
        ("tsmom_then_budget_mo", {"lookback": 60, "neg_scale": 0.20, "target_mo": 0.012, "lookback_months": 6}),
        ("tsmom_then_budget_daily", {"lookback": 60, "neg_scale": 0.25, "target_mo": 0.01}),
        ("tsmom_then_budget_daily", {"lookback": 90, "neg_scale": 0.35, "target_mo": 0.01}),
        ("budget_mo_then_tsmom", {"lookback": 60, "neg_scale": 0.25, "target_mo": 0.01}),
        # Also on top of locked VT base
    ]
    # Variants: base_vt in {None, 0.0025}
    overlay_board = []
    for base_vt in (None, 0.0025):
        for stack, kw in stacks:
            idea = f"eq_{stack}"
            params = json.dumps({"base_vt": base_vt, **kw}, sort_keys=True)
            is_stats: list[WindowStats] = []
            is_rows: list[EvalRow] = []
            ok = True
            for label, s, e in is_wins:
                base, tr = locked_base_port(lock, cfg, s, e, vt=base_vt)
                if base.empty:
                    ok = False
                    break
                port = apply_overlay_stack(base, stack, **kw)
                row = eval_port(port, label, tr, idea, params, "is_select")
                if row is None:
                    ok = False
                    break
                is_rows.append(row)
                is_stats.append(to_stats(row))
            if not ok:
                continue
            sc = score_is_windows(
                is_stats,
                min_pct_pos=MIN_PCT_POS,
                max_top3_hard=MAX_TOP3_HARD,
                max_top3_soft=MAX_TOP3_SOFT,
                max_p2t=MAX_P2T,
            )
            overlay_board.append(
                {
                    "idea": idea,
                    "params": params,
                    "score": sc.score,
                    "min_mean_mo": sc.min_mean_mo,
                    "max_top3": sc.max_top3,
                    "min_pct_pos": sc.min_pct_pos,
                    "hard_pass": sc.hard_pass,
                    "soft_pass": sc.soft_pass,
                    "stack": stack,
                    "kw": kw,
                    "base_vt": base_vt,
                    "is_rows": is_rows,
                }
            )
            rows.extend(is_rows)
            tag = "HARD" if sc.hard_pass else ("SOFT" if sc.soft_pass else "fail")
            print(
                f"  {idea:28s} base_vt={base_vt} score={sc.score if sc.score==sc.score else float('nan'):.4f} "
                f"min_mo={sc.min_mean_mo*100 if sc.min_mean_mo==sc.min_mean_mo else float('nan'):.2f}% "
                f"max_t3={sc.max_top3*100 if sc.max_top3==sc.max_top3 else float('nan'):.0f}% "
                f"min_pos={sc.min_pct_pos*100 if sc.min_pct_pos==sc.min_pct_pos else float('nan'):.0f}% [{tag}]",
                flush=True,
            )

    # --- 2. Pairs residual sleeve ---
    print("=== Pairs residual MR (D1, IS hedge, hard risk caps) ===", flush=True)
    wins_grid = [40, 60, 90]
    entry_grid = [1.5, 1.75, 2.0, 2.25]
    fits = []
    for ps in DEFAULT_PAIRS:
        for win in wins_grid:
            fit = fit_pair_is(ps.a, ps.b, "D1", is_fit_start, is_fit_end, win=win)
            if fit is None:
                continue
            # Prefer cointegrated-ish (ADF < -2.5 rough)
            fits.append(fit)
            print(
                f"  fit {ps.a}/{ps.b} win={win} beta={fit['beta']:.3f} adf={fit['adf']:.2f} n={fit['n_is']}",
                flush=True,
            )

    # Rank pairs by ADF (more negative better), unique by (a,b) taking best win
    fits_sorted = sorted(fits, key=lambda f: f["adf"])
    best_by_pair: dict[tuple, dict] = {}
    for f in fits_sorted:
        key = (f["a"], f["b"])
        if key not in best_by_pair:
            best_by_pair[key] = f
    ranked_pairs = list(best_by_pair.values())
    print(f"unique pairs ranked by ADF: {len(ranked_pairs)}", flush=True)

    pairs_board = []
    for n_pairs in (2, 3, 4):
        chosen = ranked_pairs[:n_pairs]
        if len(chosen) < n_pairs:
            continue
        risk_per = per_leg_risk(RF, n_pairs, legs_per_pair=1)  # synthetic residual book
        # Cap residual risk further for monthly focus
        risk_per = min(risk_per, 0.01)
        for entry in entry_grid:
            for sleeve_budget in (0.25, 0.40, 0.55, 1.0):
                idea = f"pairs_n{n_pairs}"
                params = json.dumps(
                    {
                        "pairs": [f"{c['a']}/{c['b']}:w{c['win']}" for c in chosen],
                        "entry": entry,
                        "risk_frac": risk_per,
                        "sleeve_budget": sleeve_budget,
                        "betas": {f"{c['a']}/{c['b']}": c["beta"] for c in chosen},
                    },
                    sort_keys=True,
                )
                is_stats = []
                is_rows = []
                ok = True
                for label, s, e in is_wins:
                    curves = []
                    for c in chosen:
                        eq = pair_equity_window(c, s, e, entry=entry, risk_frac=risk_per)
                        if eq.empty:
                            ok = False
                            break
                        curves.append(eq)
                    if not ok or not curves:
                        ok = False
                        break
                    port = combine_sleeve_curves(curves, max_gross=float(sleeve_budget))
                    row = eval_port(port, label, 0, idea, params, "is_select")
                    if row is None:
                        ok = False
                        break
                    is_rows.append(row)
                    is_stats.append(to_stats(row))
                if not ok:
                    continue
                sc = score_is_windows(
                    is_stats,
                    min_pct_pos=MIN_PCT_POS,
                    max_top3_hard=MAX_TOP3_HARD,
                    max_top3_soft=MAX_TOP3_SOFT,
                    max_p2t=MAX_P2T,
                )
                pairs_board.append(
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
                        "entry": entry,
                        "risk_frac": risk_per,
                        "sleeve_budget": sleeve_budget,
                        "is_rows": is_rows,
                    }
                )
                rows.extend(is_rows)

    soft_pairs = [p for p in pairs_board if p["soft_pass"]]
    print(
        f"pairs candidates: {len(pairs_board)} soft_pass={len(soft_pairs)} "
        f"hard_pass={sum(1 for p in pairs_board if p['hard_pass'])}",
        flush=True,
    )
    for p in sorted(pairs_board, key=lambda x: -x["score"])[:8]:
        print(
            f"  {p['idea']} entry={p['entry']} bud={p['sleeve_budget']} "
            f"score={p['score']:.4f} min_mo={p['min_mean_mo']*100:.2f}% "
            f"max_t3={p['max_top3']*100:.0f}% soft={p['soft_pass']}",
            flush=True,
        )

    # --- Blend: locked sleeve (with optional overlay) + pairs sleeve ---
    print("=== Blend locked(+overlay) + pairs sleeve (IS) ===", flush=True)
    blend_board = []
    # Top soft overlays (or best by raw min_mo even if fail) + top pairs
    top_ov = sorted(overlay_board, key=lambda x: (-x["soft_pass"], -x["score"], -x["min_mean_mo"]))[:6]
    top_pr = sorted(pairs_board, key=lambda x: (-x["soft_pass"], -x["score"], -x["min_mean_mo"]))[:6]
    # Always include plain locked base as overlay "identity"
    identity = {
        "idea": "eq_none",
        "params": '{"base_vt": 0.0025}',
        "stack": "none",
        "kw": {},
        "base_vt": 0.0025,
        "score": float("-inf"),
        "soft_pass": False,
        "hard_pass": False,
        "min_mean_mo": float("nan"),
        "max_top3": float("nan"),
        "min_pct_pos": float("nan"),
    }
    blend_ovs = [identity] + top_ov
    for ov in blend_ovs:
        for pr in top_pr:
            for lock_w in (0.70, 0.80, 0.90):
                pairs_w = 1.0 - lock_w
                idea = f"blend_{ov['idea']}_{pr['idea']}"
                params = json.dumps(
                    {
                        "overlay": json.loads(ov["params"]) if isinstance(ov["params"], str) else ov["params"],
                        "pairs": json.loads(pr["params"]),
                        "lock_w": lock_w,
                        "pairs_w": pairs_w,
                    },
                    sort_keys=True,
                )
                is_stats = []
                is_rows = []
                ok = True
                for label, s, e in is_wins:
                    base, tr = locked_base_port(lock, cfg, s, e, vt=ov.get("base_vt", 0.0025))
                    if base.empty:
                        ok = False
                        break
                    if ov["stack"] != "none":
                        base = apply_overlay_stack(base, ov["stack"], **ov["kw"])
                    curves = []
                    for c in pr["chosen"]:
                        eq = pair_equity_window(c, s, e, entry=pr["entry"], risk_frac=pr["risk_frac"])
                        if eq.empty:
                            ok = False
                            break
                        curves.append(eq)
                    if not ok:
                        break
                    pairs_port = combine_sleeve_curves(curves, max_gross=1.0)
                    # Align and mix with fixed lock_w / pairs_w (sum=1)
                    both = pd.concat(
                        [base.rename("L") / float(base.iloc[0]), pairs_port.rename("P") / float(pairs_port.iloc[0])],
                        axis=1,
                    ).dropna()
                    if len(both) < 30:
                        ok = False
                        break
                    port = (both["L"] * lock_w + both["P"] * pairs_w) * INITIAL
                    row = eval_port(port, label, tr, idea, params, "is_select")
                    if row is None:
                        ok = False
                        break
                    is_rows.append(row)
                    is_stats.append(to_stats(row))
                if not ok:
                    continue
                sc = score_is_windows(
                    is_stats,
                    min_pct_pos=MIN_PCT_POS,
                    max_top3_hard=MAX_TOP3_HARD,
                    max_top3_soft=MAX_TOP3_SOFT,
                    max_p2t=MAX_P2T,
                )
                blend_board.append(
                    {
                        "idea": idea,
                        "params": params,
                        "score": sc.score,
                        "min_mean_mo": sc.min_mean_mo,
                        "max_top3": sc.max_top3,
                        "min_pct_pos": sc.min_pct_pos,
                        "hard_pass": sc.hard_pass,
                        "soft_pass": sc.soft_pass,
                        "ov": ov,
                        "pr": pr,
                        "lock_w": lock_w,
                        "is_rows": is_rows,
                    }
                )
                rows.extend(is_rows)

    print(
        f"blend candidates: {len(blend_board)} soft={sum(1 for b in blend_board if b['soft_pass'])} "
        f"hard={sum(1 for b in blend_board if b['hard_pass'])}",
        flush=True,
    )

    # --- Select best IS soft / hard (holdout NEVER in score) ---
    all_cands = overlay_board + pairs_board + blend_board
    soft = [c for c in all_cands if c["soft_pass"]]
    hard = [c for c in all_cands if c["hard_pass"]]
    soft_sorted = sorted(soft, key=lambda x: (-x["score"], x["max_top3"], -x["min_pct_pos"]))
    hard_sorted = sorted(hard, key=lambda x: (-x["score"], x["max_top3"], -x["min_pct_pos"]))
    # Also track best by raw min_mean_mo among all (for evidence if none soft-pass)
    by_min_mo = sorted(
        [c for c in all_cands if c["min_mean_mo"] == c["min_mean_mo"]],
        key=lambda x: (-x["min_mean_mo"], x["max_top3"]),
    )

    print(
        f"TOTAL candidates={len(all_cands)} soft_pass={len(soft)} hard_pass={len(hard)}",
        flush=True,
    )

    def confirm_candidate(cand: dict) -> dict:
        """Run confirm windows; holdout not used for selection — confirmation only."""
        conf_rows = []
        year_map: dict[str, WindowStats] = {}
        ho_stat = None
        idea = cand["idea"]
        params = cand["params"]
        for label, s, e, _ov in conf_wins:
            port = pd.Series(dtype=float)
            tr = 0
            if idea.startswith("eq_"):
                base, tr = locked_base_port(lock, cfg, s, e, vt=cand.get("base_vt"))
                if not base.empty:
                    port = apply_overlay_stack(base, cand["stack"], **cand["kw"]) if cand["stack"] != "none" else base
            elif idea.startswith("pairs_"):
                curves = []
                for c in cand["chosen"]:
                    eq = pair_equity_window(c, s, e, entry=cand["entry"], risk_frac=cand["risk_frac"])
                    curves.append(eq)
                if curves and all(len(x) for x in curves):
                    port = combine_sleeve_curves(curves, max_gross=float(cand["sleeve_budget"]))
            elif idea.startswith("blend_"):
                ov, pr = cand["ov"], cand["pr"]
                lock_w = cand["lock_w"]
                base, tr = locked_base_port(lock, cfg, s, e, vt=ov.get("base_vt", 0.0025))
                if not base.empty:
                    if ov["stack"] != "none":
                        base = apply_overlay_stack(base, ov["stack"], **ov["kw"])
                    curves = []
                    for c in pr["chosen"]:
                        curves.append(pair_equity_window(c, s, e, entry=pr["entry"], risk_frac=pr["risk_frac"]))
                    if curves and all(len(x) for x in curves):
                        pairs_port = combine_sleeve_curves(curves, max_gross=1.0)
                        both = pd.concat(
                            [base.rename("L") / float(base.iloc[0]), pairs_port.rename("P") / float(pairs_port.iloc[0])],
                            axis=1,
                        ).dropna()
                        if len(both) >= 30:
                            port = (both["L"] * lock_w + both["P"] * (1.0 - lock_w)) * INITIAL
            row = eval_port(port, label, tr, idea, params, "confirm" if label != "holdout_365d" else "holdout")
            if row:
                conf_rows.append(row)
                rows.append(row)
                st = to_stats(row)
                if label in {"2024", "2025", "2026"}:
                    year_map[label] = st
                if label == "holdout_365d":
                    ho_stat = st
                print(
                    f"  CONFIRM {idea[:40]:40s} {label:14s} mo={row.mean_mo*100:5.2f}% "
                    f"pos={row.pct_pos*100:4.0f}% top3={row.top3*100:4.0f}% gates={row.gates}",
                    flush=True,
                )
        promote = False
        reason = []
        if ho_stat is None:
            reason.append("no_holdout")
        else:
            ho_ok = holdout_clears_promote(
                ho_stat, min_mean_mo=TARGET_MO, min_pct_pos=MIN_PCT_POS, max_top3=MAX_TOP3_SOFT, max_p2t=MAX_P2T
            )
            yrs_ok = years_each_clear(year_map, ("2024", "2025", "2026"), min_mean_mo=TARGET_MO, min_pct_pos=MIN_PCT_POS)
            if not cand.get("soft_pass"):
                reason.append("is_soft_fail")
            if not ho_ok:
                reason.append("holdout_fail")
            if not yrs_ok:
                reason.append("years_fail")
            promote = bool(cand.get("soft_pass")) and ho_ok and yrs_ok
        return {
            "cand": cand,
            "conf_rows": conf_rows,
            "year_map": {k: asdict(v) for k, v in year_map.items()},
            "holdout": asdict(ho_stat) if ho_stat else None,
            "promote": promote,
            "reason": reason,
        }

    promote_results = []
    # Confirm up to top soft + best raw min_mo for evidence
    to_confirm = []
    seen = set()
    for c in hard_sorted[:3] + soft_sorted[:5] + by_min_mo[:3]:
        key = (c["idea"], c["params"])
        if key in seen:
            continue
        seen.add(key)
        to_confirm.append(c)

    print(f"=== Confirm {len(to_confirm)} candidates (holdout confirmation only) ===", flush=True)
    for c in to_confirm:
        print(f"-- {c['idea']} score={c['score']:.4f} soft={c['soft_pass']} hard={c['hard_pass']}", flush=True)
        promote_results.append(confirm_candidate(c))

    any_promote = any(r["promote"] for r in promote_results)
    best_soft = soft_sorted[0] if soft_sorted else None
    best_raw = by_min_mo[0] if by_min_mo else None

    # --- Write reports ---
    board_rows = []
    for c in all_cands:
        board_rows.append(
            {
                "idea": c["idea"],
                "params": c["params"],
                "score": c["score"],
                "min_mean_mo": c["min_mean_mo"],
                "max_top3": c["max_top3"],
                "min_pct_pos": c["min_pct_pos"],
                "hard_pass": c["hard_pass"],
                "soft_pass": c["soft_pass"],
            }
        )
    board_df = pd.DataFrame(board_rows).sort_values(
        ["soft_pass", "hard_pass", "score", "min_mean_mo"], ascending=[False, False, False, False]
    )
    board_path = REPORTS / "quest_equity_tsmom_pairs_board.csv"
    board_df.to_csv(board_path, index=False)

    eval_df = pd.DataFrame([asdict(r) for r in rows])
    eval_path = REPORTS / "quest_equity_tsmom_pairs_wf.csv"
    eval_df.to_csv(eval_path, index=False)

    promo_rows = []
    for r in promote_results:
        c = r["cand"]
        ho = r["holdout"] or {}
        promo_rows.append(
            {
                "idea": c["idea"],
                "params": c["params"],
                "is_score": c["score"],
                "is_min_mo": c["min_mean_mo"],
                "is_max_top3": c["max_top3"],
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
    promo_path = REPORTS / "quest_equity_tsmom_pairs_promote.csv"
    promo_df.to_csv(promo_path, index=False)

    selected = {
        "locked_tag": LOCKED_TAG,
        "data_source": src,
        "rf": RF,
        "promote": any_promote,
        "hist_notes": hist_notes,
        "n_candidates": len(all_cands),
        "n_soft": len(soft),
        "n_hard": len(hard),
        "best_soft": None
        if best_soft is None
        else {
            "idea": best_soft["idea"],
            "params": best_soft["params"],
            "score": best_soft["score"],
            "min_mean_mo": best_soft["min_mean_mo"],
            "max_top3": best_soft["max_top3"],
            "min_pct_pos": best_soft["min_pct_pos"],
        },
        "best_raw_min_mo": None
        if best_raw is None
        else {
            "idea": best_raw["idea"],
            "params": best_raw["params"],
            "min_mean_mo": best_raw["min_mean_mo"],
            "max_top3": best_raw["max_top3"],
            "min_pct_pos": best_raw["min_pct_pos"],
            "soft_pass": best_raw["soft_pass"],
        },
        "promote_results": promo_rows,
    }
    (ROOT / "configs" / "quest_equity_tsmom_pairs_selected.json").write_text(
        json.dumps(selected, indent=2, default=str) + "\n", encoding="utf-8"
    )

    # Markdown summary
    lines = [
        "# Equity TSMOM + pairs residual WF",
        "",
        f"**data_source:** `{src}`  ",
        f"**RF:** {RF} (no hike)  ",
        f"**Locked:** `{LOCKED_TAG}` unchanged unless promote.  ",
        f"**Candidates:** {len(all_cands)} | soft_pass={len(soft)} | hard_pass={len(hard)}  ",
        f"**Promote:** **{'YES' if any_promote else 'NO'}**",
        "",
        "## History expand",
        "",
    ]
    for n in hist_notes:
        lines.append(f"- {n}")
    lines += [
        "",
        "H4 Yahoo 1h cap (~730d) unchanged — cannot add more H4 years via yfinance.",
        "",
        "## Best IS soft (selection; holdout excluded)",
        "",
    ]
    if best_soft:
        lines.append(
            f"- `{best_soft['idea']}` score={best_soft['score']:.4f} "
            f"min_mo={best_soft['min_mean_mo']*100:.2f}% max_top3={best_soft['max_top3']*100:.0f}% "
            f"min_pos={best_soft['min_pct_pos']*100:.0f}%"
        )
    else:
        lines.append("- **None** — no IS soft passer (%pos≥70% & top3 soft≤70% on both IS years).")
    lines += ["", "## Best raw IS min(mean_mo) (evidence)", ""]
    if best_raw:
        lines.append(
            f"- `{best_raw['idea']}` min_mo={best_raw['min_mean_mo']*100:.2f}% "
            f"max_top3={best_raw['max_top3']*100:.0f}% min_pos={best_raw['min_pct_pos']*100:.0f}% "
            f"soft={best_raw['soft_pass']}"
        )
    lines += ["", "## Confirmation table", "", "| Idea | 2024 mo/%pos | 2025 | 2026 | HO mo/%pos | Promote | Reason |", "|------|-------------:|-----:|-----:|-----------:|:-------:|--------|"]
    for r in promo_rows:
        def fmt(y):
            mo = r.get(f"y{y}_mo")
            pos = r.get(f"y{y}_pos")
            if mo is None or mo != mo:
                return "—"
            return f"{mo*100:.2f}/{pos*100:.0f}"

        ho = f"{r['ho_mean_mo']*100:.2f}/{r['ho_pct_pos']*100:.0f}" if r.get("ho_mean_mo") == r.get("ho_mean_mo") and r.get("ho_mean_mo") is not None else "—"
        lines.append(
            f"| `{r['idea'][:42]}` | {fmt(2024)} | {fmt(2025)} | {fmt(2026)} | {ho} | "
            f"{'YES' if r['promote'] else 'no'} | {r['reason']} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
    ]
    if any_promote:
        lines.append("At least one candidate cleared IS soft + holdout ≥1% / ≥70% pos + each year — see promote CSV.")
    else:
        lines.append(
            "No promote. Equity-TSMOM / monthly-budget overlays and pairs residual sleeves "
            "did not jointly clear ≥1% mean_mo on **each** of 2024, 2025, 2026, holdout with ≥70% "
            "pos under FTMO gates without look-ahead / RF hike on `approximate_non_ftmo`."
        )
    lines += [
        "",
        f"Artifacts: `{board_path.name}`, `{eval_path.name}`, `{promo_path.name}`, "
        "`configs/quest_equity_tsmom_pairs_selected.json`.",
        "",
    ]
    md_path = REPORTS / "quest_equity_tsmom_pairs_wf.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== DONE ===", flush=True)
    print(f"promote={any_promote} soft={len(soft)} hard={len(hard)} board={board_path}", flush=True)
    if best_raw:
        print(
            f"best_raw_min_mo={best_raw['idea']} {best_raw['min_mean_mo']*100:.2f}% "
            f"soft={best_raw['soft_pass']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
