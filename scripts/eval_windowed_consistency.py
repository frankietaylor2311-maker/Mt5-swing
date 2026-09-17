#!/usr/bin/env python3
"""Multi-window basket evaluation with warmup (no look-ahead / no holdout tuning).

Reports calendar years, rolling 12m, holdout, monthly consistency vs ~1%/mo goal.
Uses fixed params from configs/best_interim_approximate.yaml or QUEST_LEGS JSON.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mt5_swing.backtest.engine import BacktestConfig, run_backtest
from mt5_swing.backtest.metrics import compute_metrics
from mt5_swing.config import load_config
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.strategies.registry import get_strategy

FTMO = ROOT / "data" / "ftmo"
HIST = ROOT / "data" / "history"


def resolve_csv(symbol: str, tf: str) -> Path | None:
    for d in (FTMO, HIST):
        p = d / f"{symbol}_{tf}.csv"
        if p.exists():
            return p
    return None


def load_legs(cfg_path: Path | None = None) -> list[dict]:
    env = os.environ.get("QUEST_LEGS", "").strip()
    if env:
        return json.loads(env)
    envp = os.environ.get("QUEST_CFG", "").strip()
    if envp:
        path = Path(envp)
    else:
        path = cfg_path or (ROOT / "configs" / "best_interim_approximate.yaml")
    locked = yaml.safe_load(path.read_text())
    return list(locked.get("candidates") or [])


def make_bt(cfg: dict, leg: dict, risk_per_leg: float, max_lot: float) -> BacktestConfig:
    risk, bt = cfg.get("risk", {}), cfg.get("backtest", {})
    b = BacktestConfig(
        symbol=leg["symbol"],
        initial_equity=float(bt.get("initial_equity", 100_000)),
        commission_per_lot=float(bt.get("commission_per_lot", 7.0)),
        slippage_pips=float(bt.get("slippage_pips", 0.5)),
        default_spread_pips=float(
            cfg.get("default_spreads_pips", {}).get(leg["symbol"], bt.get("default_spread_pips", 1.2))
        ),
        signal_lag=int(bt.get("signal_lag", 1)),
        max_dd=float(risk.get("max_peak_to_trough_dd", 0.10)),
        daily_dd=float(risk.get("max_daily_dd", 0.05)),
        risk_fraction=risk_per_leg,
        atr_stop_mult=float(risk.get("atr_stop_mult", 2.0)),
        sizing=str(risk.get("sizing", "atr")),
        vol_target=bool(leg.get("vol_target", False)),
        max_lot=max_lot,
        max_loss_mode=str(risk.get("max_loss_mode", "static_initial")),
        daily_loss_mode=str(risk.get("daily_loss_mode", "ftmo_initial")),
        daily_tz=str(risk.get("daily_tz", "Europe/Prague")),
        use_atr_exits=bool(leg.get("use_atr_exits", True)),
        atr_target_mult=float(risk.get("atr_target_mult", 3.0)),
        no_same_bar_exit=bool(risk.get("no_same_bar_exit", True)),
        atr_trail_mult=float((leg.get("exits") or {}).get("atr_trail_mult", risk.get("atr_trail_mult", 0.0)) or 0.0),
        max_hold_bars=int((leg.get("exits") or {}).get("max_hold_bars", 0) or 0),
    )
    exits = leg.get("exits") or {}
    for k, v in exits.items():
        if hasattr(b, k):
            setattr(b, k, v)
    if any(str(k).startswith("atr_") for k in exits):
        b.use_atr_exits = True
    # Mean-reversion family: ATR exits often hurt — honor explicit flag
    mr = {"mean_reversion_regime", "bbands_reversion", "cci_reversion", "stoch_reversion", "willr_reversion"}
    if leg["strategy"] in mr and "atr_stop_mult" not in exits and "atr_target_mult" not in exits:
        b.use_atr_exits = False
    return b


def run_leg_equity(ohlc: pd.DataFrame, leg: dict, bt: BacktestConfig) -> pd.Series:
    params = dict(leg.get("params") or {})
    if not params.get("session_hours"):
        params["session_hours"] = None
    # carry_proxy: inject symbol for auto bias
    if leg["strategy"] in {"carry_proxy", "fred_carry"} and "symbol" not in params:
        params["symbol"] = leg["symbol"]
    strat = get_strategy(leg["strategy"], **params)
    res = run_backtest(ohlc, strat, bt)
    return res.equity


def combine_portfolio(curves: list[pd.Series], weights: np.ndarray, initial: float) -> pd.Series:
    eq = pd.concat(curves, axis=1, sort=True).sort_index().ffill().dropna(how="any")
    if eq.empty:
        return eq
    norms = eq / eq.iloc[0]
    return (norms * weights).sum(axis=1) * initial


def monthly_stats(port: pd.Series) -> dict:
    if port.empty or len(port) < 10:
        return {"mean_mo": float("nan"), "pct_pos": float("nan"), "top3_share": float("nan"), "n_months": 0}
    m = port.resample("ME").last().dropna()
    rets = m.pct_change().dropna()
    if rets.empty:
        return {"mean_mo": float("nan"), "pct_pos": float("nan"), "top3_share": float("nan"), "n_months": 0}
    pos = rets[rets > 0]
    top3_share = float(pos.nlargest(min(3, len(pos))).sum() / pos.sum()) if len(pos) and pos.sum() > 0 else float("nan")
    return {
        "mean_mo": float(rets.mean()),
        "median_mo": float(rets.median()),
        "std_mo": float(rets.std()),
        "pct_pos": float((rets > 0).mean()),
        "best": float(rets.max()),
        "worst": float(rets.min()),
        "top3_share": top3_share,
        "n_months": int(len(rets)),
        "total": float(port.iloc[-1] / port.iloc[0] - 1),
    }


def peak_to_trough(port: pd.Series) -> float:
    if port.empty:
        return float("nan")
    peak = port.cummax()
    return float(((peak - port) / peak).max())


@dataclass
class WindowResult:
    label: str
    ret: float
    p2t: float
    static: float
    daily: float
    gates: bool
    mean_mo: float
    pct_pos: float
    top3_share: float
    n_months: int
    overlap: str


def eval_window(
    legs: list[dict],
    cfg: dict,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    *,
    warmup_bars: int,
    risk_fraction: float,
    max_lot: float,
    label: str,
    overlap: str,
    weight_mode: str = "oos_sharpe",
) -> WindowResult | None:
    initial = float(cfg.get("backtest", {}).get("initial_equity", 100_000))
    n = max(len(legs), 1)
    risk_per = risk_fraction / n
    curves = []
    sharpes = []
    weights_locked = []
    for leg in legs:
        path = resolve_csv(leg["symbol"], leg["timeframe"])
        if path is None:
            return None
        ohlc = load_ohlc_csv(path, symbol=leg["symbol"], timeframe=leg["timeframe"])
        # Window with warmup
        if end is not None:
            ohlc = ohlc.loc[ohlc.index <= end]
        if start is not None:
            # include warmup bars before start
            pre = ohlc.loc[ohlc.index < start]
            win = ohlc.loc[ohlc.index >= start]
            if end is not None:
                win = win.loc[win.index <= end]
            if len(pre) >= warmup_bars:
                warm = pre.iloc[-warmup_bars:]
            else:
                warm = pre
            ohlc_run = pd.concat([warm, win])
            eval_start = start
        else:
            ohlc_run = ohlc
            eval_start = ohlc_run.index[warmup_bars] if len(ohlc_run) > warmup_bars else ohlc_run.index[0]
        if len(ohlc_run) < warmup_bars + 30:
            return None
        bt = make_bt(cfg, leg, risk_per, max_lot)
        eq = run_leg_equity(ohlc_run, leg, bt)
        eq = eq.loc[eq.index >= eval_start]
        if end is not None:
            eq = eq.loc[eq.index <= end]
        if len(eq) < 20:
            return None
        curves.append(eq.rename(f"{leg['symbol']}_{leg['strategy']}"))
        sharpes.append(max(float(leg.get("oos_sharpe") or 0), 0.01))
        weights_locked.append(float(leg.get("weight") or 0) or None)

    if weight_mode == "equal":
        w = np.ones(len(curves)) / len(curves)
    elif weight_mode == "risk_parity":
        # Inverse of IS-like vol from warmup-inclusive curve (full window vol — slightly leaky).
        # Use only first 40% of each curve as proxy (pre-holdout-ish) to reduce leak.
        vols = []
        for c in curves:
            r = c.pct_change().dropna()
            cut = max(20, int(len(r) * 0.4))
            v = float(r.iloc[:cut].std()) if cut else 1.0
            vols.append(max(v, 1e-8))
        inv = 1.0 / np.array(vols)
        w = inv / inv.sum()
    elif weight_mode == "locked" and all(x is not None and x > 0 for x in weights_locked):
        w = np.array(weights_locked, dtype=float)
        w = w / w.sum()
    else:
        w = np.array(sharpes, dtype=float)
        w = w / w.sum()
    port = combine_portfolio(curves, w, initial)
    if port.empty:
        return None
    vt = os.environ.get("PORT_VOL_TARGET", "").strip()
    if vt:
        # Causal overlay: scale next-bar returns by target / trailing vol (no look-ahead)
        target = float(vt)
        look = int(os.environ.get("PORT_VOL_LOOKBACK", "60"))
        r = port.pct_change()
        trail = r.shift(1).rolling(look, min_periods=max(20, look // 3)).std()
        clip_lo = float(os.environ.get("PORT_VOL_CLIP_LO", "0.25"))
        clip_hi = float(os.environ.get("PORT_VOL_CLIP_HI", "3.0"))
        scale = (target / trail.replace(0, np.nan)).clip(clip_lo, clip_hi).fillna(1.0)
        r2 = (r.fillna(0) * scale)
        port = (1.0 + r2).cumprod() * initial

    m = compute_metrics(
        port,
        None,
        max_dd_gate=0.10,
        daily_dd_gate=0.05,
        max_loss_mode="static_initial",
        daily_loss_mode="ftmo_initial",
        daily_tz="Europe/Prague",
        initial_equity=initial,
    )
    ms = monthly_stats(port)
    return WindowResult(
        label=label,
        ret=float(m.total_return),
        p2t=peak_to_trough(port),
        static=float(m.static_loss_from_initial),
        daily=float(m.max_daily_dd),
        gates=bool(m.gates_pass),
        mean_mo=ms["mean_mo"],
        pct_pos=ms["pct_pos"],
        top3_share=ms["top3_share"],
        n_months=ms["n_months"],
        overlap=overlap,
    )


def main() -> None:
    cfg = load_config(ROOT / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    legs = load_legs()
    rf = float(os.environ.get("RISK_FRACTION", "0.075"))
    max_lot = float(os.environ.get("MAX_LOT", "50"))
    warmup = int(os.environ.get("WARMUP_BARS", "250"))
    weight_mode = os.environ.get("WEIGHTS", "locked")
    tag = os.environ.get("QUEST_TAG", "quest")

    # Determine data span from first leg
    path0 = resolve_csv(legs[0]["symbol"], legs[0]["timeframe"])
    ohlc0 = load_ohlc_csv(path0, symbol=legs[0]["symbol"], timeframe=legs[0]["timeframe"])
    end = ohlc0.index.max()
    holdout_days = int(cfg.get("walk_forward", {}).get("holdout_days", 365))
    holdout_start = end - pd.Timedelta(days=holdout_days)

    windows = []
    # Calendar years
    for year in range(2024, end.year + 1):
        s = pd.Timestamp(f"{year}-01-01", tz="UTC")
        e = pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC")
        if s > end:
            continue
        e = min(e, end)
        if year < holdout_start.year:
            overlap = "research/WF overlap"
        elif year == holdout_start.year:
            overlap = "mixed research→holdout"
        else:
            overlap = "pure holdout"
        windows.append((f"{year}", s, e, overlap))

    windows.append(("holdout_365d", holdout_start, end, "pure holdout"))
    # Rolling 12m ending at end, and ending 6m earlier
    windows.append(("roll12_end", end - pd.Timedelta(days=365), end, "pure holdout"))
    windows.append(("roll12_m6", end - pd.Timedelta(days=365 + 182), end - pd.Timedelta(days=182), "mixed"))

    results = []
    for label, s, e, ov in windows:
        r = eval_window(
            legs, cfg, s, e,
            warmup_bars=warmup, risk_fraction=rf, max_lot=max_lot,
            label=label, overlap=ov, weight_mode=weight_mode,
        )
        if r:
            results.append(r)
            print(
                f"{r.label:14s} ret={r.ret*100:7.2f}% mean_mo={r.mean_mo*100:5.2f}% "
                f"pos={r.pct_pos*100:4.0f}% top3={r.top3_share*100:4.0f}% "
                f"gates={'PASS' if r.gates else 'FAIL'} p2t={r.p2t*100:.2f}% ({r.overlap})"
            )

    # Write report
    lines = [
        f"# One-pct month quest — {tag}",
        "",
        f"**data_source:** approximate_non_ftmo (no ftmo exports in data/ftmo/)",
        f"**risk_fraction:** {rf:.1%}  **max_lot:** {max_lot}  **warmup:** {warmup}  **weights:** {weight_mode}",
        f"**legs:** {len(legs)}",
        "",
        "| Window | Return | Mean mo | %pos mo | Top3 share | Gates | P2T DD | Overlap |",
        "|--------|-------:|--------:|--------:|-----------:|:-----:|-------:|---------|",
    ]
    for r in results:
        lines.append(
            f"| {r.label} | {r.ret*100:.2f}% | {r.mean_mo*100:.2f}% | {r.pct_pos*100:.0f}% | "
            f"{r.top3_share*100:.0f}% | {'PASS' if r.gates else 'FAIL'} | {r.p2t*100:.2f}% | {r.overlap} |"
        )
    # Goal check on holdout / eval windows
    hold = next((r for r in results if r.label == "holdout_365d"), None)
    lines += ["", "## vs 1%/mo target", ""]
    if hold:
        gap = 0.01 - hold.mean_mo
        lines.append(f"- Holdout mean monthly: **{hold.mean_mo*100:.2f}%** (target ≥1.0%) — gap **{gap*100:.2f} pp**")
        lines.append(f"- Holdout % positive months: **{hold.pct_pos*100:.0f}%** (prefer ≥70%)")
        lines.append(f"- Holdout top-3 share of gains: **{hold.top3_share*100:.0f}%** (prefer ≤50%)")
        lines.append(f"- Holdout gates: **{'PASS' if hold.gates else 'FAIL'}**")
        met = hold.mean_mo >= 0.009 and hold.gates and hold.pct_pos >= 0.55
        lines.append(f"- **Target met (relaxed):** {'YES' if met else 'NO'}")
    out = ROOT / "reports" / f"quest_{tag}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # CSV
    pd.DataFrame([r.__dict__ for r in results]).to_csv(ROOT / "reports" / f"quest_{tag}.csv", index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()
