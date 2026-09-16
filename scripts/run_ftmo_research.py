#!/usr/bin/env python3
"""
FTMO 2-Step Challenge walk-forward research runner (swing H4/D1).

- Prefers data/ftmo/ (MT5 exports). If missing, uses data/history/ interim public
  and labels every row/report as approximate_non_ftmo.
- Param selection uses ONLY pre-holdout bars (last holdout_days excluded).
- Final holdout is confirmation only — never used to pick params.
- Does NOT claim FTMO go-live readiness on approximate_non_ftmo data.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mt5_swing.backtest.engine import BacktestConfig, run_backtest
from mt5_swing.config import load_config
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.optimize import constrained_grid_search
from mt5_swing.strategies.registry import get_strategy, list_strategies
from mt5_swing.validation.walk_forward import WalkForwardConfig, run_walk_forward

FTMO_DIR = ROOT / "data" / "ftmo"
HIST_DIR = ROOT / "data" / "history"
REPORTS = ROOT / "reports"


def data_source_for(path: Path) -> str:
    meta = path.with_suffix(".meta.json")
    if meta.exists():
        try:
            return json.loads(meta.read_text()).get("data_source", "unknown")
        except Exception:
            pass
    if "ftmo" in path.parts:
        return "ftmo_mt5_export"
    if "history" in path.parts:
        return "approximate_non_ftmo"
    return "unknown"


def resolve_csv(symbol: str, tf: str) -> Path | None:
    for d in (FTMO_DIR, HIST_DIR):
        p = d / f"{symbol}_{tf}.csv"
        if p.exists():
            return p
    return None


def bt_cfg(cfg: dict, symbol: str) -> BacktestConfig:
    risk, bt = cfg.get("risk", {}), cfg.get("backtest", {})
    return BacktestConfig(
        symbol=symbol,
        initial_equity=float(bt.get("initial_equity", 100_000)),
        commission_per_lot=float(bt.get("commission_per_lot", 7.0)),
        slippage_pips=float(bt.get("slippage_pips", 0.5)),
        default_spread_pips=float(bt.get("default_spread_pips", 1.2)),
        signal_lag=int(bt.get("signal_lag", 1)),
        max_dd=float(risk.get("max_peak_to_trough_dd", 0.10)),
        daily_dd=float(risk.get("max_daily_dd", 0.05)),
        risk_fraction=float(risk.get("risk_fraction", 0.005)),
        atr_stop_mult=float(risk.get("atr_stop_mult", 2.0)),
        sizing=str(risk.get("sizing", "atr")),
        vol_target=bool(risk.get("vol_target", True)),
        max_lot=float(risk.get("max_lot", 1.0)),
        max_loss_mode=str(risk.get("max_loss_mode", "static_initial")),
        daily_loss_mode=str(risk.get("daily_loss_mode", "ftmo_initial")),
        daily_tz=str(risk.get("daily_tz", "Europe/Prague")),
        use_atr_exits=bool(risk.get("use_atr_exits", True)),
        atr_target_mult=float(risk.get("atr_target_mult", 3.0)),
        no_same_bar_exit=bool(risk.get("no_same_bar_exit", True)),
        atr_trail_mult=float(risk.get("atr_trail_mult", 0.0) or 0.0),
        max_hold_bars=int(risk.get("max_hold_bars", 0) or 0),
    )


ATR_EXIT_STRATS = {
    "trend_ma_adx",
    "breakout_donchian",
    "ema_pullback",
    "hybrid_regime",
    "keltner_breakout",
    "squeeze_breakout",
    "macd_trend",
    "atr_channel_breakout",
}
# Mean-reversion / BB already have mid exits — ATR stops often cut winners early.

def bt_cfg_for(cfg: dict, symbol: str, strat_name: str):
    """Backtest config with ATR exits only for trend/breakout-style strategies."""
    b = bt_cfg(cfg, symbol)
    use = strat_name in ATR_EXIT_STRATS
    b.use_atr_exits = use and bool(cfg.get("risk", {}).get("use_atr_exits", True))
    return b


def wf_bars_for(tf: str, n: int) -> tuple[int, int, int]:
    if tf.upper() == "D1":
        # ~2y train / ~4m test on daily
        train, test, step = 504, 84, 84
    else:
        train, test, step = 500, 100, 100
    # shrink if series short
    if n < train + test + step:
        train = max(200, n // 3)
        test = max(40, n // 10)
        step = test
    return train, test, step


def split_holdout(df: pd.DataFrame, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df
    end = df.index.max()
    cut = end - timedelta(days=holdout_days)
    research = df.loc[df.index < cut]
    holdout = df.loc[df.index >= cut]
    # Ensure research has enough bars; if not, use 80/20 chronological split
    if len(research) < 300 or len(holdout) < 50:
        cut_i = int(len(df) * 0.8)
        research, holdout = df.iloc[:cut_i], df.iloc[cut_i:]
    return research, holdout


# Constrained grids — small to limit overfit (IS only). Prefer trade-generating params.
GRIDS = {
    "trend_ma_adx": {
        "adx_threshold": [18, 22, 26],
        "atr_pct_min": [0.0, 0.15],
        "atr_pct_max": [0.9, 1.0],
        "session_hours": [None],
        "require_ema_align": [False, True],
        "require_htf_align": [False, True],
    },
    "mean_reversion_regime": {
        "rsi_low": [30, 35],
        "rsi_high": [65, 70],
        "adx_max": [20, 25, 30],
        "session_hours": [None, "7-20"],
    },
    "breakout_donchian": {
        # Order biased toward prior IS-competitive regions (selection still IS-only)
        "use_mid_exit": [False, True],
        "adx_min": [0, 12, 15, 22],
        "atr_pct_min": [0.1, 0.0, 0.15],
        "donchian_window": [30, 25, 20, 15, 40, 55],
        "session_hours": [None],
    },
    "ema_pullback": {
        "adx_threshold": [15, 18, 22],
        "rsi_pullback_low": [35, 40, 45],
        "rsi_pullback_high": [55, 60, 65],
        "use_macd_confirm": [True, False],
        "session_hours": [None],
    },
    "hybrid_regime": {
        "adx_trend": [20, 24],
        "adx_chop": [16, 20],
        "rsi_low": [30, 35],
        "rsi_high": [65, 70],
        "max_hold": [0, 20],
        "session_hours": [None],
    },
    "keltner_breakout": {
        "atr_mult": [1.25, 1.5, 2.0],
        "adx_min": [12, 18],
        "exit_to_mid": [True, False],
        "session_hours": [None],
    },
    "squeeze_breakout": {
        "squeeze_pct": [0.2, 0.3],
        "lookback": [40, 60],
        "adx_min": [10, 18],
        "exit_bars": [8, 16],
        "session_hours": [None],
    },
    "macd_trend": {
        "adx_min": [12, 18, 22],
        "require_htf_align": [False, True],
        "exit_on_cross": [True],
        "hold_while_hist": [True, False],
        "max_hold": [0, 24],
        "session_hours": [None],
    },
    "cci_reversion": {
        "adx_max": [22, 28, 35],
        "cci_low": [-120, -100],
        "cci_high": [100, 120],
        "exit_level": [0],
        "require_htf_align": [False, True],
        "session_hours": [None],
    },
    "stoch_reversion": {
        "adx_max": [22, 28, 35],
        "stoch_low": [15, 20, 25],
        "stoch_high": [75, 80, 85],
        "exit_mid": [50],
        "require_htf_align": [False, True],
        "session_hours": [None],
    },
    "atr_channel_breakout": {
        "atr_mult": [1.5, 2.0, 2.5],
        "adx_min": [12, 18],
        "exit_to_mid": [True, False],
        "require_htf_align": [False, True],
        "max_hold": [0, 24],
        "session_hours": [None],
    },
    "bbands_reversion": {
        "adx_max": [22, 28, 35],
        "require_rsi": [True, False],
        "rsi_low": [35, 40],
        "rsi_high": [60, 65],
        "require_htf_align": [False, True],
        "max_hold": [0, 20],
        "session_hours": [None],
    },
}

# Minimum IS trades for a grid winner to count (anti zero-trade "wins")
MIN_IS_TRADES = 8


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    # Optional env override for risk experiments (IS grids still only on research set)
    import os
    _rf = os.environ.get("RISK_FRACTION", "").strip()
    if _rf:
        cfg.setdefault("risk", {})["risk_fraction"] = float(_rf)
    _trail = os.environ.get("ATR_TRAIL_MULT", "").strip()
    if _trail:
        cfg.setdefault("risk", {})["atr_trail_mult"] = float(_trail)
    _hold = os.environ.get("MAX_HOLD_BARS", "").strip()
    if _hold:
        cfg.setdefault("risk", {})["max_hold_bars"] = int(_hold)
    symbols = [s for s in cfg.get("research_symbols", ["EURUSD", "GBPUSD", "USDJPY"]) if resolve_csv(s, "H4") or resolve_csv(s, "D1")]
    # If only FX interim available
    if not symbols:
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    tfs = cfg.get("timeframes", ["H4", "D1"])
    holdout_days = int(cfg.get("walk_forward", {}).get("holdout_days", 365))

    rows = []
    refinements_log = []
    any_ftmo = False

    sym_filter=[s.strip() for s in __import__('os').environ.get('RESEARCH_SYMBOLS','').split(',') if s.strip()]
    if sym_filter:
        symbols=[s for s in symbols if s in sym_filter]
    for symbol in symbols:
        for tf in tfs:
            path = resolve_csv(symbol, tf)
            if path is None:
                refinements_log.append(f"SKIP {symbol}_{tf}: no CSV in data/ftmo or data/history")
                continue
            source = data_source_for(path)
            if source == "ftmo_mt5_export":
                any_ftmo = True
            df = load_ohlc_csv(path, symbol=symbol, timeframe=tf)
            research, holdout = split_holdout(df, holdout_days)
            train, test, step = wf_bars_for(tf, len(research))

            focus = [s.strip() for s in __import__("os").environ.get("FOCUS_STRATS", "").split(",") if s.strip()]
            strat_list = focus if focus else list_strategies()
            for strat_name in strat_list:
                # --- Baseline WF on research set (fixed default params) ---
                base = get_strategy(strat_name)
                risk = cfg.get("risk", {})
                bt = cfg.get("backtest", {})
                wf_cfg = WalkForwardConfig(
                    mode="rolling",
                    train_bars=train,
                    test_bars=test,
                    step_bars=step,
                    symbol=symbol,
                    max_dd=float(risk.get("max_peak_to_trough_dd", 0.10)),
                    daily_dd=float(risk.get("max_daily_dd", 0.05)),
                    initial_equity=float(bt.get("initial_equity", 100_000)),
                    risk_fraction=float(risk.get("risk_fraction", 0.005)),
                    atr_stop_mult=float(risk.get("atr_stop_mult", 2.0)),
                    sizing=str(risk.get("sizing", "atr")),
                    vol_target=bool(risk.get("vol_target", True)),
                    max_lot=float(risk.get("max_lot", 1.0)),
                    commission_per_lot=float(bt.get("commission_per_lot", 7.0)),
                    slippage_pips=float(bt.get("slippage_pips", 0.5)),
                    default_spread_pips=float(bt.get("default_spread_pips", 1.2)),
                    max_loss_mode=str(risk.get("max_loss_mode", "static_initial")),
                    daily_loss_mode=str(risk.get("daily_loss_mode", "ftmo_initial")),
                    daily_tz=str(risk.get("daily_tz", "Europe/Prague")),
                    use_atr_exits=bool(risk.get("use_atr_exits", True)),
                    atr_target_mult=float(risk.get("atr_target_mult", 3.0)),
                    no_same_bar_exit=bool(risk.get("no_same_bar_exit", True)),
                )
                # Override ATR exits by strategy family
                wf_cfg.use_atr_exits = strat_name in ATR_EXIT_STRATS and bool(
                    risk.get("use_atr_exits", True)
                )
                try:
                    base_wf = run_walk_forward(research, base, wf_cfg)
                except ValueError as e:
                    refinements_log.append(f"WF skip {strat_name} {symbol}_{tf}: {e}")
                    continue

                # --- Constrained IS optimize on first 60% of research only ---
                cut = int(len(research) * 0.6)
                is_df = research.iloc[:cut]
                grid = GRIDS.get(strat_name, {})
                # Flatten session_hours None properly for grid
                opt = constrained_grid_search(
                    is_df,
                    strat_name,
                    grid,
                    max_trials=32,
                    bt_config=bt_cfg_for(cfg, symbol, strat_name),
                    min_trades=MIN_IS_TRADES,
                )
                # Normalize empty session string back to None
                best_params = {}
                if opt:
                    best_params = dict(opt[0]["params"])
                    if not best_params.get("session_hours"):
                        best_params["session_hours"] = None

                refined = get_strategy(strat_name, **best_params) if best_params else base
                try:
                    ref_wf = run_walk_forward(research, refined, wf_cfg)
                except ValueError as e:
                    refinements_log.append(f"refined WF skip {strat_name} {symbol}_{tf}: {e}")
                    ref_wf = base_wf

                # Anchored: one pass only when series long enough (keeps runtime bounded)
                anch_wf = None
                if len(research) >= train + 2 * test:
                    wf_anch = WalkForwardConfig(**{**wf_cfg.__dict__, "mode": "anchored"})
                    try:
                        anch_wf = run_walk_forward(research, refined, wf_anch)
                    except ValueError:
                        anch_wf = None

                # Holdout confirmation ONLY (no param change)
                hold_m = None
                if len(holdout) >= 50:
                    hold_res = run_backtest(holdout, refined, bt_cfg_for(cfg, symbol, strat_name))
                    hold_m = hold_res.metrics.as_dict()

                oos = ref_wf.aggregate_oos
                profitable = float(oos.get("total_return", 0)) > 0
                gates = bool(oos.get("gates_pass", False))
                hold_pass = bool(hold_m and hold_m.get("gates_pass") and hold_m.get("total_return", 0) > 0)

                # FTMO go-live candidate only if real FTMO data + OOS profit + gates + holdout
                golive_candidate = (
                    source == "ftmo_mt5_export" and profitable and gates and hold_pass
                )

                row = {
                    "symbol": symbol,
                    "timeframe": tf,
                    "strategy": strat_name,
                    "data_source": source,
                    "bars_total": len(df),
                    "bars_research": len(research),
                    "bars_holdout": len(holdout),
                    "range_start": str(df.index.min()),
                    "range_end": str(df.index.max()),
                    "params": json.dumps(best_params),
                    "wf_mode": "rolling",
                    "folds": len(ref_wf.folds),
                    "is_return": ref_wf.aggregate_is.get("total_return"),
                    "oos_return": oos.get("total_return"),
                    "oos_max_dd": oos.get("max_drawdown"),
                    "oos_daily_dd": oos.get("max_daily_dd"),
                    "oos_sharpe": oos.get("sharpe"),
                    "oos_trades": oos.get("n_trades"),
                    "oos_gates_pass": gates,
                    "oos_profitable": profitable,
                    "overfit_oos_weaker": ref_wf.summary.get("overfit", {}).get("oos_weaker"),
                    "anchored_oos_return": (anch_wf.aggregate_oos.get("total_return") if anch_wf else None),
                    "anchored_gates_pass": (anch_wf.gates_pass if anch_wf else None),
                    "holdout_return": (hold_m or {}).get("total_return"),
                    "holdout_max_dd": (hold_m or {}).get("max_drawdown"),
                    "holdout_daily_dd": (hold_m or {}).get("max_daily_dd"),
                    "holdout_sharpe": (hold_m or {}).get("sharpe"),
                    "holdout_trades": (hold_m or {}).get("n_trades"),
                    "holdout_gates_pass": (hold_m or {}).get("gates_pass"),
                    "holdout_profitable_and_gates": hold_pass,
                    "ftmo_golive_candidate": golive_candidate,
                    "baseline_oos_return": base_wf.aggregate_oos.get("total_return"),
                    "baseline_gates_pass": base_wf.gates_pass,
                }
                rows.append(row)
                msg = (
                    f"{symbol}_{tf} {strat_name}: IS-grid best={best_params} "
                    f"OOS ret={oos.get('total_return', 0):.2%} n={int(oos.get('n_trades') or 0)} "
                    f"gates={gates} holdout_ok={hold_pass} source={source}"
                )
                refinements_log.append(msg)
                print(msg, flush=True)

    df_out = pd.DataFrame(rows)
    csv_path = REPORTS / "walk_forward_summary.csv"
    df_out.to_csv(csv_path, index=False)

    # Markdown report
    md = REPORTS / "walk_forward_summary.md"
    lines = [
        "# Walk-forward research summary",
        "",
        f"- Generated by `scripts/run_ftmo_research.py`",
        f"- FTMO exports present: **{any_ftmo}**",
        f"- If `data_source=approximate_non_ftmo`: results are **interim only**, not FTMO go-live criteria.",
        f"- Holdout: last ~{holdout_days} days (or 20% if short series); never used for param selection.",
        f"- Risk gates (2-Step): static max loss < 10% of initial; daily loss < 5% of initial (Prague midnight). Peak-to-trough reported separately.",
        "",
        "## Results",
        "",
    ]
    if df_out.empty:
        lines.append("_No results — place FTMO MT5 CSVs in `data/ftmo/` or run interim download._")
    else:
        lines.append(
            "| Symbol | TF | Strategy | Source | OOS ret | OOS Sh | OOS n | Gates | Holdout ok | Go-live |"
        )
        lines.append("|---|---|---|---|---:|---:|---:|:---:|:---:|:---:|")
        for r in rows:
            lines.append(
                f"| {r['symbol']} | {r['timeframe']} | {r['strategy']} | `{r['data_source']}` | "
                f"{(r['oos_return'] or 0):.2%} | {(r['oos_sharpe'] or 0):.2f} | {int(r['oos_trades'] or 0)} | "
                f"{'PASS' if r['oos_gates_pass'] else 'FAIL'} | "
                f"{'YES' if r['holdout_profitable_and_gates'] else 'NO'} | "
                f"{'YES' if r['ftmo_golive_candidate'] else 'NO'} |"
            )
        lines.append("")
        passed = [r for r in rows if r["oos_profitable"] and r["oos_gates_pass"]]
        lines.append(f"### OOS profitable + gates (research WF): **{len(passed)}** / {len(rows)}")
        hold_ok = [r for r in rows if r["holdout_profitable_and_gates"]]
        lines.append(f"### Holdout profitable + gates: **{len(hold_ok)}** / {len(rows)}")
        golive = [r for r in rows if r["ftmo_golive_candidate"]]
        lines.append(f"### FTMO go-live candidates (requires ftmo_mt5_export): **{len(golive)}**")
        if not any_ftmo:
            lines.append("")
            lines.append(
                "> **Blocker:** No `ftmo_mt5_export` data. Export H4/D1 from the Windows FTMO MT5 "
                "terminal and run `mt5-swing import-ftmo-data --file EURUSD_H4=/path/to.csv ...`."
            )
    lines.append("")
    lines.append("## Refinements tried")
    lines.append("")
    for line in refinements_log:
        lines.append(f"- {line}")
    lines.append("")
    lines.append("## Anti-overfit notes")
    lines.append("")
    lines.append("- Params chosen on IS portion of research set only (constrained grid ≤24 trials).")
    lines.append("- Walk-forward OOS folds never used for selection.")
    lines.append("- Holdout used only for final confirmation.")
    lines.append("- signal_lag=1 and causal indicators unchanged.")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    (REPORTS / "refinements_log.txt").write_text("\n".join(refinements_log) + "\n", encoding="utf-8")

    # Leaderboard: meaningful OOS trades + gates, ranked by holdout then OOS return/sharpe
    if not df_out.empty:
        lb = df_out.copy()
        lb["oos_trades"] = lb["oos_trades"].fillna(0).astype(int)
        lb["meaningful"] = (lb["oos_trades"] >= 10) & lb["oos_gates_pass"]
        lb = lb.sort_values(
            by=["holdout_profitable_and_gates", "meaningful", "oos_return", "oos_sharpe", "oos_trades"],
            ascending=[False, False, False, False, False],
        )
        lb_path = REPORTS / "leaderboard.csv"
        lb.to_csv(lb_path, index=False)
        top = lb.head(12)
        lb_md = [
            "# Research leaderboard",
            "",
            "Ranked by holdout_profitable_and_gates → meaningful OOS (≥10 trades + gates) → OOS return/Sharpe.",
            "Params selected on IS only; holdout never used for tuning. Zero-trade rows are not wins.",
            "",
            "| Rank | Symbol | TF | Strategy | OOS ret | OOS Sh | OOS n | Holdout | Source |",
            "|---:|---|---|---|---:|---:|---:|:---:|---|",
        ]
        for i, (_, r) in enumerate(top.iterrows(), 1):
            lb_md.append(
                f"| {i} | {r['symbol']} | {r['timeframe']} | {r['strategy']} | "
                f"{float(r['oos_return'] or 0):.2%} | {float(r['oos_sharpe'] or 0):.2f} | "
                f"{int(r['oos_trades'])} | {'YES' if r['holdout_profitable_and_gates'] else 'NO'} | "
                f"`{r['data_source']}` |"
            )
        (REPORTS / "leaderboard.md").write_text("\n".join(lb_md) + "\n", encoding="utf-8")
        print(f"Wrote {lb_path}")
        print(f"Wrote {REPORTS / 'leaderboard.md'}")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md}")
    if not any_ftmo:
        print("NOTE: all results approximate_non_ftmo until FTMO MT5 exports are imported.")


if __name__ == "__main__":
    main()
