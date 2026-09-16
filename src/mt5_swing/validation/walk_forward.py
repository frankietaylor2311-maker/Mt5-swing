"""
Walk-forward validation: rolling (default) and anchored modes.

Always evaluates out-of-sample folds. Optimization (if any) is constrained to IS
windows; OOS is never used for parameter selection in this module's default path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from mt5_swing.backtest.engine import BacktestConfig, run_backtest
from mt5_swing.validation.overfit import summarize_is_oos


@dataclass
class WalkForwardConfig:
    mode: Literal["rolling", "anchored"] = "rolling"
    train_bars: int = 500
    test_bars: int = 100
    step_bars: int = 100  # rolling step; ignored for pure anchored single-step lists
    min_train_bars: int = 200
    max_dd: float = 0.10
    daily_dd: float = 0.05
    symbol: str = "EURUSD"
    signal_lag: int = 1
    initial_equity: float = 10_000.0


@dataclass
class FoldResult:
    fold: int
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
    is_metrics: dict
    oos_metrics: dict


@dataclass
class WalkForwardResult:
    mode: str
    folds: list[FoldResult] = field(default_factory=list)
    aggregate_is: dict = field(default_factory=dict)
    aggregate_oos: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    gates_pass: bool = False

    def print_report(self) -> None:
        print(f"=== Walk-forward ({self.mode}) ===")
        print(f"Folds: {len(self.folds)}")
        print("--- In-sample (aggregate) ---")
        _print_metrics(self.aggregate_is)
        print("--- Out-of-sample (aggregate) ---")
        _print_metrics(self.aggregate_oos)
        print("--- Gates (OOS) ---")
        mdd = self.aggregate_oos.get("max_drawdown", float("nan"))
        ddd = self.aggregate_oos.get("max_daily_dd", float("nan"))
        pass_mdd = self.aggregate_oos.get("passed_max_dd_gate", False)
        pass_ddd = self.aggregate_oos.get("passed_daily_dd_gate", False)
        print(f"  Max peak-to-trough DD < 10%: {mdd:.2%} -> {'PASS' if pass_mdd else 'FAIL'}")
        print(f"  Max daily DD < 5%:          {ddd:.2%} -> {'PASS' if pass_ddd else 'FAIL'}")
        overall = "PASS" if self.gates_pass else "FAIL"
        print(f"  Overall risk gates:         {overall}")
        ov = self.summary.get("overfit", {})
        if ov:
            print("--- Overfit checks (informational) ---")
            print(f"  return_gap (IS-OOS): {ov.get('return_gap', 0):.2%}")
            print(f"  sharpe_gap:          {ov.get('sharpe_gap', 0):.3f}")
            print(f"  oos_weaker flag:     {ov.get('oos_weaker')}")
            print(f"  note: {ov.get('note')}")
        print(
            "Disclaimer: baseline strategies are research templates; "
            "OOS profitability is NOT claimed."
        )


def _print_metrics(m: dict) -> None:
    if not m:
        print("  (empty)")
        return
    print(f"  total_return: {m.get('total_return', 0):.2%}")
    print(f"  max_drawdown: {m.get('max_drawdown', 0):.2%}")
    print(f"  max_daily_dd: {m.get('max_daily_dd', 0):.2%}")
    print(f"  sharpe:       {m.get('sharpe', 0):.3f}")
    print(f"  n_trades:     {m.get('n_trades', 0)}")
    print(f"  win_rate:     {m.get('win_rate', 0):.2%}")


def _bt_config(wf: WalkForwardConfig) -> BacktestConfig:
    return BacktestConfig(
        symbol=wf.symbol,
        initial_equity=wf.initial_equity,
        signal_lag=wf.signal_lag,
        max_dd=wf.max_dd,
        daily_dd=wf.daily_dd,
    )


def _aggregate_metrics(metric_dicts: list[dict], max_dd: float, daily_dd: float) -> dict:
    if not metric_dicts:
        return {}
    # Equity-weighted-ish: average returns; take worst (max) drawdowns across folds
    avg_ret = sum(m["total_return"] for m in metric_dicts) / len(metric_dicts)
    max_dd_v = max(m["max_drawdown"] for m in metric_dicts)
    max_ddd_v = max(m["max_daily_dd"] for m in metric_dicts)
    avg_sh = sum(m["sharpe"] for m in metric_dicts) / len(metric_dicts)
    n_trades = sum(m["n_trades"] for m in metric_dicts)
    # Win rate weighted by trades
    if n_trades:
        win_rate = sum(m["win_rate"] * m["n_trades"] for m in metric_dicts) / n_trades
    else:
        win_rate = 0.0
    pass_mdd = max_dd_v < max_dd
    pass_ddd = max_ddd_v < daily_dd
    return {
        "total_return": avg_ret,
        "max_drawdown": max_dd_v,
        "max_daily_dd": max_ddd_v,
        "sharpe": avg_sh,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": sum(m.get("profit_factor", 0) for m in metric_dicts) / len(metric_dicts),
        "final_equity": metric_dicts[-1].get("final_equity", 0),
        "start_equity": metric_dicts[0].get("start_equity", 0),
        "passed_max_dd_gate": pass_mdd,
        "passed_daily_dd_gate": pass_ddd,
        "gates_pass": pass_mdd and pass_ddd,
    }


def run_walk_forward(
    ohlc: pd.DataFrame,
    strategy,
    config: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    wf = config or WalkForwardConfig()
    n = len(ohlc)
    if n < wf.min_train_bars + wf.test_bars:
        raise ValueError(
            f"Need at least {wf.min_train_bars + wf.test_bars} bars; got {n}"
        )

    folds: list[FoldResult] = []
    fold_i = 0

    if wf.mode == "rolling":
        start = 0
        while start + wf.train_bars + wf.test_bars <= n:
            tr_a, tr_b = start, start + wf.train_bars
            te_a, te_b = tr_b, tr_b + wf.test_bars
            folds.append(_run_fold(ohlc, strategy, wf, fold_i, tr_a, tr_b, te_a, te_b))
            fold_i += 1
            start += wf.step_bars
    elif wf.mode == "anchored":
        # Growing train window; fixed test length stepping forward
        train_end = wf.train_bars
        while train_end + wf.test_bars <= n:
            tr_a, tr_b = 0, train_end
            te_a, te_b = tr_b, tr_b + wf.test_bars
            folds.append(_run_fold(ohlc, strategy, wf, fold_i, tr_a, tr_b, te_a, te_b))
            fold_i += 1
            train_end += wf.step_bars
    else:
        raise ValueError(f"Unknown mode: {wf.mode}")

    if not folds:
        raise ValueError("No walk-forward folds produced; check train/test/step sizes")

    agg_is = _aggregate_metrics([f.is_metrics for f in folds], wf.max_dd, wf.daily_dd)
    agg_oos = _aggregate_metrics([f.oos_metrics for f in folds], wf.max_dd, wf.daily_dd)
    summary = summarize_is_oos(agg_is, agg_oos)
    return WalkForwardResult(
        mode=wf.mode,
        folds=folds,
        aggregate_is=agg_is,
        aggregate_oos=agg_oos,
        summary=summary,
        gates_pass=bool(agg_oos.get("gates_pass", False)),
    )


def _run_fold(
    ohlc: pd.DataFrame,
    strategy,
    wf: WalkForwardConfig,
    fold_i: int,
    tr_a: int,
    tr_b: int,
    te_a: int,
    te_b: int,
) -> FoldResult:
    train = ohlc.iloc[tr_a:tr_b]
    test = ohlc.iloc[te_a:te_b]
    # Features computed separately per split to avoid leaking test into train indicators
    # (rolling windows only use data within the slice — correct for OOS purity)
    bt_cfg = _bt_config(wf)
    is_res = run_backtest(train, strategy, bt_cfg)
    oos_res = run_backtest(test, strategy, bt_cfg)
    return FoldResult(
        fold=fold_i,
        train_start=train.index[0],
        train_end=train.index[-1],
        test_start=test.index[0],
        test_end=test.index[-1],
        is_metrics=is_res.metrics.as_dict(),
        oos_metrics=oos_res.metrics.as_dict(),
    )
