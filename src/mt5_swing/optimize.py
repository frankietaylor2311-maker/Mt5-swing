"""Constrained in-sample optimization — OOS never used for selection."""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

from mt5_swing.backtest.engine import BacktestConfig, run_backtest
from mt5_swing.strategies.registry import get_strategy


def constrained_grid_search(
    ohlc: pd.DataFrame,
    strategy_name: str,
    param_grid: dict[str, list[Any]],
    *,
    bt_config: BacktestConfig | None = None,
    max_trials: int = 20,
    max_dd: float = 0.10,
    daily_dd: float = 0.05,
) -> list[dict[str, Any]]:
    """
    Exhaustive grid over ``param_grid``, capped at ``max_trials``.

    Candidates that breach IS risk gates are discarded. Ranking uses IS Sharpe
    among gate-passing trials only. Caller must validate winners on walk-forward OOS.
    """
    cfg = bt_config or BacktestConfig(max_dd=max_dd, daily_dd=daily_dd)
    keys = sorted(param_grid.keys())
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    if len(combos) > max_trials:
        combos = combos[:max_trials]

    results: list[dict[str, Any]] = []
    for combo in combos:
        params = dict(zip(keys, combo))
        strat = get_strategy(strategy_name, **params)
        res = run_backtest(ohlc, strat, cfg)
        m = res.metrics.as_dict()
        results.append(
            {
                "params": params,
                "metrics": m,
                "gates_pass": m["gates_pass"],
                "sharpe": m["sharpe"],
            }
        )

    passing = [r for r in results if r["gates_pass"]]
    passing.sort(key=lambda r: r["sharpe"], reverse=True)
    # Append failing at end for transparency
    failing = [r for r in results if not r["gates_pass"]]
    return passing + failing
