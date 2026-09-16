"""Constrained in-sample optimization — OOS never used for selection."""

from __future__ import annotations

import itertools
import math
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
    min_trades: int = 8,
) -> list[dict[str, Any]]:
    """
    Exhaustive grid over ``param_grid``, capped at ``max_trials``.

    Candidates that breach IS risk gates are discarded. Zero/near-zero trade
    systems are deprioritized (not treated as wins). Ranking among gate-passing
    trials with enough trades uses IS Sharpe then total return.
    Caller must validate winners on walk-forward OOS — never select on OOS here.
    """
    cfg = bt_config or BacktestConfig(max_dd=max_dd, daily_dd=daily_dd)
    keys = sorted(param_grid.keys())
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    if len(combos) > max_trials:
        # Prefer diverse first slice; still deterministic
        combos = combos[:max_trials]

    results: list[dict[str, Any]] = []
    for combo in combos:
        params = dict(zip(keys, combo))
        # Normalize empty session_hours
        if "session_hours" in params and not params["session_hours"]:
            params["session_hours"] = None
        strat = get_strategy(strategy_name, **params)
        res = run_backtest(ohlc, strat, cfg)
        m = res.metrics.as_dict()
        n_tr = int(m.get("n_trades") or 0)
        gates = bool(m["gates_pass"])
        enough = n_tr >= min_trades
        # Composite: only gate+enough get real score; else heavily penalize
        if gates and enough:
            score = float(m["sharpe"]) + 0.25 * math.log1p(n_tr) + 2.0 * float(m["total_return"])
        elif gates and n_tr > 0:
            score = float(m["sharpe"]) - 5.0  # soft penalty
        else:
            score = -999.0
        results.append(
            {
                "params": params,
                "metrics": m,
                "gates_pass": gates,
                "sharpe": m["sharpe"],
                "n_trades": n_tr,
                "score": score,
                "enough_trades": enough,
            }
        )

    passing = [r for r in results if r["gates_pass"] and r["enough_trades"]]
    passing.sort(key=lambda r: r["score"], reverse=True)
    soft = [r for r in results if r["gates_pass"] and not r["enough_trades"]]
    soft.sort(key=lambda r: (r["n_trades"], r["sharpe"]), reverse=True)
    failing = [r for r in results if not r["gates_pass"]]
    # Prefer enough-trade winners; fall back to soft only if none
    ordered = passing + soft + failing
    return ordered
