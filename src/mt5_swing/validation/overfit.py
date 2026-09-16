"""Simple overfit heuristics comparing in-sample vs out-of-sample metrics."""

from __future__ import annotations

from typing import Any


def overfit_score(is_metrics: dict[str, Any], oos_metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Return diagnostics — not a pass claim.

    Flags:
    - return_gap: IS total_return - OOS total_return (large positive → possible overfit)
    - sharpe_gap: same for Sharpe
    - oos_weaker: True if OOS return < 50% of IS return when IS > 0
    """
    is_ret = float(is_metrics.get("total_return", 0.0))
    oos_ret = float(oos_metrics.get("total_return", 0.0))
    is_sh = float(is_metrics.get("sharpe", 0.0))
    oos_sh = float(oos_metrics.get("sharpe", 0.0))
    return_gap = is_ret - oos_ret
    sharpe_gap = is_sh - oos_sh
    oos_weaker = bool(is_ret > 0 and oos_ret < 0.5 * is_ret)
    return {
        "return_gap": return_gap,
        "sharpe_gap": sharpe_gap,
        "oos_weaker": oos_weaker,
        "note": "Baselines may fail risk gates; gaps do not prove edge.",
    }


def summarize_is_oos(is_metrics: dict[str, Any], oos_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "is": is_metrics,
        "oos": oos_metrics,
        "overfit": overfit_score(is_metrics, oos_metrics),
        "gates_pass_oos": bool(oos_metrics.get("gates_pass", False)),
    }
