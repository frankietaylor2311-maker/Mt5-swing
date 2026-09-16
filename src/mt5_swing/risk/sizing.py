"""Position sizing helpers (lots)."""

from __future__ import annotations


def atr_position_size(
    equity: float,
    atr: float,
    *,
    risk_fraction: float = 0.01,
    atr_stop_mult: float = 2.0,
    pip_size: float = 0.0001,
    pip_value: float = 10.0,
    min_lot: float = 0.01,
    max_lot: float = 10.0,
    lot_step: float = 0.01,
) -> float:
    """
    Size so that a stop of ``atr_stop_mult * atr`` risks ``risk_fraction`` of equity.

    Approximate: risk_cash / (stop_distance_in_pips * pip_value).
    """
    if equity <= 0 or atr <= 0 or pip_size <= 0 or pip_value <= 0:
        return 0.0
    stop_dist = atr_stop_mult * atr
    stop_pips = stop_dist / pip_size
    if stop_pips <= 0:
        return 0.0
    risk_cash = equity * risk_fraction
    lots = risk_cash / (stop_pips * pip_value)
    lots = max(min_lot, min(max_lot, lots))
    # Round down to lot_step
    steps = int(lots / lot_step)
    return round(steps * lot_step, 2)


def fixed_fractional_size(
    equity: float,
    *,
    fraction: float = 0.02,
    ref_equity: float = 10_000.0,
    ref_lot: float = 0.1,
    min_lot: float = 0.01,
    max_lot: float = 10.0,
    lot_step: float = 0.01,
) -> float:
    """Scale lot size linearly with equity vs a reference."""
    if equity <= 0 or ref_equity <= 0:
        return 0.0
    lots = ref_lot * (equity / ref_equity) * (fraction / 0.02)
    lots = max(min_lot, min(max_lot, lots))
    steps = int(lots / lot_step)
    return round(steps * lot_step, 2)
