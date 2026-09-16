"""Strategy name → instance factory."""

from __future__ import annotations

from typing import Any

from mt5_swing.strategies.breakout import BreakoutDonchian
from mt5_swing.strategies.mean_reversion import MeanReversionRegime
from mt5_swing.strategies.trend_ma_adx import TrendMAADX

_REGISTRY = {
    TrendMAADX.name: TrendMAADX,
    MeanReversionRegime.name: MeanReversionRegime,
    BreakoutDonchian.name: BreakoutDonchian,
}


def list_strategies() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_strategy(name: str, **kwargs: Any):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Available: {list_strategies()}")
    return _REGISTRY[name](**kwargs)
