"""Strategy protocol and baseline swing strategies."""

from mt5_swing.strategies.base import Strategy, Signal
from mt5_swing.strategies.trend_ma_adx import TrendMAADX
from mt5_swing.strategies.mean_reversion import MeanReversionRegime
from mt5_swing.strategies.breakout import BreakoutDonchian
from mt5_swing.strategies.registry import get_strategy, list_strategies

__all__ = [
    "Strategy",
    "Signal",
    "TrendMAADX",
    "MeanReversionRegime",
    "BreakoutDonchian",
    "get_strategy",
    "list_strategies",
]
