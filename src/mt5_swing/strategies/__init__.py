"""Strategy protocol and baseline swing strategies."""

from mt5_swing.strategies.base import Strategy, Signal
from mt5_swing.strategies.trend_ma_adx import TrendMAADX
from mt5_swing.strategies.mean_reversion import MeanReversionRegime
from mt5_swing.strategies.breakout import BreakoutDonchian
from mt5_swing.strategies.ema_pullback import EmaPullback
from mt5_swing.strategies.hybrid_regime import HybridRegime
from mt5_swing.strategies.bbands_reversion import BBandsReversion
from mt5_swing.strategies.keltner_breakout import KeltnerBreakout
from mt5_swing.strategies.registry import get_strategy, list_strategies

__all__ = [
    "Strategy",
    "Signal",
    "TrendMAADX",
    "MeanReversionRegime",
    "BreakoutDonchian",
    "EmaPullback",
    "HybridRegime",
    "BBandsReversion",
    "KeltnerBreakout",
    "get_strategy",
    "list_strategies",
]
