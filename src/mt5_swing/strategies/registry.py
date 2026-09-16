"""Strategy name → instance factory."""

from __future__ import annotations

from typing import Any

from mt5_swing.strategies.bbands_reversion import BBandsReversion
from mt5_swing.strategies.breakout import BreakoutDonchian
from mt5_swing.strategies.ema_pullback import EmaPullback
from mt5_swing.strategies.hybrid_regime import HybridRegime
from mt5_swing.strategies.keltner_breakout import KeltnerBreakout
from mt5_swing.strategies.squeeze_breakout import SqueezeBreakout
from mt5_swing.strategies.macd_trend import MacdTrend
from mt5_swing.strategies.stoch_reversion import StochReversion
from mt5_swing.strategies.atr_channel import AtrChannelBreakout
from mt5_swing.strategies.cci_reversion import CciReversion
from mt5_swing.strategies.mean_reversion import MeanReversionRegime
from mt5_swing.strategies.trend_ma_adx import TrendMAADX

_REGISTRY = {
    TrendMAADX.name: TrendMAADX,
    MeanReversionRegime.name: MeanReversionRegime,
    BreakoutDonchian.name: BreakoutDonchian,
    EmaPullback.name: EmaPullback,
    HybridRegime.name: HybridRegime,
    BBandsReversion.name: BBandsReversion,
    KeltnerBreakout.name: KeltnerBreakout,
    SqueezeBreakout.name: SqueezeBreakout,
    MacdTrend.name: MacdTrend,
    StochReversion.name: StochReversion,
    AtrChannelBreakout.name: AtrChannelBreakout,
    CciReversion.name: CciReversion,
}


def list_strategies() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_strategy(name: str, **kwargs: Any):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Available: {list_strategies()}")
    return _REGISTRY[name](**kwargs)
