"""MetaTrader 5 client (optional dependency; stub when unavailable)."""

from mt5_swing.broker.mt5.client import MT5Client, MT5Unavailable

__all__ = ["MT5Client", "MT5Unavailable"]
