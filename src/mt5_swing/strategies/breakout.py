"""Donchian channel breakout baseline."""

from __future__ import annotations

import pandas as pd

from mt5_swing.strategies.base import Signal


class BreakoutDonchian:
    name = "breakout_donchian"

    def __init__(self, use_mid_exit: bool = True):
        self.use_mid_exit = use_mid_exit

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        required = {"close", "donchian_upper", "donchian_lower", "donchian_mid"}
        # close is unlagged OHLC; channels are lagged features — compare close.shift(0)
        # against lagged channels so we don't use same-bar channel that includes this bar's high
        # Features pipeline already lagged channels when signal_lag=1.
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        # Use lagged close for apples-to-apples with lagged channels if present
        px = data["close"]
        if "signal_close" in data.columns:
            px = data["signal_close"]
        long_cond = px > data["donchian_upper"]
        short_cond = px < data["donchian_lower"]
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        last = int(Signal.FLAT)
        for i in range(len(sig)):
            if pd.isna(data["donchian_upper"].iloc[i]):
                last = int(Signal.FLAT)
            elif bool(long_cond.iloc[i]):
                last = int(Signal.LONG)
            elif bool(short_cond.iloc[i]):
                last = int(Signal.SHORT)
            elif self.use_mid_exit:
                if last == int(Signal.LONG) and px.iloc[i] < data["donchian_mid"].iloc[i]:
                    last = int(Signal.FLAT)
                elif last == int(Signal.SHORT) and px.iloc[i] > data["donchian_mid"].iloc[i]:
                    last = int(Signal.FLAT)
            sig.iloc[i] = last
        return sig.astype(int)
