"""Mean-reversion baseline with ADX regime filter (trade only when ADX is low)."""

from __future__ import annotations

import pandas as pd

from mt5_swing.strategies.base import Signal


class MeanReversionRegime:
    name = "mean_reversion_regime"

    def __init__(
        self,
        rsi_low: float = 30.0,
        rsi_high: float = 70.0,
        adx_max: float = 20.0,
    ):
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.adx_max = adx_max

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        required = {"rsi", "adx"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        calm = data["adx"] < self.adx_max
        long_cond = calm & (data["rsi"] < self.rsi_low)
        short_cond = calm & (data["rsi"] > self.rsi_high)
        # Exit toward mid: when RSI crosses back through 50, flatten via signal=0
        # (handled by target-position model: no signal → flat)
        mid = (data["rsi"] >= 45) & (data["rsi"] <= 55)
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        sig = sig.mask(long_cond, int(Signal.LONG))
        sig = sig.mask(short_cond, int(Signal.SHORT))
        # Hold previous until mid-zone flatten — simple sticky signals
        sticky = sig.copy()
        last = int(Signal.FLAT)
        for i in range(len(sticky)):
            v = int(sig.iloc[i])
            if pd.isna(data["rsi"].iloc[i]) or pd.isna(data["adx"].iloc[i]):
                last = int(Signal.FLAT)
            elif v != int(Signal.FLAT):
                last = v
            elif bool(mid.iloc[i]):
                last = int(Signal.FLAT)
            sticky.iloc[i] = last
        return sticky.astype(int)
