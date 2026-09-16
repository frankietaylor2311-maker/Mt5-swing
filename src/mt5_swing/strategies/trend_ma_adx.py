"""Trend-following baseline: SMA crossover filtered by ADX."""

from __future__ import annotations

import pandas as pd

from mt5_swing.strategies.base import Signal


class TrendMAADX:
    name = "trend_ma_adx"

    def __init__(self, adx_threshold: float = 20.0):
        self.adx_threshold = adx_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        required = {"sma_fast", "sma_slow", "adx"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        long_cond = (data["sma_fast"] > data["sma_slow"]) & (data["adx"] >= self.adx_threshold)
        short_cond = (data["sma_fast"] < data["sma_slow"]) & (data["adx"] >= self.adx_threshold)
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        sig = sig.mask(long_cond, int(Signal.LONG))
        sig = sig.mask(short_cond, int(Signal.SHORT))
        # Flat when features NaN
        nan_mask = data[["sma_fast", "sma_slow", "adx"]].isna().any(axis=1)
        sig = sig.mask(nan_mask, int(Signal.FLAT))
        return sig
