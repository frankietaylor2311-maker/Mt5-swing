"""Trend EMA with RSI pullback entries — higher trade frequency than SMA cross."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mt5_swing.strategies.base import Signal


class EmaPullback:
    name = "ema_pullback"

    def __init__(
        self,
        adx_threshold: float = 18.0,
        rsi_pullback_low: float = 40.0,
        rsi_pullback_high: float = 60.0,
        atr_pct_min: float = 0.0,
        atr_pct_max: float = 1.0,
        atr_lookback: int = 100,
        session_hours: str | None = None,
        use_macd_confirm: bool = False,
    ):
        self.adx_threshold = float(adx_threshold)
        self.rsi_pullback_low = float(rsi_pullback_low)
        self.rsi_pullback_high = float(rsi_pullback_high)
        self.atr_pct_min = float(atr_pct_min)
        self.atr_pct_max = float(atr_pct_max)
        self.atr_lookback = int(atr_lookback)
        self.session_hours = session_hours or None
        self.use_macd_confirm = bool(use_macd_confirm)

    def _session_mask(self, index: pd.DatetimeIndex) -> pd.Series:
        if not self.session_hours:
            return pd.Series(True, index=index)
        start_s, end_s = self.session_hours.split("-")
        start_h, end_h = int(start_s), int(end_s)
        hours = index.tz_convert("UTC").hour if index.tz is not None else index.hour
        if start_h <= end_h:
            ok = (hours >= start_h) & (hours <= end_h)
        else:
            ok = (hours >= start_h) | (hours <= end_h)
        return pd.Series(ok, index=index)

    def _atr_rank_ok(self, data: pd.DataFrame) -> pd.Series:
        if "atr" not in data.columns:
            return pd.Series(True, index=data.index)
        if self.atr_pct_min <= 0 and self.atr_pct_max >= 1:
            return pd.Series(True, index=data.index)
        atr = data["atr"]
        roll_min = atr.rolling(self.atr_lookback, min_periods=max(20, self.atr_lookback // 5)).min()
        roll_max = atr.rolling(self.atr_lookback, min_periods=max(20, self.atr_lookback // 5)).max()
        span = (roll_max - roll_min).replace(0, np.nan)
        rank = ((atr - roll_min) / span).clip(0, 1)
        return (rank >= self.atr_pct_min) & (rank <= self.atr_pct_max)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        need = {"ema_fast", "sma_slow", "adx", "rsi"}
        missing = need - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        sess = self._session_mask(data.index)
        atr_ok = self._atr_rank_ok(data)
        trend_up = data["ema_fast"] > data["sma_slow"]
        trend_dn = data["ema_fast"] < data["sma_slow"]
        strong = data["adx"] >= self.adx_threshold
        long_entry = trend_up & strong & sess & atr_ok & (data["rsi"] <= self.rsi_pullback_low)
        short_entry = trend_dn & strong & sess & atr_ok & (data["rsi"] >= self.rsi_pullback_high)
        if self.use_macd_confirm and "macd" in data.columns and "macd_signal" in data.columns:
            long_entry = long_entry & (data["macd"] >= data["macd_signal"])
            short_entry = short_entry & (data["macd"] <= data["macd_signal"])
        exit_long = (~trend_up) | (data["rsi"] >= 55)
        exit_short = (~trend_dn) | (data["rsi"] <= 45)
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        last = int(Signal.FLAT)
        for i in range(len(sig)):
            if pd.isna(data["ema_fast"].iloc[i]) or pd.isna(data["adx"].iloc[i]):
                last = int(Signal.FLAT)
            elif bool(long_entry.iloc[i]):
                last = int(Signal.LONG)
            elif bool(short_entry.iloc[i]):
                last = int(Signal.SHORT)
            elif last == int(Signal.LONG) and bool(exit_long.iloc[i]):
                last = int(Signal.FLAT)
            elif last == int(Signal.SHORT) and bool(exit_short.iloc[i]):
                last = int(Signal.FLAT)
            sig.iloc[i] = last
        return sig.astype(int)
