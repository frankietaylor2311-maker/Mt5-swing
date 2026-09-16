"""Hybrid: trend-follow when ADX high, mean-revert when ADX low."""

from __future__ import annotations

import pandas as pd

from mt5_swing.strategies.base import Signal


class HybridRegime:
    name = "hybrid_regime"

    def __init__(
        self,
        adx_trend: float = 22.0,
        adx_chop: float = 18.0,
        rsi_low: float = 32.0,
        rsi_high: float = 68.0,
        session_hours: str | None = None,
        max_hold: int = 0,
    ):
        self.adx_trend = float(adx_trend)
        self.adx_chop = float(adx_chop)
        self.rsi_low = float(rsi_low)
        self.rsi_high = float(rsi_high)
        self.session_hours = session_hours or None
        self.max_hold = int(max_hold)

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

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        need = {"sma_fast", "sma_slow", "adx", "rsi"}
        missing = need - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        sess = self._session_mask(data.index)
        trend_up = data["sma_fast"] > data["sma_slow"]
        trend_dn = data["sma_fast"] < data["sma_slow"]
        trending = data["adx"] >= self.adx_trend
        chop = data["adx"] <= self.adx_chop
        long_t = trending & trend_up & sess
        short_t = trending & trend_dn & sess
        long_m = chop & sess & (data["rsi"] < self.rsi_low)
        short_m = chop & sess & (data["rsi"] > self.rsi_high)
        mid = (data["rsi"] >= 48) & (data["rsi"] <= 52)
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        last = int(Signal.FLAT)
        mode = "flat"
        held = 0
        for i in range(len(sig)):
            if pd.isna(data["adx"].iloc[i]):
                last = int(Signal.FLAT)
                mode = "flat"
            elif bool(long_t.iloc[i]):
                last = int(Signal.LONG)
                mode = "trend"
            elif bool(short_t.iloc[i]):
                last = int(Signal.SHORT)
                mode = "trend"
            elif bool(long_m.iloc[i]):
                last = int(Signal.LONG)
                mode = "mr"
            elif bool(short_m.iloc[i]):
                last = int(Signal.SHORT)
                mode = "mr"
            elif mode == "trend":
                if not bool(trending.iloc[i]):
                    last = int(Signal.FLAT)
                    mode = "flat"
                elif last == int(Signal.LONG) and not bool(trend_up.iloc[i]):
                    last = int(Signal.FLAT)
                    mode = "flat"
                elif last == int(Signal.SHORT) and not bool(trend_dn.iloc[i]):
                    last = int(Signal.FLAT)
                    mode = "flat"
            elif mode == "mr" and bool(mid.iloc[i]):
                last = int(Signal.FLAT)
                mode = "flat"
            if last != int(Signal.FLAT):
                held += 1
                if self.max_hold > 0 and held >= self.max_hold:
                    last = int(Signal.FLAT)
                    mode = "flat"
                    held = 0
            else:
                held = 0
            sig.iloc[i] = last
        return sig.astype(int)
