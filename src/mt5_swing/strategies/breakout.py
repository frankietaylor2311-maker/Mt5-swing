"""Donchian channel breakout (+ optional ADX / ATR / session filters)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mt5_swing.strategies.base import Signal


class BreakoutDonchian:
    name = "breakout_donchian"

    def __init__(
        self,
        use_mid_exit: bool = True,
        adx_min: float = 0.0,
        atr_pct_min: float = 0.0,
        atr_pct_max: float = 1.0,
        atr_lookback: int = 100,
        session_hours: str | None = None,
    ):
        self.use_mid_exit = bool(use_mid_exit)
        self.adx_min = float(adx_min)
        self.atr_pct_min = float(atr_pct_min)
        self.atr_pct_max = float(atr_pct_max)
        self.atr_lookback = int(atr_lookback)
        self.session_hours = session_hours or None

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

    def _filters(self, data: pd.DataFrame) -> pd.Series:
        ok = pd.Series(True, index=data.index)
        if self.adx_min > 0 and "adx" in data.columns:
            ok &= data["adx"] >= self.adx_min
        if "atr" in data.columns and not (self.atr_pct_min <= 0 and self.atr_pct_max >= 1):
            atr = data["atr"]
            roll_min = atr.rolling(self.atr_lookback, min_periods=20).min()
            roll_max = atr.rolling(self.atr_lookback, min_periods=20).max()
            span = (roll_max - roll_min).replace(0, np.nan)
            rank = ((atr - roll_min) / span).clip(0, 1)
            ok &= (rank >= self.atr_pct_min) & (rank <= self.atr_pct_max)
        ok &= self._session_mask(data.index)
        return ok

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        required = {"close", "donchian_upper", "donchian_lower", "donchian_mid"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        px = data["close"]
        if "signal_close" in data.columns:
            px = data["signal_close"]
        filt = self._filters(data)
        long_cond = (px > data["donchian_upper"]) & filt
        short_cond = (px < data["donchian_lower"]) & filt
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
