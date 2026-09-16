"""Mean-reversion with ADX regime filter (+ optional ATR band / session)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mt5_swing.strategies.base import Signal


class MeanReversionRegime:
    name = "mean_reversion_regime"

    def __init__(
        self,
        rsi_low: float = 35.0,
        rsi_high: float = 65.0,
        adx_max: float = 25.0,
        exit_low: float = 48.0,
        exit_high: float = 52.0,
        atr_pct_max: float = 1.0,
        atr_lookback: int = 100,
        session_hours: str | None = None,
    ):
        self.rsi_low = float(rsi_low)
        self.rsi_high = float(rsi_high)
        self.adx_max = float(adx_max)
        self.exit_low = float(exit_low)
        self.exit_high = float(exit_high)
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

    def _atr_ok(self, data: pd.DataFrame) -> pd.Series:
        if "atr" not in data.columns or self.atr_pct_max >= 1.0:
            return pd.Series(True, index=data.index)
        atr = data["atr"]
        roll_min = atr.rolling(self.atr_lookback, min_periods=20).min()
        roll_max = atr.rolling(self.atr_lookback, min_periods=20).max()
        span = (roll_max - roll_min).replace(0, np.nan)
        rank = ((atr - roll_min) / span).clip(0, 1)
        return rank <= self.atr_pct_max

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        required = {"rsi", "adx"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        calm = data["adx"] < self.adx_max
        sess = self._session_mask(data.index)
        atr_ok = self._atr_ok(data)
        long_cond = calm & sess & atr_ok & (data["rsi"] < self.rsi_low)
        short_cond = calm & sess & atr_ok & (data["rsi"] > self.rsi_high)
        mid = (data["rsi"] >= self.exit_low) & (data["rsi"] <= self.exit_high)
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        sig = sig.mask(long_cond, int(Signal.LONG))
        sig = sig.mask(short_cond, int(Signal.SHORT))
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
