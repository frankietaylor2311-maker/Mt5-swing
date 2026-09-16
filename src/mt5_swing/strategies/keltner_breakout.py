"""Keltner / ATR-channel breakout (EMA mid ± ATR width)."""

from __future__ import annotations

import pandas as pd

from mt5_swing.strategies.base import Signal


class KeltnerBreakout:
    name = "keltner_breakout"

    def __init__(
        self,
        atr_mult: float = 1.5,
        adx_min: float = 15.0,
        session_hours: str | None = None,
        exit_to_mid: bool = True,
    ):
        self.atr_mult = float(atr_mult)
        self.adx_min = float(adx_min)
        self.session_hours = session_hours or None
        self.exit_to_mid = bool(exit_to_mid)

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
        need = {"ema_fast", "atr", "adx"}
        missing = need - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        px = data["signal_close"] if "signal_close" in data.columns else data["close"]
        mid = data["ema_fast"]
        upper = mid + self.atr_mult * data["atr"]
        lower = mid - self.atr_mult * data["atr"]
        sess = self._session_mask(data.index)
        strong = data["adx"] >= self.adx_min
        long_cond = sess & strong & (px > upper)
        short_cond = sess & strong & (px < lower)
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        last = int(Signal.FLAT)
        for i in range(len(sig)):
            if pd.isna(mid.iloc[i]) or pd.isna(data["atr"].iloc[i]):
                last = int(Signal.FLAT)
            elif bool(long_cond.iloc[i]):
                last = int(Signal.LONG)
            elif bool(short_cond.iloc[i]):
                last = int(Signal.SHORT)
            elif self.exit_to_mid:
                if last == int(Signal.LONG) and px.iloc[i] < mid.iloc[i]:
                    last = int(Signal.FLAT)
                elif last == int(Signal.SHORT) and px.iloc[i] > mid.iloc[i]:
                    last = int(Signal.FLAT)
            sig.iloc[i] = last
        return sig.astype(int)
