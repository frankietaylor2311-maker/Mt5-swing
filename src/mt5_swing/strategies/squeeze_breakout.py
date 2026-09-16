"""Volatility squeeze breakout: enter when BB width expands after a quiet period."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mt5_swing.strategies.base import Signal


class SqueezeBreakout:
    name = "squeeze_breakout"

    def __init__(
        self,
        squeeze_pct: float = 0.25,
        lookback: int = 50,
        adx_min: float = 12.0,
        session_hours: str | None = None,
        exit_bars: int = 12,
    ):
        self.squeeze_pct = float(squeeze_pct)
        self.lookback = int(lookback)
        self.adx_min = float(adx_min)
        self.session_hours = session_hours or None
        self.exit_bars = int(exit_bars)

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
        need = {"bb_upper", "bb_lower", "bb_mid", "adx", "close"}
        missing = need - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        px = data["signal_close"] if "signal_close" in data.columns else data["close"]
        width = (data["bb_upper"] - data["bb_lower"]) / data["bb_mid"].replace(0, np.nan)
        roll_min = width.rolling(self.lookback, min_periods=max(10, self.lookback // 5)).min()
        roll_max = width.rolling(self.lookback, min_periods=max(10, self.lookback // 5)).max()
        span = (roll_max - roll_min).replace(0, np.nan)
        rank = ((width - roll_min) / span).clip(0, 1)
        # Squeeze: width in bottom quantile recently, then expand
        was_squeeze = rank.shift(1) <= self.squeeze_pct
        expanding = width > width.shift(1)
        sess = self._session_mask(data.index)
        strong = data["adx"] >= self.adx_min
        long_cond = was_squeeze & expanding & sess & strong & (px > data["bb_mid"])
        short_cond = was_squeeze & expanding & sess & strong & (px < data["bb_mid"])
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        last = int(Signal.FLAT)
        held = 0
        for i in range(len(sig)):
            if pd.isna(width.iloc[i]) or pd.isna(data["adx"].iloc[i]):
                last = int(Signal.FLAT)
                held = 0
            elif bool(long_cond.iloc[i]):
                last = int(Signal.LONG)
                held = 0
            elif bool(short_cond.iloc[i]):
                last = int(Signal.SHORT)
                held = 0
            elif last != int(Signal.FLAT):
                held += 1
                if held >= self.exit_bars:
                    last = int(Signal.FLAT)
                    held = 0
            sig.iloc[i] = last
        return sig.astype(int)
