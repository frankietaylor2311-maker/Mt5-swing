"""Volatility-regime breakout: trade Donchian/ATR breaks only in high-vol regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mt5_swing.strategies.base import Signal


class VolBreakout:
    name = "vol_breakout"

    def __init__(
        self,
        atr_pct_min: float = 0.55,
        atr_lookback: int = 100,
        channel: int = 20,
        adx_min: float = 12.0,
        session_hours: str | None = None,
        exit_bars: int = 16,
        use_mid_exit: bool = True,
    ):
        self.atr_pct_min = float(atr_pct_min)
        self.atr_lookback = int(atr_lookback)
        self.channel = int(channel)
        self.adx_min = float(adx_min)
        self.session_hours = session_hours or None
        self.exit_bars = int(exit_bars)
        self.use_mid_exit = bool(use_mid_exit)

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
        need = {"atr", "adx", "high", "low"}
        missing = need - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        px = data["signal_close"] if "signal_close" in data.columns else data["close"]
        atr = data["atr"]
        roll_min = atr.rolling(self.atr_lookback, min_periods=20).min()
        roll_max = atr.rolling(self.atr_lookback, min_periods=20).max()
        span = (roll_max - roll_min).replace(0, np.nan)
        rank = ((atr - roll_min) / span).clip(0, 1)
        high_vol = rank >= self.atr_pct_min
        # Prior-window channel (exclude current bar) + use lagged features already in pipeline when channel==20
        if self.channel == 20 and {"donchian_upper", "donchian_lower", "donchian_mid"}.issubset(data.columns):
            upper, lower, mid = data["donchian_upper"], data["donchian_lower"], data["donchian_mid"]
        else:
            from mt5_swing.features.indicators import donchian, lag
            lag_n = int(getattr(data, "attrs", {}).get("signal_lag", 1) or 1)
            u, l, m = donchian(data["high"], data["low"], self.channel)
            upper, lower, mid = lag(u, lag_n), lag(l, lag_n), lag(m, lag_n)
        sess = self._session_mask(data.index)
        strong = data["adx"] >= self.adx_min
        filt = high_vol & sess & strong
        long_cond = filt & (px > upper)
        short_cond = filt & (px < lower)
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        last = int(Signal.FLAT)
        held = 0
        for i in range(len(sig)):
            if pd.isna(upper.iloc[i]) or pd.isna(rank.iloc[i]):
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
                if self.use_mid_exit:
                    if last == int(Signal.LONG) and px.iloc[i] < mid.iloc[i]:
                        last = int(Signal.FLAT)
                        held = 0
                    elif last == int(Signal.SHORT) and px.iloc[i] > mid.iloc[i]:
                        last = int(Signal.FLAT)
                        held = 0
                if self.exit_bars > 0 and held >= self.exit_bars:
                    last = int(Signal.FLAT)
                    held = 0
            sig.iloc[i] = last
        return sig.astype(int)
