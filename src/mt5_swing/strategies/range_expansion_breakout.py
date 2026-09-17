"""ABORTED mid-wave (2026-09-17): Frankie steered away from technical
indicator/strategy number-hunting toward literature-backed macro / geopolitics.
Left on disk unregistered — do not promote or grid-tune this wave.
"""

"""Range-expansion breakout — different features from vol_breakout.

Uses prior-bar true-range vs a longer median TR (expansion ratio) and a break of
the prior N-bar high/low. vol_breakout uses ATR percentile rank + Donchian(20).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mt5_swing.strategies.base import Signal


class RangeExpansionBreakout:
    name = "range_expansion_breakout"

    def __init__(
        self,
        expand_lookback: int = 20,
        expand_mult: float = 1.4,
        break_window: int = 5,
        adx_min: float = 12.0,
        session_hours: str | None = None,
        max_hold: int = 12,
        use_mid_exit: bool = True,
    ):
        self.expand_lookback = int(expand_lookback)
        self.expand_mult = float(expand_mult)
        self.break_window = int(break_window)
        self.adx_min = float(adx_min)
        self.session_hours = session_hours or None
        self.max_hold = int(max_hold)
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
        need = {"high", "low", "adx"}
        missing = need - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        px = data["signal_close"] if "signal_close" in data.columns else data["close"]
        # Causal TR from lagged OHLC vs prior close (features already lag atr; rebuild TR from raw)
        prev_close = data["close"].shift(1)
        tr = pd.concat(
            [
                (data["high"] - data["low"]).abs(),
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        # Use lag-1 TR vs lag-1 median so expansion decision is causal vs fill
        lag = int(getattr(data, "attrs", {}).get("signal_lag", 1) or 1)
        tr_lag = tr.shift(lag)
        med = tr_lag.rolling(self.expand_lookback, min_periods=max(5, self.expand_lookback // 2)).median()
        expand = tr_lag >= (self.expand_mult * med)
        # Prior N-bar high/low excluding current (shift 1 then rolling)
        prior_hi = data["high"].shift(1).rolling(self.break_window, min_periods=self.break_window).max()
        prior_lo = data["low"].shift(1).rolling(self.break_window, min_periods=self.break_window).min()
        # Also lag the channel by signal_lag for consistency with signal_close
        if lag > 0:
            prior_hi = prior_hi.shift(lag - 1) if lag > 1 else prior_hi
            prior_lo = prior_lo.shift(lag - 1) if lag > 1 else prior_lo
        # When signal_lag=1, signal_close is close.shift(1); prior_hi from high.shift(1).rolling
        # already excludes current raw bar — align with signal_close by shifting channel once more
        # so we compare lagged close to a fully prior channel:
        upper = prior_hi.shift(1) if lag >= 1 else prior_hi
        lower = prior_lo.shift(1) if lag >= 1 else prior_lo
        mid = (upper + lower) / 2.0
        sess = self._session_mask(data.index)
        strong = data["adx"] >= self.adx_min
        # expand already lag-aligned; combine
        filt = expand & sess & strong
        long_cond = filt & (px > upper)
        short_cond = filt & (px < lower)
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        last = int(Signal.FLAT)
        held = 0
        for i in range(len(sig)):
            if pd.isna(upper.iloc[i]) or pd.isna(tr_lag.iloc[i]):
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
                if self.max_hold > 0 and held >= self.max_hold:
                    last = int(Signal.FLAT)
                    held = 0
            sig.iloc[i] = last
        return sig.astype(int)
