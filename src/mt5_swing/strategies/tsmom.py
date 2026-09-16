"""Time-series momentum (TSMOM): sign of past return, optional vol/ADX gate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mt5_swing.strategies.base import Signal


class TsMom:
    name = "tsmom"

    def __init__(
        self,
        lookback: int = 48,
        adx_min: float = 0.0,
        atr_pct_min: float = 0.0,
        atr_lookback: int = 100,
        session_hours: str | None = None,
        max_hold: int = 0,
        skip_zero: bool = True,
    ):
        self.lookback = int(lookback)
        self.adx_min = float(adx_min)
        self.atr_pct_min = float(atr_pct_min)
        self.atr_lookback = int(atr_lookback)
        self.session_hours = session_hours or None
        self.max_hold = int(max_hold)
        self.skip_zero = bool(skip_zero)

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

    def _atr_rank(self, data: pd.DataFrame) -> pd.Series:
        atr = data["atr"]
        roll_min = atr.rolling(self.atr_lookback, min_periods=20).min()
        roll_max = atr.rolling(self.atr_lookback, min_periods=20).max()
        span = (roll_max - roll_min).replace(0, np.nan)
        return ((atr - roll_min) / span).clip(0, 1)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        px = data["signal_close"] if "signal_close" in data.columns else data["close"]
        ret = px / px.shift(self.lookback) - 1.0
        sess = self._session_mask(data.index)
        ok = sess.copy()
        if self.adx_min > 0 and "adx" in data.columns:
            ok &= data["adx"] >= self.adx_min
        if self.atr_pct_min > 0 and "atr" in data.columns:
            ok &= self._atr_rank(data) >= self.atr_pct_min
        long_cond = ok & (ret > 0)
        short_cond = ok & (ret < 0)
        if self.skip_zero:
            flat_mom = ret.abs() < 1e-12
            long_cond &= ~flat_mom
            short_cond &= ~flat_mom
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        last = int(Signal.FLAT)
        held = 0
        for i in range(len(sig)):
            if pd.isna(ret.iloc[i]):
                last = int(Signal.FLAT)
                held = 0
            elif bool(long_cond.iloc[i]):
                last = int(Signal.LONG)
                held = 0
            elif bool(short_cond.iloc[i]):
                last = int(Signal.SHORT)
                held = 0
            elif self.max_hold > 0 and last != int(Signal.FLAT):
                held += 1
                if held >= self.max_hold:
                    last = int(Signal.FLAT)
                    held = 0
            elif not bool(ok.iloc[i]):
                last = int(Signal.FLAT)
                held = 0
            sig.iloc[i] = last
        return sig.astype(int)
