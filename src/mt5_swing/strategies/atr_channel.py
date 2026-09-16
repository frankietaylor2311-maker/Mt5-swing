"""ATR channel breakout: enter when close breaks SMA ± ATR*mult; exit on mid reversion or opposite."""

from __future__ import annotations

import pandas as pd

from mt5_swing.strategies.base import Signal


class AtrChannelBreakout:
    name = "atr_channel_breakout"

    def __init__(
        self,
        atr_mult: float = 2.0,
        adx_min: float = 15.0,
        exit_to_mid: bool = True,
        require_htf_align: bool = False,
        session_hours: str | None = None,
        max_hold: int = 0,
    ):
        self.atr_mult = float(atr_mult)
        self.adx_min = float(adx_min)
        self.exit_to_mid = bool(exit_to_mid)
        self.require_htf_align = bool(require_htf_align)
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
        need = {"sma_fast", "atr", "adx", "signal_close"}
        missing = need - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        px = data["signal_close"]
        mid = data["sma_fast"]
        upper = mid + self.atr_mult * data["atr"]
        lower = mid - self.atr_mult * data["atr"]
        sess = self._session_mask(data.index)
        strong = data["adx"] >= self.adx_min
        long_cond = strong & sess & (px > upper)
        short_cond = strong & sess & (px < lower)
        if self.require_htf_align and "htf_sma_fast" in data.columns and "htf_sma_slow" in data.columns:
            htf_up = data["htf_sma_fast"] > data["htf_sma_slow"]
            htf_dn = data["htf_sma_fast"] < data["htf_sma_slow"]
            long_cond = long_cond & htf_up
            short_cond = short_cond & htf_dn
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        last = int(Signal.FLAT)
        held = 0
        for i in range(len(sig)):
            if pd.isna(upper.iloc[i]) or pd.isna(data["adx"].iloc[i]):
                last = int(Signal.FLAT)
                held = 0
            elif bool(long_cond.iloc[i]):
                last = int(Signal.LONG)
                held = 0
            elif bool(short_cond.iloc[i]):
                last = int(Signal.SHORT)
                held = 0
            elif self.exit_to_mid:
                if last == int(Signal.LONG) and px.iloc[i] <= mid.iloc[i]:
                    last = int(Signal.FLAT)
                    held = 0
                elif last == int(Signal.SHORT) and px.iloc[i] >= mid.iloc[i]:
                    last = int(Signal.FLAT)
                    held = 0
            if last != int(Signal.FLAT):
                held += 1
                if self.max_hold > 0 and held >= self.max_hold:
                    last = int(Signal.FLAT)
                    held = 0
            sig.iloc[i] = last
        return sig.astype(int)
