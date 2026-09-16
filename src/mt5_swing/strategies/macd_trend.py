"""MACD trend-follow with ADX filter and optional HTF SMA alignment."""

from __future__ import annotations

import pandas as pd

from mt5_swing.strategies.base import Signal


class MacdTrend:
    name = "macd_trend"

    def __init__(
        self,
        adx_min: float = 18.0,
        require_htf_align: bool = False,
        exit_on_cross: bool = True,
        session_hours: str | None = None,
        max_hold: int = 0,
    ):
        self.adx_min = float(adx_min)
        self.require_htf_align = bool(require_htf_align)
        self.exit_on_cross = bool(exit_on_cross)
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
        need = {"macd", "macd_signal", "adx"}
        missing = need - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        sess = self._session_mask(data.index)
        macd = data["macd"]
        sig_line = data["macd_signal"]
        hist = macd - sig_line
        prev_hist = hist.shift(1)
        strong = data["adx"] >= self.adx_min
        htf_ok_long = True
        htf_ok_short = True
        if self.require_htf_align and "htf_sma_fast" in data.columns and "htf_sma_slow" in data.columns:
            htf_ok_long = data["htf_sma_fast"] > data["htf_sma_slow"]
            htf_ok_short = data["htf_sma_fast"] < data["htf_sma_slow"]
        # Cross up / down using lagged hist (features already signal_lagged)
        cross_up = (prev_hist <= 0) & (hist > 0)
        cross_dn = (prev_hist >= 0) & (hist < 0)
        long_cond = sess & strong & cross_up & htf_ok_long
        short_cond = sess & strong & cross_dn & htf_ok_short
        out = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        last = int(Signal.FLAT)
        held = 0
        for i in range(len(out)):
            if pd.isna(macd.iloc[i]) or pd.isna(data["adx"].iloc[i]):
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
                exit_cross = False
                if self.exit_on_cross:
                    if last == int(Signal.LONG) and bool(cross_dn.iloc[i]):
                        exit_cross = True
                    elif last == int(Signal.SHORT) and bool(cross_up.iloc[i]):
                        exit_cross = True
                if exit_cross or (self.max_hold > 0 and held >= self.max_hold):
                    last = int(Signal.FLAT)
                    held = 0
            out.iloc[i] = last
        return out.astype(int)
