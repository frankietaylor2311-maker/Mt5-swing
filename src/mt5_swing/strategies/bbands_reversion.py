"""Bollinger band mean-reversion with optional ADX calm filter."""

from __future__ import annotations

import pandas as pd

from mt5_swing.strategies.base import Signal


class BBandsReversion:
    name = "bbands_reversion"

    def __init__(
        self,
        adx_max: float = 28.0,
        exit_to_mid: bool = True,
        session_hours: str | None = None,
        require_rsi: bool = True,
        rsi_low: float = 40.0,
        rsi_high: float = 60.0,
        require_htf_align: bool = False,
    ):
        self.adx_max = float(adx_max)
        self.exit_to_mid = bool(exit_to_mid)
        self.session_hours = session_hours or None
        self.require_rsi = bool(require_rsi)
        self.rsi_low = float(rsi_low)
        self.rsi_high = float(rsi_high)
        self.require_htf_align = bool(require_htf_align)

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
        need = {"bb_upper", "bb_lower", "bb_mid", "adx"}
        missing = need - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        px = data["signal_close"] if "signal_close" in data.columns else data["close"]
        sess = self._session_mask(data.index)
        calm = data["adx"] <= self.adx_max
        long_cond = calm & sess & (px < data["bb_lower"])
        short_cond = calm & sess & (px > data["bb_upper"])
        if self.require_rsi and "rsi" in data.columns:
            long_cond = long_cond & (data["rsi"] < self.rsi_low)
            short_cond = short_cond & (data["rsi"] > self.rsi_high)
        if self.require_htf_align and "htf_sma_fast" in data.columns and "htf_sma_slow" in data.columns:
            # Fade only with (or against) HTF — long dips in HTF uptrend, shorts in HTF downtrend
            htf_up = data["htf_sma_fast"] > data["htf_sma_slow"]
            htf_dn = data["htf_sma_fast"] < data["htf_sma_slow"]
            long_cond = long_cond & htf_up
            short_cond = short_cond & htf_dn
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        last = int(Signal.FLAT)
        for i in range(len(sig)):
            if pd.isna(data["bb_mid"].iloc[i]) or pd.isna(data["adx"].iloc[i]):
                last = int(Signal.FLAT)
            elif bool(long_cond.iloc[i]):
                last = int(Signal.LONG)
            elif bool(short_cond.iloc[i]):
                last = int(Signal.SHORT)
            elif self.exit_to_mid:
                if last == int(Signal.LONG) and px.iloc[i] >= data["bb_mid"].iloc[i]:
                    last = int(Signal.FLAT)
                elif last == int(Signal.SHORT) and px.iloc[i] <= data["bb_mid"].iloc[i]:
                    last = int(Signal.FLAT)
            sig.iloc[i] = last
        return sig.astype(int)
