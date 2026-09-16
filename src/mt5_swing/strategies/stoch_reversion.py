"""Stochastic oscillator mean-reversion with optional ADX calm filter.

Distinct from bbands_reversion: fades %K extremes rather than price vs bands.
"""

from __future__ import annotations

import pandas as pd

from mt5_swing.strategies.base import Signal


class StochReversion:
    name = "stoch_reversion"

    def __init__(
        self,
        adx_max: float = 28.0,
        stoch_low: float = 20.0,
        stoch_high: float = 80.0,
        exit_mid: float = 50.0,
        require_htf_align: bool = False,
        session_hours: str | None = None,
    ):
        self.adx_max = float(adx_max)
        self.stoch_low = float(stoch_low)
        self.stoch_high = float(stoch_high)
        self.exit_mid = float(exit_mid)
        self.require_htf_align = bool(require_htf_align)
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

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        need = {"stoch_k", "adx"}
        missing = need - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        k = data["stoch_k"]
        sess = self._session_mask(data.index)
        calm = data["adx"] <= self.adx_max
        long_cond = calm & sess & (k < self.stoch_low)
        short_cond = calm & sess & (k > self.stoch_high)
        if self.require_htf_align and "htf_sma_fast" in data.columns and "htf_sma_slow" in data.columns:
            htf_up = data["htf_sma_fast"] > data["htf_sma_slow"]
            htf_dn = data["htf_sma_fast"] < data["htf_sma_slow"]
            long_cond = long_cond & htf_up
            short_cond = short_cond & htf_dn
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        last = int(Signal.FLAT)
        for i in range(len(sig)):
            if pd.isna(k.iloc[i]) or pd.isna(data["adx"].iloc[i]):
                last = int(Signal.FLAT)
            elif bool(long_cond.iloc[i]):
                last = int(Signal.LONG)
            elif bool(short_cond.iloc[i]):
                last = int(Signal.SHORT)
            elif last == int(Signal.LONG) and k.iloc[i] >= self.exit_mid:
                last = int(Signal.FLAT)
            elif last == int(Signal.SHORT) and k.iloc[i] <= self.exit_mid:
                last = int(Signal.FLAT)
            sig.iloc[i] = last
        return sig.astype(int)
