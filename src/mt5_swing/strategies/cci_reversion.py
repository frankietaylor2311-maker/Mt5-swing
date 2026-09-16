"""CCI mean-reversion with ADX calm filter (distinct from BB/stoch)."""

from __future__ import annotations

import pandas as pd

from mt5_swing.strategies.base import Signal


class CciReversion:
    name = "cci_reversion"

    def __init__(
        self,
        adx_max: float = 28.0,
        cci_low: float = -100.0,
        cci_high: float = 100.0,
        exit_level: float = 0.0,
        require_htf_align: bool = False,
        session_hours: str | None = None,
    ):
        self.adx_max = float(adx_max)
        self.cci_low = float(cci_low)
        self.cci_high = float(cci_high)
        self.exit_level = float(exit_level)
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
        need = {"cci", "adx"}
        missing = need - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        cci = data["cci"]
        sess = self._session_mask(data.index)
        calm = data["adx"] <= self.adx_max
        long_cond = calm & sess & (cci < self.cci_low)
        short_cond = calm & sess & (cci > self.cci_high)
        if self.require_htf_align and "htf_sma_fast" in data.columns and "htf_sma_slow" in data.columns:
            htf_up = data["htf_sma_fast"] > data["htf_sma_slow"]
            htf_dn = data["htf_sma_fast"] < data["htf_sma_slow"]
            long_cond = long_cond & htf_up
            short_cond = short_cond & htf_dn
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        last = int(Signal.FLAT)
        for i in range(len(sig)):
            if pd.isna(cci.iloc[i]) or pd.isna(data["adx"].iloc[i]):
                last = int(Signal.FLAT)
            elif bool(long_cond.iloc[i]):
                last = int(Signal.LONG)
            elif bool(short_cond.iloc[i]):
                last = int(Signal.SHORT)
            elif last == int(Signal.LONG) and cci.iloc[i] >= self.exit_level:
                last = int(Signal.FLAT)
            elif last == int(Signal.SHORT) and cci.iloc[i] <= self.exit_level:
                last = int(Signal.FLAT)
            sig.iloc[i] = last
        return sig.astype(int)
