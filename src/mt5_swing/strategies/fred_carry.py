"""FRED rate-differential carry — Lustig–Verdelhan / Menkhoff-style sign.

Uses lagged monthly short-rate differentials (not price-trend carry_proxy).
Long the high-yielding currency in the pair when |diff| ≥ min_diff; optional
slow MA agreement filter. Publication lag on rates is mandatory (no look-ahead).
"""

from __future__ import annotations

import pandas as pd

from mt5_swing.features.macro_factors import align_to_index, pair_currencies, rate_differential
from mt5_swing.strategies.base import Signal


class FredCarry:
    name = "fred_carry"

    def __init__(
        self,
        symbol: str | None = None,
        min_diff: float = 0.25,
        monthly_lag: int = 1,
        require_trend_agree: bool = True,
        trend_fast: int = 48,
        trend_slow: int = 120,
        session_hours: str | None = None,
        max_hold: int = 0,
    ):
        self.symbol = (symbol or "").upper() or None
        self.min_diff = float(min_diff)
        self.monthly_lag = int(monthly_lag)
        self.require_trend_agree = bool(require_trend_agree)
        self.trend_fast = int(trend_fast)
        self.trend_slow = int(trend_slow)
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
        if not self.symbol:
            sym = getattr(data, "attrs", {}).get("symbol")
            if sym:
                self.symbol = str(sym).upper()
        if not self.symbol:
            raise ValueError("fred_carry requires symbol")
        base, quote = pair_currencies(self.symbol)
        diff = rate_differential(base, quote, monthly_lag=self.monthly_lag)
        diff_on = align_to_index(diff, data.index)
        # One more bar shift so month-end rate is known before next session
        diff_on = diff_on.shift(1)
        px = data["signal_close"] if "signal_close" in data.columns else data["close"]
        sess = self._session_mask(data.index)
        long_carry = (diff_on >= self.min_diff) & sess
        short_carry = (diff_on <= -self.min_diff) & sess
        if self.require_trend_agree:
            sma_f = px.rolling(self.trend_fast, min_periods=self.trend_fast).mean()
            sma_s = px.rolling(self.trend_slow, min_periods=self.trend_slow).mean()
            long_carry = long_carry & (sma_f > sma_s)
            short_carry = short_carry & (sma_f < sma_s)
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        last = int(Signal.FLAT)
        held = 0
        for i in range(len(sig)):
            if pd.isna(diff_on.iloc[i]):
                last = int(Signal.FLAT)
                held = 0
            elif bool(long_carry.iloc[i]):
                last = int(Signal.LONG)
                held = 0
            elif bool(short_carry.iloc[i]):
                last = int(Signal.SHORT)
                held = 0
            elif last != int(Signal.FLAT):
                # Flat when differential collapses inside band
                if abs(float(diff_on.iloc[i])) < self.min_diff:
                    last = int(Signal.FLAT)
                    held = 0
                elif self.max_hold > 0:
                    held += 1
                    if held >= self.max_hold:
                        last = int(Signal.FLAT)
                        held = 0
            sig.iloc[i] = last
        return sig.astype(int)
