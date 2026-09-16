"""Carry-proxy swing: slow trend with optional long-bias on classic FX carry pairs.

No rate feeds — uses price trend as a funding/carry proxy. Pair bias is a prior
(not fit on holdout): AUD/NZD vs JPY/CHF style symbols prefer long; funding
currencies prefer short when trend agrees.
"""

from __future__ import annotations

import pandas as pd

from mt5_swing.strategies.base import Signal

# Soft directional prior by symbol (research prior, not optimized on OOS/holdout)
_LONG_BIAS = {
    "AUDJPY", "NZDJPY", "AUDCHF", "NZDCHF", "AUDUSD", "NZDUSD", "AUDCAD", "EURAUD",
}
_SHORT_BIAS = {
    "USDJPY", "EURJPY", "GBPJPY", "CADJPY", "USDCHF", "EURCHF",
}


class CarryProxy:
    name = "carry_proxy"

    def __init__(
        self,
        fast: int = 48,
        slow: int = 120,
        adx_min: float = 14.0,
        bias_mode: str = "auto",  # auto | long | short | neutral
        symbol: str | None = None,
        session_hours: str | None = None,
        require_adx: bool = True,
        max_hold: int = 0,
    ):
        self.fast = int(fast)
        self.slow = int(slow)
        self.adx_min = float(adx_min)
        self.bias_mode = str(bias_mode)
        self.symbol = (symbol or "").upper() or None
        self.session_hours = session_hours or None
        self.require_adx = bool(require_adx)
        self.max_hold = int(max_hold)

    def _bias(self) -> str:
        if self.bias_mode != "auto":
            return self.bias_mode
        if self.symbol in _LONG_BIAS:
            return "long"
        if self.symbol in _SHORT_BIAS:
            return "short"
        return "neutral"

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
        px = data["signal_close"] if "signal_close" in data.columns else data["close"]
        # Causal SMAs on lagged signal_close (already lagged close)
        sma_f = px.rolling(self.fast, min_periods=self.fast).mean()
        sma_s = px.rolling(self.slow, min_periods=self.slow).mean()
        sess = self._session_mask(data.index)
        ok = sess.copy()
        if self.require_adx and "adx" in data.columns:
            ok &= data["adx"] >= self.adx_min
        up = sma_f > sma_s
        down = sma_f < sma_s
        bias = self._bias()
        if bias == "long":
            long_cond = ok & up
            short_cond = ok & down & (sma_f < sma_s * 0.998)  # stricter for shorts
        elif bias == "short":
            short_cond = ok & down
            long_cond = ok & up & (sma_f > sma_s * 1.002)
        else:
            long_cond = ok & up
            short_cond = ok & down
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        last = int(Signal.FLAT)
        held = 0
        for i in range(len(sig)):
            if pd.isna(sma_f.iloc[i]) or pd.isna(sma_s.iloc[i]):
                last = int(Signal.FLAT)
                held = 0
            elif bool(long_cond.iloc[i]):
                last = int(Signal.LONG)
                held = 0
            elif bool(short_cond.iloc[i]):
                last = int(Signal.SHORT)
                held = 0
            elif last != int(Signal.FLAT):
                # Exit on MA cross against position
                if last == int(Signal.LONG) and bool(down.iloc[i]):
                    last = int(Signal.FLAT)
                    held = 0
                elif last == int(Signal.SHORT) and bool(up.iloc[i]):
                    last = int(Signal.FLAT)
                    held = 0
                elif self.max_hold > 0:
                    held += 1
                    if held >= self.max_hold:
                        last = int(Signal.FLAT)
                        held = 0
            sig.iloc[i] = last
        return sig.astype(int)
