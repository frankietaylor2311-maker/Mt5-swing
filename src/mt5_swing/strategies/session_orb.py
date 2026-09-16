"""Session opening-range breakout adapted for H4 swing (London 07–11 UTC).

Uses the first completed H4 bar in the session window as the OR; subsequent
same-day bars break above/below. Flat at next session open (time diversification).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mt5_swing.strategies.base import Signal


class SessionORB:
    name = "session_orb"

    def __init__(
        self,
        or_hour_utc: int = 7,
        session_end_utc: int = 20,
        adx_min: float = 10.0,
        atr_pct_min: float = 0.2,
        atr_lookback: int = 100,
        buffer_atr: float = 0.1,
        max_hold: int = 6,
        session_hours: str | None = None,  # accepted for grid API compat; unused
    ):
        self.or_hour_utc = int(or_hour_utc)
        self.session_end_utc = int(session_end_utc)
        self.adx_min = float(adx_min)
        self.atr_pct_min = float(atr_pct_min)
        self.atr_lookback = int(atr_lookback)
        self.buffer_atr = float(buffer_atr)
        self.max_hold = int(max_hold)
        self.session_hours = session_hours or None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        need = {"high", "low", "atr"}
        missing = need - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        px = data["signal_close"] if "signal_close" in data.columns else data["close"]
        # Use lagged OHLC for OR formation to avoid same-bar peek on unfinished OR
        # high/low in features pipeline are unlagged for fills; for OR we use signal_close
        # and lagged high/low via shift(1) of raw — but signal_lag already applied to atr/adx.
        # Conservative: form OR from prior bar's high/low when hour matches.
        idx = data.index
        hours = idx.tz_convert("UTC").hour if idx.tz is not None else idx.hour
        dates = idx.tz_convert("UTC").date if idx.tz is not None else idx.date
        atr = data["atr"]
        roll_min = atr.rolling(self.atr_lookback, min_periods=20).min()
        roll_max = atr.rolling(self.atr_lookback, min_periods=20).max()
        span = (roll_max - roll_min).replace(0, np.nan)
        rank = ((atr - roll_min) / span).clip(0, 1)
        adx_ok = data["adx"] >= self.adx_min if "adx" in data.columns else pd.Series(True, index=idx)
        vol_ok = rank >= self.atr_pct_min

        # Lagged high/low for OR (causal vs fill)
        hi = data["high"].shift(1)
        lo = data["low"].shift(1)

        or_high = {}
        or_low = {}
        sig = pd.Series(int(Signal.FLAT), index=idx, dtype=int)
        last = int(Signal.FLAT)
        held = 0
        cur_day = None
        for i in range(len(sig)):
            d = dates[i]
            h = int(hours[i])
            if cur_day != d:
                cur_day = d
                last = int(Signal.FLAT)
                held = 0
            # Capture OR when we first see the OR hour bar (using prior completed H/L)
            if h == self.or_hour_utc and d not in or_high:
                if hi.iloc[i] == hi.iloc[i] and lo.iloc[i] == lo.iloc[i]:
                    or_high[d] = float(hi.iloc[i])
                    or_low[d] = float(lo.iloc[i])
            if d not in or_high:
                sig.iloc[i] = int(Signal.FLAT)
                continue
            if h < self.or_hour_utc or h > self.session_end_utc:
                last = int(Signal.FLAT)
                held = 0
                sig.iloc[i] = last
                continue
            buf = self.buffer_atr * float(atr.iloc[i]) if atr.iloc[i] == atr.iloc[i] else 0.0
            up = or_high[d] + buf
            dn = or_low[d] - buf
            filt = bool(adx_ok.iloc[i]) and bool(vol_ok.iloc[i])
            if filt and px.iloc[i] == px.iloc[i]:
                if px.iloc[i] > up:
                    last = int(Signal.LONG)
                    held = 0
                elif px.iloc[i] < dn:
                    last = int(Signal.SHORT)
                    held = 0
                elif last != int(Signal.FLAT):
                    held += 1
                    if self.max_hold > 0 and held >= self.max_hold:
                        last = int(Signal.FLAT)
                        held = 0
            else:
                if last != int(Signal.FLAT):
                    held += 1
                    if self.max_hold > 0 and held >= self.max_hold:
                        last = int(Signal.FLAT)
                        held = 0
            sig.iloc[i] = last
        return sig.astype(int)
