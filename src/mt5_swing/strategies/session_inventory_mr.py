"""ABORTED mid-wave (2026-09-17): Frankie steered away from technical
indicator/strategy number-hunting toward literature-backed macro / geopolitics.
Left on disk unregistered — do not promote or grid-tune this wave.
"""

"""Session inventory mean-reversion — fade stretch from session open.

Structurally distinct from session_orb (breakout of opening range) and from
bbands_reversion (Bollinger bands). Tracks London-session open inventory and
fades when price is stretched by ATR multiples while ADX is calm; flat at
session end or when inventory mean-reverts to the open.
"""

from __future__ import annotations

import pandas as pd

from mt5_swing.strategies.base import Signal


class SessionInventoryMR:
    name = "session_inventory_mr"

    def __init__(
        self,
        session_open_utc: int = 7,
        session_end_utc: int = 20,
        stretch_atr: float = 0.8,
        adx_max: float = 25.0,
        max_hold: int = 8,
        session_hours: str | None = None,  # grid API compat; unused (session_* used)
    ):
        self.session_open_utc = int(session_open_utc)
        self.session_end_utc = int(session_end_utc)
        self.stretch_atr = float(stretch_atr)
        self.adx_max = float(adx_max)
        self.max_hold = int(max_hold)
        self.session_hours = session_hours or None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        need = {"atr", "adx"}
        missing = need - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        px = data["signal_close"] if "signal_close" in data.columns else data["close"]
        # Lagged open for inventory anchor (causal)
        open_px = data["open"].shift(1) if "open" in data.columns else px
        idx = data.index
        hours = idx.tz_convert("UTC").hour if idx.tz is not None else idx.hour
        dates = idx.tz_convert("UTC").date if idx.tz is not None else idx.date
        atr = data["atr"]
        calm = data["adx"] <= self.adx_max

        sess_open: dict = {}
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
            # Capture session open on first bar at/after open hour
            if d not in sess_open and h >= self.session_open_utc:
                if open_px.iloc[i] == open_px.iloc[i]:
                    sess_open[d] = float(open_px.iloc[i])
            if d not in sess_open:
                sig.iloc[i] = int(Signal.FLAT)
                continue
            if h < self.session_open_utc or h > self.session_end_utc:
                last = int(Signal.FLAT)
                held = 0
                sig.iloc[i] = last
                continue
            anchor = sess_open[d]
            a = float(atr.iloc[i]) if atr.iloc[i] == atr.iloc[i] else 0.0
            stretch = self.stretch_atr * a
            if stretch <= 0 or px.iloc[i] != px.iloc[i]:
                sig.iloc[i] = last
                continue
            # Fade inventory: long when below open - stretch, short when above open + stretch
            if bool(calm.iloc[i]) and px.iloc[i] <= anchor - stretch:
                last = int(Signal.LONG)
                held = 0
            elif bool(calm.iloc[i]) and px.iloc[i] >= anchor + stretch:
                last = int(Signal.SHORT)
                held = 0
            elif last == int(Signal.LONG) and px.iloc[i] >= anchor:
                last = int(Signal.FLAT)
                held = 0
            elif last == int(Signal.SHORT) and px.iloc[i] <= anchor:
                last = int(Signal.FLAT)
                held = 0
            elif last != int(Signal.FLAT):
                held += 1
                if self.max_hold > 0 and held >= self.max_hold:
                    last = int(Signal.FLAT)
                    held = 0
            sig.iloc[i] = last
        return sig.astype(int)
