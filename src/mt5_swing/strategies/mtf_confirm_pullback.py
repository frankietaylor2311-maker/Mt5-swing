"""ABORTED mid-wave (2026-09-17): Frankie steered away from technical
indicator/strategy number-hunting toward literature-backed macro / geopolitics.
Left on disk unregistered — do not promote or grid-tune this wave.
"""

"""Multi-TF confirmation pullback — structurally distinct from ema_pullback.

Primary gate is HTF trend (htf_sma_fast vs htf_sma_slow). Entries only on LTF
RSI pullbacks *with* HTF alignment; exits on HTF flip or RSI mean-revert.
ema_pullback uses same-TF ema_fast vs sma_slow and does not require HTF.
"""

from __future__ import annotations

import pandas as pd

from mt5_swing.strategies.base import Signal


class MtfConfirmPullback:
    name = "mtf_confirm_pullback"

    def __init__(
        self,
        adx_min: float = 16.0,
        rsi_pull_low: float = 42.0,
        rsi_pull_high: float = 58.0,
        rsi_exit_long: float = 58.0,
        rsi_exit_short: float = 42.0,
        session_hours: str | None = None,
        max_hold: int = 0,
    ):
        self.adx_min = float(adx_min)
        self.rsi_pull_low = float(rsi_pull_low)
        self.rsi_pull_high = float(rsi_pull_high)
        self.rsi_exit_long = float(rsi_exit_long)
        self.rsi_exit_short = float(rsi_exit_short)
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
        need = {"htf_sma_fast", "htf_sma_slow", "rsi", "adx"}
        missing = need - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        sess = self._session_mask(data.index)
        htf_up = data["htf_sma_fast"] > data["htf_sma_slow"]
        htf_dn = data["htf_sma_fast"] < data["htf_sma_slow"]
        strong = data["adx"] >= self.adx_min
        long_entry = htf_up & strong & sess & (data["rsi"] <= self.rsi_pull_low)
        short_entry = htf_dn & strong & sess & (data["rsi"] >= self.rsi_pull_high)
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        last = int(Signal.FLAT)
        held = 0
        for i in range(len(sig)):
            if pd.isna(data["htf_sma_fast"].iloc[i]) or pd.isna(data["rsi"].iloc[i]):
                last = int(Signal.FLAT)
                held = 0
            elif bool(long_entry.iloc[i]):
                last = int(Signal.LONG)
                held = 0
            elif bool(short_entry.iloc[i]):
                last = int(Signal.SHORT)
                held = 0
            elif last == int(Signal.LONG):
                # Exit: HTF flips OR RSI mean-reverts up through exit level
                if (not bool(htf_up.iloc[i])) or data["rsi"].iloc[i] >= self.rsi_exit_long:
                    last = int(Signal.FLAT)
                    held = 0
                else:
                    held += 1
            elif last == int(Signal.SHORT):
                if (not bool(htf_dn.iloc[i])) or data["rsi"].iloc[i] <= self.rsi_exit_short:
                    last = int(Signal.FLAT)
                    held = 0
                else:
                    held += 1
            if last != int(Signal.FLAT) and self.max_hold > 0 and held >= self.max_hold:
                last = int(Signal.FLAT)
                held = 0
            sig.iloc[i] = last
        return sig.astype(int)
