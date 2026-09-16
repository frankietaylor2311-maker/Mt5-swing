"""Trend-following: SMA crossover filtered by ADX (+ optional ATR / session filters)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mt5_swing.strategies.base import Signal


class TrendMAADX:
    name = "trend_ma_adx"

    def __init__(
        self,
        adx_threshold: float = 22.0,
        atr_pct_min: float = 0.0,
        atr_pct_max: float = 1.0,
        atr_lookback: int = 100,
        session_hours: str | None = None,
        require_ema_align: bool = False,
        require_htf_align: bool = False,
    ):
        """
        Parameters
        ----------
        adx_threshold : minimum ADX to allow trend entries
        atr_pct_min/max : keep trades when ATR rank in [min, max] over lookback
        session_hours : optional ``start-end`` UTC hours inclusive, e.g. ``7-20``
        require_ema_align : also require close vs ema_fast alignment with SMA side
        """
        self.adx_threshold = float(adx_threshold)
        self.atr_pct_min = float(atr_pct_min)
        self.atr_pct_max = float(atr_pct_max)
        self.atr_lookback = int(atr_lookback)
        self.session_hours = session_hours or None
        self.require_ema_align = bool(require_ema_align)
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

    def _atr_rank_ok(self, data: pd.DataFrame) -> pd.Series:
        if "atr" not in data.columns:
            return pd.Series(True, index=data.index)
        if self.atr_pct_min <= 0 and self.atr_pct_max >= 1:
            return pd.Series(True, index=data.index)
        atr = data["atr"]
        # Causal percentile rank of current ATR vs past lookback (excludes future)
        roll_min = atr.rolling(self.atr_lookback, min_periods=max(20, self.atr_lookback // 5)).min()
        roll_max = atr.rolling(self.atr_lookback, min_periods=max(20, self.atr_lookback // 5)).max()
        span = (roll_max - roll_min).replace(0, np.nan)
        rank = ((atr - roll_min) / span).clip(0, 1)
        return (rank >= self.atr_pct_min) & (rank <= self.atr_pct_max)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        required = {"sma_fast", "sma_slow", "adx"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"{self.name} missing features: {missing}")
        sess = self._session_mask(data.index)
        atr_ok = self._atr_rank_ok(data)
        trend_up = data["sma_fast"] > data["sma_slow"]
        trend_dn = data["sma_fast"] < data["sma_slow"]
        strong = data["adx"] >= self.adx_threshold
        if self.require_ema_align and "ema_fast" in data.columns:
            px = data["signal_close"] if "signal_close" in data.columns else data["close"]
            # Prefer lagged close if present via features — use sma_fast as proxy when needed
            trend_up = trend_up & (data["ema_fast"] > data["sma_slow"])
            trend_dn = trend_dn & (data["ema_fast"] < data["sma_slow"])
            _ = px  # reserved for future filters
        if self.require_htf_align and "htf_sma_fast" in data.columns:
            htf_up = data["htf_sma_fast"] > data["htf_sma_slow"]
            htf_dn = data["htf_sma_fast"] < data["htf_sma_slow"]
            trend_up = trend_up & htf_up
            trend_dn = trend_dn & htf_dn
        long_cond = trend_up & strong & sess & atr_ok
        short_cond = trend_dn & strong & sess & atr_ok
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        sig = sig.mask(long_cond, int(Signal.LONG))
        sig = sig.mask(short_cond, int(Signal.SHORT))
        nan_mask = data[["sma_fast", "sma_slow", "adx"]].isna().any(axis=1)
        sig = sig.mask(nan_mask, int(Signal.FLAT))
        return sig
