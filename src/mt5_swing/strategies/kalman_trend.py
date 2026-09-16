"""Simple 1-D Kalman level/velocity trend — causal recursive filter on close."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mt5_swing.strategies.base import Signal


class KalmanTrend:
    name = "kalman_trend"

    def __init__(
        self,
        process_var: float = 1e-5,
        measure_var: float = 1e-3,
        vel_thresh: float = 0.0,
        adx_min: float = 12.0,
        session_hours: str | None = None,
        max_hold: int = 0,
        exit_on_flip: bool = True,
    ):
        self.process_var = float(process_var)
        self.measure_var = float(measure_var)
        self.vel_thresh = float(vel_thresh)
        self.adx_min = float(adx_min)
        self.session_hours = session_hours or None
        self.max_hold = int(max_hold)
        self.exit_on_flip = bool(exit_on_flip)

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

    def _filter(self, z: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        n = len(z)
        level = np.full(n, np.nan)
        vel = np.full(n, np.nan)
        x = np.array([np.nan, 0.0])  # level, velocity
        P = np.eye(2) * 1.0
        Q = np.eye(2) * self.process_var
        R = self.measure_var
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        for i, zi in enumerate(z.values):
            if zi != zi:  # nan
                continue
            if x[0] != x[0]:
                x[:] = [zi, 0.0]
                P[:] = np.eye(2)
                level[i] = zi
                vel[i] = 0.0
                continue
            # Predict
            x = F @ x
            P = F @ P @ F.T + Q
            # Update (scalar measurement)
            y = float(zi - (H @ x).item())
            S = float((H @ P @ H.T).item()) + R
            K = (P @ H.T) / S  # shape (2,1)
            x = x + (K.flatten() * y)
            P = (np.eye(2) - K @ H) @ P
            level[i] = float(x[0])
            vel[i] = float(x[1])
        return level, vel

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        px = data["signal_close"] if "signal_close" in data.columns else data["close"]
        # Scale threshold relative to price level for FX vs JPY
        med = float(px.dropna().median()) if px.notna().any() else 1.0
        scale = med if med > 0 else 1.0
        thr = self.vel_thresh * scale * 1e-4 if self.vel_thresh > 0 else 0.0
        # Use unlagged close? No — signal_close is lagged; filter on lagged px only.
        _level, vel = self._filter(px)
        sess = self._session_mask(data.index)
        ok = sess.copy()
        if self.adx_min > 0 and "adx" in data.columns:
            ok &= data["adx"] >= self.adx_min
        long_cond = ok & (vel > thr)
        short_cond = ok & (vel < -thr)
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        last = int(Signal.FLAT)
        held = 0
        for i in range(len(sig)):
            if vel[i] != vel[i]:
                last = int(Signal.FLAT)
                held = 0
            elif bool(long_cond.iloc[i]):
                last = int(Signal.LONG)
                held = 0
            elif bool(short_cond.iloc[i]):
                last = int(Signal.SHORT)
                held = 0
            elif last != int(Signal.FLAT):
                if self.exit_on_flip:
                    if last == int(Signal.LONG) and vel[i] < 0:
                        last = int(Signal.FLAT)
                        held = 0
                    elif last == int(Signal.SHORT) and vel[i] > 0:
                        last = int(Signal.FLAT)
                        held = 0
                if self.max_hold > 0 and last != int(Signal.FLAT):
                    held += 1
                    if held >= self.max_hold:
                        last = int(Signal.FLAT)
                        held = 0
            sig.iloc[i] = last
        return sig.astype(int)
