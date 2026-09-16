"""Intraday-to-swing + IS monthly-PnL stack helpers (causal, no HO peek).

Used by the quest pivot away from pairs Δz proxy:
- point-in-time closed-bar intraday → swing holds
- IS-only vol scale to ~1%/mo expected (confirm OOS frozen)
- missing-month diversifier gates (must lift 2024 AND 2025 %pos on IS)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.portfolio.overlays import apply_vol_target
from mt5_swing.portfolio.smooth_select import WindowStats, expected_monthly_from_daily_vol


def monthly_returns(port: pd.Series) -> pd.Series:
    """Month-end pct-change series (NaN-safe)."""
    if port is None or len(port) < 10:
        return pd.Series(dtype=float)
    m = port.resample("ME").last().dropna()
    return m.pct_change().dropna()


def missing_months(port: pd.Series, *, threshold: float = 0.0) -> list[pd.Timestamp]:
    """Months where portfolio return <= threshold (causal list for diversifier search)."""
    rets = monthly_returns(port)
    if rets.empty:
        return []
    return [pd.Timestamp(t) for t in rets.index[rets <= float(threshold)]]


def month_hit_rate(port: pd.Series, months: list[pd.Timestamp]) -> float:
    """Fraction of listed months with positive return (nan → 0)."""
    if not months:
        return float("nan")
    rets = monthly_returns(port)
    if rets.empty:
        return float("nan")
    hits = 0
    n = 0
    for t in months:
        if t in rets.index:
            n += 1
            if float(rets.loc[t]) > 0:
                hits += 1
    return float(hits / n) if n else float("nan")


@dataclass(frozen=True)
class DiversifierGate:
    """IS gate: diversifier must improve %pos on BOTH 2024 and 2025 vs locked."""

    ok: bool
    pct_pos_2024_base: float
    pct_pos_2024_new: float
    pct_pos_2025_base: float
    pct_pos_2025_new: float
    miss_hit_2024: float
    miss_hit_2025: float
    reason: str


def diversifier_improves_is_pct_pos(
    base_2024: WindowStats,
    new_2024: WindowStats,
    base_2025: WindowStats,
    new_2025: WindowStats,
    *,
    min_delta: float = 0.0,
) -> DiversifierGate:
    """Require strict %pos improvement on both IS years (missing-month diversifier)."""
    b24, n24 = float(base_2024.pct_pos), float(new_2024.pct_pos)
    b25, n25 = float(base_2025.pct_pos), float(new_2025.pct_pos)
    if any(x != x for x in (b24, n24, b25, n25)):
        return DiversifierGate(False, b24, n24, b25, n25, float("nan"), float("nan"), "nan_pct_pos")
    if not (new_2024.gates and new_2025.gates):
        return DiversifierGate(False, b24, n24, b25, n25, float("nan"), float("nan"), "gates_fail")
    if n24 <= b24 + float(min_delta) + 1e-12:
        return DiversifierGate(False, b24, n24, b25, n25, float("nan"), float("nan"), "2024_pct_pos_not_up")
    if n25 <= b25 + float(min_delta) + 1e-12:
        return DiversifierGate(False, b24, n24, b25, n25, float("nan"), float("nan"), "2025_pct_pos_not_up")
    return DiversifierGate(True, b24, n24, b25, n25, float("nan"), float("nan"), "ok")


def is_daily_vol(port: pd.Series, look: int = 60) -> float:
    """Trailing IS daily vol estimate (median of causal rolling std, lagged)."""
    if port is None or len(port) < max(40, look):
        return float("nan")
    r = port.pct_change().fillna(0.0)
    trail = r.shift(1).rolling(int(look), min_periods=max(20, look // 3)).std().dropna()
    if trail.empty:
        return float("nan")
    return float(trail.median())


def is_scale_for_monthly_target(
    port_is: pd.Series,
    *,
    target_mo: float = 0.01,
    assumed_sharpe: float = 1.0,
    look: int = 60,
    lo: float = 0.25,
    hi: float = 1.0,
) -> float:
    """Constant scale from IS vol only so E[mo] ≈ target under assumed Sharpe.

    ``hi≤1`` by default → never hike risk vs the input sleeve (RF fixed).
    Freeze this scalar and apply to OOS/confirm without retuning.
    """
    vol = is_daily_vol(port_is, look=look)
    if vol != vol or vol <= 1e-12:
        return 1.0
    # Expected mo from current vol
    exp_mo = expected_monthly_from_daily_vol(vol, sharpe=assumed_sharpe)
    if exp_mo <= 1e-12:
        return float(lo)
    scale = float(target_mo) / float(exp_mo)
    return float(np.clip(scale, lo, hi))


def apply_frozen_scale(port: pd.Series, scale: float) -> pd.Series:
    """Apply a frozen (IS-calibrated) scalar to bar returns — causal constant."""
    if port is None or len(port) < 2:
        return port if port is not None else pd.Series(dtype=float)
    s = float(np.clip(scale, 0.0, 1.0))
    r = port.pct_change().fillna(0.0)
    return (1.0 + r * s).cumprod() * float(port.iloc[0])


def apply_is_calibrated_monthly_vt(
    port: pd.Series,
    *,
    scale: float | None = None,
    port_is: pd.Series | None = None,
    target_mo: float = 0.01,
    assumed_sharpe: float = 1.0,
    look: int = 60,
    lo: float = 0.25,
    hi: float = 1.0,
) -> pd.Series:
    """Scale portfolio to ~target_mo using frozen IS scale (or calibrate from port_is)."""
    if scale is None:
        if port_is is None:
            raise ValueError("pass scale= or port_is= for IS calibration")
        scale = is_scale_for_monthly_target(
            port_is, target_mo=target_mo, assumed_sharpe=assumed_sharpe, look=look, lo=lo, hi=hi
        )
    return apply_frozen_scale(port, float(scale))


def swing_exits_for_intraday(timeframe: str, strategy: str) -> dict:
    """Map H1/M15 closed-bar strategies to swing-style hold exits (more trades/mo)."""
    tf = timeframe.upper()
    mr = {
        "mean_reversion_regime",
        "bbands_reversion",
        "cci_reversion",
        "stoch_reversion",
        "willr_reversion",
    }
    if tf == "H1":
        if strategy in mr:
            return {"max_hold_bars": 48}  # ~2 trading days
        return {"atr_stop_mult": 1.5, "atr_target_mult": 4.0, "max_hold_bars": 72}
    if tf == "M15":
        if strategy in mr:
            return {"max_hold_bars": 96}  # ~1 trading day
        return {"atr_stop_mult": 1.5, "atr_target_mult": 4.0, "max_hold_bars": 160}
    return {}


def yahoo_m15_depth_note() -> str:
    return (
        "Yahoo 15m FX history is ~60 calendar days — cannot cover 2024/2025/2026 "
        "calendars; M15 is probe-only on approximate_non_ftmo. FTMO MT5 M15 export "
        "is required for multi-year intraday-to-swing."
    )
