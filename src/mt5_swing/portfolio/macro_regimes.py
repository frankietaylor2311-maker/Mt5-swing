"""Literature-backed macro / geopolitical regime scales (not equity-curve cools).

Fixed a-priori thresholds from scholar designs — NOT a knob grid on locked FX4+.
- Menkhoff et al. (2012): high global FX / equity vol → cut carry risk
- Caldara & Iacoviello (2022) / Liu & Zhang (2024): elevated GPR → risk-off / safe tilt
All scales use lagged factors only; hi≤1 (no RF hike).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mt5_swing.features.macro_factors import load_gpr_daily, load_vix, rolling_z, align_to_index


def apply_vix_risk_gate(
    port: pd.Series,
    *,
    z_thresh: float = 1.0,
    cool_scale: float = 0.35,
    lookback: int = 252,
    lo: float = 0.0,
    hi: float = 1.0,
) -> pd.Series:
    """When lag-1 VIX z-score ≥ thresh, scale next-bar returns by cool_scale (≤1)."""
    if port is None or len(port) < 40:
        return port if port is not None else pd.Series(dtype=float)
    eq = port.astype(float)
    r = eq.pct_change().fillna(0.0)
    vix = load_vix(bar_lag=1)
    z = rolling_z(vix, lookback=lookback)
    z_on = align_to_index(z, eq.index)
    # Extra shift so bar t uses info known at t-1 close (VIX already lag=1)
    scale = pd.Series(1.0, index=eq.index)
    hot = z_on.shift(1) >= float(z_thresh)
    scale = scale.where(~hot.fillna(False), float(cool_scale))
    scale = scale.clip(lower=float(lo), upper=float(hi)).fillna(1.0)
    return (1.0 + r * scale).cumprod() * float(eq.iloc[0])


def apply_gpr_risk_gate(
    port: pd.Series,
    *,
    z_thresh: float = 1.5,
    cool_scale: float = 0.35,
    lookback: int = 252,
    lo: float = 0.0,
    hi: float = 1.0,
) -> pd.Series:
    """When lag-1 GPR z-score ≥ thresh, scale next-bar returns (geopolitical risk-off)."""
    if port is None or len(port) < 40:
        return port if port is not None else pd.Series(dtype=float)
    eq = port.astype(float)
    r = eq.pct_change().fillna(0.0)
    gpr = load_gpr_daily(bar_lag=1)
    z = rolling_z(gpr, lookback=lookback)
    z_on = align_to_index(z, eq.index)
    scale = pd.Series(1.0, index=eq.index)
    hot = z_on.shift(1) >= float(z_thresh)
    scale = scale.where(~hot.fillna(False), float(cool_scale))
    scale = scale.clip(lower=float(lo), upper=float(hi)).fillna(1.0)
    return (1.0 + r * scale).cumprod() * float(eq.iloc[0])


def apply_vix_gpr_stack(
    port: pd.Series,
    *,
    vix_z: float = 1.0,
    gpr_z: float = 1.5,
    cool_scale: float = 0.35,
    lookback: int = 252,
) -> pd.Series:
    """Sequential literature gates: VIX then GPR (both hi≤1)."""
    return apply_gpr_risk_gate(
        apply_vix_risk_gate(port, z_thresh=vix_z, cool_scale=cool_scale, lookback=lookback),
        z_thresh=gpr_z,
        cool_scale=cool_scale,
        lookback=lookback,
    )


SAFE_HAVEN = {"USDCHF", "USDJPY", "EURCHF", "EURJPY"}  # long quote-safe or short risk
RISK_ON = {"AUDUSD", "NZDUSD", "AUDJPY", "EURAUD", "GBPJPY"}


def gpr_safe_haven_weight_mult(symbol: str, gpr_z: float, *, z_thresh: float = 1.5) -> float:
    """A priori tilt: in high-GPR regimes overweight safe / underweight risk FX."""
    if gpr_z != gpr_z or gpr_z < float(z_thresh):
        return 1.0
    sym = symbol.upper()
    if sym in SAFE_HAVEN:
        return 1.25  # relative tilt within basket renormalization — not RF hike
    if sym in RISK_ON:
        return 0.5
    return 0.85
