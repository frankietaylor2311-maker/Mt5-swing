"""Capital-sleeve mix helpers for FTMO scholarly FX research.

Separate capital / risk sleeves (not overlays on locked equity). Each sleeve
contributes ``weight_i * r_i``; weights are gross capital shares summing to 1.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def validate_weights(weights: Mapping[str, float], *, tol: float = 1e-9) -> dict[str, float]:
    """Return normalised weights; raise if empty, negative, or sum≈0."""
    if not weights:
        raise ValueError("weights must be non-empty")
    out = {str(k): float(v) for k, v in weights.items()}
    for k, v in out.items():
        if not np.isfinite(v) or v < -tol:
            raise ValueError(f"weight for {k!r} must be finite and ≥0, got {v}")
    s = sum(out.values())
    if s <= tol:
        raise ValueError("weights must sum to a positive number")
    return {k: v / s for k, v in out.items()}


def mix_sleeve_returns(
    sleeves: Mapping[str, pd.Series],
    weights: Mapping[str, float],
    *,
    fill_missing: float = 0.0,
) -> pd.Series:
    """Capital-share mix: ``sum_i w_i * r_i`` on the union daily index.

    Missing sleeve observations are filled with ``fill_missing`` (default 0)
    so a satellite that starts later does not drop core capital days.
    """
    w = validate_weights(weights)
    missing = set(w) - set(sleeves)
    if missing:
        raise KeyError(f"weights reference unknown sleeves: {sorted(missing)}")
    frames = []
    for name, wi in w.items():
        r = sleeves[name]
        if not isinstance(r, pd.Series):
            raise TypeError(f"sleeve {name!r} must be a Series")
        frames.append((r.astype(float) * float(wi)).rename(name))
    if not frames:
        return pd.Series(dtype=float)
    df = pd.concat(frames, axis=1, sort=True)
    # Fill only after align so early NaNs become 0 contribution (idle capital)
    mixed = df.fillna(fill_missing).sum(axis=1)
    mixed.name = "mixed"
    return mixed


def equity_to_daily_returns(equity: pd.Series) -> pd.Series:
    """Last-per-calendar-day equity → daily pct-change (UTC-normalised index)."""
    if equity is None or equity.empty:
        return pd.Series(dtype=float)
    eq = equity.astype(float).sort_index()
    if eq.index.tz is None:
        eq.index = eq.index.tz_localize("UTC")
    else:
        eq.index = eq.index.tz_convert("UTC")
    daily = eq.resample("1D").last().dropna()
    r = daily.pct_change()
    r.name = equity.name or "daily_ret"
    return r


def select_mix_is_only(
    mix_is_mean_mo: Mapping[str, float],
    *,
    require_positive: bool = True,
) -> str:
    """Pick primary mix from IS means only (never holdout).

    Among finite means (optionally >0), choose the max. If none qualify,
    fall back to the max finite mean; if still empty, raise.
    """
    finite = {
        k: float(v)
        for k, v in mix_is_mean_mo.items()
        if v is not None and np.isfinite(float(v))
    }
    if not finite:
        raise ValueError("no finite IS mean_mo to select from")
    pool = {k: v for k, v in finite.items() if v > 0} if require_positive else finite
    if not pool:
        pool = finite
    return max(pool.items(), key=lambda kv: kv[1])[0]


# Pre-registered §53 mix grid (fixed a priori; do not expand from HO).
PRE_REGISTERED_MIXES: dict[str, dict[str, float]] = {
    "core_100": {"core": 1.0},
    "core85_bci15": {"core": 0.85, "bci_chg_xs": 0.15},
    "core70_bci30": {"core": 0.70, "bci_chg_xs": 0.30},
    "core60_bci40": {"core": 0.60, "bci_chg_xs": 0.40},
    "core80_bci10_ppi10": {"core": 0.80, "bci_chg_xs": 0.10, "high_ppi_xs": 0.10},
    "core70_bci20_ppi10": {"core": 0.70, "bci_chg_xs": 0.20, "high_ppi_xs": 0.10},
}

SATELLITE_A = "bci_chg_xs"
SATELLITE_B = "high_ppi_xs"  # a priori vs high_bci_z_xs; cross-family hard board §51
CORE_NAME = "core"
LOCKED_TAG = "fx4plus_gbpcad_d1_voltarget_0025"
