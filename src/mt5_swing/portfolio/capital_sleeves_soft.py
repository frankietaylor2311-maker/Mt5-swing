"""Capital-sleeve soft-stack mix helpers for FTMO scholarly FX research (§70).

Keep locked fx4plus as **core**; allocate smaller fixed capital shares to
pre-registered soft-stack satellites (soft_ew_macro5 / soft_ew7_cip).
Separate capital sleeves — NOT overlays/coolers on locked equity.

Does **not** mutate ``capital_sleeves.PRE_REGISTERED_MIXES`` (§53).
"""

from __future__ import annotations

from mt5_swing.portfolio.capital_sleeves import (
    CORE_NAME,
    LOCKED_TAG,
    equity_to_daily_returns,
    mix_sleeve_returns,
    select_mix_is_only,
    validate_weights,
)

# ---------------------------------------------------------------------------
# PRE-REGISTERED §70 mix grid — fixed a priori BEFORE any holdout peek.
# Satellite A = soft_ew_macro5 (§66 / §69 ~+11.8 bp/mo NW t≈+3.63)
# Satellite B = soft_ew7_cip (§69 ~+10.4 bp/mo NW t≈+3.94)
# Prefer NOT a third satellite series; grid ≤6 mixes like §53.
# ---------------------------------------------------------------------------
SATELLITE_A = "soft_ew_macro5"
SATELLITE_B = "soft_ew7_cip"

PRE_REGISTERED_MIXES_SOFT: dict[str, dict[str, float]] = {
    "core_100": {CORE_NAME: 1.0},
    "core85_soft15": {CORE_NAME: 0.85, SATELLITE_A: 0.15},
    "core70_soft30": {CORE_NAME: 0.70, SATELLITE_A: 0.30},
    "core60_soft40": {CORE_NAME: 0.60, SATELLITE_A: 0.40},
    "core80_soft10_cip10": {
        CORE_NAME: 0.80,
        SATELLITE_A: 0.10,
        SATELLITE_B: 0.10,
    },
    "core70_soft20_cip10": {
        CORE_NAME: 0.70,
        SATELLITE_A: 0.20,
        SATELLITE_B: 0.10,
    },
}

# Alias kept for callers that expect PRE_REGISTERED_MIXES in this module.
PRE_REGISTERED_MIXES = PRE_REGISTERED_MIXES_SOFT

__all__ = [
    "CORE_NAME",
    "LOCKED_TAG",
    "PRE_REGISTERED_MIXES",
    "PRE_REGISTERED_MIXES_SOFT",
    "SATELLITE_A",
    "SATELLITE_B",
    "equity_to_daily_returns",
    "mix_sleeve_returns",
    "select_mix_is_only",
    "validate_weights",
]
