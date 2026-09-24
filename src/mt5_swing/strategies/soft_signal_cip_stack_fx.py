"""Dahlquist-style soft-signal EW + CIP soft-leg enrichment (§69).

Pre-registered BEFORE any holdout peek (2026-09-24)
----------------------------------------------------
SOFT_LEGS_CIP = macro soft5 (§66) + two CIP softs that cleared soft |NW t|≥1.5
with **positive** full-sample mean on the §67 board:

1. ``bci_chg_xs`` (§39 OECD BCI)
2. ``high_ip_xs`` (§42 IP)
3. ``high_ppi_xs`` (§51 PPI)
4. ``low_gdp_xs`` (§58 GDP honesty reverse)
5. ``low_cars_xs`` (§60 cars honesty reverse)
6. ``ust_premium_haven_usd`` (§67 soft-best) — ONE haven seed only;
   do NOT also include ``cip_stress_haven_usd`` (same UST-premium channel)
7. ``cip_ew`` (§67 soft) — ONE CIP XS EW seed;
   do NOT also include ``cip_carry_ew`` (already a blend)

Canonical soft CIP stack (do not HO-tune membership)::

    SOFT_LEGS_CIP = [
        "bci_chg_xs", "high_ip_xs", "high_ppi_xs", "low_gdp_xs", "low_cars_xs",
        "ust_premium_haven_usd", "cip_ew",
    ]

Literature: Dahlquist & Hasseltoft (2020) *Economic momentum* EW recombination
+ Du–Keerati–Schreger / Du–Tepper–Verdelhan CIP / Treasury-premium channel.

Distinct from soft_ew §66 (macro5 only), CIP XS §67, CIP-conditioned carry §68,
capital-sleeve §53, combo §8. This module equal-weights previously soft-boarded
scholarly XS *daily returns*, rebuilt with the same PIT lags as source waves —
not a capital mix with the locked sleeve; does **not** mutate §66 SOFT_LEGS.

Board combinations (n≈7, fixed a priori)
----------------------------------------
- ``soft_ew7_cip`` — EW of all 7 SOFT_LEGS_CIP (**primary**)
- ``soft_ew_macro5`` — original §66 five only (baseline)
- ``soft_ew_cip2`` — ust_premium_haven_usd + cip_ew only
- ``soft_ew6_no_haven`` / ``soft_ew6_no_cip_ew`` — leave-one-out CIP legs
- ``soft_ew_growth_cip`` — growth3 (bci_chg, high_ip, high_ppi) + both CIP softs
- ``soft_ew_honesty_cip`` — honesty2 (low_gdp, low_cars) + both CIP softs

Row-wise ``nanmean`` of available legs; require ≥ ``min_legs`` (default 2)
non-null legs on a day. Costs already applied inside source factor returns
(1.5 bps/side). Explicit: do **not** overlay on locked fx4plus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pandas as pd

from mt5_swing.strategies.soft_signal_stack_fx import equal_weight_daily

# ---------------------------------------------------------------------------
# PRE-REGISTERED soft CIP legs — fixed BEFORE any holdout read. Do not edit from HO.
# ---------------------------------------------------------------------------
MACRO_SOFT5: list[str] = [
    "bci_chg_xs",  # §39 OECD BCI soft
    "high_ip_xs",  # §42 IP soft
    "high_ppi_xs",  # §51 PPI soft
    "low_gdp_xs",  # §58 GDP honesty reverse soft
    "low_cars_xs",  # §60 cars honesty reverse soft
]

CIP_SOFT2: list[str] = [
    "ust_premium_haven_usd",  # §67 soft-best (~+7.6 bp NW t≈+1.94)
    "cip_ew",  # §67 soft (~+6.1 bp NW t≈+1.68)
]

SOFT_LEGS_CIP: list[str] = list(MACRO_SOFT5) + list(CIP_SOFT2)

GROWTH3: list[str] = ["bci_chg_xs", "high_ip_xs", "high_ppi_xs"]
HONESTY2: list[str] = ["low_gdp_xs", "low_cars_xs"]

# Named board specs: name → ordered subset (fixed a priori)
BOARD_SPECS: dict[str, list[str]] = {
    "soft_ew7_cip": list(SOFT_LEGS_CIP),
    "soft_ew_macro5": list(MACRO_SOFT5),
    "soft_ew_cip2": list(CIP_SOFT2),
    "soft_ew6_no_haven": [x for x in SOFT_LEGS_CIP if x != "ust_premium_haven_usd"],
    "soft_ew6_no_cip_ew": [x for x in SOFT_LEGS_CIP if x != "cip_ew"],
    "soft_ew_growth_cip": list(GROWTH3) + list(CIP_SOFT2),
    "soft_ew_honesty_cip": list(HONESTY2) + list(CIP_SOFT2),
}

PRIMARY = "soft_ew7_cip"


@dataclass
class SoftSignalCipStackConfig:
    """Fixed research priors — do not grid / holdout-tune membership."""

    soft_legs: list[str] = field(default_factory=lambda: list(SOFT_LEGS_CIP))
    min_legs: int = 2  # require ≥2 available legs on a day
    cost_bps_side: float = 1.5  # informational; costs live in source factors
    # Board membership is BOARD_SPECS (pre-registered); not tunable here.


def soft_signal_cip_stack_factor_returns(
    leg_returns: Mapping[str, pd.Series],
    *,
    cfg: SoftSignalCipStackConfig | None = None,
    board_specs: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, pd.Series]:
    """Build CIP-enriched soft-stack board EW combinations from named daily returns.

    ``leg_returns`` should contain SOFT_LEGS_CIP keys (missing legs skipped
    day-wise via nanmean). Membership is fixed a priori in ``BOARD_SPECS`` /
    ``cfg.soft_legs`` — do not select from holdout.
    """
    cfg = cfg or SoftSignalCipStackConfig()
    specs = dict(board_specs) if board_specs is not None else dict(BOARD_SPECS)
    allowed = set(cfg.soft_legs)
    out: dict[str, pd.Series] = {}
    for fname, names in specs.items():
        use = [n for n in names if n in allowed]
        if not use:
            continue
        out[fname] = equal_weight_daily(
            leg_returns, use, min_legs=cfg.min_legs, name=fname
        )
    return out


__all__ = [
    "BOARD_SPECS",
    "CIP_SOFT2",
    "GROWTH3",
    "HONESTY2",
    "MACRO_SOFT5",
    "PRIMARY",
    "SOFT_LEGS_CIP",
    "SoftSignalCipStackConfig",
    "equal_weight_daily",
    "soft_signal_cip_stack_factor_returns",
]
