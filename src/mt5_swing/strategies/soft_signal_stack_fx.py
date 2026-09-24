"""Dahlquist-style equal-weight soft-signal recombination FX (§66).

Pre-registered BEFORE any holdout peek (2026-09-24)
----------------------------------------------------
SOFT_LEGS fixed a priori from prior documented soft boards that cleared
soft |NW t|≥1.5 with **positive** full-sample mean on their own boards:

1. ``bci_chg_xs`` (§39 OECD BCI) — ~+14.8 bp/mo NW t≈+2.34  (primary BCI seed;
   do NOT also include ``high_bci_z_xs`` in the same EW — correlated companion)
2. ``high_ip_xs`` (§42 IP) — ~+10.1 bp/mo NW t≈+1.88
3. ``high_ppi_xs`` (§51 PPI) — ~+11.6 bp/mo NW t≈+2.00
4. ``low_gdp_xs`` (§58 GDP honesty reverse) — ~+10.2 bp/mo NW t≈+1.55
5. ``low_cars_xs`` (§60 cars honesty reverse) — ~+12.0 bp/mo NW t≈+1.93

Canonical soft stack (do not HO-tune membership)::

    SOFT_LEGS = ["bci_chg_xs", "high_ip_xs", "high_ppi_xs", "low_gdp_xs", "low_cars_xs"]

Literature: Dahlquist & Hasseltoft (2020) *Economic momentum* — equal-weight
of macro-momentum signals; recombination after single-series promote=0.

Distinct from combo §8 (carry+mom+dollar × VIX/GPR), capital-sleeve mix §53
(locked fx4plus core + BCI/PPI capital shares), and macro_diff blend
(CPI+IP+UR EW inside one loader). This module equal-weights previously
soft-boarded scholarly XS *daily returns*, rebuilt with the same PIT lags as
source waves — not a capital mix with the locked sleeve.

Board combinations (n≈7, fixed a priori)
----------------------------------------
- ``soft_ew5`` — EW of all 5 SOFT_LEGS (**primary**)
- ``soft_ew_growth`` — EW of growth-channel only: high_ip, high_ppi, bci_chg
- ``soft_ew_honesty`` — EW of low_gdp, low_cars only
- ``soft_ew4_no_bci`` / ``soft_ew4_no_ip`` / ``soft_ew4_no_ppi`` — leave-one-out
- ``soft_ew3_core`` — EW of bci_chg + high_ip + high_ppi (same-sign growth priors)

Row-wise ``nanmean`` of available legs; require ≥ ``min_legs`` (default 2)
non-null legs on a day. Costs already applied inside source factor returns
(1.5 bps/side). Explicit: do **not** overlay on locked fx4plus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PRE-REGISTERED soft legs — fixed BEFORE any holdout read. Do not edit from HO.
# ---------------------------------------------------------------------------
SOFT_LEGS: list[str] = [
    "bci_chg_xs",  # §39 OECD BCI soft (~+14.8 bp/mo NW t≈+2.34)
    "high_ip_xs",  # §42 IP soft (~+10.1 bp/mo NW t≈+1.88)
    "high_ppi_xs",  # §51 PPI soft (~+11.6 bp/mo NW t≈+2.00)
    "low_gdp_xs",  # §58 GDP honesty reverse soft (~+10.2 bp/mo NW t≈+1.55)
    "low_cars_xs",  # §60 cars honesty reverse soft (~+12.0 bp/mo NW t≈+1.93)
]

GROWTH_LEGS: list[str] = ["high_ip_xs", "high_ppi_xs", "bci_chg_xs"]
HONESTY_LEGS: list[str] = ["low_gdp_xs", "low_cars_xs"]
CORE_GROWTH_LEGS: list[str] = ["bci_chg_xs", "high_ip_xs", "high_ppi_xs"]

# Named board specs: name → ordered subset of SOFT_LEGS (or documented subset)
BOARD_SPECS: dict[str, list[str]] = {
    "soft_ew5": list(SOFT_LEGS),
    "soft_ew_growth": list(GROWTH_LEGS),
    "soft_ew_honesty": list(HONESTY_LEGS),
    "soft_ew4_no_bci": [x for x in SOFT_LEGS if x != "bci_chg_xs"],
    "soft_ew4_no_ip": [x for x in SOFT_LEGS if x != "high_ip_xs"],
    "soft_ew4_no_ppi": [x for x in SOFT_LEGS if x != "high_ppi_xs"],
    "soft_ew3_core": list(CORE_GROWTH_LEGS),
}

PRIMARY = "soft_ew5"


@dataclass
class SoftSignalStackConfig:
    """Fixed research priors — do not grid / holdout-tune membership."""

    soft_legs: list[str] = field(default_factory=lambda: list(SOFT_LEGS))
    min_legs: int = 2  # require ≥2 available legs on a day
    cost_bps_side: float = 1.5  # informational; costs live in source factors
    # Board membership is BOARD_SPECS (pre-registered); not tunable here.


def equal_weight_daily(
    legs: Mapping[str, pd.Series],
    names: Sequence[str],
    *,
    min_legs: int = 2,
    name: str = "ew",
) -> pd.Series:
    """Row-wise nanmean of named daily return series; NaN when < min_legs present."""
    cols = []
    for n in names:
        if n not in legs:
            continue
        s = legs[n]
        if not isinstance(s, pd.Series):
            raise TypeError(f"leg {n!r} must be a Series")
        cols.append(s.astype(float).rename(n))
    if not cols:
        return pd.Series(dtype=float, name=name)
    df = pd.concat(cols, axis=1, sort=True)
    count = df.notna().sum(axis=1)
    ew = df.mean(axis=1, skipna=True)
    ew = ew.where(count >= int(min_legs))
    ew.name = name
    return ew


def soft_signal_stack_factor_returns(
    leg_returns: Mapping[str, pd.Series],
    *,
    cfg: SoftSignalStackConfig | None = None,
    board_specs: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, pd.Series]:
    """Build soft-stack board EW combinations from named daily factor returns.

    ``leg_returns`` must contain the SOFT_LEGS keys (missing legs are skipped
    day-wise via nanmean; a board factor is empty if no matching legs exist).
    Membership is fixed a priori in ``BOARD_SPECS`` / ``cfg.soft_legs`` —
    do not select from holdout.
    """
    cfg = cfg or SoftSignalStackConfig()
    specs = dict(board_specs) if board_specs is not None else dict(BOARD_SPECS)
    # Restrict leave-one-out / growth / honesty to the configured soft_legs
    # when custom soft_legs provided (still a priori, not HO).
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
