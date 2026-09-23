"""Country-GPR bilateral (relative-to-US) panel helpers for scholarly FX §45.

Reuses ``load_country_gpr`` from ``macro_uncertainty`` (Caldara–Iacoviello
GPRC_* → currency panel; EUR = EW of DEU/FRA/ITA/ESP/NLD/BEL). Does **not**
alter §9 absolute home-GPR loaders or sorts.

Gap: no ``GPRC_NZL`` in the standard 44-country file → NZD omitted.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mt5_swing.data.macro_uncertainty import (
    CURRENCY_USD_PAIR,
    EUR_GPRC_ISO3,
    load_country_gpr,
)

DEFAULT_PUB_LAG_MONTHS = 1
FOREIGN_CCY = ("EUR", "GBP", "JPY", "CAD", "AUD", "CHF")  # NZD absent


def load_country_gpr_levels(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Publication-lagged country GPR levels by currency (incl. USD, excl. NZD)."""
    panel = load_country_gpr(
        download=download,
        pub_lag_months=int(pub_lag_months),
        force=force,
    )
    keep = [c for c in panel.columns if c.upper() != "NZD"]
    out = panel[keep].copy()
    out.attrs.update(panel.attrs)
    out.attrs["missing_currencies"] = list(
        dict.fromkeys(list(panel.attrs.get("missing_currencies") or []) + ["NZD"])
    )
    out.attrs["eur_constituents"] = list(EUR_GPRC_ISO3)
    out.attrs["bilateral_note"] = "rel_to_US = home − USD; NZD gap (no GPRC_NZL)"
    return out


def relative_gpr_panel(
    country_gpr: pd.DataFrame | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Foreign currency panel of (home − US) GPR levels.

    Columns are foreign ISO currencies present in both the level panel and
    ``CURRENCY_USD_PAIR`` (EUR/GBP/JPY/CAD/AUD/CHF when available). USD column
    is not included (differential is vs US).
    """
    levels = (
        country_gpr
        if country_gpr is not None
        else load_country_gpr_levels(
            pub_lag_months=pub_lag_months, download=download, force=force
        )
    )
    if "USD" not in levels.columns:
        raise ValueError("country GPR panel missing USD column (GPRC_USA)")
    us = levels["USD"]
    foreign = [
        c
        for c in levels.columns
        if c.upper() != "USD" and c in CURRENCY_USD_PAIR and c.upper() != "NZD"
    ]
    rel = pd.DataFrame({c: levels[c] - us for c in foreign}, index=levels.index)
    rel = rel.dropna(how="all").sort_index()
    rel.attrs.update(getattr(levels, "attrs", {}) or {})
    rel.attrs["unit"] = "home_minus_us_gpr_level"
    rel.attrs["missing_currencies"] = ["NZD"]
    rel.attrs["foreign_panel"] = list(rel.columns)
    return rel


def country_gpr_bilateral_coverage(rel: pd.DataFrame) -> pd.DataFrame:
    """Per-currency observation counts for the relative panel."""
    rows = []
    for c in list(FOREIGN_CCY) + ["NZD"]:
        if c == "NZD":
            rows.append(
                {
                    "currency": "NZD",
                    "n_obs": 0,
                    "start": None,
                    "end": None,
                    "mapped": False,
                    "note": "no GPRC_NZL",
                }
            )
            continue
        if c not in rel.columns:
            rows.append(
                {
                    "currency": c,
                    "n_obs": 0,
                    "start": None,
                    "end": None,
                    "mapped": False,
                    "note": "missing",
                }
            )
            continue
        s = rel[c].dropna()
        rows.append(
            {
                "currency": c,
                "n_obs": int(len(s)),
                "start": str(s.index.min().date()) if len(s) else None,
                "end": str(s.index.max().date()) if len(s) else None,
                "mapped": True,
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def country_gpr_csv_hint() -> Path:
    from mt5_swing.data.macro_uncertainty import country_gpr_csv_path

    return country_gpr_csv_path()
