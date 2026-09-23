"""FRED / OECD MEI broad-money growth panels for monetary-approach FX (free, PIT).

Literature
----------
- Frenkel (1976) / Bilson (1978) monetary approach; Dornbusch (1976) sticky-price
  (overshooting) monetary model: higher *relative* money growth → depreciation of
  the expanding currency (ceteris paribus income / rates).
- Primary prior: **low relative money growth** (monetary tightness) → subsequent
  appreciation. Honesty alternate: high-growth / liquidity-premium sort.
- Distinct from CB balance-sheet / QE assets (§22 WALCL), real-rate / breakeven
  (§23), debt/GDP (§29), fiscal (§28), TB (§30), BIS REER (§31), BIS credit (§32),
  funding-liq, macro-diff, equity-diff.

Free data (no Bloomberg / OECD portal key)
-----------------------------------------
OECD MEI **broad money** growth rate via FRED ``MABMM301{cc}M657S`` (measure 657
= growth rate same period previous year). Preferred over levels ``*189S`` because
657S is already a growth rate and stays current into 2025–26 for core G10 while
SA levels often end ~2023-11.

| Ccy | FRED id (primary)   | Notes                                         |
|-----|---------------------|-----------------------------------------------|
| USD | MABMM301USM657S     | OECD US broad-money growth; ``M2SL`` alt level|
| EUR | MABMM301EZM657S     | Euro-area (not Germany)                       |
| GBP | MABMM301GBM657S     |                                               |
| JPY | MABMM301JPM657S     |                                               |
| CAD | MABMM301CAM657S     |                                               |
| AUD | MABMM301AUM657S     |                                               |
| NZD | None (primary)      | ``MABMM301NZM657S`` ends 2018-12 — stale      |
| CHF | None (primary)      | ``MABMM301CHM657S`` ends 2018-12 — stale      |

Tried / documented:
- ``MABMM301*M189S`` (SA levels) — live but many end ~2023-11; not primary.
- ``MABMM301NZM189N`` / ``MABMM301CHM189N`` (NSA levels to ~2023-11) — archival
  YoY possible but **not mixed** into primary 657S panel (scale / definition
  mismatch vs measure 657).
- ``MABMM301NZM657S`` / ``MABMM301CHM657S`` — live historically, end **2018-12**.
- ``MYAGM2*`` / ``MYAGM3*`` IMF-style — mostly **404** on FRED for G10.
- US ``M2SL`` — classic US M2 level (through 2026-08); documented alt for US
  tilt research; primary panel uses OECD 657S for cross-country consistency.

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Monthly OECD / national money aggregates typically land ~4–8 weeks after
month-end. Default ``pub_lag_months=2``: observation dated month-start ``t`` is
first known at ``t + 2 months``. Strategy modules add ``signal_lag`` months on top.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED OECD broad-money YoY growth (measure 657)
MONEY_GROWTH_SERIES: dict[str, str | None] = {
    "USD": "MABMM301USM657S",
    "EUR": "MABMM301EZM657S",  # Euro-area
    "GBP": "MABMM301GBM657S",
    "JPY": "MABMM301JPM657S",
    "CAD": "MABMM301CAM657S",
    "AUD": "MABMM301AUM657S",
    "NZD": None,  # MABMM301NZM657S ends 2018-12 — unmapped on primary
    "CHF": None,  # MABMM301CHM657S ends 2018-12 — unmapped on primary
}

# Documented level / alt / failed mnemonics (not primary growth panel)
MONEY_LEVEL_ALT: dict[str, str | None] = {
    "USD": "M2SL",  # classic US M2 SA level
    "EUR": "MABMM301EZM189S",
    "GBP": "MABMM301GBM189S",
    "JPY": "MABMM301JPM189S",
    "CAD": "MABMM301CAM189S",
    "AUD": "MABMM301AUM189S",
    "NZD": "MABMM301NZM189N",  # NSA (SA ends 2018)
    "CHF": "MABMM301CHM189N",
}

MONEY_FAILED_OR_ALT: dict[str, list[str]] = {
    "nzd_chf_657s_stale_2018": ["MABMM301NZM657S", "MABMM301CHM657S"],
    "sa_levels_end_~2023": [
        "MABMM301EZM189S",
        "MABMM301GBM189S",
        "MABMM301JPM189S",
        "MABMM301CAM189S",
        "MABMM301AUM189S",
        "MABMM301USM189S",
    ],
    "nzd_chf_nsa_levels_archival": ["MABMM301NZM189N", "MABMM301CHM189N"],
    "imf_myagm_mostly_404": [
        "MYAGM2GBM196N",
        "MYAGM3GBM196N",
        "MYAGM2CAM196N",
        "MYAGM2AUM196N",
        "MYAGM2NZM196N",
        "MYAGM2CHM196N",
    ],
    "us_m2_level_alt": ["M2SL", "M2NS", "BOGMBASE"],
}

US_M2_LEVEL = "M2SL"
EUR_EA_GROWTH = "MABMM301EZM657S"

DEFAULT_PUB_LAG_MONTHS = 2  # conservative monthly money-release lag


def _to_month_start(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = pd.DatetimeIndex(s.index)
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    ms = pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start")).tz_localize("UTC")
    s = s.copy()
    s.index = ms
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _load_monthly(series_id: str, *, download: bool, force: bool) -> pd.Series:
    if download:
        download_fred_series(series_id, force=force)
    return _to_month_start(load_fred_series(series_id, download=False))


def _apply_pub_lag(df: pd.DataFrame, *, pub_lag_months: int) -> pd.DataFrame:
    if df.empty or pub_lag_months <= 0:
        return df
    out = df.copy()
    out.index = out.index + pd.DateOffset(months=int(pub_lag_months))
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def load_money_growth_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Monthly OECD broad-money growth (measure 657) → PIT panel.

    Columns = ISO currency codes. Values = OECD MEI growth rate (measure 657).
    Index = UTC month-start of the PIT *known* date after ``pub_lag_months``.
    """
    curs = [c.upper() for c in (currencies or MONEY_GROWTH_SERIES.keys())]
    cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}

    for c in curs:
        sid = MONEY_GROWTH_SERIES.get(c)
        if sid is None:
            notes[c] = (
                "unmapped_on_primary_657s "
                f"(stale/alt: {MONEY_LEVEL_ALT.get(c)}; "
                f"657S ends 2018 for NZD/CHF)"
            )
            continue
        try:
            s = _load_monthly(sid, download=download, force=force)
            series_map[c] = sid
            if c == "EUR":
                notes[c] = f"Euro-area via {sid}"
            elif c == "USD":
                notes[c] = f"OECD US broad-money growth {sid}; M2SL level alt documented"
            cols[c] = s
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    df = pd.DataFrame(cols).sort_index()
    df = _apply_pub_lag(df, pub_lag_months=pub_lag_months)

    df.attrs["factor"] = "oecd_broad_money_growth_657"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI MABMM301*M657S broad-money growth "
        "(NZD/CHF 657S stale 2018 — unmapped on primary; M2SL US level alt)"
    )
    df.attrs["unit"] = "oecd_mei_growth_rate_657"
    df.attrs["unmapped"] = [c for c in curs if c not in cols]
    df.attrs["failed_or_alt"] = MONEY_FAILED_OR_ALT
    df.attrs["level_alt_map"] = {k: v for k, v in MONEY_LEVEL_ALT.items() if v}
    return df


def load_us_money_growth(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US OECD broad-money growth (MABMM301USM657S), PIT-lagged."""
    panel = load_money_growth_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US money-growth series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_MONEY_GROWTH"
    out.attrs["series_id"] = MONEY_GROWTH_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def load_us_m2_level(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """Classic US M2 (M2SL) level, PIT-lagged — documented alt, not primary XS."""
    s = _load_monthly(US_M2_LEVEL, download=download, force=force)
    df = _apply_pub_lag(s.to_frame("USD"), pub_lag_months=pub_lag_months)
    out = df["USD"].dropna()
    out.name = "US_M2SL"
    out.attrs["series_id"] = US_M2_LEVEL
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def money_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_money_growth_panel(download=False)
    rows = []
    notes = panel.attrs.get("notes", {})
    smap = panel.attrs.get("series_map", {})
    for c in panel.columns:
        s = panel[c].dropna()
        rows.append(
            {
                "currency": c,
                "fred_id": smap.get(c, ""),
                "n_obs": int(len(s)),
                "start": str(s.index.min().date()) if len(s) else None,
                "end": str(s.index.max().date()) if len(s) else None,
                "last": float(s.iloc[-1]) if len(s) else float("nan"),
                "note": notes.get(c, ""),
            }
        )
    for c in panel.attrs.get("unmapped", []):
        rows.append(
            {
                "currency": c,
                "fred_id": "",
                "n_obs": 0,
                "start": None,
                "end": None,
                "last": float("nan"),
                "note": notes.get(c, "unmapped_on_primary_657s"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
