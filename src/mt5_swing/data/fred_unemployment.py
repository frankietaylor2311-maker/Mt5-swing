"""FRED / OECD MEI unemployment-rate panels for dedicated UR-differential FX.

Literature
----------
- Dahlquist & Hasseltoft (2020, JFE) economic momentum — labour channel via
  **unemployment rates** (distinct from employment *persons* growth §41).
- Primary prior: currencies with *low* relative UR (strong labour markets)
  subsequently appreciate vs high-UR peers. Honesty alternate: long high UR /
  labour-stress debtor premium.
- Distinct from: employment §41 (LFEMTTTT persons YoY), macro-diff §12 EW
  CPI+IP+UR *blend* (UR only as one EW leg — no dedicated UR XS board until
  this wave), OECD CLI/CCI/BCI, WUI, EPU/TPU, CA/TB, fiscal/debt, BIS,
  reserves, house-price, money, equity, IG OAS, PPI/CPI, ULC/LP/CU.

Free data (mapped in ``fred_macro_diff.UR_SERIES``; re-exported here)
--------------------------------------------------------------------
OECD MEI / national unemployment **rate levels** (%):

| Ccy | FRED id            | Notes                                              |
|-----|--------------------|----------------------------------------------------|
| USD | LRHUTTTTUSM156S    | live                                               |
| EUR | LRHUTTTTEZM156S    | EA UR — sparse tail ~2023-01 on FRED (documented)  |
| GBP | LRHUTTTTGBM156S    | live                                               |
| JPY | LRHUTTTTJPM156S    | live                                               |
| AUD | LRHUTTTTAUM156S    | live                                               |
| CAD | LRHUTTTTCAM156S    | live                                               |
| CHF | LMUNRRTTCHM156S    | registered UR (not LRHUTTTT*)                      |
| NZD | —                  | **unmapped** — no clean FRED multi-country UR      |

Unit choice (fixed a priori)
----------------------------
Panel stores **unemployment rate levels (%)**. Strategy scores XS-rank on
levels (not YoY of persons). Acceleration = −Δ12 of the level (falling UR).

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Matches ``fred_macro_diff`` UR default: ``pub_lag_months=1``. Observation
dated month-start ``t`` is first known at ``t + 1 month``. Strategy modules
add ``signal_lag=1`` month + a **1 trading-day** weight lag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_macro_diff import UR_SERIES, load_ur_panel

# Re-export map (NZD deliberately absent — document, do not invent)
UNEMPLOYMENT_SERIES: dict[str, str | None] = {
    "USD": UR_SERIES.get("USD"),
    "EUR": UR_SERIES.get("EUR"),
    "GBP": UR_SERIES.get("GBP"),
    "JPY": UR_SERIES.get("JPY"),
    "AUD": UR_SERIES.get("AUD"),
    "CAD": UR_SERIES.get("CAD"),
    "CHF": UR_SERIES.get("CHF"),
    "NZD": None,  # missing on FRED — leave unmapped
}

UNEMPLOYMENT_FAILED_OR_ALT: dict[str, list[str]] = {
    "nzd_unmapped_on_fred": [],
    "chf_registered_ur_not_lrhutttt": ["LMUNRRTTCHM156S"],
    "eur_ea_sparse_tail_~2023_01": ["LRHUTTTTEZM156S"],
    "macro_diff_ur_series_reused": list(UR_SERIES.values()),
    "employment_persons_lfemtttt_distinct_41": [
        "LFEMTTTTUSM647S",
        "LFEMTTTTDEQ647S",
    ],
}

DEFAULT_PUB_LAG_MONTHS = 1  # a priori: matches macro_diff ur lag


def load_unemployment_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,  # noqa: ARG001 — API parity with siblings; load_ur_panel has no force
) -> pd.DataFrame:
    """OECD / national unemployment **rate levels** → monthly PIT panel.

    Thin wrapper around ``fred_macro_diff.load_ur_panel`` + ``UR_SERIES``.
    Columns = ISO currency codes. Values = UR **levels (%)**. Index = UTC
    month-start of the PIT *known* date after ``pub_lag_months``. NZD is
    requested so coverage docs show the unmapped gap; it will not appear as
    a column unless a future FRED series is added.
    """
    curs = [c.upper() for c in (currencies or list(UNEMPLOYMENT_SERIES.keys()))]
    # Include NZD in the request so load_ur_panel notes it as unmapped
    panel = load_ur_panel(
        currencies=curs,
        pub_lag_months=pub_lag_months,
        download=download,
    )
    # Enrich attrs for dedicated wave reporting
    notes = dict(panel.attrs.get("notes", {}))
    series_map = dict(panel.attrs.get("series_map", {}))
    for c in curs:
        if c not in series_map and c not in notes:
            notes[c] = UNEMPLOYMENT_SERIES.get(c) and "load_fail" or "unmapped_or_unavailable_on_fred"
    if "NZD" in curs:
        notes.setdefault("NZD", "unmapped_or_unavailable_on_fred")
    if "CHF" in series_map:
        notes.setdefault("CHF", "LMUNRRTTCHM156S registered UR (not LRHUTTTT*)")
    if "EUR" in series_map:
        notes.setdefault("EUR", "EA UR ends ~2023-01 on FRED — sparse tail")

    panel.attrs["factor"] = "oecd_mei_unemployment_rate"
    panel.attrs["pub_lag_months"] = int(pub_lag_months)
    panel.attrs["series_map"] = series_map
    panel.attrs["notes"] = notes
    panel.attrs["source"] = (
        "FRED OECD MEI / national unemployment rate levels "
        "(LRHUTTTT*M156S; CHF=LMUNRRTTCHM156S registered; "
        "NZD unmapped on FRED; reuses fred_macro_diff.UR_SERIES / load_ur_panel)"
    )
    panel.attrs["unit"] = "unemployment_rate_pct_level"
    panel.attrs["score_basis"] = "ur_level_pct"
    panel.attrs["unmapped"] = [c for c in curs if c not in series_map]
    panel.attrs["missing"] = panel.attrs.get("missing", ["NZD"])
    panel.attrs["failed_or_alt"] = UNEMPLOYMENT_FAILED_OR_ALT
    panel.attrs["stale"] = bool("EUR" in series_map)  # EA sparse tail honesty
    panel.attrs["stale_note"] = (
        "EUR EA UR on free FRED ends ~2023-01 (sparse tail; post-end ffilled); "
        "NZD unmapped; CHF registered UR; USD/GBP/JPY/AUD/CAD live"
    )
    return panel


def load_us_unemployment(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US unemployment rate (LRHUTTTTUSM156S), PIT-lagged."""
    panel = load_unemployment_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US unemployment series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_UR"
    out.attrs["series_id"] = UNEMPLOYMENT_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def unemployment_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_unemployment_panel(download=False)
    rows = []
    notes = panel.attrs.get("notes", {})
    smap = panel.attrs.get("series_map", {})
    for c in panel.columns:
        s = panel[c].dropna()
        rows.append(
            {
                "currency": c,
                "fred_id": smap.get(c, ""),
                "freq": "M",
                "n_obs": int(len(s)),
                "start": str(s.index.min().date()) if len(s) else None,
                "end": str(s.index.max().date()) if len(s) else None,
                "last": float(s.iloc[-1]) if len(s) else float("nan"),
                "note": notes.get(c, ""),
            }
        )
    for c in panel.attrs.get("unmapped", []):
        if c in panel.columns:
            continue
        rows.append(
            {
                "currency": c,
                "fred_id": "",
                "freq": "",
                "n_obs": 0,
                "start": None,
                "end": None,
                "last": float("nan"),
                "note": notes.get(c, "unmapped_or_unavailable_on_fred"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
