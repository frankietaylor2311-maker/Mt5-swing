"""FRED / OECD MEI Consumer Confidence (CCI) panels for sentiment FX.

Literature
----------
- OECD Consumer Confidence Indicators summarise household sentiment (vs
  coincident IP/UR/CPI and vs *leading* OECD CLI).
- Ludvigson-style consumer sentiment / confidence → risk appetite and asset
  returns; FX inherits relative-sentiment differentials (Dahlquist-style
  macro–FX framing for *sentiment*, not coincident activity and not leading CLI).
- Primary prior: currencies with *high* relative lagged CCI (or CCI change)
  appreciate vs low-CCI peers. Honesty alternate: long low CCI.
- Distinct from: OECD CLI (§37), macro-diff CPI/IP/UR, house-price, money-growth,
  CA/TB, BIS, reserves, IG OAS, equity-diff, GPR/EPU.

Free data (no OECD portal key)
------------------------------
OECD MEI **consumer confidence balance** via FRED ``CSCICP02*M460S`` (live),
plus US OECD standardised CCI ``USACSCICP02STSAM`` (``CSCICP02USM460S`` **404**).

| Ccy | FRED id (primary)   | Notes                                                 |
|-----|---------------------|-------------------------------------------------------|
| USD | USACSCICP02STSAM    | US OECD CCI standardised; CSCICP02USM460S 404         |
| EUR | CSCICP02EZM460S     | EZ aggregate (live through ~2026-01); DE/FR alts      |
| GBP | CSCICP02GBM460S     | through ~2026-08                                      |
| JPY | CSCICP02JPM460S     | through ~2026-08                                      |
| AUD | CSCICP02AUM460S     | through ~2026-08                                      |
| CAD | None (primary)      | CSCICP02CAM460S 404; CSCICP03CAM665S ends 2017-12     |
| NZD | None (primary)      | CSCICP02NZM460S 404; CSCICP03NZM665S ends 2023-09     |
| CHF | None (primary)      | CSCICP02CHM460S 404; CSCICP03CHM665S ends 2023-11     |

Tried / documented:
- ``CSCICP03*M665S`` amplitude-adjusted CCI — live IDs but **stale** ends
  ~2024-01 for core G10 (CAD 2017). Not used as primary (too stale for
  2025/2026/HO evaluation). Prefer live ``CSCICP02*M460S`` balances.
- ``CSCICP03USM460S`` / ``CSCICP02USM460S`` — **404**.
- ``CSCICP02FRM665S`` / DE alts — coverage / report only.
- ``UMCSENT`` — US Michigan sentiment (companion / coverage only; not primary XS).

Unit choice (fixed a priori)
----------------------------
Foreign CSCICP02 balances (~percentage balance). Strategy scores use **YoY
diff** ``CCI_t − CCI_{t−12}`` for XS ranks (change in consumer confidence),
plus trailing z of that YoY and Δ12 acceleration. US tilt uses
``USACSCICP02STSAM`` (same YoY-z machinery; scale differs from foreign
balances but tilt is univariate). Documented in panel attrs.

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Monthly OECD CCI typically lands ~1–2 months after reference. Default
``pub_lag_months=2``: observation dated month-start ``t`` is first known at
``t + 2 months``. Strategy modules add a **1 trading-day** weight lag
(``signal_lag_days=1``); no extra month signal lag (a priori for this wave).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED OECD CCI (live CSCICP02 balances; US standardised)
CCI_SERIES: dict[str, str | None] = {
    "USD": "USACSCICP02STSAM",  # CSCICP02USM460S 404
    "EUR": "CSCICP02EZM460S",  # EZ aggregate (preferred over DE)
    "GBP": "CSCICP02GBM460S",
    "JPY": "CSCICP02JPM460S",
    "CAD": None,  # CSCICP02CAM460S 404; CSCICP03CAM665S ends 2017
    "AUD": "CSCICP02AUM460S",
    "NZD": None,  # CSCICP02NZM460S 404; CSCICP03NZM665S ends 2023-09
    "CHF": None,  # CSCICP02CHM460S 404; CSCICP03CHM665S ends 2023-11
}

CCI_FAILED_OR_ALT: dict[str, list[str]] = {
    "usd_cscicp02_usm460s_404": ["CSCICP02USM460S"],
    "usd_primary_standardised": ["USACSCICP02STSAM"],
    "usd_michigan_coverage": ["UMCSENT"],
    "usd_amplitude_stale_2024": ["CSCICP03USM665S"],
    "eur_ez_primary": ["CSCICP02EZM460S"],
    "eur_germany_alt": ["CSCICP02DEM460S"],
    "eur_france_alt": ["CSCICP02FRM460S"],
    "eur_amplitude_stale": ["CSCICP03EZM665S", "CSCICP03DEM665S"],
    "cad_404_and_amplitude_stale_2017": ["CSCICP02CAM460S", "CSCICP03CAM665S"],
    "nzd_404_and_amplitude_stale_2023": ["CSCICP02NZM460S", "CSCICP03NZM665S"],
    "chf_404_and_amplitude_stale_2023": ["CSCICP02CHM460S", "CSCICP03CHM665S"],
    "amplitude_family_stale_2024": [
        "CSCICP03GBM665S",
        "CSCICP03JPM665S",
        "CSCICP03AUM665S",
        "CSCICP03O9M665S",
    ],
}

EUR_SECONDARY = "CSCICP02DEM460S"
EUR_FRANCE = "CSCICP02FRM460S"

DEFAULT_PUB_LAG_MONTHS = 2  # conservative monthly OECD CCI lag


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


def load_oecd_cci_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Monthly OECD CCI → PIT panel.

    Columns = ISO currency codes. Foreign values = CSCICP02 percentage
    balances; USD = USACSCICP02STSAM. Index = UTC month-start of the PIT
    *known* date after ``pub_lag_months``. Strategy modules convert to YoY
    diff for XS ranks (foreign only — same CSCICP02 unit).
    """
    curs = [c.upper() for c in (currencies or CCI_SERIES.keys())]
    cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}

    for c in curs:
        sid = CCI_SERIES.get(c)
        if sid is None:
            notes[c] = (
                "unmapped_on_primary_cci "
                "(CSCICP02 404; CSCICP03 amplitude stale — CAD 2017 / NZD+CHF 2023)"
            )
            continue
        try:
            s = _load_monthly(sid, download=download, force=force)
            series_map[c] = sid
            if c == "EUR":
                notes[c] = (
                    f"EZ aggregate via {sid} "
                    f"(DE/FR CSCICP02*M460S alts; CSCICP03 amplitude ends 2024-01)"
                )
            elif c == "USD":
                notes[c] = (
                    f"US OECD standardised CCI {sid} "
                    f"(CSCICP02USM460S 404; CSCICP03USM665S ends 2024-01)"
                )
            else:
                notes[c] = f"OECD CCI balance {sid}"
            cols[c] = s
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    df = pd.DataFrame(cols).sort_index()
    df = _apply_pub_lag(df, pub_lag_months=pub_lag_months)

    df.attrs["factor"] = "oecd_cci_consumer_confidence"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI CSCICP02*M460S consumer-confidence balances "
        "(EUR=EZ; USD=USACSCICP02STSAM; CAD/NZD/CHF unmapped; "
        "CSCICP03*M665S amplitude stale ~2024-01 — not primary)"
    )
    df.attrs["unit"] = "oecd_mei_cci_balance_or_us_standardised"
    df.attrs["unmapped"] = [c for c in curs if c not in cols]
    df.attrs["failed_or_alt"] = CCI_FAILED_OR_ALT
    return df


def load_us_oecd_cci(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US OECD CCI (USACSCICP02STSAM), PIT-lagged."""
    panel = load_oecd_cci_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US OECD CCI series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_OECD_CCI"
    out.attrs["series_id"] = CCI_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def cci_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_oecd_cci_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_primary_cci"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
