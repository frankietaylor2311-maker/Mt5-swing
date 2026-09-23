"""FRED / OECD MEI Business Confidence (BCI) panels for manufacturing-confidence FX.

Literature
----------
- OECD Business Confidence Indicators summarise manufacturing / business
  sentiment (PMI-style balances) — distinct from *consumer* CCI (§38) and
  *leading* OECD CLI (§37).
- ISM manufacturing literature; Dahlquist–Hasseltoft-style macro–FX
  differential framing applied to *business* confidence.
- Primary prior: currencies with *high* relative lagged BCI (or BCI YoY
  change) appreciate vs low-BCI peers. Honesty alternate: long low BCI.
- Distinct from: OECD CLI (§37), OECD CCI (§38), macro-diff CPI/IP/UR,
  house-price, money-growth, CA/TB, BIS, reserves, IG OAS, equity-diff,
  GPR/EPU.

Free data (no OECD portal key; ISM/NAPM 404 on FRED)
---------------------------------------------------
OECD MEI **business confidence balance** via FRED ``BSCICP02*M460S`` (live).
ISM / NAPM manufacturing PMI series are **404** on FRED — BCI is the free
fallback (confirmed a priori).

| Ccy | FRED id (primary)   | Notes                                                 |
|-----|---------------------|-------------------------------------------------------|
| USD | BSCICP02USM460S     | live through ~2026-08                                 |
| EUR | BSCICP02EZM460S     | EZ aggregate (live through ~2026-01); DE/FR/IT/ES alts|
| GBP | BSCICP02GBM460S     | through ~2026-08                                      |
| CHF | BSCICP02CHM460S     | **live** (unlike CCI CHF which was unmapped)          |
| JPY | None (primary)      | BSCICP02JPM460S 404; BSCICP03JPM665S ends 2023-12     |
| AUD | None (primary)      | BSCICP02AUM460S 404; BSCICP03AUM665S ends 2023-11     |
| CAD | None (primary)      | BSCICP02CAM460S 404                                   |
| NZD | None (primary)      | BSCICP02NZM460S 404; BSCICP03NZM665S ends 2023-12     |

Tried / documented:
- ``BSCICP03*M665S`` amplitude-adjusted BCI — live IDs but **stale** ends
  ~2023-11..2024-01 for AU/JP/NZ/CH/US/EZ. Not used as primary (too stale for
  2025/2026/HO evaluation). Prefer live ``BSCICP02*M460S`` balances.
- ``NAPM`` / ISM manufacturing — **404** on FRED.
- ``BSCICP02AUM460S`` / ``CAM`` / ``NZM`` / ``JPM`` — **404**.
- FR/IT/ES ``BSCICP02*M460S`` — coverage / report only (EZ primary for EUR).

Unit choice (fixed a priori)
----------------------------
BSCICP02 balances (~percentage balance). Strategy scores use **YoY diff**
``BCI_t − BCI_{t−12}`` for XS ranks, plus trailing z of that YoY and Δ12
acceleration. Documented in panel attrs.

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Monthly OECD BCI typically lands ~1–2 months after reference. Default
``pub_lag_months=2``: observation dated month-start ``t`` is first known at
``t + 2 months``. Strategy modules add a **1 trading-day** weight lag
(``signal_lag_days=1``); no extra month signal lag (a priori for this wave).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED OECD BCI (live BSCICP02 balances)
BCI_SERIES: dict[str, str | None] = {
    "USD": "BSCICP02USM460S",
    "EUR": "BSCICP02EZM460S",  # EZ aggregate (preferred over DE)
    "GBP": "BSCICP02GBM460S",
    "CHF": "BSCICP02CHM460S",  # live — unlike CCI
    "JPY": None,  # BSCICP02JPM460S 404; BSCICP03JPM665S ends 2023-12
    "CAD": None,  # BSCICP02CAM460S 404
    "AUD": None,  # BSCICP02AUM460S 404; BSCICP03AUM665S ends 2023-11
    "NZD": None,  # BSCICP02NZM460S 404; BSCICP03NZM665S ends 2023-12
}

BCI_FAILED_OR_ALT: dict[str, list[str]] = {
    "ism_napm_404": ["NAPM"],
    "usd_primary_balance": ["BSCICP02USM460S"],
    "usd_amplitude_stale_2024": ["BSCICP03USM665S"],
    "eur_ez_primary": ["BSCICP02EZM460S"],
    "eur_germany_alt": ["BSCICP02DEM460S"],
    "eur_france_alt": ["BSCICP02FRM460S"],
    "eur_italy_coverage": ["BSCICP02ITM460S"],
    "eur_spain_coverage": ["BSCICP02ESM460S"],
    "eur_amplitude_stale": ["BSCICP03EZM665S", "BSCICP03DEM665S"],
    "jpy_404_and_amplitude_stale_2023": ["BSCICP02JPM460S", "BSCICP03JPM665S"],
    "aud_404_and_amplitude_stale_2023": ["BSCICP02AUM460S", "BSCICP03AUM665S"],
    "cad_404": ["BSCICP02CAM460S"],
    "nzd_404_and_amplitude_stale_2023": ["BSCICP02NZM460S", "BSCICP03NZM665S"],
    "chf_primary_live": ["BSCICP02CHM460S"],
    "chf_amplitude_stale_2023": ["BSCICP03CHM665S"],
    "amplitude_family_stale_2023_2024": [
        "BSCICP03GBM665S",
        "BSCICP03JPM665S",
        "BSCICP03AUM665S",
        "BSCICP03NZM665S",
        "BSCICP03USM665S",
        "BSCICP03EZM665S",
    ],
}

EUR_SECONDARY = "BSCICP02DEM460S"
EUR_FRANCE = "BSCICP02FRM460S"
EUR_ITALY = "BSCICP02ITM460S"
EUR_SPAIN = "BSCICP02ESM460S"

DEFAULT_PUB_LAG_MONTHS = 2  # conservative monthly OECD BCI lag


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


def load_oecd_bci_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Monthly OECD BCI → PIT panel.

    Columns = ISO currency codes. Values = BSCICP02 percentage balances.
    Index = UTC month-start of the PIT *known* date after ``pub_lag_months``.
    Strategy modules convert to YoY diff for XS ranks (foreign only).
    """
    curs = [c.upper() for c in (currencies or BCI_SERIES.keys())]
    cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}

    for c in curs:
        sid = BCI_SERIES.get(c)
        if sid is None:
            notes[c] = (
                "unmapped_on_primary_bci "
                "(BSCICP02 404; BSCICP03 amplitude stale ~2023-11..2024-01 — "
                "AUD/JPY/NZD; CAD 404)"
            )
            continue
        try:
            s = _load_monthly(sid, download=download, force=force)
            series_map[c] = sid
            if c == "EUR":
                notes[c] = (
                    f"EZ aggregate via {sid} "
                    f"(DE/FR/IT/ES BSCICP02*M460S alts; BSCICP03 amplitude ends ~2024-01)"
                )
            elif c == "CHF":
                notes[c] = f"OECD BCI balance {sid} (live — unlike CCI CHF)"
            elif c == "USD":
                notes[c] = (
                    f"OECD BCI balance {sid} "
                    f"(NAPM/ISM 404; BSCICP03USM665S ends 2024-01)"
                )
            else:
                notes[c] = f"OECD BCI balance {sid}"
            cols[c] = s
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    df = pd.DataFrame(cols).sort_index()
    df = _apply_pub_lag(df, pub_lag_months=pub_lag_months)

    df.attrs["factor"] = "oecd_bci_business_confidence"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI BSCICP02*M460S business-confidence balances "
        "(EUR=EZ; CHF live; AUD/CAD/NZD/JPY unmapped; "
        "BSCICP03*M665S amplitude stale ~2023-11..2024-01 — not primary; "
        "NAPM/ISM 404)"
    )
    df.attrs["unit"] = "oecd_mei_bci_balance"
    df.attrs["unmapped"] = [c for c in curs if c not in cols]
    df.attrs["failed_or_alt"] = BCI_FAILED_OR_ALT
    return df


def load_us_oecd_bci(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US OECD BCI (BSCICP02USM460S), PIT-lagged."""
    panel = load_oecd_bci_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US OECD BCI series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_OECD_BCI"
    out.attrs["series_id"] = BCI_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def bci_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_oecd_bci_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_primary_bci"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
