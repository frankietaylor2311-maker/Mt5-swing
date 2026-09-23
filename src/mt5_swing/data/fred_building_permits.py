"""FRED / OECD MEI building permits (ODCNPI03) panels for housing-activity FX.

Literature
----------
- Housing starts / building permits as *leading real-activity / construction*
  channel for FX (growth-channel / Taylor-rule FX; country housing-activity
  differentials). Distinct from house-*price* §35 (BIS real HPI levels/growth).
- Primary prior: currencies with *high* relative permits YoY growth subsequently
  appreciate vs low-activity peers. Honesty alternate: long low housing /
  activity-stress.
- Distinct from: house-price §35, OECD CLI/CCI/BCI, IP §42, employment §41,
  WUI, EPU/TPU, CA/TB, fiscal/debt, BIS, reserves, money, equity, IG OAS,
  commodity, retail-sales (stale).

Free data (verified live on FRED ~2026-09-23)
--------------------------------------------
Prefer OECD MEI **permits for dwellings** YoY growth via FRED
``{ISO3}ODCNPI03GYSAM`` (monthly SA, growth rate same period previous year).
USD: OECD ``USAODCNPI03GYSAM`` is **404** → US Census ``PERMIT`` (levels) with
YoY % computed in-loader. EUR: Germany ``DEUODCNPI03GYSAM`` (``EA19ODCNPI03GYSAM``
ends 2023-08 — stale; France ``FRAODCNPI03GYSAM`` live alt).

| Ccy | FRED id (primary)   | Freq | Notes                                              |
|-----|---------------------|------|----------------------------------------------------|
| USD | PERMIT (→ YoY %)    | M    | US Census permits; USAODCNPI03GYSAM **404**        |
| EUR | DEUODCNPI03GYSAM    | M    | Germany proxy (EA19 GYSAM stale ~2023-08)          |
| GBP | —                   | —    | **unmapped** (GBRODCNPI03* / WSCNDW01* 404)        |
| JPY | —                   | —    | **unmapped** (JPNODCNPI03* 404; WSCNDW01 stale)    |
| CAD | CANODCNPI03GYSAM    | M    | through ~2026-06                                   |
| AUD | AUSODCNPI03GYSAM    | M    | through ~2026-07                                   |
| NZD | NZLODCNPI03GYSAM    | M    | through ~2026-04                                   |
| CHF | —                   | —    | **unmapped** (CHEODCNPI03* 404)                    |

Tried / documented:
- ``USA/GBR/JPN/CHE ODCNPI03GYSAM`` / ``IXOBSAM`` / ``GYSAQ`` — **404**.
- ``*ODCNPI02*`` housing-starts OECD — **404** on probed G10.
- ``*ODCNPI03MLM`` / ``*WSCNDW01IXOBSAM`` — end ~**2023-10..2024-04** (stale).
- ``EA19ODCNPI03GYSAM`` / ``IXOBSAM`` — end **2023-08** (stale) → DEU EUR proxy.
- ``NLD/ITA/ESP/AUT ODCNPI03GYSAM`` — 404 or stale; BEL/PRT live alts (not primary).
- US ``HOUST`` / ``HOUST1F`` / ``PERMIT1`` — live national alts (not mixed into
  OECD panel; PERMIT chosen a priori as permits to match ODCNPI03).
- ACM fallback series live (``THREEFYTP10`` etc.) — **not used**: foreign panel
  has 4 live currencies (EUR/CAD/AUD/NZD) + USD ≥ ~4–5 threshold for honest XS.

Unit choice (fixed a priori)
----------------------------
Panel stores **YoY growth %**. OECD GYSAM values used as reported; US ``PERMIT``
levels converted via ``pct_change(12)*100`` before pub lag. Strategy scores use
these YoY values directly for XS ranks.

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Building permits / housing typically land ~1–2 months after reference. Default
``pub_lag_months=2`` (picked once a priori — same conservative band as IP §42).
Observation dated month-start ``t`` is first known at ``t + 2 months``.
Strategy modules add a **1 trading-day** weight lag (``signal_lag_days=1``);
no extra month signal lag (a priori for §43).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED building-permits YoY (OECD MEI ODCNPI03) or US PERMIT levels
PERMITS_SERIES: dict[str, str | None] = {
    "USD": "PERMIT",  # levels → YoY in loader; USAODCNPI03GYSAM 404
    "EUR": "DEUODCNPI03GYSAM",  # Germany proxy; EA19 stale
    "GBP": None,  # unmapped
    "JPY": None,  # unmapped
    "CAD": "CANODCNPI03GYSAM",
    "AUD": "AUSODCNPI03GYSAM",
    "NZD": "NZLODCNPI03GYSAM",
    "CHF": None,  # unmapped
}

# Native series unit: "yoy" already growth %; "level" needs pct_change(12)*100
PERMITS_UNIT: dict[str, str] = {
    "USD": "level",
    "EUR": "yoy",
    "CAD": "yoy",
    "AUD": "yoy",
    "NZD": "yoy",
}

PERMITS_FREQ: dict[str, str] = {
    "USD": "M",
    "EUR": "M",
    "CAD": "M",
    "AUD": "M",
    "NZD": "M",
}

PERMITS_FAILED_OR_ALT: dict[str, list[str]] = {
    "usd_oecd_gysam_404": ["USAODCNPI03GYSAM", "USAODCNPI03IXOBSAM", "USAODCNPI03GYSAQ"],
    "usd_national_primary_permit": ["PERMIT"],
    "usd_national_alts_not_primary": ["HOUST", "HOUST1F", "HOUST5F", "PERMIT1", "COMPUTSA"],
    "gbp_all_404": [
        "GBRODCNPI03GYSAM",
        "GBRODCNPI03IXOBSAM",
        "GBRODCNPI03GYSAQ",
        "GBRWSCNDW01GYSAM",
        "GBRWSCNDW01IXOBSAM",
        "ODCNPI03GBM661N",
        "ODCNPI03GBM657S",
        "WSCNDW01GBM661N",
    ],
    "jpy_odcnpi03_404_wscndw_stale": [
        "JPNODCNPI03GYSAM",
        "JPNODCNPI03IXOBSAM",
        "JPNWSCNDW01GYSAM",  # ends ~2023-11
        "JPNWSCNDW01IXOBSAM",  # ends ~2024-03
    ],
    "chf_all_404": [
        "CHEODCNPI03GYSAM",
        "CHEODCNPI03IXOBSAM",
        "CHEODCNPI03GYSAQ",
        "CHEWSCNDW01IXOBSAM",
    ],
    "odcnpi02_starts_g10_404": [
        "CANODCNPI02GYSAM",
        "AUSODCNPI02GYSAM",
        "NZLODCNPI02GYSAM",
        "DEUODCNPI02GYSAM",
        "FRAODCNPI02GYSAM",
        "USAODCNPI02GYSAM",
    ],
    "odcnpi03_mlm_stale_~2023_10": [
        "CANODCNPI03MLM",
        "AUSODCNPI03MLM",
        "NZLODCNPI03MLM",
    ],
    "eur_ea19_gysam_stale_2023_08": ["EA19ODCNPI03GYSAM", "EA19ODCNPI03IXOBSAM"],
    "eur_germany_proxy_primary": ["DEUODCNPI03GYSAM"],
    "eur_france_alt": ["FRAODCNPI03GYSAM"],
    "eur_bel_prt_alts": ["BELODCNPI03GYSAM", "PRTODCNPI03GYSAM"],
    "eur_ita_esp_aut_nld": [
        "ITAODCNPI03GYSAM",
        "ESPODCNPI03GYSAM",
        "AUTODCNPI03GYSAM",
        "NLDODCNPI03GYSAM",
    ],
    "acm_fallback_live_not_used": [
        "THREEFYTP10",
        "THREEFYTP5",
        "THREEFYTP2",
        "THREEFYTP1",
    ],
    "acm_alts_404": [
        "TENEXPPECTERMPREMIUM",
        "T5YTP",
        "T10YTP",
        "ACMTP10",
        "ACMTP05",
    ],
}

EUR_FRANCE_ALT = "FRAODCNPI03GYSAM"
EUR_EA19_STALE = "EA19ODCNPI03GYSAM"
US_HOUST_ALT = "HOUST"
US_PERMIT1_ALT = "PERMIT1"

DEFAULT_PUB_LAG_MONTHS = 2  # a priori: housing ~1–2m; use 2 (same band as IP)


def _to_month_start(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = pd.DatetimeIndex(s.index)
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    ms = pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start")).tz_localize("UTC")
    s = s.copy()
    s.index = ms
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _load_series(series_id: str, *, download: bool, force: bool) -> pd.Series:
    if download:
        download_fred_series(series_id, force=force)
    return _to_month_start(load_fred_series(series_id, download=False))


def _apply_pub_lag_monthly(df: pd.DataFrame, *, pub_lag_months: int) -> pd.DataFrame:
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


def load_building_permits_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """OECD MEI / US Census building-permits YoY → monthly PIT panel.

    Columns = ISO currency codes. Values = permits **YoY growth %**.
    Index = UTC month-start of the PIT *known* date after ``pub_lag_months``.
    US ``PERMIT`` levels are converted to YoY % before the pub lag.
    Strategy modules XS-rank on these YoY values (foreign only).
    """
    curs = [c.upper() for c in (currencies or PERMITS_SERIES.keys())]
    monthly_cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}
    freq_map: dict[str, str] = {}
    unit_map: dict[str, str] = {}

    for c in curs:
        sid = PERMITS_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_odcnpi03_permits"
            continue
        try:
            s = _load_series(sid, download=download, force=force)
            unit = PERMITS_UNIT.get(c, "yoy")
            if unit == "level":
                s = (s.pct_change(12) * 100.0).dropna()
                notes[c] = (
                    f"US Census {sid} levels → YoY % "
                    f"(USAODCNPI03GYSAM 404; HOUST/PERMIT1 alts)"
                )
            elif c == "EUR":
                notes[c] = (
                    f"Germany proxy via {sid} (M) "
                    f"(EA19ODCNPI03GYSAM ends 2023-08; FRAODCNPI03GYSAM alt)"
                )
            else:
                notes[c] = f"OECD MEI permits YoY {sid} (monthly)"
            series_map[c] = sid
            freq_map[c] = PERMITS_FREQ.get(c, "M")
            unit_map[c] = unit
            monthly_cols[c] = s
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    if not monthly_cols:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(monthly_cols).sort_index()
        df = _apply_pub_lag_monthly(df, pub_lag_months=pub_lag_months)
        df = df.dropna(how="all")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

    df.attrs["factor"] = "oecd_mei_building_permits_odcnpi03"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["freq_map"] = freq_map
    df.attrs["unit_map"] = unit_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI {ISO3}ODCNPI03GYSAM permits YoY monthly "
        "(CAD/AUD/NZD; EUR=DEU Germany proxy — EA19 GYSAM stale; "
        "USD=PERMIT levels→YoY — USAODCNPI03GYSAM 404; "
        "GBP/JPY/CHF unmapped 404; ODCNPI02 starts 404; "
        "ACM THREEFYTP10 live fallback not used — panel ≥4 foreign)"
    )
    df.attrs["unit"] = "permits_yoy_pct"
    df.attrs["score_basis"] = "yoy_growth_as_reported_or_derived"
    df.attrs["unmapped"] = [c for c in curs if c not in series_map]
    df.attrs["failed_or_alt"] = PERMITS_FAILED_OR_ALT
    df.attrs["acm_fallback_available"] = True
    df.attrs["acm_fallback_used"] = False
    df.attrs["acm_fallback_reason"] = (
        "not_needed: foreign live cols EUR/CAD/AUD/NZD = 4 (≥ ~4–5 threshold); "
        "USD PERMIT for stress/haven; GBP/JPY/CHF remain unmapped"
    )
    return df


def load_us_building_permits(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US Census PERMIT YoY %, PIT-lagged."""
    panel = load_building_permits_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US building permits series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_PERMITS"
    out.attrs["series_id"] = PERMITS_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def building_permits_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_building_permits_panel(download=False)
    rows = []
    notes = panel.attrs.get("notes", {})
    smap = panel.attrs.get("series_map", {})
    fmap = panel.attrs.get("freq_map", {})
    for c in panel.columns:
        s = panel[c].dropna()
        rows.append(
            {
                "currency": c,
                "fred_id": smap.get(c, ""),
                "freq": fmap.get(c, ""),
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
                "freq": "",
                "n_obs": 0,
                "start": None,
                "end": None,
                "last": float("nan"),
                "note": notes.get(c, "unmapped_on_fred_odcnpi03_permits"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
