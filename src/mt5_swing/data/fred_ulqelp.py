"""FRED / OECD MEI labour productivity (ULQELP01) panels for LP-growth FX.

Literature
----------
- Relative labour productivity growth → competitiveness / Balassa–Samuelson-
  adjacent FX (OECD MEI early-estimate labour productivity; output per worker).
- Primary prior: currencies with *high* relative LP YoY growth appreciate vs
  low-LP peers (productivity / competitiveness). Honesty alternate: long low LP
  / productivity-stress debtor.
- Natural companion to ULC §48 (competitiveness = ULC vs productivity). Distinct
  from: employment §41 (persons YoY), IP §42 (industrial volume), ULC §48
  (unit labour *cost*), Balassa–Samuelson §14 (IP *levels* productivity proxy +
  real FX residual), retail §46, DSR §47, CLI/CCI/BCI, WUI, EPU/TPU.

Free data (verified live on FRED ~2026-09-23)
--------------------------------------------
Prefer OECD MEI **early-estimate quarterly labour-productivity YoY growth** via
FRED ``ULQELP01{ISO2}Q657S`` (seasonally adjusted, growth rate same period
previous year). **Panel is STALE** — all G10 Q657S / Q661S / Q659S end
~**2023-04..07** (no live refresh on free FRED as of probe). Country
``{ISO3}ULQELP01GYSAQ`` / classic UV mnemonics **404**. ``OECDULQELP01GYSAQ``
aggregate ends **2023-01**. US national ``OPHNFB`` / ``PRS85006092`` live through
~2026-04 (alts — not mixed into OECD panel).

| Ccy | FRED id (primary)   | Freq | Notes                                              |
|-----|---------------------|------|----------------------------------------------------|
| USD | ULQELP01USQ657S     | Q    | ends ~2023-07 (stale)                              |
| EUR | ULQELP01DEQ657S     | Q    | Germany proxy (EZ also ends 2023-07)               |
| GBP | ULQELP01GBQ657S     | Q    | ends ~2023-04 (stale)                              |
| JPY | ULQELP01JPQ657S     | Q    | ends ~2023-07 (stale)                              |
| CAD | ULQELP01CAQ657S     | Q    | ends ~2023-07 (stale)                              |
| AUD | ULQELP01AUQ657S     | Q    | ends ~2023-07 (stale)                              |
| NZD | ULQELP01NZQ657S     | Q    | ends ~2023-04 (stale)                              |
| CHF | ULQELP01CHQ657S     | Q    | ends ~2023-07 (stale)                              |

Tried / documented:
- ``{ISO3}ULQELP01GYSAQ`` / ``IXOBSAQ`` / ``GPSAQ`` — country **404**.
- ``ULQELP01*Q661S`` index / ``*Q659S`` — same ~2023-07 stale end (not fresher).
- ``ULQELP01EZQ657S`` EZ YoY — ends **2023-07** (same stale) → Germany
  ``ULQELP01DEQ657S`` EUR proxy (FR/IT/ES/NL Q657S alts, also stale).
- US ``OPHNFB`` / ``PRS85006092`` — live national alts (not mixed into OECD panel).
- IMF IFS trade unit-value / ToT fallback (§49 optional): classic UV / ToT
  mnemonics **404** on FRED; OECD ``LOCOTTOR`` only Korea (ends 2023-11);
  ``PIEAMP01*`` export-price indices end ~2022-12 — not boarded as primary.

Unit choice (fixed a priori)
----------------------------
Panel stores **LP YoY growth rates as reported** (Q657S). Strategy scores use
these YoY values directly for XS ranks. Acceleration = Δ12 of that YoY growth
(after quarterly→monthly ffill).

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
OECD early-estimate quarterly labour productivity typically lands ~1–3 months
after reference quarter. Default ``pub_lag_months=3`` (picked once a priori —
labour-adjacent, same band as ULC §48 / employment §41). Observation dated
quarter-start ``t`` is first known at ``t + 3 months``. Quarterly series are
pub-lagged then forward-filled to month-start (no intra-quarter leak). Strategy
modules add ``signal_lag=1`` month + a **1 trading-day** weight lag.

**Staleness honesty:** free FRED ULQELP panel ends mid-2023; post-2023 ranks
are ffilled frozen. Full-sample still informative; 2025/2026 windows reflect
that limitation. No go-live claim under ``approximate_non_ftmo``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED OECD MEI early-estimate labour productivity YoY (quarterly Q657S)
# NOTE: all end ~2023-04..07 on free FRED (stale) — documented, still boarded.
LP_SERIES: dict[str, str | None] = {
    "USD": "ULQELP01USQ657S",
    "EUR": "ULQELP01DEQ657S",  # Germany proxy; EZ also ends 2023-07
    "GBP": "ULQELP01GBQ657S",
    "JPY": "ULQELP01JPQ657S",
    "CAD": "ULQELP01CAQ657S",
    "AUD": "ULQELP01AUQ657S",
    "NZD": "ULQELP01NZQ657S",
    "CHF": "ULQELP01CHQ657S",
}

LP_FREQ: dict[str, str] = {
    "USD": "Q",
    "EUR": "Q",
    "GBP": "Q",
    "JPY": "Q",
    "CAD": "Q",
    "AUD": "Q",
    "NZD": "Q",
    "CHF": "Q",
}

LP_FAILED_OR_ALT: dict[str, list[str]] = {
    "iso3_ulqelp01_gysaq_country_404": [
        "USAULQELP01GYSAQ",
        "GBRULQELP01GYSAQ",
        "JPNULQELP01GYSAQ",
        "CANULQELP01GYSAQ",
        "AUSULQELP01GYSAQ",
        "NZLULQELP01GYSAQ",
        "CHEULQELP01GYSAQ",
        "DEUULQELP01GYSAQ",
        "EA19ULQELP01GYSAQ",
        "OECDULQELP01GYSAQ",  # ends 2023-01
    ],
    "q657s_q661s_q659s_stale_~2023_07": [
        "ULQELP01USQ657S",
        "ULQELP01USQ661S",
        "ULQELP01USQ659S",
        "ULQELP01GBQ657S",
        "ULQELP01JPQ657S",
        "ULQELP01CAQ657S",
        "ULQELP01AUQ657S",
        "ULQELP01NZQ657S",
        "ULQELP01CHQ657S",
        "ULQELP01DEQ657S",
        "ULQELP01EZQ657S",
    ],
    "eur_germany_proxy_primary": ["ULQELP01DEQ657S"],
    "eur_fr_it_es_nl_alts_also_stale": [
        "ULQELP01FRQ657S",
        "ULQELP01ITQ657S",
        "ULQELP01ESQ657S",
        "ULQELP01NLQ657S",
    ],
    "us_national_alts_live_not_primary": ["OPHNFB", "PRS85006092"],
    "imf_ifs_unit_value_tot_fallback_404": [
        "XTEXUV01USM661N",
        "XTIMUV01USM661N",
        "USAXRTTOT01IXOBSAM",
        "TNTTUS",
        "XTUVUS",
        "KORLOCOTTORSTSAM",  # only Korea LOCOTTOR on FRED; ends 2023-11
    ],
    "oecd_pieamp_export_price_stale_2022_12": [
        "PIEAMP01USM661N",
        "PIEAMP01GBM661N",
        "PIEAMP01CAM661N",
        "PIEAMP01DEM661N",
        "PIEAMP01CHM661N",
    ],
}

EUR_FRANCE = "ULQELP01FRQ657S"
EUR_ITALY = "ULQELP01ITQ657S"
EUR_SPAIN = "ULQELP01ESQ657S"
EUR_NLD = "ULQELP01NLQ657S"
EUR_EZ_STALE = "ULQELP01EZQ657S"
US_OPHNFB_ALT = "OPHNFB"

DEFAULT_PUB_LAG_MONTHS = 3  # a priori: quarterly LP labour-adjacent (ULC §48 / emp §41)

# Reuse ULC helper implementations via identical private helpers
from mt5_swing.data.fred_ulc import (  # noqa: E402
    _apply_pub_lag_monthly,
    _expand_quarterly_to_monthly,
    _to_month_start,
    _to_quarter_start,
)


def _load_series(series_id: str, *, freq: str, download: bool, force: bool) -> pd.Series:
    if download:
        download_fred_series(series_id, force=force)
    raw = load_fred_series(series_id, download=False)
    if freq.upper().startswith("Q"):
        return _to_quarter_start(raw)
    return _to_month_start(raw)


def load_labour_prod_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """OECD MEI labour productivity YoY growth → monthly PIT panel.

    Columns = ISO currency codes. Values = LP **YoY growth %** as reported
    (Q657S). Index = UTC month-start of the PIT *known* date after
    ``pub_lag_months``. Quarterly series are pub-lagged first, then
    forward-filled to month-start. Strategy modules XS-rank on these YoY values
    (foreign only).

    **Staleness:** free FRED ends ~2023-04..07 for all G10 — post-2023 values
    are ffilled. Documented in attrs.
    """
    curs = [c.upper() for c in (currencies or LP_SERIES.keys())]
    monthly_cols: dict[str, pd.Series] = {}
    quarterly_cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}
    freq_map: dict[str, str] = {}

    for c in curs:
        sid = LP_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_ulqelp01"
            continue
        freq = LP_FREQ.get(c, "Q")
        try:
            s = _load_series(sid, freq=freq, download=download, force=force)
            series_map[c] = sid
            freq_map[c] = freq
            end = str(s.dropna().index.max().date()) if len(s.dropna()) else "?"
            if c == "EUR":
                notes[c] = (
                    f"Germany proxy via {sid} (Q) stale_end={end} "
                    f"(EZ also ~2023-07; FR/IT/ES/NL Q657S alts)"
                )
            else:
                notes[c] = (
                    f"OECD MEI early-estimate LP YoY {sid} (quarterly) "
                    f"stale_end={end}"
                )
            if freq == "Q":
                quarterly_cols[c] = s
            else:
                monthly_cols[c] = s
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    frames: list[pd.DataFrame] = []
    if monthly_cols:
        mdf = pd.DataFrame(monthly_cols).sort_index()
        mdf = _apply_pub_lag_monthly(mdf, pub_lag_months=pub_lag_months)
        frames.append(mdf)
    if quarterly_cols:
        qdf = pd.DataFrame(quarterly_cols).sort_index()
        qdf = _expand_quarterly_to_monthly(qdf, pub_lag_months=pub_lag_months)
        frames.append(qdf)

    if not frames:
        df = pd.DataFrame()
    else:
        df = pd.concat(frames, axis=1).sort_index()
        df = df.dropna(how="all")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

    df.attrs["factor"] = "oecd_mei_labour_prod_ulqelp01"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["freq_map"] = freq_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI ULQELP01{ISO2}Q657S early-estimate labour-productivity "
        "YoY quarterly (full G10; EUR=DE Germany proxy; "
        "ALL G10 end ~2023-04..07 STALE on free FRED; "
        "{ISO3}ULQELP01GYSAQ country 404; OECD agg ends 2023-01; "
        "OPHNFB/PRS85006092 US alts live not mixed; "
        "IMF IFS UV/ToT fallback 404 on FRED)"
    )
    df.attrs["unit"] = "labour_prod_yoy_pct_as_reported"
    df.attrs["score_basis"] = "yoy_growth_as_reported"
    df.attrs["stale"] = True
    df.attrs["stale_note"] = "ULQELP01*Q657S ends ~2023-04..07 on free FRED for all G10"
    df.attrs["unmapped"] = [c for c in curs if c not in series_map]
    df.attrs["failed_or_alt"] = LP_FAILED_OR_ALT
    return df


def load_us_labour_prod(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US OECD MEI LP YoY (ULQELP01USQ657S), PIT-lagged (stale ~2023-07)."""
    panel = load_labour_prod_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US labour productivity series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_LP"
    out.attrs["series_id"] = LP_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def labour_prod_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_labour_prod_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_fred_ulqelp01"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
