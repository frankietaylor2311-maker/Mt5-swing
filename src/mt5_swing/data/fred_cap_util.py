"""FRED / OECD MEI capacity utilization (ULQECU01) panels for CU-growth FX.

Literature
----------
- Relative capacity utilization → cyclical strength / demand pressure / FX
  (OECD MEI early-estimate CU; Phillips-curve / output-gap adjacent).
- Primary prior: currencies with *high* relative CU YoY growth appreciate vs
  low-CU peers (strong activity / demand pressure). Honesty alternate: long
  low CU / slack / stress debtor.
- Natural companion completing the ULC (§48) / labour-productivity (§49)
  trilogy. Distinct from: employment §41, IP §42 (industrial volume, not CU),
  ULC §48, LP §49, Balassa–Samuelson §14, retail §46, DSR §47, CLI/CCI/BCI,
  building-permits §43, WUI, EPU/TPU.

Free data (verified live on FRED ~2026-09-23)
--------------------------------------------
Prefer OECD MEI **early-estimate quarterly capacity-utilization YoY growth**
via FRED ``ULQECU01{ISO2}Q657S`` (seasonally adjusted, growth rate same period
previous year). **Panel is STALE** — all G10 Q657S end ~**2023-04..07** (no
live refresh on free FRED as of probe). Country ``{ISO3}ULQECU01GYSAQ`` /
classic mnemonics **404**. ``OECDULQECU01GYSAQ`` aggregate ends **2023-01**.
US national ``TCU`` (total industry capacity utilization, level) live through
~2026-08 (alt — not mixed into OECD YoY panel).

| Ccy | FRED id (primary)   | Freq | Notes                                              |
|-----|---------------------|------|----------------------------------------------------|
| USD | ULQECU01USQ657S     | Q    | ends ~2023-07 (stale)                              |
| EUR | ULQECU01DEQ657S     | Q    | Germany proxy (EZ also ends 2023-07)               |
| GBP | ULQECU01GBQ657S     | Q    | ends ~2023-04 (stale)                              |
| JPY | ULQECU01JPQ657S     | Q    | ends ~2023-07 (stale)                              |
| CAD | ULQECU01CAQ657S     | Q    | ends ~2023-04 (stale)                              |
| AUD | ULQECU01AUQ657S     | Q    | ends ~2023-07 (stale)                              |
| NZD | ULQECU01NZQ657S     | Q    | ends ~2023-04 (stale)                              |
| CHF | ULQECU01CHQ657S     | Q    | ends ~2023-07 (stale)                              |

Tried / documented:
- ``{ISO3}ULQECU01GYSAQ`` / ``IXOBSAQ`` — country **404** (OECD agg ends 2023-01).
- ``ULQECU01*Q661S`` index — same ~2023-07 stale end (not fresher).
- ``ULQECU01EZQ657S`` EZ YoY — ends **2023-07** (same stale) → Germany
  ``ULQECU01DEQ657S`` EUR proxy (FR/IT/NL Q657S alts, also stale).
- US ``TCU`` — live national level alt through ~2026-08 (not mixed into OECD
  YoY panel; YoY of TCU not boarded as primary).

Unit choice (fixed a priori)
----------------------------
Panel stores **CU YoY growth rates as reported** (Q657S). Strategy scores use
these YoY values directly for XS ranks. Acceleration = Δ12 of that YoY growth
(after quarterly→monthly ffill).

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
OECD early-estimate quarterly capacity utilization typically lands ~1–3 months
after reference quarter. Default ``pub_lag_months=3`` (picked once a priori —
labour/activity quarterly band, same as ULC §48 / LP §49). Observation dated
quarter-start ``t`` is first known at ``t + 3 months``. Quarterly series are
pub-lagged then forward-filled to month-start (no intra-quarter leak). Strategy
modules add ``signal_lag=1`` month + a **1 trading-day** weight lag.

**Staleness honesty:** free FRED ULQECU panel ends mid-2023; post-2023 ranks
are ffilled frozen. Full-sample still informative; 2025/2026 windows reflect
that limitation. No go-live claim under ``approximate_non_ftmo``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED OECD MEI early-estimate capacity utilization YoY (quarterly Q657S)
# NOTE: all end ~2023-04..07 on free FRED (stale) — documented, still boarded.
CU_SERIES: dict[str, str | None] = {
    "USD": "ULQECU01USQ657S",
    "EUR": "ULQECU01DEQ657S",  # Germany proxy; EZ also ends 2023-07
    "GBP": "ULQECU01GBQ657S",
    "JPY": "ULQECU01JPQ657S",
    "CAD": "ULQECU01CAQ657S",
    "AUD": "ULQECU01AUQ657S",
    "NZD": "ULQECU01NZQ657S",
    "CHF": "ULQECU01CHQ657S",
}

CU_FREQ: dict[str, str] = {
    "USD": "Q",
    "EUR": "Q",
    "GBP": "Q",
    "JPY": "Q",
    "CAD": "Q",
    "AUD": "Q",
    "NZD": "Q",
    "CHF": "Q",
}

CU_FAILED_OR_ALT: dict[str, list[str]] = {
    "iso3_ulqecu01_gysaq_country_404": [
        "USAULQECU01GYSAQ",
        "GBRULQECU01GYSAQ",
        "JPNULQECU01GYSAQ",
        "CANULQECU01GYSAQ",
        "AUSULQECU01GYSAQ",
        "NZLULQECU01GYSAQ",
        "CHEULQECU01GYSAQ",
        "DEUULQECU01GYSAQ",
        "EA19ULQECU01GYSAQ",
        "OECDULQECU01GYSAQ",  # ends 2023-01
    ],
    "q657s_q661s_stale_~2023_07": [
        "ULQECU01USQ657S",
        "ULQECU01USQ661S",
        "ULQECU01GBQ657S",
        "ULQECU01JPQ657S",
        "ULQECU01CAQ657S",
        "ULQECU01AUQ657S",
        "ULQECU01NZQ657S",
        "ULQECU01CHQ657S",
        "ULQECU01DEQ657S",
        "ULQECU01EZQ657S",
    ],
    "eur_germany_proxy_primary": ["ULQECU01DEQ657S"],
    "eur_fr_it_nl_alts_also_stale": [
        "ULQECU01FRQ657S",
        "ULQECU01ITQ657S",
        "ULQECU01NLQ657S",
    ],
    "us_national_tcu_live_not_primary": ["TCU"],
}

EUR_FRANCE = "ULQECU01FRQ657S"
EUR_ITALY = "ULQECU01ITQ657S"
EUR_NLD = "ULQECU01NLQ657S"
EUR_EZ_STALE = "ULQECU01EZQ657S"
US_TCU_ALT = "TCU"

DEFAULT_PUB_LAG_MONTHS = 3  # a priori: quarterly CU labour/activity band (ULC §48 / LP §49)

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


def load_cap_util_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """OECD MEI capacity utilization YoY growth → monthly PIT panel.

    Columns = ISO currency codes. Values = CU **YoY growth %** as reported
    (Q657S). Index = UTC month-start of the PIT *known* date after
    ``pub_lag_months``. Quarterly series are pub-lagged first, then
    forward-filled to month-start. Strategy modules XS-rank on these YoY values
    (foreign only).

    **Staleness:** free FRED ends ~2023-04..07 for all G10 — post-2023 values
    are ffilled. Documented in attrs.
    """
    curs = [c.upper() for c in (currencies or CU_SERIES.keys())]
    monthly_cols: dict[str, pd.Series] = {}
    quarterly_cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}
    freq_map: dict[str, str] = {}

    for c in curs:
        sid = CU_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_ulqecu01"
            continue
        freq = CU_FREQ.get(c, "Q")
        try:
            s = _load_series(sid, freq=freq, download=download, force=force)
            series_map[c] = sid
            freq_map[c] = freq
            end = str(s.dropna().index.max().date()) if len(s.dropna()) else "?"
            if c == "EUR":
                notes[c] = (
                    f"Germany proxy via {sid} (Q) stale_end={end} "
                    f"(EZ also ~2023-07; FR/IT/NL Q657S alts)"
                )
            else:
                notes[c] = (
                    f"OECD MEI early-estimate CU YoY {sid} (quarterly) "
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

    df.attrs["factor"] = "oecd_mei_cap_util_ulqecu01"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["freq_map"] = freq_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI ULQECU01{ISO2}Q657S early-estimate capacity-utilization "
        "YoY quarterly (full G10; EUR=DE Germany proxy; "
        "ALL G10 end ~2023-04..07 STALE on free FRED; "
        "{ISO3}ULQECU01GYSAQ country 404; OECD agg ends 2023-01; "
        "TCU US national level alt live not mixed)"
    )
    df.attrs["unit"] = "cap_util_yoy_pct_as_reported"
    df.attrs["score_basis"] = "yoy_growth_as_reported"
    df.attrs["stale"] = True
    df.attrs["stale_note"] = "ULQECU01*Q657S ends ~2023-04..07 on free FRED for all G10"
    df.attrs["unmapped"] = [c for c in curs if c not in series_map]
    df.attrs["failed_or_alt"] = CU_FAILED_OR_ALT
    return df


def load_us_cap_util(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US OECD MEI CU YoY (ULQECU01USQ657S), PIT-lagged (stale ~2023-07)."""
    panel = load_cap_util_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US capacity utilization series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_CU"
    out.attrs["series_id"] = CU_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def cap_util_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_cap_util_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_fred_ulqecu01"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
