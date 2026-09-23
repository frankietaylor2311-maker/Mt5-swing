"""FRED / OECD MEI unit labour cost (ULQEUL01) panels for ULC-growth FX.

Literature
----------
- Relative unit labour costs / wage growth → competitiveness / REER-adjacent
  labour-cost FX (OECD MEI early-estimate ULC; Balassa–Samuelson-adjacent
  labour-cost channel; relative-cost prior).
- Primary prior: currencies with *low* relative ULC YoY growth appreciate vs
  high-ULC peers (competitiveness). Honesty alternate: long high ULC /
  labour-cost stress debtor.
- Distinct from: employment §41 (persons YoY), IP §42, retail §46, house-price
  §35, REER §31, money §33, credit §32, DSR §47, CLI/CCI/BCI, WUI, EPU/TPU.

Free data (verified live on FRED ~2026-09-23)
--------------------------------------------
Prefer OECD MEI **early-estimate quarterly ULC YoY growth** via FRED
``ULQEUL01{ISO2}Q657S`` (seasonally adjusted, growth rate same period previous
year). Index ``*Q661S`` ends ~**2023-07** (stale — not primary). Country
``{ISO3}ULQEUL01GYSAQ`` / ``ULQULC01*`` / ``ULCQEM01*`` mostly **404**.
``OECDULQEUL01GYSAQ`` aggregate live but ends **2023-01** (not primary).
US national ``ULCNFB`` / ``PRS85006112`` live alts (not mixed into OECD panel).

| Ccy | FRED id (primary)   | Freq | Notes                                              |
|-----|---------------------|------|----------------------------------------------------|
| USD | ULQEUL01USQ657S     | Q    | through ~2025-07                                   |
| EUR | ULQEUL01DEQ657S     | Q    | Germany proxy (``ULQEUL01EZQ657S`` ends 2024-01)   |
| GBP | ULQEUL01GBQ657S     | Q    | through ~2025-10                                   |
| JPY | ULQEUL01JPQ657S     | Q    | through ~2026-01                                   |
| CAD | ULQEUL01CAQ657S     | Q    | through ~2026-01                                   |
| AUD | ULQEUL01AUQ657S     | Q    | through ~2026-01                                   |
| NZD | ULQEUL01NZQ657S     | Q    | through ~2026-01                                   |
| CHF | ULQEUL01CHQ657S     | Q    | through ~2025-10                                   |

Tried / documented:
- ``ULQULC01*`` / ``ULCQEM01*`` / ``ULQMHN01*`` classic — mostly **404**.
- ``{ISO3}ULQEUL01GYSAQ`` / ``IXOBSAQ`` — country **404** (only ``OECDULQEUL01GYSAQ`` live, ends 2023-01).
- ``ULQEUL01*Q661S`` index — ends ~**2023-04..07** (stale — not primary).
- ``ULQEUL01EZQ657S`` EZ YoY — ends **2024-01** (stale) → Germany
  ``ULQEUL01DEQ657S`` EUR proxy (FR/IT/ES/NL Q657S alts).
- Wage fallback ``LCEAMN01*`` manufacturing earnings — live for several G10
  but **not primary** once full ULC YoY panel found (documented as alt).
- US ``ULCNFB`` / ``PRS85006112`` — live national alts (not mixed into OECD panel).

Unit choice (fixed a priori)
----------------------------
Panel stores **ULC YoY growth rates as reported** (Q657S). Strategy scores use
these YoY values directly for XS ranks. Acceleration = Δ12 of that YoY growth
(after quarterly→monthly ffill).

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
OECD early-estimate quarterly ULC typically lands ~1–3 months after reference
quarter. Default ``pub_lag_months=3`` (picked once a priori — labour-adjacent,
same band as employment §41). Observation dated quarter-start ``t`` is first
known at ``t + 3 months``. Quarterly series are pub-lagged then forward-filled
to month-start (no intra-quarter leak). Strategy modules add ``signal_lag=1``
month (quarterly-ish) + a **1 trading-day** weight lag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED OECD MEI early-estimate ULC YoY (quarterly Q657S)
ULC_SERIES: dict[str, str | None] = {
    "USD": "ULQEUL01USQ657S",
    "EUR": "ULQEUL01DEQ657S",  # Germany proxy; EZ Q657S ends 2024-01
    "GBP": "ULQEUL01GBQ657S",
    "JPY": "ULQEUL01JPQ657S",
    "CAD": "ULQEUL01CAQ657S",
    "AUD": "ULQEUL01AUQ657S",
    "NZD": "ULQEUL01NZQ657S",
    "CHF": "ULQEUL01CHQ657S",
}

# Native frequency tag for each primary id (all quarterly)
ULC_FREQ: dict[str, str] = {
    "USD": "Q",
    "EUR": "Q",
    "GBP": "Q",
    "JPY": "Q",
    "CAD": "Q",
    "AUD": "Q",
    "NZD": "Q",
    "CHF": "Q",
}

ULC_FAILED_OR_ALT: dict[str, list[str]] = {
    "ulqulc01_ulcqem01_ulqmhn01_mostly_404": [
        "ULQULC01USQ657S",
        "ULCQEM01USQ657S",
        "ULQMHN01USQ657S",
        "USAULC",
        "USAULQEUL01GYSAQ",
        "GBRULQEUL01GYSAQ",
        "DEUULQEUL01GYSAQ",
    ],
    "iso3_ulqeul01_gysaq_country_404_oecd_agg_stale": [
        "OECDULQEUL01GYSAQ",  # ends 2023-01
        "USAULQEUL01GYSAQ",
        "GBRULQEUL01GYSAQ",
        "JPNULQEUL01GYSAQ",
        "CANULQEUL01GYSAQ",
        "AUSULQEUL01GYSAQ",
        "NZLULQEUL01GYSAQ",
        "CHEULQEUL01GYSAQ",
        "DEUULQEUL01GYSAQ",
        "EA19ULQEUL01GYSAQ",
    ],
    "q661s_index_stale_~2023_07": [
        "ULQEUL01USQ661S",
        "ULQEUL01GBQ661S",
        "ULQEUL01JPQ661S",
        "ULQEUL01CAQ661S",
        "ULQEUL01AUQ661S",
        "ULQEUL01NZQ661S",
        "ULQEUL01CHQ661S",
        "ULQEUL01DEQ661S",
        "ULQEUL01EZQ661S",
    ],
    "eur_ez_yoy_stale_2024_01": ["ULQEUL01EZQ657S"],
    "eur_germany_proxy_primary": ["ULQEUL01DEQ657S"],
    "eur_fr_it_es_nl_alts": [
        "ULQEUL01FRQ657S",
        "ULQEUL01ITQ657S",
        "ULQEUL01ESQ657S",
        "ULQEUL01NLQ657S",
    ],
    "wage_lceamn_fallback_not_primary": [
        "LCEAMN01USM657S",
        "LCEAMN01CAM657S",
        "LCEAMN01USQ657S",
        "LCEAMN01CAQ657S",
        "LCEAMN01DEQ657S",
        "LCEAMN01NZQ657S",
        "LCEAMN01GBQ661S",
        "LCEAMN01JPQ661S",
        "LCEAMN01AUQ661S",
    ],
    "us_national_alts_not_primary": ["ULCNFB", "PRS85006112", "COMPRNFB"],
    "chf_wage_lceamn_404": [
        "LCEAMN01CHQ657S",
        "LCEAMN01CHQ661S",
        "LCEAMN01CHM657S",
        "LCEAMN01CHM661S",
    ],
}

EUR_FRANCE = "ULQEUL01FRQ657S"
EUR_ITALY = "ULQEUL01ITQ657S"
EUR_SPAIN = "ULQEUL01ESQ657S"
EUR_NLD = "ULQEUL01NLQ657S"
EUR_EZ_STALE = "ULQEUL01EZQ657S"
US_ULCNFB_ALT = "ULCNFB"

DEFAULT_PUB_LAG_MONTHS = 3  # a priori: quarterly ULC labour-adjacent (emp §41 band)


def _to_month_start(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = pd.DatetimeIndex(s.index)
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    ms = pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start")).tz_localize("UTC")
    s = s.copy()
    s.index = ms
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _to_quarter_start(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = pd.DatetimeIndex(s.index)
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    qs = pd.DatetimeIndex(naive.to_period("Q").to_timestamp(how="start")).tz_localize("UTC")
    s = s.copy()
    s.index = qs
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _load_series(series_id: str, *, freq: str, download: bool, force: bool) -> pd.Series:
    if download:
        download_fred_series(series_id, force=force)
    raw = load_fred_series(series_id, download=False)
    if freq.upper().startswith("Q"):
        return _to_quarter_start(raw)
    return _to_month_start(raw)


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


def _expand_quarterly_to_monthly(
    df: pd.DataFrame,
    *,
    pub_lag_months: int,
) -> pd.DataFrame:
    """Shift quarterly index by pub_lag, then ffill to month-start (no leak)."""
    if df.empty:
        return df
    out = df.copy()
    if pub_lag_months > 0:
        out.index = out.index + pd.DateOffset(months=int(pub_lag_months))
        out = out[~out.index.duplicated(keep="last")].sort_index()
    naive_min = out.index.min().tz_convert(None) if out.index.tz is not None else out.index.min()
    naive_max = out.index.max().tz_convert(None) if out.index.tz is not None else out.index.max()
    start = naive_min.to_period("M").to_timestamp(how="start")
    end = naive_max.to_period("M").to_timestamp(how="start")
    monthly_idx = pd.date_range(start, end, freq="MS", tz="UTC")
    out = out.reindex(monthly_idx.union(out.index)).sort_index()
    out = out.ffill()
    out = out.loc[monthly_idx]
    out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def load_ulc_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """OECD MEI ULC YoY growth → monthly PIT panel.

    Columns = ISO currency codes. Values = ULC **YoY growth %** as reported
    (Q657S). Index = UTC month-start of the PIT *known* date after
    ``pub_lag_months``. Quarterly series are pub-lagged first, then
    forward-filled to month-start. Strategy modules XS-rank on these YoY values
    (foreign only).
    """
    curs = [c.upper() for c in (currencies or ULC_SERIES.keys())]
    monthly_cols: dict[str, pd.Series] = {}
    quarterly_cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}
    freq_map: dict[str, str] = {}

    for c in curs:
        sid = ULC_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_ulqeul01"
            continue
        freq = ULC_FREQ.get(c, "Q")
        try:
            s = _load_series(sid, freq=freq, download=download, force=force)
            series_map[c] = sid
            freq_map[c] = freq
            if c == "EUR":
                notes[c] = (
                    f"Germany proxy via {sid} (Q) "
                    f"(ULQEUL01EZQ657S ends 2024-01; FR/IT/ES/NL Q657S alts)"
                )
            else:
                notes[c] = f"OECD MEI early-estimate ULC YoY {sid} (quarterly)"
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

    df.attrs["factor"] = "oecd_mei_ulc_ulqeul01"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["freq_map"] = freq_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI ULQEUL01{ISO2}Q657S early-estimate ULC YoY quarterly "
        "(full G10; EUR=DE Germany proxy — EZ Q657S ends 2024-01; "
        "Q661S index stale ~2023-07 — not primary; "
        "{ISO3}ULQEUL01GYSAQ country 404; ULQULC01/ULCQEM01 404; "
        "LCEAMN wage fallback not primary; ULCNFB/PRS85006112 US alts)"
    )
    df.attrs["unit"] = "ulc_yoy_pct_as_reported"
    df.attrs["score_basis"] = "yoy_growth_as_reported"
    df.attrs["unmapped"] = [c for c in curs if c not in series_map]
    df.attrs["failed_or_alt"] = ULC_FAILED_OR_ALT
    return df


def load_us_ulc(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US OECD MEI ULC YoY (ULQEUL01USQ657S), PIT-lagged."""
    panel = load_ulc_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US ULC series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_ULC"
    out.attrs["series_id"] = ULC_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def ulc_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_ulc_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_fred_ulqeul01"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
