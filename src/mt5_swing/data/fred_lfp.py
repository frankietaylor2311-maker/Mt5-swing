"""FRED / OECD MEI labour-force participation / activity-rate (LRAC64TT) panels.

Literature
----------
- Dahlquist & Hasseltoft (2020, JFE) economic-momentum labour channel companion —
  labour-force **participation / activity rate** as the extensive labour-supply
  margin (growth / labour-engagement prior).
- Primary prior: currencies with *high* relative LFP / activity rate
  subsequently appreciate vs low-LFP peers. Honesty alternate: long low LFP /
  labour-disengagement debtor premium.
- Distinct from: employment persons §41 (LFEMTTTT YoY), UR levels §54
  (jobless rate), wages §55 (LCEAMN earnings YoY), ULC/LP/CU.

Free data (verified live on FRED probe 2026-09-24)
-------------------------------------------------
Activity rate ages 15–64 (%), OECD MEI / national:

| Ccy | FRED id            | Freq | Notes                                      |
|-----|--------------------|------|--------------------------------------------|
| USD | LRAC64TTUSM156S    | M    | live ~2026-07                              |
| JPY | LRAC64TTJPM156S    | M    | live ~2026-07                              |
| CAD | LRAC64TTCAM156S    | M    | live ~2026-08                              |
| AUD | LRAC64TTAUM156S    | M    | live ~2026-07                              |
| GBP | LRAC64TTGBQ156S    | Q    | live ~2026-04                              |
| EUR | LRAC64TTDEQ156S    | Q    | **Germany proxy** (EZQ156S stale ~2022-10) |
| CHF | LRAC64TTCHQ156S    | Q    | live ~2026-04                              |
| NZD | LRAC64TTNZQ156S    | Q    | live ~2026-04                              |

Tried / documented (not this wave):
- ``LRAC64TTEZQ156S`` — EA stale ~2022-10 (Germany proxy used instead).
- Monthly GB/DE/CH/NZ ``*M156S`` — mostly 404.
- Vacancies ``LMJVTTUV*`` — too stale (~2023) for 2025/26.
- Hours ``HOHWMN02*`` — G10 404.

Unit choice (fixed a priori)
----------------------------
Panel stores **activity / LFP rate levels (%)**. Strategy scores XS-rank on
levels (not YoY of persons). Rising participation = +Δ12 of the level.

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Participation / labour stats lag similar to other labour releases. Default
``pub_lag_months=2``. Observation dated month-/quarter-start ``t`` is first
known at ``t + 2 months``. Quarterly series are pub-lagged then forward-filled
to month-start (no intra-quarter leak) — mirror ``fred_wage_earnings`` mixed
M+Q. Strategy modules add ``signal_lag=1`` month + a **1 trading-day** weight
lag. Mixed M+Q keeps pub_lag=2 (not raised to 3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED OECD MEI activity rate 15–64 (LRAC64TT)
LFP_SERIES: dict[str, str | None] = {
    "USD": "LRAC64TTUSM156S",
    "JPY": "LRAC64TTJPM156S",
    "CAD": "LRAC64TTCAM156S",
    "AUD": "LRAC64TTAUM156S",
    "GBP": "LRAC64TTGBQ156S",
    "EUR": "LRAC64TTDEQ156S",  # Germany proxy; EZQ156S stale ~2022-10
    "CHF": "LRAC64TTCHQ156S",
    "NZD": "LRAC64TTNZQ156S",
}

LFP_FREQ: dict[str, str] = {
    "USD": "M",
    "JPY": "M",
    "CAD": "M",
    "AUD": "M",
    "GBP": "Q",
    "EUR": "Q",
    "CHF": "Q",
    "NZD": "Q",
}

LFP_FAILED_OR_ALT: dict[str, list[str]] = {
    "eur_ez_stale_germany_proxy": [
        "LRAC64TTEZQ156S",  # stale ~2022-10
        "LRAC64TTDEQ156S",  # Germany proxy (primary)
    ],
    "monthly_gb_de_ch_nz_mostly_404": [
        "LRAC64TTGBM156S",
        "LRAC64TTDEM156S",
        "LRAC64TTCHM156S",
        "LRAC64TTNZM156S",
    ],
    "vacancies_lmjvttuv_stale_not_this_wave": [
        "LMJVTTUVUSM647S",
        "LMJVTTUVEZM647S",
    ],
    "hours_hohwmn02_g10_404": [
        "HOHWMN02USM065S",
        "HOHWMN02GBM065S",
    ],
    "employment_persons_lfemtttt_distinct_41": [
        "LFEMTTTTUSM647S",
        "LFEMTTTTDEQ647S",
    ],
    "unemployment_rate_lrhutttt_distinct_54": [
        "LRHUTTTTUSM156S",
        "LRHUTTTTEZM156S",
    ],
    "wage_lceamn_distinct_55": [
        "LCEAMN01USM657S",
        "LCEAMN01EZQ657S",
    ],
}

EUR_EZ_ALT_STALE = "LRAC64TTEZQ156S"

DEFAULT_PUB_LAG_MONTHS = 2  # a priori: labour-participation lag; mixed M+Q


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


def load_lfp_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """OECD MEI activity / LFP rate levels (%) → monthly PIT panel.

    Columns = ISO currency codes. Values = activity rate **levels (%)**.
    Index = UTC month-start of the PIT *known* date after ``pub_lag_months``.
    Quarterly series are pub-lagged first, then forward-filled to month-start
    (no intra-quarter leak). Full G10 mapped (EUR = Germany proxy).
    """
    curs = [c.upper() for c in (currencies or list(LFP_SERIES.keys()))]
    monthly_cols: dict[str, pd.Series] = {}
    quarterly_cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}
    freq_map: dict[str, str] = {}

    for c in curs:
        sid = LFP_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_lrac64tt"
            continue
        freq = LFP_FREQ.get(c, "M")
        try:
            s = _load_series(sid, freq=freq, download=download, force=force)
            if c == "EUR":
                notes[c] = (
                    f"Germany proxy {sid} (Q levels %); "
                    f"EA {EUR_EZ_ALT_STALE} stale ~2022-10 — not primary"
                )
            elif freq == "Q":
                notes[c] = f"OECD MEI activity rate {sid} quarterly levels %"
            else:
                notes[c] = f"OECD MEI activity rate {sid} monthly levels %"
            series_map[c] = sid
            freq_map[c] = freq
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
    # Expand each quarterly series alone so a longer leg cannot ffill a shorter
    # one past its true last obs (mirror wage mixed-freq honesty).
    for c, s in quarterly_cols.items():
        qdf = pd.DataFrame({c: s}).sort_index()
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

    stale_note_parts = []
    if "EUR" in series_map and series_map["EUR"] == "LRAC64TTDEQ156S":
        stale_note_parts.append(
            "EUR EA LRAC64TTEZQ156S stale ~2022-10 — Germany DEQ156S proxy used"
        )

    df.attrs["factor"] = "oecd_mei_lfp_activity_rate_lrac64tt"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["freq_map"] = freq_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI LRAC64TT activity rate ages 15–64 (%) "
        "(USD/JPY/CAD/AUD *M156S; GBP/CHF/NZD *Q156S; "
        "EUR=LRAC64TTDEQ156S Germany proxy — EZQ156S stale ~2022-10; "
        "full G10 mapped; vacancies/hours alts documented not boarded)"
    )
    df.attrs["unit"] = "lfp_activity_rate_pct_level"
    df.attrs["score_basis"] = "lfp_level_pct"
    df.attrs["unmapped"] = [c for c in curs if c not in series_map]
    df.attrs["failed_or_alt"] = LFP_FAILED_OR_ALT
    df.attrs["stale"] = bool(
        "EUR" in series_map and series_map.get("EUR") == "LRAC64TTDEQ156S"
    )
    df.attrs["stale_note"] = (
        "; ".join(stale_note_parts)
        if stale_note_parts
        else "USD/JPY/CAD/AUD/GBP/CHF/NZD live into 2025–2026; EUR=DE proxy"
    )
    return df


def load_us_lfp(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US activity rate 15–64 (LRAC64TTUSM156S), PIT-lagged."""
    panel = load_lfp_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US LFP / activity-rate series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_LFP"
    out.attrs["series_id"] = LFP_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def lfp_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_lfp_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_fred_lrac64tt"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
