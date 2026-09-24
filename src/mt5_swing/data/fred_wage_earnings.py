"""FRED / OECD MEI manufacturing earnings (LCEAMN01) panels for wage-growth FX.

Literature
----------
- Relative wage / manufacturing earnings growth → competitiveness /
  REER-adjacent labour-cost channel (Balassa–Samuelson wage side;
  Dahlquist–Hasseltoft labour-momentum companion).
- Primary prior: currencies with *low* relative wage YoY growth appreciate vs
  high-wage peers (competitiveness). Honesty alternate: long high wage growth /
  labour-cost stress debtor.
- Distinct from: ULC §48 (ULQEUL01 = wages/productivity unit labour cost),
  employment persons §41, UR levels §54, LP §49, CU §50, PPI/CPI.

Free data (verified live on FRED cache ~2026-09-24)
-------------------------------------------------
Prefer OECD MEI manufacturing earnings **YoY** ``LCEAMN01*657S`` where live;
else YoY of ``*661S`` / ``*661N`` index levels. Documented as ULC §48 wage
fallback "not primary" — now boarded as a **dedicated** scholarly sleeve.

| Ccy | FRED id (primary)   | Freq | Unit   | Notes                                      |
|-----|---------------------|------|--------|--------------------------------------------|
| USD | LCEAMN01USM657S     | M    | growth | live ~2026-07                              |
| EUR | LCEAMN01EZQ657S     | Q    | growth | EA YoY; DE ``LCEAMN01DEQ657S`` / FR Q661S alts |
| GBP | LCEAMN01GBM661S→YoY | M    | level  | monthly index live ~2026-03                |
| JPY | LCEAMN01JPM661S→YoY | M    | level  | monthly index live ~2026-06                |
| AUD | LCEAMN01AUQ661S→YoY | Q    | level  | quarterly index → YoY ~2026-01             |
| CAD | LCEAMN01CAM657S     | M    | growth | live ~2026-04                              |
| NZD | LCEAMN01NZQ657S     | Q    | growth | quarterly YoY live ~2026-01                |
| CHF | —                   | —    | —      | **unmapped** (LCEAMN01CH* 404)             |

Tried / documented:
- ``LCEAMN01CH*`` CHF — **404** (leave unmapped; do not invent).
- ``LCEAMN01DEQ657S`` / ``LCEAMN01FRQ661S`` — EUR alts (DE fresher ~2026-01;
  FR Q index → YoY gap-fill).
- ``LCEAMN01USM661N`` / CES AHET — US alts (not mixed into OECD XS panel).
- ``LCEAMN01USQ657S`` / ``CAQ657S`` — quarterly companions (monthly preferred).

Unit choice (fixed a priori)
----------------------------
Panel stores **manufacturing earnings YoY growth %**. Native ``*657S`` used as
reported. Index ``*661S``/``*661N`` convert to YoY before pub lag (monthly
``pct_change(12)*100``; quarterly ``pct_change(4)*100``). Strategy scores
XS-rank on these YoY values.

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Earnings often lag UR/CPI. Default ``pub_lag_months=2``. Observation dated
month-/quarter-start ``t`` is first known at ``t + 2 months``. Quarterly
series are pub-lagged then forward-filled to month-start (no intra-quarter
leak). Strategy modules add ``signal_lag=1`` month + a **1 trading-day**
weight lag. Mixed M+Q panel keeps pub_lag=2 (not raised to 3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED OECD MEI manufacturing earnings (LCEAMN01)
WAGE_SERIES: dict[str, str | None] = {
    "USD": "LCEAMN01USM657S",
    "EUR": "LCEAMN01EZQ657S",  # EA YoY; DE/FR alts documented
    "GBP": "LCEAMN01GBM661S",  # index → YoY
    "JPY": "LCEAMN01JPM661S",  # index → YoY
    "AUD": "LCEAMN01AUQ661S",  # Q index → YoY
    "CAD": "LCEAMN01CAM657S",
    "NZD": "LCEAMN01NZQ657S",
    "CHF": None,  # LCEAMN01CH* 404 — leave unmapped
}

WAGE_UNIT: dict[str, str] = {
    "USD": "growth",
    "EUR": "growth",
    "GBP": "level",
    "JPY": "level",
    "AUD": "level",
    "CAD": "growth",
    "NZD": "growth",
}

WAGE_FREQ: dict[str, str] = {
    "USD": "M",
    "EUR": "Q",
    "GBP": "M",
    "JPY": "M",
    "AUD": "Q",
    "CAD": "M",
    "NZD": "Q",
}

WAGE_FAILED_OR_ALT: dict[str, list[str]] = {
    "chf_lceamn_404_unmapped": [
        "LCEAMN01CHQ657S",
        "LCEAMN01CHQ661S",
        "LCEAMN01CHM657S",
        "LCEAMN01CHM661S",
    ],
    "eur_ez_primary_de_fr_alts": [
        "LCEAMN01EZQ657S",
        "LCEAMN01DEQ657S",
        "LCEAMN01FRQ661S",
        "LCEAMN01DEM661N",
        "LCEAMN01EZQ661S",
    ],
    "gbp_jpy_aud_index_to_yoy": [
        "LCEAMN01GBM661S",
        "LCEAMN01JPM661S",
        "LCEAMN01AUQ661S",
        "LCEAMN01GBQ661S",
        "LCEAMN01JPQ661S",
    ],
    "us_national_alts_not_mixed": [
        "LCEAMN01USM661N",
        "LCEAMN01USQ657S",
        "CES0500000003",  # AHET note only
    ],
    "ulc_48_lceamn_was_fallback_not_primary": [
        "LCEAMN01USM657S",
        "LCEAMN01CAM657S",
        "LCEAMN01NZQ657S",
    ],
}

EUR_DE_ALT = "LCEAMN01DEQ657S"
EUR_FR_ALT = "LCEAMN01FRQ661S"
US_Q_ALT = "LCEAMN01USQ657S"

DEFAULT_PUB_LAG_MONTHS = 2  # a priori: earnings often lag UR/CPI


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


def load_wage_earnings_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """OECD MEI manufacturing earnings YoY → monthly PIT panel.

    Columns = ISO currency codes. Values = wage/earnings **YoY %**. Index =
    UTC month-start of the PIT *known* date after ``pub_lag_months``. Level
    series convert to YoY before the pub lag. Quarterly YoY series are
    pub-lagged first, then forward-filled to month-start. CHF is requested so
    coverage docs show the unmapped gap.
    """
    curs = [c.upper() for c in (currencies or list(WAGE_SERIES.keys()))]
    monthly_cols: dict[str, pd.Series] = {}
    quarterly_cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}
    freq_map: dict[str, str] = {}
    unit_map: dict[str, str] = {}

    for c in curs:
        sid = WAGE_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_lceamn_404"
            continue
        freq = WAGE_FREQ.get(c, "M")
        unit = WAGE_UNIT.get(c, "growth")
        try:
            s = _load_series(sid, freq=freq, download=download, force=force)
            if unit == "level":
                if freq == "Q":
                    s = (s.pct_change(4) * 100.0).dropna()
                else:
                    s = (s.pct_change(12) * 100.0).dropna()
                notes[c] = (
                    f"OECD MEI {sid} {'quarterly' if freq == 'Q' else 'monthly'} "
                    f"index → YoY %"
                )
            elif c == "EUR":
                notes[c] = (
                    f"EA YoY via {sid} (Q growth as reported); "
                    f"DE {EUR_DE_ALT} / FR {EUR_FR_ALT}→YoY alts"
                )
            else:
                notes[c] = f"OECD MEI manufacturing earnings YoY {sid} (as reported)"
            series_map[c] = sid
            freq_map[c] = freq
            unit_map[c] = unit
            if freq == "Q":
                quarterly_cols[c] = s
            else:
                monthly_cols[c] = s
        except Exception as exc:  # noqa: BLE001
            # EUR fallback: try DE then FR→YoY
            if c == "EUR":
                loaded = False
                for alt_sid, alt_freq, alt_unit, alt_note in (
                    (EUR_DE_ALT, "Q", "growth", "Germany proxy DEQ657S (EZ load fail)"),
                    (EUR_FR_ALT, "Q", "level", "France FRQ661S→YoY gap-fill (EZ/DE fail)"),
                ):
                    try:
                        s = _load_series(
                            alt_sid, freq=alt_freq, download=download, force=force
                        )
                        if alt_unit == "level":
                            s = (s.pct_change(4) * 100.0).dropna()
                        series_map[c] = alt_sid
                        freq_map[c] = alt_freq
                        unit_map[c] = alt_unit
                        notes[c] = alt_note
                        quarterly_cols[c] = s
                        loaded = True
                        break
                    except Exception:  # noqa: BLE001
                        continue
                if not loaded:
                    notes[c] = f"load_fail:{exc}"[:120]
            else:
                notes[c] = f"load_fail:{exc}"[:120]

    frames: list[pd.DataFrame] = []
    if monthly_cols:
        mdf = pd.DataFrame(monthly_cols).sort_index()
        mdf = _apply_pub_lag_monthly(mdf, pub_lag_months=pub_lag_months)
        frames.append(mdf)
    # Expand each quarterly series alone so a longer leg (AUD/NZD) cannot
    # ffill a shorter one (EUR EZ ends ~2025-07) past its true last obs.
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

    # Staleness honesty: EUR EZ quarterly ends ~2025-07 (ffill into 2026)
    stale_note_parts = []
    if "EUR" in series_map and series_map["EUR"] == "LCEAMN01EZQ657S":
        stale_note_parts.append(
            "EUR EA LCEAMN01EZQ657S ends ~2025-07 (post-end ffilled; DE/FR alts)"
        )
    if "CHF" in curs and "CHF" not in series_map:
        stale_note_parts.append("CHF LCEAMN unmapped (404)")

    df.attrs["factor"] = "oecd_mei_manufacturing_earnings_lceamn"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["freq_map"] = freq_map
    df.attrs["unit_map"] = unit_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI LCEAMN01 manufacturing earnings YoY "
        "(USD/CAD *M657S; EUR=*EZQ657S; NZD=*NZQ657S; "
        "GBP/JPY *M661S→YoY; AUD *AUQ661S→YoY; CHF unmapped 404; "
        "ULC §48 documented this as wage fallback — now dedicated sleeve)"
    )
    df.attrs["unit"] = "wage_earnings_yoy_pct"
    df.attrs["score_basis"] = "yoy_growth_pct"
    df.attrs["unmapped"] = [c for c in curs if c not in series_map]
    df.attrs["failed_or_alt"] = WAGE_FAILED_OR_ALT
    df.attrs["stale"] = bool(
        "EUR" in series_map and series_map.get("EUR") == "LCEAMN01EZQ657S"
    )
    df.attrs["stale_note"] = (
        "; ".join(stale_note_parts)
        if stale_note_parts
        else "USD/GBP/JPY/AUD/CAD/NZD live into 2025–2026; CHF unmapped"
    )
    return df


def load_us_wage_earnings(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US manufacturing earnings YoY (LCEAMN01USM657S), PIT-lagged."""
    panel = load_wage_earnings_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US wage/earnings series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_WAGE"
    out.attrs["series_id"] = WAGE_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def wage_earnings_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_wage_earnings_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_fred_lceamn_404"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
