"""FRED / OECD MEI retail-sales (SLRTTO / RSAFS) panels for consumer-sales FX.

Literature
----------
- Dahlquist & Hasseltoft (2020, JFE) economic momentum: currencies with strong
  past macro trends (incl. retail sales) subsequently appreciate.
- Primary prior: currencies with *high* relative retail-sales growth appreciate
  vs low-growth peers. Honesty alternate: long low retail / consumer-stress.
- Distinct from: IP §42, employment §41, building-permits §43, OECD CLI/CCI/BCI,
  macro_diff EW blend, CA/TB, fiscal/debt, house-price, money, equity, IG OAS,
  commodity, GPR family, ACM TP, WUI, EPU/TPU.

Free data (verified live on FRED ~2026-09-23)
--------------------------------------------
Prefer OECD MEI **retail trade volume growth** ``SLRTTO01*Q657S`` where live.
USD: OECD ``USAWSCNDW01GYSAM`` ends ~2023-11 (stale) / ``SLRTTO01USQ657S`` ends
2018 → US Census ``RSAFS`` Advance Retail Sales (levels → YoY %). EUR: Germany
``SLRTTO01DEQ657S`` (``EA19WSCNDW01GYSAM`` **404**; ``FRAWSCNDW01*`` ends 2018;
``SLRTTO01EZQ657S`` ends 2023-07 — stale). CAD: ``SLRTTO01CAQ657S`` ends ~2022
→ ``CANWSCNDW01IXOBSAM`` index → YoY (ends ~2024-04 — still a gap vs peers).

| Ccy | FRED id (primary)      | Freq | Notes                                           |
|-----|------------------------|------|-------------------------------------------------|
| USD | RSAFS (→ YoY %)        | M    | Advance Retail Sales; USAWSCNDW GYSAM stale     |
| EUR | SLRTTO01DEQ657S        | Q    | Germany proxy (EA19 WSCNDW 404; FRA stale)      |
| GBP | SLRTTO01GBQ657S        | Q    | through ~2026-04                                |
| JPY | SLRTTO01JPQ657S        | Q    | through ~2026-04 (JPNWSCNDW GYSAM stale 2023)   |
| CAD | CANWSCNDW01IXOBSAM→YoY | M    | index→YoY; ends ~2024-04 (SLRTTO CAQ ends 2022) |
| AUD | SLRTTO01AUQ657S        | Q    | through ~2025-04                                |
| NZD | SLRTTO01NZQ657S        | Q    | through ~2026-04                                |
| CHF | SLRTTO01CHQ657S        | Q    | through ~2026-04                                |

Tried / documented:
- ``USAWSCNDW01GYSAM`` — end **2023-11** (stale); ``USAWSCNDW01IXOBSAM`` end **2024-03**.
- ``SLRTTO01USA657S`` annual — end **2024-01**; ``SLRTTO01USQ657S`` end **2018**.
- ``EA19WSCNDW01GYSAM`` — **404**; ``FRAWSCNDW01GYSAM`` / ``IXOBSAM`` end **2018-09**.
- ``SLRTTO01EZQ657S`` — end **2023-07** (stale) → DEU EUR proxy.
- ``JPNWSCNDW01GYSAM`` end **2023-11**; ``JPNWSCNDW01IXOBSAM`` end **2024-03**.
- ``SLRTTO01CAQ657S`` end **2022-01**; ``CANWSCNDW01GYSAM`` end **2023-11**.
- ``AUS/NZL/CHE/DEU/GBR WSCNDW01*`` — mostly **404**.
- ``SLRTTO01*M657S`` monthly growth — mostly **404**.
- US ``RSXFS`` / ``RETAILIMSA`` / ``MRTSSM44X72USS`` — live national alts (not mixed).

Unit choice (fixed a priori)
----------------------------
Panel stores **YoY / growth % as comparable units**. ``RSAFS`` / ``CANWSCNDW01IXOBSAM``
levels/index → ``pct_change(12)*100`` before pub lag. ``SLRTTO01*Q657S`` growth
series used **as reported**. Strategy scores XS-rank on these values directly.

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Retail sales typically land ~1–2 months after reference. Default
``pub_lag_months=2`` (same band as IP §42). Observation dated month-/quarter-start
``t`` is first known at ``t + 2 months``. Quarterly series are pub-lagged then
forward-filled to month-start (no intra-quarter leak). Strategy modules add a
**1 trading-day** weight lag (``signal_lag_days=1``); no extra month signal lag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED retail / consumer-sales series
RETAIL_SERIES: dict[str, str | None] = {
    "USD": "RSAFS",  # levels → YoY; USAWSCNDW / SLRTTO US stale
    "EUR": "SLRTTO01DEQ657S",  # Germany proxy
    "GBP": "SLRTTO01GBQ657S",
    "JPY": "SLRTTO01JPQ657S",
    "CAD": "CANWSCNDW01IXOBSAM",  # index → YoY; SLRTTO CAQ ends 2022
    "AUD": "SLRTTO01AUQ657S",
    "NZD": "SLRTTO01NZQ657S",
    "CHF": "SLRTTO01CHQ657S",
}

# Native series unit: "growth" already %; "level" needs pct_change(12)*100
RETAIL_UNIT: dict[str, str] = {
    "USD": "level",
    "EUR": "growth",
    "GBP": "growth",
    "JPY": "growth",
    "CAD": "level",
    "AUD": "growth",
    "NZD": "growth",
    "CHF": "growth",
}

RETAIL_FREQ: dict[str, str] = {
    "USD": "M",
    "EUR": "Q",
    "GBP": "Q",
    "JPY": "Q",
    "CAD": "M",
    "AUD": "Q",
    "NZD": "Q",
    "CHF": "Q",
}

RETAIL_FAILED_OR_ALT: dict[str, list[str]] = {
    "usd_oecd_wscndw_stale": ["USAWSCNDW01GYSAM", "USAWSCNDW01IXOBSAM"],
    "usd_slrtto_stale": ["SLRTTO01USA657S", "SLRTTO01USQ657S"],
    "usd_national_primary_rsafs": ["RSAFS"],
    "usd_national_alts_not_primary": ["RSXFS", "RETAILIMSA", "MRTSSM44X72USS"],
    "eur_ea19_wscndw_404": ["EA19WSCNDW01GYSAM"],
    "eur_fra_wscndw_stale_2018": ["FRAWSCNDW01GYSAM", "FRAWSCNDW01IXOBSAM"],
    "eur_ez_slrtto_stale_2023_07": ["SLRTTO01EZQ657S"],
    "eur_germany_proxy_primary": ["SLRTTO01DEQ657S"],
    "jpy_wscndw_stale": ["JPNWSCNDW01GYSAM", "JPNWSCNDW01IXOBSAM"],
    "cad_slrtto_ends_2022": ["SLRTTO01CAQ657S"],
    "cad_wscndw_gysam_stale_2023_11": ["CANWSCNDW01GYSAM"],
    "cad_index_yoy_primary_gap_2024_04": ["CANWSCNDW01IXOBSAM"],
    "wscndw_g10_mostly_404": [
        "AUSWSCNDW01GYSAM",
        "NZLWSCNDW01GYSAM",
        "CHEWSCNDW01GYSAM",
        "DEUWSCNDW01GYSAM",
        "GBRWSCNDW01GYSAM",
    ],
    "slrtto_monthly_m657s_mostly_404": [
        "SLRTTO01DEM657S",
        "SLRTTO01GBM657S",
        "SLRTTO01JPM657S",
        "SLRTTO01AUM657S",
        "SLRTTO01NZM657S",
        "SLRTTO01CHM657S",
        "SLRTTO01CAM657S",
    ],
}

DEFAULT_PUB_LAG_MONTHS = 2  # a priori: retail ~1–2m; same band as IP §42


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


def load_retail_sales_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """OECD MEI / US Census retail-sales growth → monthly PIT panel.

    Columns = ISO currency codes. Values = retail **YoY / growth %** (comparable).
    Index = UTC month-start of the PIT *known* date after ``pub_lag_months``.
    Level/index series (``RSAFS``, ``CANWSCNDW01IXOBSAM``) are converted to YoY %
    before the pub lag. Quarterly growth series are pub-lagged first, then
    forward-filled to month-start. Strategy modules XS-rank on these values
    (foreign only).
    """
    curs = [c.upper() for c in (currencies or RETAIL_SERIES.keys())]
    monthly_cols: dict[str, pd.Series] = {}
    quarterly_cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}
    freq_map: dict[str, str] = {}
    unit_map: dict[str, str] = {}

    for c in curs:
        sid = RETAIL_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_retail_sales"
            continue
        freq = RETAIL_FREQ.get(c, "M")
        unit = RETAIL_UNIT.get(c, "growth")
        try:
            s = _load_series(sid, freq=freq, download=download, force=force)
            if unit == "level":
                s = (s.pct_change(12) * 100.0).dropna()
                if c == "USD":
                    notes[c] = (
                        f"US Census {sid} levels → YoY % "
                        f"(USAWSCNDW01GYSAM ends 2023-11; SLRTTO01USQ657S ends 2018; "
                        f"RSXFS/RETAILIMSA alts)"
                    )
                elif c == "CAD":
                    notes[c] = (
                        f"OECD MEI {sid} index → YoY % "
                        f"(ends ~2024-04 gap; SLRTTO01CAQ657S ends 2022; "
                        f"CANWSCNDW01GYSAM ends 2023-11)"
                    )
                else:
                    notes[c] = f"{sid} levels/index → YoY %"
            elif c == "EUR":
                notes[c] = (
                    f"Germany proxy via {sid} (Q growth as reported) "
                    f"(EA19WSCNDW 404; FRAWSCNDW ends 2018; SLRTTO01EZQ657S ends 2023-07)"
                )
            else:
                notes[c] = f"OECD MEI retail volume growth {sid} (quarterly as reported)"
            series_map[c] = sid
            freq_map[c] = freq
            unit_map[c] = unit
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
        df = pd.concat(frames, axis=1, sort=True).sort_index()
        df = df.dropna(how="all")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

    df.attrs["factor"] = "oecd_mei_retail_sales_slrtto"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["freq_map"] = freq_map
    df.attrs["unit_map"] = unit_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI SLRTTO01*Q657S retail volume growth (GBP/JPY/AUD/NZD/CHF; "
        "EUR=DEU Germany proxy — EA19 WSCNDW 404 / FRA stale / EZ SLRTTO stale; "
        "USD=RSAFS levels→YoY — USAWSCNDW/SLRTTO US stale; "
        "CAD=CANWSCNDW01IXOBSAM index→YoY ends ~2024-04 — SLRTTO CAQ ends 2022)"
    )
    df.attrs["unit"] = "retail_yoy_growth_pct"
    df.attrs["score_basis"] = "yoy_growth_as_reported_or_derived"
    df.attrs["unmapped"] = [c for c in curs if c not in series_map]
    df.attrs["failed_or_alt"] = RETAIL_FAILED_OR_ALT
    df.attrs["gaps"] = {
        "CAD": "CANWSCNDW01IXOBSAM ends ~2024-04 after pub_lag — thin 2025/2026 CAD coverage",
        "AUD": "SLRTTO01AUQ657S ends ~2025-04 — thinner than peers into 2026",
    }
    return df


def load_us_retail_sales(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US RSAFS YoY %, PIT-lagged."""
    panel = load_retail_sales_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US retail sales series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_RETAIL"
    out.attrs["series_id"] = RETAIL_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def retail_sales_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_retail_sales_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_fred_retail_sales"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
