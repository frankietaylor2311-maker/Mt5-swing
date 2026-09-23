"""FRED / OECD MEI employment (LFEMTTTT) panels for labour-growth FX differentials.

Literature
----------
- Labour / employment growth as a macro–FX differential (Dahlquist–Hasseltoft-style
  framing; labour-market strength → subsequent appreciation prior).
- Primary prior: currencies with *high* relative employment growth appreciate vs
  low-growth peers. Honesty alternate: long low emp / labour-stress debtor premium.
- Distinct from: macro-diff UR/CPI/IP, OECD CLI/CCI/BCI, WUI, EPU/TPU, CA/TB,
  fiscal/debt, BIS REER/credit, reserves, house-price, money-growth, equity-diff,
  IG OAS. (Retail-sales OECD SARTMISMEI probed separately and **stale** — not
  re-run as primary.)

Free data (verified live on FRED ~2026-09-23)
--------------------------------------------
OECD MEI **civilian employment** persons via FRED ``LFEMTTTT*``. Prefer live
**levels** ``*M647S`` / ``*Q647S`` and compute YoY in the strategy — the YoY
measure ``*M657S`` / ``*Q657S`` is mostly **404** or **stale** (~ends 2023-10
…2024-02).

| Ccy | FRED id (primary)   | Freq | Notes                                              |
|-----|---------------------|------|----------------------------------------------------|
| USD | LFEMTTTTUSM647S     | M    | persons; through ~2025-09                          |
| EUR | LFEMTTTTDEQ647S     | Q    | Germany proxy (``LFEMTTTTEZQ647S`` ends 2022-10)   |
| GBP | LFEMTTTTGBQ647S     | Q    | through ~2026-04                                   |
| JPY | LFEMTTTTJPM647S     | M    | through ~2026-07                                   |
| CAD | LFEMTTTTCAM647S     | M    | through ~2026-08                                   |
| AUD | LFEMTTTTAUM647S     | M    | through ~2026-07                                   |
| NZD | LFEMTTTTNZQ647S     | Q    | through ~2026-04                                   |
| CHF | LFEMTTTTCHQ647S     | Q    | through ~2026-04                                   |

Tried / documented:
- ``LFEMTTTT*M657S`` YoY growth — mostly **404**; ``LFEMTTTTAUM657S`` ends
  **2024-02** (stale — not primary).
- ``LFEMTTTT*Q657S`` YoY — live historically, ends ~**2023-10** / 2024-01 (stale).
- ``LFEMTTTTEZQ647S`` EZ persons — ends **2022-10** (stale) → Germany
  ``LFEMTTTTDEQ647S`` EUR proxy (FR/IT/ES Q647S alts).
- ``LFEMTTTT*M159N`` levels — mostly **404**.
- US ``PAYEMS`` / ``CE16OV`` — live national alts (not mixed into OECD persons
  panel for cross-country consistency).
- ``LREM64TT*`` employment *rate* — live for several G10 but **distinct** concept
  (rate vs employment growth); not primary for this LFEMTTTT wave.

Unit choice (fixed a priori)
----------------------------
Panel stores **employment persons levels**. Strategy scores use **YoY % change**
``P_t / P_{t−12} − 1`` (or equiv. after quarterly→monthly ffill) for XS ranks —
not raw levels. Acceleration = Δ12 of that YoY growth. Documented in panel attrs.

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
National / OECD labour releases often land 1–3 months after reference (monthly
payrolls faster; quarterly LFS slower). Default ``pub_lag_months=3`` (picked
once a priori — labour slower than BCI/money's 2). Observation dated
month-/quarter-start ``t`` is first known at ``t + 3 months``. Quarterly series
are pub-lagged then forward-filled to month-start (no intra-quarter leak).
Strategy modules add a **1 trading-day** weight lag (``signal_lag_days=1``);
no extra month signal lag (a priori for §41).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED OECD MEI employment persons (levels; YoY in strategy)
EMPLOYMENT_SERIES: dict[str, str | None] = {
    "USD": "LFEMTTTTUSM647S",
    "EUR": "LFEMTTTTDEQ647S",  # Germany proxy; EZ Q647S ends 2022-10
    "GBP": "LFEMTTTTGBQ647S",
    "JPY": "LFEMTTTTJPM647S",
    "CAD": "LFEMTTTTCAM647S",
    "AUD": "LFEMTTTTAUM647S",
    "NZD": "LFEMTTTTNZQ647S",
    "CHF": "LFEMTTTTCHQ647S",
}

# Native frequency tag for each primary id (M=monthly, Q=quarterly)
EMPLOYMENT_FREQ: dict[str, str] = {
    "USD": "M",
    "EUR": "Q",
    "GBP": "Q",
    "JPY": "M",
    "CAD": "M",
    "AUD": "M",
    "NZD": "Q",
    "CHF": "Q",
}

EMPLOYMENT_FAILED_OR_ALT: dict[str, list[str]] = {
    "yoy_m657s_mostly_404_or_stale": [
        "LFEMTTTTUSM657S",
        "LFEMTTTTGBM657S",
        "LFEMTTTTJPM657S",
        "LFEMTTTCAM657S",
        "LFEMTTTTAUM657S",  # ends 2024-02
        "LFEMTTTTNZM657S",
        "LFEMTTTTCHM657S",
        "LFEMTTTEZM657S",
        "LFEMTTTTDEM657S",
    ],
    "yoy_q657s_stale_~2023_10": [
        "LFEMTTTTUSQ657S",
        "LFEMTTTTGBQ657S",
        "LFEMTTTTJPQ657S",
        "LFEMTTTTCAQ657S",
        "LFEMTTTTAUQ657S",
        "LFEMTTTTNZQ657S",
        "LFEMTTTTCHQ657S",
        "LFEMTTTTDEQ657S",
        "LFEMTTTTEZQ657S",
    ],
    "eur_ez_persons_stale_2022": ["LFEMTTTTEZQ647S"],
    "eur_germany_proxy_primary": ["LFEMTTTTDEQ647S"],
    "eur_fr_it_es_alts": ["LFEMTTTTFRQ647S", "LFEMTTTTITQ647S", "LFEMTTTTESQ647S"],
    "m159n_levels_mostly_404": [
        "LFEMTTTTUSM159N",
        "LFEMTTTTGBM159N",
        "LFEMTTTTJPM159N",
        "LFEMTTTCAM159N",
        "LFEMTTTTAUM159N",
        "LFEMTTTTNZM159N",
        "LFEMTTTTCHM159N",
        "LFEMTTTEZM159N",
        "LFEMTTTTDEM159N",
    ],
    "us_national_alts_not_primary": ["PAYEMS", "CE16OV"],
    "employment_rate_lrem_not_primary": [
        "LREM64TTUSM156S",
        "LREM64TTJPM156S",
        "LREM64TTCAM156S",
        "LREM64TTAUM156S",
        "LREM64TTGBQ156N",
        "LREM64TTDEQ156N",
        "LREM64TTNZQ156N",
        "LREM64TTCHQ156N",
    ],
    "retail_sales_sartmismei_stale_skipped": ["SARTMISMEI"],
}

EUR_FRANCE = "LFEMTTTTFRQ647S"
EUR_ITALY = "LFEMTTTTITQ647S"
EUR_SPAIN = "LFEMTTTTESQ647S"
EUR_EZ_STALE = "LFEMTTTTEZQ647S"
US_PAYEMS_ALT = "PAYEMS"

DEFAULT_PUB_LAG_MONTHS = 3  # a priori: labour slower than BCI/money (2)


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


def load_employment_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """OECD MEI employment persons → monthly PIT panel.

    Columns = ISO currency codes. Values = employment persons **levels**.
    Index = UTC month-start of the PIT *known* date after ``pub_lag_months``.
    Quarterly series are pub-lagged first, then forward-filled to month-start.
    Strategy modules convert to YoY % for XS ranks (foreign only).
    """
    curs = [c.upper() for c in (currencies or EMPLOYMENT_SERIES.keys())]
    monthly_cols: dict[str, pd.Series] = {}
    quarterly_cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}
    freq_map: dict[str, str] = {}

    for c in curs:
        sid = EMPLOYMENT_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_lfemtttt"
            continue
        freq = EMPLOYMENT_FREQ.get(c, "M")
        try:
            s = _load_series(sid, freq=freq, download=download, force=force)
            series_map[c] = sid
            freq_map[c] = freq
            if c == "EUR":
                notes[c] = (
                    f"Germany proxy via {sid} (Q) "
                    f"(LFEMTTTTEZQ647S EZ ends 2022-10; FR/IT/ES Q647S alts)"
                )
            elif freq == "Q":
                notes[c] = f"OECD MEI employment persons {sid} (quarterly)"
            else:
                notes[c] = f"OECD MEI employment persons {sid} (monthly)"
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
        # Align to common monthly grid via outer join already done; drop all-NaN rows
        df = df.dropna(how="all")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

    df.attrs["factor"] = "oecd_mei_employment_lfemtttt"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["freq_map"] = freq_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI LFEMTTTT* employment persons "
        "(USD/JPY/CAD/AUD=*M647S monthly; EUR=DE Q647S Germany proxy — "
        "EZ Q647S stale 2022-10; GBP/NZD/CHF=*Q647S; "
        "M657S/Q657S YoY stale/404 — not primary; PAYEMS/CE16OV US alts; "
        "LREM64TT rate not primary)"
    )
    df.attrs["unit"] = "employment_persons_level"
    df.attrs["score_basis"] = "yoy_pct_of_levels"
    df.attrs["unmapped"] = [c for c in curs if c not in series_map]
    df.attrs["failed_or_alt"] = EMPLOYMENT_FAILED_OR_ALT
    return df


def load_us_employment(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US OECD MEI employment persons (LFEMTTTTUSM647S), PIT-lagged."""
    panel = load_employment_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US employment series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_EMPLOYMENT"
    out.attrs["series_id"] = EMPLOYMENT_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def employment_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_employment_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_fred_lfemtttt"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
