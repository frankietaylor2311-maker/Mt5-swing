"""FRED / OECD MEI consumer-price (CPI / HICP) panels for CPI-YoY FX.

Literature
----------
- Dahlquist & Hasseltoft (2020, JFE) economic momentum: currencies with strong
  past macro trends (incl. CPI) subsequently appreciate.
- Primary prior: currencies with *high* relative CPI YoY appreciate vs
  low-CPI peers. Honesty alternate: long low CPI / purchasing-power /
  high-inflation short (PPP-style).
- Distinct from: PPI §51 (producer prices), IP §42 (volume), retail §46
  (consumer sales), employment §41, ULC §48 / LP §49 / CU §50, building-permits
  §43, CLI/CCI/BCI, macro_diff EW CPI/IP/UR *blend*, PPP real FX, CA/TB,
  fiscal/debt, BIS, reserves, house-price, money, equity, IG OAS, commodity,
  GPR family, ACM TP, WUI, EPU/TPU.

Free data (verified on FRED ~2026-09-24)
---------------------------------------
Reuse live series from ``fred_macro_diff.CPI_SERIES`` (index → YoY). Prefer
growth/YoY panel. **Coverage honesty:** USD/EUR live through mid-2026;
GBP/CAD ~2025-03, CHF ~2025-04, AUD/NZD Q ~2025-01 (mildly stale);
**JPY** OECD MEI ``JPNCPIALLMINMEI`` / ``CPALTT01JPM659N`` end ~**2021-06**
(hard STALE — post-2021 ranks ffilled frozen for JPY).

| Ccy | FRED id (primary)       | Freq | Unit  | Notes                                      |
|-----|-------------------------|------|-------|--------------------------------------------|
| USD | CPIAUCSL                | M    | level | live ~2026-08                              |
| EUR | CP0000EZ19M086NEST      | M    | level | EA19 HICP; live ~2026-08                   |
| GBP | GBRCPIALLMINMEI         | M    | level | ends ~2025-03                              |
| JPY | JPNCPIALLMINMEI         | M    | level | **STALE** ends ~2021-06                    |
| CAD | CANCPIALLMINMEI         | M    | level | ends ~2025-03                              |
| AUD | AUSCPIALLQINMEI         | Q    | level | quarterly; ends ~2025-01                   |
| NZD | NZLCPIALLQINMEI         | Q    | level | quarterly; ends ~2025-01                   |
| CHF | CHECPIALLMINMEI         | M    | level | ends ~2025-04                              |

Tried / documented:
- ``CPALTT01{ISO2}M659N`` YoY growth — same end dates as CPIALLMINMEI (JPY
  still ends 2021-06); index→YoY preferred for unit consistency with USD/EUR.
- ``CPHPTT01EZM659N`` EA HICP YoY — ends ~2023-01 (staler than NEST index).
- Japan ``CP0000JPM086NEST`` / ``CPIJPALTT01GYM`` / ``JPNCPIALLGYM`` — **404**.
- ``FPCPITOTLZGJPN`` World Bank annual inflation — annual only (not mixed).

Unit choice (fixed a priori)
----------------------------
Panel stores **CPI YoY %**. All primaries are index levels → YoY before pub
lag (monthly ``pct_change(12)*100``; quarterly ``pct_change(4)*100``).
Strategy scores XS-rank on these YoY values.

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
CPI / HICP typically lands ~1 month after reference (matches
``fred_macro_diff`` CPI default). Default ``pub_lag_months=1``. Observation
dated month-/quarter-start ``t`` is first known at ``t + 1 month``.
Quarterly series are pub-lagged then forward-filled to month-start (no
intra-quarter leak). Strategy modules add ``signal_lag=1`` month + a
**1 trading-day** weight lag (PPI sibling convention).

**Staleness honesty:** JPY free FRED OECD CPI ends mid-2021; GBP/CAD/CHF /
AUD/NZD end early-2025; USD/EUR live. Post-end ranks are ffilled frozen.
No go-live claim under ``approximate_non_ftmo``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED CPI / HICP (reuse fred_macro_diff.CPI_SERIES ids)
# NOTE: JPY ends ~2021-06 on free FRED (hard stale) — documented, still boarded.
CPI_SERIES: dict[str, str | None] = {
    "USD": "CPIAUCSL",
    "EUR": "CP0000EZ19M086NEST",  # EA19 HICP index
    "GBP": "GBRCPIALLMINMEI",
    "JPY": "JPNCPIALLMINMEI",  # STALE ~2021-06
    "CAD": "CANCPIALLMINMEI",
    "AUD": "AUSCPIALLQINMEI",  # quarterly index → YoY
    "NZD": "NZLCPIALLQINMEI",  # quarterly index → YoY
    "CHF": "CHECPIALLMINMEI",
}

# Native series unit: all primaries are index levels → YoY
CPI_UNIT: dict[str, str] = {
    "USD": "level",
    "EUR": "level",
    "GBP": "level",
    "JPY": "level",
    "CAD": "level",
    "AUD": "level",
    "NZD": "level",
    "CHF": "level",
}

CPI_FREQ: dict[str, str] = {
    "USD": "M",
    "EUR": "M",
    "GBP": "M",
    "JPY": "M",
    "CAD": "M",
    "AUD": "Q",
    "NZD": "Q",
    "CHF": "M",
}

CPI_FAILED_OR_ALT: dict[str, list[str]] = {
    "cpaltt01_yoy_same_ends_not_mixed": [
        "CPALTT01USM659N",
        "CPALTT01GBM659N",
        "CPALTT01CAM659N",
        "CPALTT01CHM659N",
        "CPALTT01JPM659N",
        "CPALTT01DEM659N",
    ],
    "eur_hicp_yoy_staler_than_nest": ["CPHPTT01EZM659N"],
    "jpy_oecd_stale_2021_06": [
        "JPNCPIALLMINMEI",
        "CPALTT01JPM659N",
        "JPNCPIALLQINMEI",
        "CPALTT01JPQ659N",
        "JPNCPICORMINMEI",
    ],
    "jpy_alts_404_or_annual_not_mixed": [
        "CP0000JPM086NEST",
        "CPIJPALTT01GYM",
        "JPNCPIALLGYM",
        "FPCPITOTLZGJPN",
    ],
    "aud_nzd_quarterly_primary": ["AUSCPIALLQINMEI", "NZLCPIALLQINMEI"],
    "macro_diff_cpi_series_reused": [
        "CPIAUCSL",
        "CP0000EZ19M086NEST",
        "GBRCPIALLMINMEI",
        "JPNCPIALLMINMEI",
        "CANCPIALLMINMEI",
        "CHECPIALLMINMEI",
        "AUSCPIALLQINMEI",
        "NZLCPIALLQINMEI",
    ],
}

DEFAULT_PUB_LAG_MONTHS = 1  # a priori: CPI/HICP ~1m (matches macro_diff)


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


def load_cpi_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """OECD MEI / national CPI YoY → monthly PIT panel.

    Columns = ISO currency codes. Values = CPI **YoY %**. Index = UTC
    month-start of the PIT *known* date after ``pub_lag_months``. Index
    series convert to YoY before the pub lag. Quarterly YoY series are
    pub-lagged first, then forward-filled to month-start. Strategy modules
    XS-rank on these values (foreign only).
    """
    curs = [c.upper() for c in (currencies or CPI_SERIES.keys())]
    monthly_cols: dict[str, pd.Series] = {}
    quarterly_cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}
    freq_map: dict[str, str] = {}
    unit_map: dict[str, str] = {}

    for c in curs:
        sid = CPI_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_cpi"
            continue
        freq = CPI_FREQ.get(c, "M")
        unit = CPI_UNIT.get(c, "level")
        try:
            s = _load_series(sid, freq=freq, download=download, force=force)
            if unit == "level":
                if freq == "Q":
                    s = (s.pct_change(4) * 100.0).dropna()
                else:
                    s = (s.pct_change(12) * 100.0).dropna()
                if c == "JPY":
                    notes[c] = (
                        f"OECD MEI {sid} levels → YoY % "
                        f"(STALE ends ~2021-06; CPALTT01JPM659N same end; "
                        f"CP0000JPM086NEST 404)"
                    )
                elif c in ("AUD", "NZD"):
                    notes[c] = (
                        f"OECD MEI {sid} quarterly index → YoY % "
                        f"(ends ~2025-01)"
                    )
                elif c == "EUR":
                    notes[c] = (
                        f"EA19 HICP {sid} levels → YoY % (live; "
                        f"CPHPTT01EZM659N YoY staler ~2023-01 — not mixed)"
                    )
                elif c == "USD":
                    notes[c] = f"US CPI {sid} levels → YoY % (live ~2026-08)"
                else:
                    notes[c] = (
                        f"OECD MEI {sid} levels → YoY % "
                        f"(ends ~2025-03..04 mild stale)"
                    )
            else:
                notes[c] = f"{sid} as reported"
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

    df.attrs["factor"] = "oecd_mei_cpi_inflation_yoy"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["freq_map"] = freq_map
    df.attrs["unit_map"] = unit_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED CPI/HICP index→YoY (CPIAUCSL; CP0000EZ19M086NEST EA19 HICP; "
        "GBP/JPY/CAD/CHF *CPIALLMINMEI; AUD/NZD *CPIALLQINMEI Q→YoY; "
        "reuses fred_macro_diff.CPI_SERIES; JPY STALE ~2021-06; "
        "GBP/CAD/CHF/AUD/NZD mild stale ~2025-01..04; USD/EUR live)"
    )
    df.attrs["unit"] = "cpi_yoy_pct"
    df.attrs["score_basis"] = "yoy_growth_derived_from_index"
    df.attrs["unmapped"] = [c for c in curs if c not in series_map]
    df.attrs["failed_or_alt"] = CPI_FAILED_OR_ALT
    # Panel-level stale flag: True because JPY hard-stale + several mild
    df.attrs["stale"] = True
    df.attrs["stale_note"] = (
        "JPY OECD MEI CPI on free FRED ends ~2021-06 (hard); "
        "GBP/CAD ~2025-03, CHF ~2025-04, AUD/NZD Q ~2025-01 (mild); "
        "USD/EUR live through mid-2026; post-end ranks ffilled frozen"
    )
    return df


def load_us_cpi(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US CPI YoY (CPIAUCSL→YoY), PIT-lagged."""
    panel = load_cpi_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US CPI series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_CPI"
    out.attrs["series_id"] = CPI_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def cpi_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_cpi_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_fred_cpi"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
