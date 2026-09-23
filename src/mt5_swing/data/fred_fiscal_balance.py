"""FRED / IMF WEO fiscal-balance panels for twin-deficits / fiscal-FX (free, PIT).

Literature
----------
- Twin deficits: fiscal deficit co-moves with current-account deficit and is
  associated with eventual currency adjustment (Abell 1990; related surveys).
- Fiscal sustainability / credibility: relatively stronger fiscal balances
  (surplus / less deficit) support real appreciation vs peers.
- Related: Corsetti–Dedola–Leduc on fiscal shocks and exchange rates; Dai &
  Philippon on fiscal deficits and risk premia.

Free data (no Bloomberg / OECD portal key)
-----------------------------------------
IMF World Economic Outlook **general government net lending/borrowing**
(% of GDP; surplus > 0) via FRED ``GGNLBA*188N``:

| Ccy | FRED id          | Notes                                      |
|-----|------------------|--------------------------------------------|
| USD | GGNLBAUSA188N    | primary US state variable                  |
| EUR | GGNLBADEA188N    | Germany proxy (EA19/EZA ends ~2010 on FRED)|
| GBP | GGNLBAGBA188N    | IMF WEO "GBA" code                         |
| JPY | GGNLBAJPA188N    | IMF WEO "JPA" code                         |
| CAD | GGNLBACNA188N    | IMF WEO "CNA" code                         |
| AUD | GGNLBAAUA188N    | IMF WEO "AUA" code                         |
| NZD | —                | no free GGNLBA* on FRED as of 2026-09      |
| CHF | —                | no free GGNLBA* on FRED as of 2026-09      |

Higher-frequency US overlays (US tilt legs only):
- ``FYFSGDA188S`` — Federal Surplus or Deficit [-] as % of GDP (annual).
- ``MTSDS133FMS`` — Monthly Treasury Statement federal surplus/deficit ($ mn).

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
IMF WEO annual releases (April / October). Year-``Y`` actuals are first usable
in the Spring WEO of ``Y+1`` (~April). Default ``pub_lag_months=15``: observation
dated year-start ``Y`` is first known at ``Y + 15 months``. Strategy modules add
``signal_lag`` months on top. Monthly MTS uses ``pub_lag_months=1``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED IMF WEO GGNLBA*188N (annual % GDP; surplus > 0)
FISCAL_GDP_SERIES: dict[str, str | None] = {
    "USD": "GGNLBAUSA188N",
    "EUR": "GGNLBADEA188N",  # Germany proxy for EUR
    "GBP": "GGNLBAGBA188N",
    "JPY": "GGNLBAJPA188N",
    "CAD": "GGNLBACNA188N",
    "AUD": "GGNLBAAUA188N",
    "NZD": None,  # unavailable on free FRED WEO panel
    "CHF": None,  # unavailable on free FRED WEO panel
}

# Optional secondary EUR reference (France) — coverage / report only
EUR_SECONDARY = "GGNLBAFRA188N"

US_FEDERAL_SURPLUS_PCT_GDP = "FYFSGDA188S"  # annual
US_MTS_SURPLUS_MN = "MTSDS133FMS"  # monthly, $ millions (surplus > 0 historically rare)

DEFAULT_PUB_LAG_MONTHS = 15  # ~April of Y+1 for year-Y WEO actual
DEFAULT_MTS_PUB_LAG_MONTHS = 1


def _to_year_start(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = pd.DatetimeIndex(s.index)
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    ys = pd.DatetimeIndex(naive.to_period("Y").to_timestamp(how="start")).tz_localize("UTC")
    s = s.copy()
    s.index = ys
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _to_month_start(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = pd.DatetimeIndex(s.index)
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    ms = pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start")).tz_localize("UTC")
    s = s.copy()
    s.index = ms
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _load_annual(series_id: str, *, download: bool, force: bool) -> pd.Series:
    if download:
        download_fred_series(series_id, force=force)
    return _to_year_start(load_fred_series(series_id, download=False))


def _load_monthly(series_id: str, *, download: bool, force: bool) -> pd.Series:
    if download:
        download_fred_series(series_id, force=force)
    return _to_month_start(load_fred_series(series_id, download=False))


def _expand_annual_to_monthly(
    df: pd.DataFrame,
    *,
    pub_lag_months: int,
) -> pd.DataFrame:
    """Shift annual year-start index by pub_lag, then ffill to month-start."""
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


def load_fiscal_gdp_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Annual IMF WEO fiscal balance / GDP (%) → monthly PIT panel.

    Columns = ISO currency codes. Values are percent of GDP (surplus > 0).
    Index = UTC month-start of the PIT *known* date after ``pub_lag_months``.
    """
    curs = [c.upper() for c in (currencies or FISCAL_GDP_SERIES.keys())]
    cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}

    for c in curs:
        sid = FISCAL_GDP_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_ggnlba"
            continue
        try:
            s = _load_annual(sid, download=download, force=force)
            series_map[c] = sid
            if c == "EUR":
                notes[c] = f"Germany proxy via {sid} (EA19/EZA discontinued ~2010)"
            cols[c] = s
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    df = pd.DataFrame(cols).sort_index()
    df = _expand_annual_to_monthly(df, pub_lag_months=pub_lag_months)

    df.attrs["factor"] = "fiscal_gdp_pct"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["notes"] = notes
    df.attrs["source"] = "FRED IMF WEO GGNLBA*188N"
    df.attrs["unit"] = "percent_of_gdp"
    df.attrs["unmapped"] = [c for c in curs if c not in cols]
    return df


def load_us_fiscal_gdp(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
    prefer_fyfsgda: bool = False,
) -> pd.Series:
    """US fiscal balance / GDP (%) with PIT publication lag.

    Default: IMF WEO ``GGNLBAUSA188N``. Optional ``FYFSGDA188S`` (BEA federal
    surplus/deficit % GDP) when ``prefer_fyfsgda=True``.
    """
    if prefer_fyfsgda:
        s = _load_annual(US_FEDERAL_SURPLUS_PCT_GDP, download=download, force=force)
        df = pd.DataFrame({"USD": s})
        df = _expand_annual_to_monthly(df, pub_lag_months=pub_lag_months)
        out = df["USD"].dropna()
        out.name = "US_FISCAL_GDP"
        out.attrs["series_id"] = US_FEDERAL_SURPLUS_PCT_GDP
    else:
        panel = load_fiscal_gdp_panel(
            ["USD"],
            pub_lag_months=pub_lag_months,
            download=download,
            force=force,
        )
        if "USD" not in panel.columns:
            raise RuntimeError("US fiscal/GDP series failed to load")
        out = panel["USD"].dropna()
        out.name = "US_FISCAL_GDP"
        out.attrs["series_id"] = FISCAL_GDP_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def load_us_mts_surplus(
    *,
    pub_lag_months: int = DEFAULT_MTS_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """Monthly Treasury Statement federal surplus/deficit ($ millions), PIT-lagged.

    Positive = surplus. Used for higher-frequency US fiscal *change* tilts only;
    units are absolute $, so strategies should use trailing z / sign, not levels
    cross-sectionally vs peers.
    """
    s = _load_monthly(US_MTS_SURPLUS_MN, download=download, force=force)
    if pub_lag_months > 0:
        s.index = s.index + pd.DateOffset(months=int(pub_lag_months))
        s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = "US_MTS_SURPLUS_MN"
    s.attrs["pub_lag_months"] = int(pub_lag_months)
    s.attrs["series_id"] = US_MTS_SURPLUS_MN
    s.attrs["unit"] = "usd_millions"
    return s


def fiscal_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_fiscal_gdp_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_fred_ggnlba"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
