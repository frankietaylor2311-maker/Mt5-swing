"""FRED / IMF WEO government debt/GDP panels for fiscal-sustainability FX (free, PIT).

Literature
----------
- Debt overhang / fiscal sustainability: higher public debt/GDP associated with
  weaker credibility and eventual currency adjustment (Reinhart–Rogoff lineage;
  twin-deficits / external-adjustment surveys).
- Della Corte–Riddiough–Sarno-style debtor premium honesty alternate: long
  high-debt / short low-debt as a risk-premium sort.

Free data (no Bloomberg / OECD portal key)
-----------------------------------------
IMF World Economic Outlook **general government gross debt** (% of GDP) via
FRED. Preferred mnemonic in the brief was ``GGXWDG*188N`` (IMF code GGXWDG_NGDP);
those IDs **404 on FRED** as of 2026-09. Live FRED actuals use ``GGGDTA*188N``
(same WEO release / country codes as ``GGNLBA*`` fiscal balance):

| Ccy | FRED id (actual) | Notes                                      |
|-----|------------------|--------------------------------------------|
| USD | GGGDTAUSA188N    | primary US state variable                  |
| EUR | GGGDTADEA188N    | Germany proxy (parallel to fiscal)         |
| GBP | GGGDTAGBA188N    | IMF WEO "GBA" code                         |
| JPY | GGGDTAJPA188N    | IMF WEO "JPA" code                         |
| CAD | GGGDTACNA188N    | IMF WEO "CNA" code                         |
| AUD | GGGDTAAUA188N    | IMF WEO "AUA" code                         |
| NZD | —                | no free GGGDTA* on FRED as of 2026-09      |
| CHF | —                | no free GGGDTA* on FRED as of 2026-09      |

Tried and failed (404): ``GGXWDGUSA188N`` / ``GGXWDGDEA188N`` / … and NZD/CHF
``GGGDTANZA188N`` / ``GGGDTACHE188N`` / ``GGGDTANZL188N``.

Higher-frequency US overlay (US tilt legs only):
- ``GFDEGDQ188S`` — Federal Debt: Total Public Debt as % of GDP (quarterly).

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
IMF WEO annual releases (April / October). Year-``Y`` actuals are first usable
in the Spring WEO of ``Y+1`` (~April). Default ``pub_lag_months=15``: observation
dated year-start ``Y`` is first known at ``Y + 15 months``. Strategy modules add
``signal_lag`` months on top. Quarterly US debt uses ``pub_lag_months=4``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED IMF WEO GGGDTA*188N (annual % GDP; gross debt)
# Note: brief suggested GGXWDG*188N — those 404; GGGDTA* is the live FRED mnemonic.
DEBT_GDP_SERIES: dict[str, str | None] = {
    "USD": "GGGDTAUSA188N",
    "EUR": "GGGDTADEA188N",  # Germany proxy for EUR
    "GBP": "GGGDTAGBA188N",
    "JPY": "GGGDTAJPA188N",
    "CAD": "GGGDTACNA188N",
    "AUD": "GGGDTAAUA188N",
    "NZD": None,  # unavailable on free FRED WEO panel
    "CHF": None,  # unavailable on free FRED WEO panel
}

# Documented 404 aliases (brief mnemonic / ISO3 attempts)
DEBT_GDP_FAILED_ALIASES: dict[str, list[str]] = {
    "GGXWDG_prefix": [
        "GGXWDGUSA188N",
        "GGXWDGDEA188N",
        "GGXWDGGBA188N",
        "GGXWDGJPA188N",
        "GGXWDGCNA188N",
        "GGXWDGAUA188N",
    ],
    "NZD_CHF": ["GGGDTANZA188N", "GGGDTANZL188N", "GGGDTACHE188N", "GGGDTACH188N"],
}

# Optional secondary EUR reference (France) — coverage / report only
EUR_SECONDARY = "GGGDTAFRA188N"

US_FEDERAL_DEBT_PCT_GDP_Q = "GFDEGDQ188S"  # quarterly
US_FEDERAL_DEBT_PCT_GDP_A = "GFDGDPA188S"  # annual (backup)

DEFAULT_PUB_LAG_MONTHS = 15  # ~April of Y+1 for year-Y WEO actual
DEFAULT_US_Q_PUB_LAG_MONTHS = 4  # ~1Q + buffer for quarterly federal debt/GDP


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


def _to_quarter_start(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = pd.DatetimeIndex(s.index)
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    qs = pd.DatetimeIndex(naive.to_period("Q").to_timestamp(how="start")).tz_localize("UTC")
    s = s.copy()
    s.index = qs
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _load_annual(series_id: str, *, download: bool, force: bool) -> pd.Series:
    if download:
        download_fred_series(series_id, force=force)
    return _to_year_start(load_fred_series(series_id, download=False))


def _load_quarterly(series_id: str, *, download: bool, force: bool) -> pd.Series:
    if download:
        download_fred_series(series_id, force=force)
    return _to_quarter_start(load_fred_series(series_id, download=False))


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


def _expand_quarterly_to_monthly(
    s: pd.Series,
    *,
    pub_lag_months: int,
) -> pd.Series:
    """Shift quarterly index by pub_lag, then ffill to month-start."""
    if s.empty:
        return s
    out = s.copy()
    if pub_lag_months > 0:
        out.index = out.index + pd.DateOffset(months=int(pub_lag_months))
        out = out[~out.index.duplicated(keep="last")].sort_index()
    df = pd.DataFrame({"v": out})
    expanded = _expand_annual_to_monthly(df, pub_lag_months=0)
    return expanded["v"]


def load_debt_gdp_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Annual IMF WEO gross debt / GDP (%) → monthly PIT panel.

    Columns = ISO currency codes. Values are percent of GDP (gross debt).
    Index = UTC month-start of the PIT *known* date after ``pub_lag_months``.
    """
    curs = [c.upper() for c in (currencies or DEBT_GDP_SERIES.keys())]
    cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}

    for c in curs:
        sid = DEBT_GDP_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_gggdta"
            continue
        try:
            s = _load_annual(sid, download=download, force=force)
            series_map[c] = sid
            if c == "EUR":
                notes[c] = f"Germany proxy via {sid}"
            cols[c] = s
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    df = pd.DataFrame(cols).sort_index()
    df = _expand_annual_to_monthly(df, pub_lag_months=pub_lag_months)

    df.attrs["factor"] = "debt_gdp_pct"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["notes"] = notes
    df.attrs["source"] = "FRED IMF WEO GGGDTA*188N (GGXWDG*188N 404 on FRED)"
    df.attrs["unit"] = "percent_of_gdp"
    df.attrs["unmapped"] = [c for c in curs if c not in cols]
    df.attrs["failed_aliases"] = DEBT_GDP_FAILED_ALIASES
    return df


def load_us_debt_gdp(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US gross debt / GDP (%) with PIT publication lag (IMF WEO GGGDTAUSA188N)."""
    panel = load_debt_gdp_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US debt/GDP series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_DEBT_GDP"
    out.attrs["series_id"] = DEBT_GDP_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def load_us_federal_debt_gdp_q(
    *,
    pub_lag_months: int = DEFAULT_US_Q_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US federal debt / GDP (%) quarterly (GFDEGDQ188S), PIT-lagged to monthly.

    Higher-frequency US-only overlay for twin / haven tilts.
    """
    s = _load_quarterly(US_FEDERAL_DEBT_PCT_GDP_Q, download=download, force=force)
    out = _expand_quarterly_to_monthly(s, pub_lag_months=pub_lag_months)
    out = out.dropna()
    out.name = "US_FEDERAL_DEBT_GDP_Q"
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    out.attrs["series_id"] = US_FEDERAL_DEBT_PCT_GDP_Q
    out.attrs["unit"] = "percent_of_gdp"
    return out


def debt_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_debt_gdp_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_fred_gggdta"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
