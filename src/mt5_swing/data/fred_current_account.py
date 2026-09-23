"""FRED / IMF BOP current-account panels for global-imbalances FX (free, PIT).

Literature
----------
- Della Corte, Riddiough & Sarno (2016), "Currency Premia and Global Imbalances,"
  *RFS* — external imbalances (NFA / CA) price currency risk premia; debtor
  currencies earn a premium on average.
- Gourinchas & Rey (2007), "International Financial Adjustment," *JPE* — US
  external imbalance / trade channel forecasts dollar adjustment.

Free data (no Bloomberg / OECD portal key)
-----------------------------------------
IMF Balance of Payments **current account / GDP (%)**, seasonally adjusted
quarterly, via FRED series ``{ISO3}B6BLTT02STSAQ``:

| Ccy | FRED id              | Notes                          |
|-----|----------------------|--------------------------------|
| USD | USAB6BLTT02STSAQ     | primary US state variable      |
| EUR | EA19B6BLTT02STSAQ    | ends ~2022-10; DEU gap-fill    |
| GBP | GBRB6BLTT02STSAQ     |                                |
| JPY | JPNB6BLTT02STSAQ     |                                |
| CAD | CANB6BLTT02STSAQ     |                                |
| AUD | AUSB6BLTT02STSAQ     |                                |
| NZD | NZLB6BLTT02STSAQ     |                                |
| CHF | CHEB6BLTT02STSAQ     |                                |

Legacy OECD MEI ``BPBLTT01*Q188S`` ends ~2013–14 — **not** used as primary.

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Quarterly BOP CA is typically released ~1 quarter after quarter-end with
revisions. Default ``pub_lag_quarters=2`` (6 months): observation dated at
quarter-start ``t`` is first usable at ``t + 6 months``. Strategy modules add
``signal_lag`` months on top.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → (primary FRED id, optional gap-fill FRED id)
CA_GDP_SERIES: dict[str, tuple[str, str | None]] = {
    "USD": ("USAB6BLTT02STSAQ", None),
    "EUR": ("EA19B6BLTT02STSAQ", "DEUB6BLTT02STSAQ"),  # DEU proxy after EA19 ends
    "GBP": ("GBRB6BLTT02STSAQ", None),
    "JPY": ("JPNB6BLTT02STSAQ", None),
    "CAD": ("CANB6BLTT02STSAQ", None),
    "AUD": ("AUSB6BLTT02STSAQ", None),
    "NZD": ("NZLB6BLTT02STSAQ", None),
    "CHF": ("CHEB6BLTT02STSAQ", None),
}

DEFAULT_PUB_LAG_QUARTERS = 2  # conservative BOP release + revision buffer


def _to_quarter_start(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = pd.DatetimeIndex(s.index)
    # Drop tz for Period conversion, then re-attach UTC
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    qs = pd.DatetimeIndex(naive.to_period("Q").to_timestamp(how="start")).tz_localize("UTC")
    s = s.copy()
    s.index = qs
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _load_one(
    series_id: str,
    *,
    download: bool,
    force: bool,
) -> pd.Series:
    if download:
        download_fred_series(series_id, force=force)
    return _to_quarter_start(load_fred_series(series_id, download=False))


def load_ca_gdp_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_quarters: int = DEFAULT_PUB_LAG_QUARTERS,
    download: bool = True,
    force: bool = False,
    gap_fill_eur: bool = True,
) -> pd.DataFrame:
    """Quarterly current-account / GDP (%) panel, publication-lagged.

    Columns = ISO currency codes. Values are percent of GDP (surplus > 0).
    Index = UTC quarter-start of the *observation* period, then shifted forward
    by ``pub_lag_quarters`` so the index becomes the PIT *known* date.
    """
    curs = [c.upper() for c in (currencies or CA_GDP_SERIES.keys())]
    cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}

    for c in curs:
        meta = CA_GDP_SERIES.get(c)
        if not meta:
            notes[c] = "unmapped"
            continue
        primary, fill = meta
        try:
            s = _load_one(primary, download=download, force=force)
            series_map[c] = primary
            if c == "EUR" and gap_fill_eur and fill:
                try:
                    s_fill = _load_one(fill, download=download, force=force)
                    # Use DEU only where EA19 missing (tail after ~2022)
                    s = s.combine_first(s_fill)
                    notes[c] = f"EA19 primary; DEU gap-fill via {fill}"
                    if s.index.max() > pd.Timestamp("2022-10-01", tz="UTC"):
                        # document that post-EA19 tail is DEU
                        notes[c] += "; post-2022-10 values may be DEU proxy"
                except Exception as exc:  # noqa: BLE001
                    notes[c] = f"EA19 only; DEU fill failed:{exc}"[:120]
            elif c == "EUR":
                notes[c] = "EA19 only (ends ~2022-10 on FRED)"
            cols[c] = s
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    df = pd.DataFrame(cols).sort_index()
    # Upsample to monthly (ffill within quarter) so strategy signal_lag months
    # align cleanly with other scholarly monthly panels.
    if not df.empty:
        # After pub lag, expand QS → MS via ffill
        if pub_lag_quarters > 0:
            df.index = df.index + pd.DateOffset(months=3 * int(pub_lag_quarters))
            df = df[~df.index.duplicated(keep="last")].sort_index()
        # Reindex to month-start covering the panel span
        naive_min = df.index.min().tz_convert(None) if df.index.tz is not None else df.index.min()
        naive_max = df.index.max().tz_convert(None) if df.index.tz is not None else df.index.max()
        start = naive_min.to_period("M").to_timestamp(how="start")
        end = naive_max.to_period("M").to_timestamp(how="start")
        monthly_idx = pd.date_range(start, end, freq="MS", tz="UTC")
        df = df.reindex(monthly_idx.union(df.index)).sort_index()
        df = df.ffill()
        df = df.loc[monthly_idx]
        df.index = pd.DatetimeIndex(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

    df.attrs["factor"] = "ca_gdp_pct"
    df.attrs["pub_lag_quarters"] = int(pub_lag_quarters)
    df.attrs["series_map"] = series_map
    df.attrs["notes"] = notes
    df.attrs["source"] = "FRED IMF BOP {ISO3}B6BLTT02STSAQ"
    df.attrs["unit"] = "percent_of_gdp"
    return df


def load_us_ca_gdp(
    *,
    pub_lag_quarters: int = DEFAULT_PUB_LAG_QUARTERS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US CA/GDP (%) with PIT publication lag (Gourinchas–Rey state variable)."""
    panel = load_ca_gdp_panel(
        ["USD"],
        pub_lag_quarters=pub_lag_quarters,
        download=download,
        force=force,
        gap_fill_eur=False,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US CA/GDP series failed to load")
    s = panel["USD"].dropna()
    s.name = "US_CA_GDP"
    s.attrs["pub_lag_quarters"] = int(pub_lag_quarters)
    return s


def ca_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_ca_gdp_panel(download=False)
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
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
