"""FRED / BIS residential property-price panels for housing-wealth FX (free, PIT).

Literature
----------
- Housing wealth / collateral channel → FX (Aoki–Proudman–Vlieghe; housing–credit
  / collateral literature; BIS residential property prices): currencies with
  *high* relative house-price momentum benefit from wealth / collateral effects
  → subsequent appreciation prior.
- Honesty alternate: long low HPI momentum (housing-stress / debtor premium).
- Distinct from: BIS private credit/GDP (§32), debt/GDP (§29), REER (§31),
  money-growth (§33), reserves (§34), CA/TB, equity-diff, funding-liq, IG OAS.

Free data (no Bloomberg / OECD portal key)
-----------------------------------------
BIS **Real Residential Property Prices** via FRED ``Q*R628BIS`` (index):

| Ccy | FRED id (primary) | Notes                                              |
|-----|-------------------|----------------------------------------------------|
| USD | QUSR628BIS        | through ~2026-Q1                                   |
| EUR | QDER628BIS        | Germany proxy (``QEUR628BIS`` **404**)             |
| GBP | QGBR628BIS        | through ~2026-Q1                                   |
| JPY | QJPR628BIS        | through ~2025-Q4                                   |
| CAD | QCAR628BIS        | through ~2026-Q1                                   |
| AUD | QAUR628BIS        | through ~2026-Q1                                   |
| NZD | QNZR628BIS        | through ~2026-Q1                                   |
| CHF | QCHR628BIS        | through ~2026-Q1                                   |

Tried / documented:
- ``QEUR628BIS`` — euro-area aggregate **404** on free FRED → Germany ``QDER628BIS``.
- ``QGER628BIS`` — 404 (Germany is ``QDER628BIS``).
- ``QFRR628BIS`` / ``QITR628BIS`` / ``QESR628BIS`` — live FR/IT/ES alts
  (coverage / report only; primary EUR = Germany).
- ``QNZL628BIS`` / ``QCHER628BIS`` / ``QCHE628BIS`` — 404; NZD/CHF use
  ``QNZR628BIS`` / ``QCHR628BIS``.

Unit choice (fixed a priori)
----------------------------
Levels are country-specific index bases. Strategy scores use **YoY log-diff**
``log(P_t) − log(P_{t−12})`` (≈ YoY % change) for XS ranks — not raw levels.
Acceleration = Δ12 of that YoY growth. Documented in panel attrs.

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Quarterly BIS residential property prices typically land ~1–2 quarters after
reference. Default ``pub_lag_months=4``: observation dated quarter-start ``t``
is first known at ``t + 4 months``. Strategy modules add ``signal_lag`` months
on top. Do **not** retune on holdout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED BIS real residential property price index (quarterly)
HPI_SERIES: dict[str, str | None] = {
    "USD": "QUSR628BIS",
    "EUR": "QDER628BIS",  # Germany proxy (QEUR628BIS 404)
    "GBP": "QGBR628BIS",
    "JPY": "QJPR628BIS",
    "CAD": "QCAR628BIS",
    "AUD": "QAUR628BIS",
    "NZD": "QNZR628BIS",
    "CHF": "QCHR628BIS",
}

# Documented failed / alt / secondary mnemonics
HPI_FAILED_OR_ALT: dict[str, list[str]] = {
    "eur_aggregate_404": ["QEUR628BIS"],
    "germany_alt_404": ["QGER628BIS"],
    "eur_secondary_fr_it_es": ["QFRR628BIS", "QITR628BIS", "QESR628BIS"],
    "nzd_chf_alt_404": ["QNZL628BIS", "QCHER628BIS", "QCHE628BIS", "QSWIR628BIS"],
    "china_coverage_only": ["QCNR628BIS"],
}

EUR_SECONDARY = "QFRR628BIS"

DEFAULT_PUB_LAG_MONTHS = 4  # conservative quarterly BIS property-price lag


def _to_quarter_start(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = pd.DatetimeIndex(s.index)
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    qs = pd.DatetimeIndex(naive.to_period("Q").to_timestamp(how="start")).tz_localize("UTC")
    s = s.copy()
    s.index = qs
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _load_quarterly(series_id: str, *, download: bool, force: bool) -> pd.Series:
    if download:
        download_fred_series(series_id, force=force)
    return _to_quarter_start(load_fred_series(series_id, download=False))


def _expand_quarterly_to_monthly(
    df: pd.DataFrame,
    *,
    pub_lag_months: int,
) -> pd.DataFrame:
    """Shift quarterly index by pub_lag, then ffill to month-start."""
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


def load_hpi_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Quarterly BIS real residential HPI → monthly PIT panel (index levels).

    Columns = ISO currency codes. Values = real residential property price
    index (country base). Index = UTC month-start of the PIT *known* date after
    ``pub_lag_months``. Strategy modules convert to YoY log-diff for XS ranks.
    """
    curs = [c.upper() for c in (currencies or HPI_SERIES.keys())]
    cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}

    for c in curs:
        sid = HPI_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_primary_q_r628bis"
            continue
        try:
            s = _load_quarterly(sid, download=download, force=force)
            series_map[c] = sid
            if c == "EUR":
                notes[c] = (
                    f"Germany proxy via {sid} "
                    "(QEUR628BIS 404 on free FRED — not primary)"
                )
            else:
                notes[c] = f"BIS real residential property prices {sid}"
            cols[c] = s
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    df = pd.DataFrame(cols).sort_index()
    df = _expand_quarterly_to_monthly(df, pub_lag_months=pub_lag_months)

    df.attrs["factor"] = "bis_real_residential_hpi"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED BIS Q*R628BIS real residential property price index (quarterly); "
        "EUR=Germany QDER628BIS (QEUR628BIS 404); ffill to month-start after "
        f"pub_lag_months={pub_lag_months}; scores use YoY log-diff not levels"
    )
    df.attrs["unit"] = "index_level_ffill_monthly"
    df.attrs["score_unit"] = "yoy_log_diff"
    df.attrs["unmapped"] = [c for c in curs if c not in cols]
    df.attrs["failed_or_alt"] = HPI_FAILED_OR_ALT
    return df


def load_us_hpi(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US BIS real residential HPI (QUSR628BIS), PIT-lagged monthly."""
    panel = load_hpi_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US HPI series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_HPI"
    out.attrs["series_id"] = HPI_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    out.attrs["unit"] = panel.attrs.get("unit")
    return out


def hpi_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_hpi_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_primary_q_r628bis"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
