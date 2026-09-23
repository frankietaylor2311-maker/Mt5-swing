"""FRED / BIS real broad effective exchange-rate (REER) panels for FX (free, PIT).

Literature
----------
- Rogoff (1996), "The Purchasing Power Parity Puzzle," *JEL* — real FX mean
  reversion is slow but present (PPP puzzle).
- Taylor / REER misalignment literature: multilateral real effective rates as
  the official gauge of real undervaluation / overvaluation vs bilateral CPI DIY.
- BIS real broad EER via FRED ``RB*BIS`` — official bank-for-international-
  settlements multilateral REER (CPI-based, broad trade weights). Distinct from
  homemade bilateral PPP real-FX in ``ppp_real_fx.py`` / wave §PPP.

Free data (no Bloomberg / BIS portal key)
-----------------------------------------
BIS real broad effective exchange rates via FRED (monthly, index):

| Ccy | FRED id   | Notes                                      |
|-----|-----------|--------------------------------------------|
| USD | RBUSBIS   | primary US state variable                  |
| EUR | RBXMBIS   | Euro-area broad REER (continuous history)  |
| GBP | RBGBBIS   |                                            |
| JPY | RBJPBIS   |                                            |
| CAD | RBCABIS   |                                            |
| AUD | RBAUBIS   |                                            |
| NZD | RBNZBIS   | mapped (unlike debt/fiscal)                |
| CHF | RBCHBIS   | mapped (unlike debt/fiscal)                |

Tried / documented:
- ``RBDEBIS`` (Germany) — live continuous; unused as primary EUR (prefer
  official euro-area ``RBXMBIS`` for multilateral EUR, parallel continuity
  choice to trade-balance Germany proxy when EZ ends).
- ``RBEZBIS`` / ``RBEMUBIS`` — **404** on FRED as of 2026-09.

Higher = real appreciation (less competitive / more overvalued in CPI-REER
terms). Primary strategy prior: long **low** REER z (undervalued) /
short high REER z (overvalued).

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Monthly BIS REER on FRED is typically lagged ~1–2 months after reference
month. Default ``pub_lag_months=2``: observation dated month-start ``t`` is
first known at ``t + 2 months``. Strategy modules add ``signal_lag`` months
on top. (As of 2026-09-23, latest point was 2026-07 — consistent with ~2m lag.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED BIS real broad EER id
BIS_REER_SERIES: dict[str, str | None] = {
    "USD": "RBUSBIS",
    "EUR": "RBXMBIS",  # Euro-area broad (not Germany)
    "GBP": "RBGBBIS",
    "JPY": "RBJPBIS",
    "CAD": "RBCABIS",
    "AUD": "RBAUBIS",
    "NZD": "RBNZBIS",
    "CHF": "RBCHBIS",
}

# Documented alternate / failed mnemonics
BIS_REER_FAILED_OR_ALT: dict[str, list[str]] = {
    "eur_germany_alt": ["RBDEBIS"],
    "eur_ez_404": ["RBEZBIS", "RBEMUBIS"],
}

EUR_GERMANY_ALT = "RBDEBIS"

DEFAULT_PUB_LAG_MONTHS = 2  # conservative monthly BIS REER release lag


def _to_month_start(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = pd.DatetimeIndex(s.index)
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    ms = pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start")).tz_localize("UTC")
    s = s.copy()
    s.index = ms
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _load_monthly(series_id: str, *, download: bool, force: bool) -> pd.Series:
    if download:
        download_fred_series(series_id, force=force)
    return _to_month_start(load_fred_series(series_id, download=False))


def _apply_pub_lag(df: pd.DataFrame, *, pub_lag_months: int) -> pd.DataFrame:
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


def load_bis_reer_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Monthly BIS real broad REER panel, publication-lagged.

    Columns = ISO currency codes. Values = BIS real broad EER index
    (higher = real appreciation / less competitive).
    Index = UTC month-start of the PIT *known* date after ``pub_lag_months``.
    """
    curs = [c.upper() for c in (currencies or BIS_REER_SERIES.keys())]
    cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}

    for c in curs:
        sid = BIS_REER_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_rb_bis"
            continue
        try:
            s = _load_monthly(sid, download=download, force=force)
            series_map[c] = sid
            if c == "EUR":
                notes[c] = (
                    f"Euro-area broad via {sid}; "
                    f"Germany alt {EUR_GERMANY_ALT} documented (unused as primary)"
                )
            cols[c] = s
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    df = pd.DataFrame(cols).sort_index()
    df = _apply_pub_lag(df, pub_lag_months=pub_lag_months)

    df.attrs["factor"] = "bis_real_broad_reer"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["notes"] = notes
    df.attrs["source"] = "FRED BIS real broad effective exchange rate RB*BIS"
    df.attrs["unit"] = "index_reer"
    df.attrs["unmapped"] = [c for c in curs if c not in cols]
    df.attrs["failed_or_alt"] = BIS_REER_FAILED_OR_ALT
    return df


def load_us_bis_reer(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US BIS real broad REER (RBUSBIS), PIT-lagged."""
    panel = load_bis_reer_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US BIS REER series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_BIS_REER"
    out.attrs["series_id"] = BIS_REER_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def reer_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_bis_reer_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_fred_rb_bis"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
