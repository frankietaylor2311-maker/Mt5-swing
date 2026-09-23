"""FRED World Uncertainty Index (WUI) country panels for uncertainty-differential FX.

Literature
----------
- Ahir, Bloom & Furceri World Uncertainty Index (quarterly, country-level from
  Economist Intelligence Unit text).
- Claim: currencies with *low* relative WUI (calmer policy/macro uncertainty)
  subsequently appreciate vs high-WUI peers; elevated global/US WUI → USD haven
  / risk-off (honesty alternate: stress / USD soft).
- Distinct from: Baker–Bloom–Davis EPU/TPU, Caldara–Iacoviello GPR / AI-GPR,
  OECD CLI/CCI/BCI, IG OAS, macro-diff CPI/IP/UR, CA/TB, fiscal/debt, BIS
  REER/credit, reserves, house-price, money-growth, equity-diff, crash-skew,
  dollar-beta.

Free data (verified live on FRED ~2026-09-23)
--------------------------------------------
Country WUI quarterly series through ~2026-04:

| Ccy | FRED id | Notes                                              |
|-----|---------|----------------------------------------------------|
| USD | WUIUSA  | live through ~2026-04                              |
| EUR | WUIDEU  | Germany proxy (no clean EA aggregate on FRED)      |
| GBP | WUIGBR  | live                                               |
| JPY | WUIJPN  | live                                               |
| CAD | WUICAN  | live                                               |
| AUD | WUIAUS  | live                                               |
| NZD | WUINZL  | live                                               |
| CHF | WUICHE  | live                                               |

Tried / documented:
- ``WUIFRA`` / ``WUIITA`` / ``WUIESP`` — France/Italy/Spain coverage alts for EUR
  (Germany ``WUIDEU`` is the a-priori EA proxy; no clean euro-area aggregate).
- Global / aggregate WUI variants — not used as primary foreign XS scores.

Unit choice (fixed a priori)
----------------------------
WUI is already an *index* (not a balance / rate that needs YoY-first). Strategy
scores use **levels** for XS ranks, plus trailing z of levels and Δ12m (≈Δ4q)
of levels. Documented in panel attrs — do **not** YoY-first by default.

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Quarterly EIU/WUI typically lands with a multi-month lag. Default
``pub_lag_months=4``: observation dated at quarter-start ``t`` is first known at
``t + 4 months``. After the lag, expand quarterly → monthly via forward-fill
(within the known tree). Strategy modules add a **1 trading-day** weight lag
(``signal_lag_days=1``); no extra month signal lag (a priori for this wave).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED country WUI (quarterly)
WUI_SERIES: dict[str, str | None] = {
    "USD": "WUIUSA",
    "EUR": "WUIDEU",  # Germany proxy; no clean EA aggregate
    "GBP": "WUIGBR",
    "JPY": "WUIJPN",
    "CAD": "WUICAN",
    "AUD": "WUIAUS",
    "NZD": "WUINZL",
    "CHF": "WUICHE",
}

WUI_FAILED_OR_ALT: dict[str, list[str]] = {
    "eur_germany_proxy": ["WUIDEU"],
    "eur_france_alt": ["WUIFRA"],
    "eur_italy_alt": ["WUIITA"],
    "eur_spain_alt": ["WUIESP"],
    "eur_no_clean_ea_aggregate": [],
    "usd_primary": ["WUIUSA"],
    "gbp_primary": ["WUIGBR"],
    "jpy_primary": ["WUIJPN"],
    "cad_primary": ["WUICAN"],
    "aud_primary": ["WUIAUS"],
    "nzd_primary": ["WUINZL"],
    "chf_primary": ["WUICHE"],
}

EUR_FRANCE = "WUIFRA"
EUR_ITALY = "WUIITA"
EUR_SPAIN = "WUIESP"

DEFAULT_PUB_LAG_MONTHS = 4  # conservative quarterly EIU/WUI lag


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


def load_wui_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Quarterly country WUI → monthly PIT panel.

    Columns = ISO currency codes. Values = WUI index levels.
    Index = UTC month-start of the PIT *known* date after ``pub_lag_months``.
    Quarterly observations are pub-lagged first, then forward-filled within
    the known tree to month-start (no within-quarter leak of unpublished values).
    Strategy modules score **levels** (WUI is already an index).
    """
    curs = [c.upper() for c in (currencies or WUI_SERIES.keys())]
    cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}

    for c in curs:
        sid = WUI_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_wui"
            continue
        try:
            s = _load_quarterly(sid, download=download, force=force)
            series_map[c] = sid
            if c == "EUR":
                notes[c] = (
                    f"Germany proxy via {sid} "
                    f"(WUIFRA/WUIITA/WUIESP alts; no clean EA aggregate on FRED)"
                )
            else:
                notes[c] = f"Ahir–Bloom–Furceri country WUI {sid} (quarterly)"
            cols[c] = s
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    df = pd.DataFrame(cols).sort_index()
    df = _expand_quarterly_to_monthly(df, pub_lag_months=pub_lag_months)

    df.attrs["factor"] = "wui_world_uncertainty_index"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED Ahir–Bloom–Furceri World Uncertainty Index (country quarterly; "
        "EUR=WUIDEU Germany proxy; WUIFRA/WUIITA/WUIESP alts; full G10 live ~2026-04)"
    )
    df.attrs["unit"] = "wui_index_level"
    df.attrs["frequency_raw"] = "quarterly"
    df.attrs["score_basis"] = "levels"  # not YoY-first — WUI is already an index
    df.attrs["unmapped"] = [c for c in curs if c not in cols]
    df.attrs["failed_or_alt"] = WUI_FAILED_OR_ALT
    return df


def load_us_wui(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US WUI (WUIUSA), PIT-lagged to monthly."""
    panel = load_wui_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US WUI series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_WUI"
    out.attrs["series_id"] = WUI_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def wui_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_wui_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_fred_wui"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
