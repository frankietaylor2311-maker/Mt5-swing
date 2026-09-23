"""FRED / BIS private credit-to-GDP panels for financial-cycle FX (free, PIT).

Literature
----------
- Borio & Drehmann / BIS early-warning: private credit/GDP and the credit gap
  (deviation from a long-run trend) predict financial-cycle stress and FX risk
  premia (Basel credit-cycle literature).
- Distinct from government debt/GDP (§29 GGGDTA*), fiscal GGNLBA (§28), CA, TB
  (§30), BIS REER (§31), CB-BS, funding-liq.

Free data (no Bloomberg / BIS portal key)
-----------------------------------------
BIS **Credit to Private Non-Financial Sector as % of GDP** via FRED
``Q*PAM770A`` (quarterly). Brief mnemonic ``CRDQ*APABIS`` is live but is
**absolute credit (bn)**, not % GDP — documented and unused as primary.

| Ccy | FRED id (actual) | Notes                                      |
|-----|------------------|--------------------------------------------|
| USD | QUSPAM770A       | primary US state variable                  |
| EUR | QXMPAM770A       | Euro-area (continuous from 1999)           |
| GBP | QGBPAM770A       |                                            |
| JPY | QJPPAM770A       |                                            |
| CAD | QCAPAM770A       |                                            |
| AUD | QAUPAM770A       |                                            |
| NZD | QNZPAM770A       | mapped (unlike debt/fiscal GGGDTA*)        |
| CHF | QCHPAM770A       | mapped                                     |

Tried / documented:
- ``CRDQ*APABIS`` (US/DE/GB/JP/CA/AU/CH) — live absolute credit; **not** % GDP.
- ``CRDQNZAPABIS`` — **404** (NZD absolute); ratio ``QNZPAM770A`` OK.
- ``CRDQEZAPABIS`` / ``CRDQEUAPABIS`` / ``CRDQUSAAPABIS`` — **404**.
- Pre-computed credit-gap IDs (``BISCRDGAPUS``, ``CRDGAPUSA``, ``CRDQUSAGAP``,
  ``BISCREDGAPUS``, …) — **404** on FRED as of 2026-09. Gap computed in-strategy
  as causal trailing-trend deviation (HP-like one-sided mean).

Germany alt for EUR: ``QDEPAM770A`` (documented; unused as primary — prefer
euro-area ``QXMPAM770A``). France ``QFRPAM770A`` coverage-only.

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
BIS quarterly credit releases typically land ~4–5 months after quarter-end.
Default ``pub_lag_months=5``: observation dated quarter-start ``t`` is first
known at ``t + 5 months``. Strategy modules add ``signal_lag`` months on top.
(As of 2026-09-23, latest point was 2025-10 — consistent with ~4–5m lag.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED BIS private credit / GDP (%) — Q*PAM770A
BIS_CREDIT_GDP_SERIES: dict[str, str | None] = {
    "USD": "QUSPAM770A",
    "EUR": "QXMPAM770A",  # Euro-area (not Germany)
    "GBP": "QGBPAM770A",
    "JPY": "QJPPAM770A",
    "CAD": "QCAPAM770A",
    "AUD": "QAUPAM770A",
    "NZD": "QNZPAM770A",
    "CHF": "QCHPAM770A",
}

# Absolute credit (bn) — live but unused as primary %GDP panel
BIS_CREDIT_ABS_SERIES: dict[str, str | None] = {
    "USD": "CRDQUSAPABIS",
    "EUR": "CRDQDEAPABIS",  # Germany absolute; euro-area abs = CRDQXMAPABIS
    "GBP": "CRDQGBAPABIS",
    "JPY": "CRDQJPAPABIS",
    "CAD": "CRDQCAAPABIS",
    "AUD": "CRDQAUAPABIS",
    "NZD": None,  # CRDQNZAPABIS 404
    "CHF": "CRDQCHAPABIS",
}

# Documented 404 / alt mnemonics
BIS_CREDIT_FAILED_OR_ALT: dict[str, list[str]] = {
    "absolute_crdq_unused_as_pct": [
        "CRDQUSAPABIS",
        "CRDQDEAPABIS",
        "CRDQGBAPABIS",
        "CRDQJPAPABIS",
        "CRDQCAAPABIS",
        "CRDQAUAPABIS",
        "CRDQCHAPABIS",
        "CRDQXMAPABIS",
    ],
    "nzd_absolute_404": ["CRDQNZAPABIS", "CRDQNZLAPABIS", "CRDQNZAAPABIS"],
    "eur_ez_404": ["CRDQEZAPABIS", "CRDQEUAPABIS", "QEUPAM770A"],
    "credit_gap_precomputed_404": [
        "BISCRDGAPUS",
        "BISCRDGAPGB",
        "BISCREDGAPUS",
        "CRDGAPUSA",
        "CRDGAPUS",
        "CRDQUSAGAP",
        "CRDQUSAAGAP",
        "CRDQUSAHPABIS",
    ],
    "eur_germany_alt_pct": ["QDEPAM770A"],
    "france_coverage_only": ["QFRPAM770A"],
}

EUR_GERMANY_ALT = "QDEPAM770A"
EUR_ABS_EA = "CRDQXMAPABIS"

DEFAULT_PUB_LAG_MONTHS = 5  # conservative quarterly BIS credit release lag


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


def load_bis_credit_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Quarterly BIS private credit / GDP (%) → monthly PIT panel.

    Columns = ISO currency codes. Values = percent of GDP.
    Index = UTC month-start of the PIT *known* date after ``pub_lag_months``.
    """
    curs = [c.upper() for c in (currencies or BIS_CREDIT_GDP_SERIES.keys())]
    cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}

    for c in curs:
        sid = BIS_CREDIT_GDP_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_q_pam770a"
            continue
        try:
            s = _load_quarterly(sid, download=download, force=force)
            series_map[c] = sid
            if c == "EUR":
                notes[c] = (
                    f"Euro-area via {sid}; "
                    f"Germany alt {EUR_GERMANY_ALT} documented (unused as primary)"
                )
            cols[c] = s
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    df = pd.DataFrame(cols).sort_index()
    df = _expand_quarterly_to_monthly(df, pub_lag_months=pub_lag_months)

    df.attrs["factor"] = "bis_private_credit_gdp_pct"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED BIS Q*PAM770A private credit/GDP % "
        "(CRDQ*APABIS = absolute credit, unused as primary)"
    )
    df.attrs["unit"] = "percent_of_gdp"
    df.attrs["unmapped"] = [c for c in curs if c not in cols]
    df.attrs["failed_or_alt"] = BIS_CREDIT_FAILED_OR_ALT
    df.attrs["absolute_series_map"] = {
        k: v for k, v in BIS_CREDIT_ABS_SERIES.items() if v is not None
    }
    return df


def load_us_bis_credit(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US private credit / GDP (%) (QUSPAM770A), PIT-lagged."""
    panel = load_bis_credit_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US BIS credit/GDP series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_BIS_CREDIT_GDP"
    out.attrs["series_id"] = BIS_CREDIT_GDP_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def credit_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_bis_credit_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_fred_q_pam770a"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
