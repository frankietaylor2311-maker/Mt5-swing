"""BIS debt-service ratio (DSR) panels for private non-financial FX (free, PIT).

Literature
----------
- Drehmann & Juselius (BIS) / Drehmann et al. (2015): the debt-service ratio
  (interest + amortisation / income) is a leading early-warning indicator of
  financial-cycle stress, distinct from stock leverage (credit/GDP).
- Channel to FX: high relative DSR → FX stress / risk premia (debtor-premium /
  fragile-borrower compensation). Honesty alternate: long low DSR (lean BS).
- Distinct from: BIS credit/GDP stock §32 (Q*PAM770A), gov debt/GDP §29,
  fiscal §28, funding-liq §20, IG OAS §36, CA/TB, REER, money, reserves.

Free data reality (verified 2026-09-23)
--------------------------------------
**FRED has no BIS DSR series** — all probed mnemonics 404 (``Q*PNFDSR``,
``BISDSR*``, ``DSR*PNF``, ``Q*DORQDSR``, …). Live free source is the BIS
SDMX REST API ``WS_DSR`` (no key):

  ``https://stats.bis.org/api/v1/data/WS_DSR/Q..P?format=csv``

``DSR_BORROWERS=P`` = private non-financial sector (PNFS). Household (H) /
NFC (N) breakdowns exist for a subset of countries but are **not** the
primary panel (thinner CHF coverage).

| Ccy | BIS ISO2 | Notes                                              |
|-----|----------|----------------------------------------------------|
| USD | US       | PNFS DSR                                           |
| EUR | DE       | **Germany proxy** (XM / EA euro-area keys missing) |
| GBP | GB       |                                                    |
| JPY | JP       |                                                    |
| CAD | CA       |                                                    |
| AUD | AU       |                                                    |
| CHF | CH       | PNFS only (H/N missing for CH)                     |
| NZD | —        | **unmapped** (NZ absent from WS_DSR)               |

Tried / documented FRED 404s: see ``BIS_DSR_FRED_404``.

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
BIS quarterly DSR releases land ~4–5 months after quarter-end (same band as
credit §32). Default ``pub_lag_months=5``: observation dated quarter-start
``t`` is first known at ``t + 5 months``. Strategy modules add
``signal_lag`` months + 1 trading-day weight lag.
(As of 2026-09-23, latest point was 2026-Q1 — consistent with ~5m lag.)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

# Currency → BIS borrowers country ISO2 (PNFS DSR)
BIS_DSR_ISO2: dict[str, str | None] = {
    "USD": "US",
    "EUR": "DE",  # Germany proxy — XM/EA missing on WS_DSR
    "GBP": "GB",
    "JPY": "JP",
    "CAD": "CA",
    "AUD": "AU",
    "NZD": None,  # absent from WS_DSR
    "CHF": "CH",
}

BIS_DSR_SDMX_URL = "https://stats.bis.org/api/v1/data/WS_DSR/Q..P?format=csv"
BIS_DSR_SDMX_URL_ALL = "https://stats.bis.org/api/v1/data/WS_DSR/Q..?format=csv"
BIS_DSR_CACHE = "bis_ws_dsr_Q_P.csv"
BIS_DSR_CACHE_ALL = "bis_ws_dsr_Q_all.csv"

# Honest FRED probe log (all 404 as of 2026-09-23) — primary idea still runs via BIS SDMX
BIS_DSR_FRED_404: dict[str, list[str]] = {
    "fred_pnf_dsr_404": [
        "QUSPNFDSR",
        "QGBPNFDSR",
        "QJPPNFDSR",
        "QCAPNFDSR",
        "QAUPNFDSR",
        "QNZPNFDSR",
        "QCHPNFDSR",
        "QDEPNFDSR",
        "QFRPNFDSR",
        "QXMPNFDSR",
    ],
    "fred_bisdsr_404": [
        "BISDSRUS",
        "BISDSRGB",
        "BISDSRJP",
        "BISDSRCA",
        "BISDSRAU",
        "BISDSRNZ",
        "BISDSRCH",
        "BISDSRDE",
        "BISDSRXM",
    ],
    "fred_dsr_pnf_404": [
        "DSRUSPNF",
        "DSRGBPNF",
        "DSRJPPNF",
        "DSRCAPNF",
        "DSRAUPNF",
        "DSRNZPNF",
        "DSRCHPNF",
        "DSRDEPNF",
        "DSRXMPNF",
    ],
    "fred_dorq_404": [
        "QUSDORQDSR",
        "QGBDORQDSR",
        "QJPDORQDSR",
        "QCADORQDSR",
        "QAUDORQDSR",
    ],
    "euro_area_keys_missing_on_bis": ["XM", "EA", "EZ"],
    "nzd_missing_on_bis_ws_dsr": ["NZ"],
}

DEFAULT_PUB_LAG_MONTHS = 5  # a priori: same band as BIS credit §32
DEFAULT_BORROWER = "P"  # private non-financial sector


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"


def bis_dsr_cache_path(*, all_borrowers: bool = False) -> Path:
    name = BIS_DSR_CACHE_ALL if all_borrowers else BIS_DSR_CACHE
    return macro_dir() / name


def download_bis_dsr(*, force: bool = False, all_borrowers: bool = False) -> Path:
    """Download BIS WS_DSR CSV into ``data/macro/`` (free SDMX, no key)."""
    path = bis_dsr_cache_path(all_borrowers=all_borrowers)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force and path.stat().st_size > 500:
        return path
    url = BIS_DSR_SDMX_URL_ALL if all_borrowers else BIS_DSR_SDMX_URL
    req = Request(url, headers={"User-Agent": "mt5-swing-research/1.0"})
    try:
        with urlopen(req, timeout=90) as resp:
            raw = resp.read()
    except URLError as exc:
        raise RuntimeError(f"BIS WS_DSR download failed: {exc}") from exc
    if len(raw) < 200 or b"OBS_VALUE" not in raw[:500]:
        raise RuntimeError(f"BIS WS_DSR payload unexpected ({len(raw)} bytes)")
    path.write_bytes(raw)
    return path


def _quarter_period_to_timestamp(period: str) -> pd.Timestamp:
    """Parse BIS ``YYYY-Qn`` → quarter-start UTC Timestamp."""
    year_s, q_s = str(period).split("-Q")
    year, q = int(year_s), int(q_s)
    month = 1 + (q - 1) * 3
    return pd.Timestamp(year=year, month=month, day=1, tz="UTC")


def _load_raw_pnfs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"BORROWERS_CTY", "TIME_PERIOD", "OBS_VALUE"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"BIS DSR CSV missing columns: {need - set(df.columns)}")
    if "DSR_BORROWERS" in df.columns:
        df = df[df["DSR_BORROWERS"] == DEFAULT_BORROWER].copy()
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["OBS_VALUE", "TIME_PERIOD", "BORROWERS_CTY"])
    return df


def _to_quarter_series(iso2: str, raw: pd.DataFrame) -> pd.Series:
    sub = raw[raw["BORROWERS_CTY"] == iso2].copy()
    if sub.empty:
        return pd.Series(dtype=float, name=iso2)
    sub = sub.sort_values("TIME_PERIOD")
    idx = [_quarter_period_to_timestamp(p) for p in sub["TIME_PERIOD"]]
    s = pd.Series(sub["OBS_VALUE"].to_numpy(dtype=float), index=pd.DatetimeIndex(idx), name=iso2)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


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


def load_bis_dsr_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """BIS PNFS debt-service ratio (%) → monthly PIT panel.

    Columns = ISO currency codes. Values = DSR percent of income.
    Index = UTC month-start of the PIT *known* date after ``pub_lag_months``.
    Strategy modules XS-rank on these levels (foreign only).
    """
    curs = [c.upper() for c in (currencies or BIS_DSR_ISO2.keys())]
    if download or force or not bis_dsr_cache_path().exists():
        download_bis_dsr(force=force, all_borrowers=False)

    raw = _load_raw_pnfs(bis_dsr_cache_path())
    cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}

    for c in curs:
        iso2 = BIS_DSR_ISO2.get(c)
        if iso2 is None:
            notes[c] = "unmapped_on_bis_ws_dsr (NZ absent)"
            continue
        s = _to_quarter_series(iso2, raw)
        if s.empty:
            notes[c] = f"missing_iso2_{iso2}_on_ws_dsr"
            continue
        series_map[c] = f"BIS:WS_DSR:Q.{iso2}.P"
        if c == "EUR":
            notes[c] = (
                f"Germany proxy via BIS WS_DSR Q.{iso2}.P "
                f"(XM/EA euro-area keys missing on WS_DSR; FR available but unused as primary)"
            )
        else:
            notes[c] = f"BIS WS_DSR Q.{iso2}.P PNFS debt-service ratio (%)"
        cols[c] = s

    df = pd.DataFrame(cols).sort_index()
    df = _expand_quarterly_to_monthly(df, pub_lag_months=pub_lag_months)

    df.attrs["factor"] = "bis_dsr_pnfs"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["iso2_map"] = {c: BIS_DSR_ISO2[c] for c in series_map}
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "BIS SDMX WS_DSR Q..P private non-financial debt-service ratio "
        "(FRED Q*PNFDSR / BISDSR* / DSR*PNF all 404 as of 2026-09; "
        "EUR=DE Germany proxy — XM/EA missing; NZD unmapped)"
    )
    df.attrs["unit"] = "dsr_pct_of_income"
    df.attrs["score_basis"] = "pnfs_dsr_level_pct"
    df.attrs["unmapped"] = [c for c in curs if c not in cols]
    df.attrs["failed_or_alt"] = BIS_DSR_FRED_404
    df.attrs["gaps"] = {
        "NZD": "NZ absent from BIS WS_DSR — unmapped",
        "EUR": "No euro-area aggregate (XM/EA) — Germany DE proxy",
    }
    return df


def load_us_bis_dsr(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US PNFS DSR (%), PIT-lagged."""
    panel = load_bis_dsr_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US BIS DSR series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_BIS_DSR"
    out.attrs["series_id"] = "BIS:WS_DSR:Q.US.P"
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def bis_dsr_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_bis_dsr_panel(download=False)
    rows = []
    notes = panel.attrs.get("notes", {})
    smap = panel.attrs.get("series_map", {})
    for c in panel.columns:
        s = panel[c].dropna()
        rows.append(
            {
                "currency": c,
                "bis_key": smap.get(c, ""),
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
                "bis_key": "",
                "n_obs": 0,
                "start": None,
                "end": None,
                "last": float("nan"),
                "note": notes.get(c, "unmapped_on_bis_ws_dsr"),
            }
        )
    return pd.DataFrame(rows)
