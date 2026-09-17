"""Point-in-time FRED short-rate loaders for FX carry differentials.

Uses free FRED CSV endpoint (no API key). Monthly OECD immediate-rate series
are preferred for cross-currency comparability; daily overnight series are
resampled to month-end when needed.

Publication delay: monthly OECD series are typically released with a short lag.
We apply ``pub_lag_months`` (default 1) so a month-t rate is first usable at
month-end t + pub_lag_months. Daily series use ``pub_lag_days`` (default 1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlretrieve

import pandas as pd

# Currency → FRED series (OECD Immediate rates monthly, % p.a.)
# USD also has DFF daily; we prefer IRSTCI01USM156N for panel alignment.
FRED_IMMEDIATE_MONTHLY: dict[str, str] = {
    "USD": "IRSTCI01USM156N",
    "EUR": "IRSTCI01EZM156N",
    "GBP": "IRSTCI01GBM156N",
    "JPY": "IRSTCI01JPM156N",
    "AUD": "IRSTCI01AUM156N",
    "CAD": "IRSTCI01CAM156N",
    "CHF": "IRSTCI01CHM156N",
    "NZD": "IRSTCI01NZM156N",
}

# Optional daily overnight overlays (research / higher frequency)
FRED_OVERNIGHT_DAILY: dict[str, str] = {
    "USD": "DFF",
    "EUR": "ECBDFR",
    "GBP": "IUDSOIA",
}

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"


def fred_csv_path(series_id: str) -> Path:
    return macro_dir() / f"fred_{series_id}.csv"


def download_fred_series(series_id: str, *, force: bool = False) -> Path:
    """Download one FRED series CSV into ``data/macro/``."""
    path = fred_csv_path(series_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force and path.stat().st_size > 50:
        return path
    url = FRED_CSV.format(series_id=series_id)
    try:
        urlretrieve(url, path)
    except URLError as exc:
        raise RuntimeError(f"FRED download failed for {series_id}: {exc}") from exc
    return path


def load_fred_series(
    series_id: str,
    *,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """Load a FRED series as a UTC-indexed float Series named ``series_id``."""
    path = fred_csv_path(series_id)
    if download or not path.exists():
        download_fred_series(series_id, force=force)
    df = pd.read_csv(path)
    # Columns: observation_date, SERIES_ID
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    val_col = series_id if series_id in df.columns else df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").values,
        index=pd.to_datetime(df[date_col], utc=True),
        name=series_id,
    )
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    return s


def apply_publication_lag(
    s: pd.Series,
    *,
    lag_months: int = 0,
    lag_days: int = 0,
) -> pd.Series:
    """Shift the *availability* index forward (point-in-time usable date).

    Value observed for period ending ``t`` becomes usable at ``t + lag``.
    """
    if lag_months <= 0 and lag_days <= 0:
        return s.copy()
    out = s.copy()
    if lag_months > 0:
        out.index = out.index + pd.DateOffset(months=int(lag_months))
    if lag_days > 0:
        out.index = out.index + pd.Timedelta(days=int(lag_days))
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def load_currency_rates(
    currencies: Iterable[str] | None = None,
    *,
    freq: str = "M",
    pub_lag_months: int = 1,
    download: bool = True,
) -> pd.DataFrame:
    """Panel of short rates (% p.a.) with publication lag applied.

    Parameters
    ----------
    currencies
        ISO currency codes (default: G10 majors we have FX history for).
    freq
        ``M`` = month-end panel from OECD immediate rates;
        ``D`` = daily overnight where available (sparse currencies dropped).
    pub_lag_months
        Months to delay monthly rate availability (default 1).
    """
    curs = [c.upper() for c in (currencies or FRED_IMMEDIATE_MONTHLY.keys())]
    if freq.upper().startswith("D"):
        cols = {}
        for c in curs:
            sid = FRED_OVERNIGHT_DAILY.get(c)
            if not sid:
                continue
            s = load_fred_series(sid, download=download)
            s = apply_publication_lag(s, lag_days=1)
            cols[c] = s
        if not cols:
            raise ValueError("No daily overnight series for requested currencies")
        df = pd.DataFrame(cols).sort_index()
        df.attrs["source"] = "fred_overnight_daily"
        df.attrs["pub_lag_days"] = 1
        return df

    cols = {}
    for c in curs:
        sid = FRED_IMMEDIATE_MONTHLY.get(c)
        if not sid:
            raise KeyError(f"No FRED monthly immediate-rate mapping for {c}")
        s = load_fred_series(sid, download=download)
        # Normalize to month-start (drop tz before Period) then apply lag
        s = s.copy()
        idx = s.index.tz_convert("UTC").tz_localize(None) if s.index.tz is not None else s.index
        s.index = pd.DatetimeIndex(idx.to_period("M").to_timestamp(how="start"), tz="UTC")
        s = apply_publication_lag(s, lag_months=pub_lag_months)
        cols[c] = s
    df = pd.DataFrame(cols).sort_index()
    df.attrs["source"] = "fred_oecd_immediate_monthly"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["units"] = "percent_per_annum"
    return df


def rate_differentials_vs_usd(rates: pd.DataFrame) -> pd.DataFrame:
    """``rate_ccy - rate_USD`` for each non-USD column (carry score vs USD)."""
    if "USD" not in rates.columns:
        raise ValueError("rates panel must include USD")
    usd = rates["USD"]
    out = rates.drop(columns=["USD"]).sub(usd, axis=0)
    out.attrs.update(getattr(rates, "attrs", {}))
    out.attrs["definition"] = "ccy_rate_minus_usd"
    return out


def download_default_rate_panel(*, force: bool = False) -> dict[str, Path]:
    """Ensure default FRED CSVs exist under ``data/macro/``."""
    paths: dict[str, Path] = {}
    for sid in set(FRED_IMMEDIATE_MONTHLY.values()) | set(FRED_OVERNIGHT_DAILY.values()):
        paths[sid] = download_fred_series(sid, force=force)
    return paths
