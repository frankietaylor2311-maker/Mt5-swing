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
# Core G10 (default carry panel — Yahoo D1 majors available)
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

# Extended OECD / EM immediate-rate series (research panel; many lack Yahoo pairs here)
FRED_IMMEDIATE_MONTHLY_EXTENDED: dict[str, str] = {
    **FRED_IMMEDIATE_MONTHLY,
    "SEK": "IRSTCI01SEM156N",
    "NOK": "IRSTCI01NOM156N",
    "DKK": "IRSTCI01DKM156N",
    "MXN": "IRSTCI01MXM156N",
    "KRW": "IRSTCI01KRM156N",
    "PLN": "IRSTCI01PLM156N",
    "CZK": "IRSTCI01CZM156N",
    "HUF": "IRSTCI01HUM156N",
    "ILS": "IRSTCI01ILM156N",
    "ZAR": "IRSTCI01ZAM156N",
    "TRY": "IRSTCI01TRM156N",
    "INR": "IRSTCI01INM156N",
    "BRL": "IRSTCI01BRM156N",
    "CLP": "IRSTCI01CLM156N",
    "ISK": "IRSTCI01ISM156N",
    "CNY": "IRSTCI01CNM156N",
    "RUB": "IRSTCI01RUM156N",
    "IDR": "IRSTCI01IDM156N",
}

# Documented missing IRSTCI01* codes (404 on FRED as of 2026-09)
FRED_IMMEDIATE_MISSING: dict[str, str] = {
    "SGD": "IRSTCI01SGM156N",
    "HKD": "IRSTCI01HKM156N",
    "THB": "IRSTCI01THM156N",
    "PHP": "IRSTCI01PHM156N",
    "MYR": "IRSTCI01MYM156N",
    "TWD": "IRSTCI01TWM156N",
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
    extended: bool = False,
) -> pd.DataFrame:
    """Panel of short rates (% p.a.) with publication lag applied.

    Parameters
    ----------
    currencies
        ISO currency codes (default: G10 majors we have FX history for,
        or the extended OECD/EM panel when ``extended=True``).
    freq
        ``M`` = month-end panel from OECD immediate rates;
        ``D`` = daily overnight where available (sparse currencies dropped).
    pub_lag_months
        Months to delay monthly rate availability (default 1).
    extended
        If True and ``currencies`` is None, load ``FRED_IMMEDIATE_MONTHLY_EXTENDED``.
    """
    mapping = FRED_IMMEDIATE_MONTHLY_EXTENDED if extended else FRED_IMMEDIATE_MONTHLY
    curs = [c.upper() for c in (currencies or mapping.keys())]
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
        sid = FRED_IMMEDIATE_MONTHLY_EXTENDED.get(c) or FRED_IMMEDIATE_MONTHLY.get(c)
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


def download_default_rate_panel(*, force: bool = False, extended: bool = False) -> dict[str, Path]:
    """Ensure default (or extended OECD) FRED CSVs exist under ``data/macro/``."""
    paths: dict[str, Path] = {}
    monthly = FRED_IMMEDIATE_MONTHLY_EXTENDED if extended else FRED_IMMEDIATE_MONTHLY
    for sid in set(monthly.values()) | set(FRED_OVERNIGHT_DAILY.values()):
        paths[sid] = download_fred_series(sid, force=force)
    return paths


def rate_panel_coverage(
    currencies: Iterable[str] | None = None,
    *,
    extended: bool = True,
    download: bool = False,
    asof: str | None = None,
) -> pd.DataFrame:
    """Document start/end dates and staleness for FRED immediate-rate series.

    Flags series whose last observation is older than ``asof`` (default: today UTC)
    by more than 4 months as ``sparse_or_stale``.
    """
    mapping = FRED_IMMEDIATE_MONTHLY_EXTENDED if extended else FRED_IMMEDIATE_MONTHLY
    curs = [c.upper() for c in (currencies or mapping.keys())]
    asof_ts = (pd.Timestamp(asof, tz='UTC') if asof else pd.Timestamp.now('UTC')).tz_convert(None)
    rows = []
    for c in curs:
        sid = mapping.get(c)
        if not sid:
            rows.append(
                {
                    "currency": c,
                    "series_id": None,
                    "status": "unmapped",
                    "n_obs": 0,
                    "start": None,
                    "end": None,
                    "months_since_end": None,
                    "sparse_or_stale": True,
                    "note": "no IRSTCI mapping",
                }
            )
            continue
        try:
            s = load_fred_series(sid, download=download)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "currency": c,
                    "series_id": sid,
                    "status": "load_fail",
                    "n_obs": 0,
                    "start": None,
                    "end": None,
                    "months_since_end": None,
                    "sparse_or_stale": True,
                    "note": str(exc)[:120],
                }
            )
            continue
        end = s.index.max()
        end_naive = end.tz_convert(None) if getattr(end, "tz", None) else end
        months_lag = (asof_ts.to_period("M") - pd.Timestamp(end_naive).to_period("M")).n
        stale = months_lag > 4
        note = ""
        if c in ("SEK",) and stale:
            note = "OECD series ends ~2020 on FRED — treat as discontinued for live carry"
        elif c in ("CHF", "NZD") and months_lag > 2:
            note = "tail lags OECD release / FRED update"
        rows.append(
            {
                "currency": c,
                "series_id": sid,
                "status": "ok",
                "n_obs": int(s.notna().sum()),
                "start": str(s.index.min().date()),
                "end": str(s.index.max().date()),
                "months_since_end": int(months_lag),
                "sparse_or_stale": bool(stale),
                "note": note,
            }
        )
    for c, sid in FRED_IMMEDIATE_MISSING.items():
        rows.append(
            {
                "currency": c,
                "series_id": sid,
                "status": "fred_404",
                "n_obs": 0,
                "start": None,
                "end": None,
                "months_since_end": None,
                "sparse_or_stale": True,
                "note": "IRSTCI01* not on FRED (404)",
            }
        )
    return pd.DataFrame(rows).sort_values("currency").reset_index(drop=True)
