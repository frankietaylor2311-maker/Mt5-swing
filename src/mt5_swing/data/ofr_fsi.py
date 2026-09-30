"""Point-in-time Office of Financial Research (OFR) Financial Stress Index loader.

Literature
----------
Office of Financial Research Financial Stress Index (OFR FSI) working paper /
methodology notes: daily US + global financial-stress aggregate from credit,
equity valuation, safe assets, funding, and volatility categories.
Brunnermeier, Nagel & Pedersen (2008): funding-liquidity spirals / FX risk-off
in elevated stress. Used here as a **distinct** free daily stress conditioner
from Chicago Fed NFCI/ANFCI, St. Louis Fed STLFSI4, and Kansas City Fed KCFSI
(different construction / panel / publisher — not a FRED series).

Free data (no paid terminals)
-----------------------------
Official daily CSV::

    https://www.financialresearch.gov/financial-stress-index/data/fsi.csv

Columns: Date, OFR FSI, Credit, Equity valuation, Safe assets, Funding,
Volatility (+ regional breakdowns). We use the headline ``OFR FSI`` column.

PIT lag (conservative, frozen — not holdout-tuned)
--------------------------------------------------
OFR publishes with data current from **two business days prior** →
``pub_lag_days=2`` (calendar-day shift, same style as TED/CPFF daily lag and
STLFSI weekly ``pub_lag_days=7``). Downstream soft/carry/value waves then
collapse to month-end and apply monthly trailing z (z_window=60m) +
``signal_lag_months=1`` + ``weight_lag_days=1`` (mirror §71–§109 soft stack).

Do **not** overlay this cooler onto the locked FTMO sleeve; evaluate as a
standalone scholarly FX conditioner.
"""

from __future__ import annotations

from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlretrieve

import pandas as pd

from mt5_swing.data.fred_rates import macro_dir

OFR_FSI_URL = (
    "https://www.financialresearch.gov/financial-stress-index/data/fsi.csv"
)
OFR_FSI_CACHE_NAME = "ofr_fsi.csv"
DEFAULT_OFR_PUB_LAG_DAYS = 2  # OFR: data current from two business days prior
VALUE_COL = "OFR FSI"
DATE_COL = "Date"


def ofr_fsi_cache_path() -> Path:
    return macro_dir() / OFR_FSI_CACHE_NAME


def download_ofr_fsi(*, force: bool = False) -> Path:
    """Download official OFR FSI CSV into ``data/macro/ofr_fsi.csv``."""
    path = ofr_fsi_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force and path.stat().st_size > 200:
        return path
    # Modest UA — some gov CDNs reject empty/default urllib agent
    req = Request(
        OFR_FSI_URL,
        headers={"User-Agent": "mt5-swing-ofr-fsi/1.0 (+research; non-commercial)"},
    )
    try:
        # urlretrieve does not take Request; use urlopen path via retrieve workaround
        from urllib.request import urlopen

        with urlopen(req, timeout=60) as resp:
            data = resp.read()
        path.write_bytes(data)
    except URLError as exc:
        raise RuntimeError(f"OFR FSI download failed: {exc}") from exc
    if path.stat().st_size < 200:
        raise RuntimeError(f"OFR FSI download too small: {path} ({path.stat().st_size} bytes)")
    return path


def _apply_pub_lag(s: pd.Series, pub_lag_days: int) -> pd.Series:
    out = s.copy()
    if pub_lag_days > 0:
        out.index = out.index + pd.Timedelta(days=int(pub_lag_days))
        out = out[~out.index.duplicated(keep="last")].sort_index()
    out.attrs["pub_lag_days"] = int(pub_lag_days)
    return out


def load_ofr_fsi_series(
    *,
    download: bool = True,
    pub_lag_days: int = DEFAULT_OFR_PUB_LAG_DAYS,
    force: bool = False,
    value_col: str = VALUE_COL,
) -> pd.Series:
    """Load headline OFR FSI (daily) with publication lag.

    Returns a UTC-indexed float Series named ``OFR_FSI``. ``pub_lag_days``
    shifts observation dates forward so the series is only known after the
    OFR publication lag (default 2 — data current from two business days prior).
    """
    path = ofr_fsi_cache_path()
    if download or force or not path.exists():
        download_ofr_fsi(force=force)
    if not path.exists():
        raise FileNotFoundError(f"OFR FSI cache missing: {path}")

    df = pd.read_csv(path)
    if DATE_COL not in df.columns:
        raise ValueError(f"OFR FSI CSV missing Date column; got {list(df.columns)}")
    if value_col not in df.columns:
        raise ValueError(
            f"OFR FSI CSV missing '{value_col}' column; got {list(df.columns)}"
        )
    s = pd.Series(
        pd.to_numeric(df[value_col], errors="coerce").values,
        index=pd.to_datetime(df[DATE_COL], utc=True),
        name="OFR_FSI",
    )
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    s = _apply_pub_lag(s, pub_lag_days)
    s.name = "OFR_FSI"
    s.attrs["source"] = "ofr_fsi_csv"
    s.attrs["frequency"] = "daily"
    s.attrs["url"] = OFR_FSI_URL
    s.attrs["value_col"] = value_col
    return s


def load_us_ofr_fsi_series(
    *,
    download: bool = True,
    pub_lag_days: int = DEFAULT_OFR_PUB_LAG_DAYS,
    force: bool = False,
) -> pd.Series:
    """Thin wrapper — US/headline OFR FSI (daily, pub_lag_days applied)."""
    return load_ofr_fsi_series(
        download=download, pub_lag_days=pub_lag_days, force=force
    )


def ensure_ofr_fsi_cached(*, force: bool = False) -> str:
    """Ensure cache exists; return absolute path string."""
    return str(download_ofr_fsi(force=force).resolve())


__all__ = [
    "OFR_FSI_URL",
    "DEFAULT_OFR_PUB_LAG_DAYS",
    "ofr_fsi_cache_path",
    "download_ofr_fsi",
    "load_ofr_fsi_series",
    "load_us_ofr_fsi_series",
    "ensure_ofr_fsi_cached",
]
