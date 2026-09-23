"""Uncertainty / geopolitical-risk loaders: VIX (yfinance) + Caldara–Iacoviello GPR.

GPR source: https://www.matteoiacoviello.com/gpr.htm (Excel monthly/daily).
We cache cleaned CSVs under ``data/macro/``. If the public download is blocked,
``load_gpr`` raises with stub instructions; ``GprIndex`` still accepts a local CSV.

Publication lag (conservative defaults for research PIT):
- VIX: lag 1 trading day (close known next session).
- Monthly GPR: lag 1 calendar month (index for month t published early t+1).
- Daily GPR: lag 1 calendar day (updates are weekly; extra lag is safer).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.error import URLError, HTTPError
from urllib.request import urlretrieve

import pandas as pd

GPR_PAGE = "https://www.matteoiacoviello.com/gpr.htm"
GPR_MONTHLY_XLS = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
GPR_DAILY_XLS = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"

# Trade-policy / economic-policy uncertainty (optional stub series ids)
TPU_NOTES = (
    "Baker–Bloom–Davis EPU / TPU: https://www.policyuncertainty.com/ — "
    "download CSV manually into data/macro/epu_tpu.csv with columns date,EPU,TPU "
    "if automated fetch is blocked."
)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"


def _ensure_macro() -> Path:
    d = macro_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# VIX
# ---------------------------------------------------------------------------

def download_vix(*, force: bool = False) -> Path:
    """Download ^VIX close history via yfinance → ``data/macro/vix_yahoo.csv``."""
    path = _ensure_macro() / "vix_yahoo.csv"
    if path.exists() and not force and path.stat().st_size > 50:
        return path
    import yfinance as yf

    raw = yf.Ticker("^VIX").history(period="max", auto_adjust=True)
    if raw is None or raw.empty:
        raise RuntimeError("Empty VIX download from yfinance")
    out = raw[["Close"]].rename(columns={"Close": "close"}).copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC")
    else:
        idx = idx.tz_localize("UTC")
    out.index = idx
    out.index.name = "date"
    out.to_csv(path)
    return path


def load_vix(
    *,
    download: bool = True,
    pub_lag_days: int = 1,
    force: bool = False,
) -> pd.Series:
    """VIX close with optional publication lag (default 1 day)."""
    path = macro_dir() / "vix_yahoo.csv"
    if download or not path.exists():
        download_vix(force=force)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    s.index = pd.to_datetime(s.index, utc=True)
    s = s.sort_index().dropna()
    s.name = "VIX"
    if pub_lag_days > 0:
        s.index = s.index + pd.Timedelta(days=int(pub_lag_days))
        s = s[~s.index.duplicated(keep="last")].sort_index()
    s.attrs["source"] = "yfinance_^VIX"
    s.attrs["pub_lag_days"] = int(pub_lag_days)
    return s


# ---------------------------------------------------------------------------
# Caldara–Iacoviello GPR
# ---------------------------------------------------------------------------

def _xls_to_monthly_csv(xls_path: Path, csv_path: Path) -> Path:
    df = pd.read_excel(xls_path, sheet_name=0)
    keep = [c for c in ("month", "GPR", "GPRT", "GPRA", "GPRH") if c in df.columns]
    out = df[keep].dropna(subset=["month"]).copy()
    out["month"] = pd.to_datetime(out["month"], utc=True)
    out = out.sort_values("month")
    out.to_csv(csv_path, index=False)
    # Side-cache country-specific GPRC_* columns (Caldara–Iacoviello country index)
    gprc = [c for c in df.columns if str(c).startswith("GPRC_")]
    if gprc and "month" in df.columns:
        cpath = csv_path.parent / "gpr_country_monthly.csv"
        cout = df[["month"] + gprc].copy()
        cout["month"] = pd.to_datetime(cout["month"], utc=True)
        cout = cout.dropna(subset=gprc, how="all").sort_values("month")
        cout.to_csv(cpath, index=False)
    return csv_path


def _xls_to_daily_csv(xls_path: Path, csv_path: Path) -> Path:
    df = pd.read_excel(xls_path, sheet_name=0)
    need = [c for c in ("DAY", "GPRD", "GPRD_ACT", "GPRD_THREAT", "GPRD_MA30", "GPRD_MA7") if c in df.columns]
    out = df[need].copy()
    out["DAY"] = pd.to_numeric(out["DAY"], errors="coerce")
    out = out.dropna(subset=["DAY"])
    out["date"] = pd.to_datetime(out["DAY"].astype(int).astype(str), format="%Y%m%d", utc=True)
    out = out.rename(columns={"GPRD": "GPR"})
    cols = ["date", "GPR"] + [c for c in ("GPRD_ACT", "GPRD_THREAT", "GPRD_MA30", "GPRD_MA7") if c in out.columns]
    out = out[cols].sort_values("date")
    out.to_csv(csv_path, index=False)
    return csv_path


def download_gpr(
    *,
    freq: Literal["M", "D"] = "M",
    force: bool = False,
) -> Path:
    """Download GPR Excel from Iacoviello site and cache a cleaned CSV.

    Raises ``RuntimeError`` with manual instructions if the HTTP fetch fails.
    """
    d = _ensure_macro()
    csv_name = "gpr_monthly.csv" if freq.upper().startswith("M") else "gpr_daily.csv"
    xls_name = "gpr_monthly.xls" if freq.upper().startswith("M") else "gpr_daily.xls"
    csv_path = d / csv_name
    xls_path = d / xls_name
    if csv_path.exists() and not force and csv_path.stat().st_size > 50:
        return csv_path
    url = GPR_MONTHLY_XLS if freq.upper().startswith("M") else GPR_DAILY_XLS
    try:
        if force or not xls_path.exists() or xls_path.stat().st_size < 1000:
            urlretrieve(url, xls_path)
    except (URLError, HTTPError, OSError) as exc:
        raise RuntimeError(
            f"GPR download blocked/failed ({exc}).\n"
            f"1. Open {GPR_PAGE}\n"
            f"2. Download monthly Excel (data_gpr_export*.xls) or daily recent Excel\n"
            f"3. Save as {xls_path} then re-run, OR place a cleaned CSV at {csv_path}\n"
            f"   with columns month|date,GPR[,GPRT,GPRA]."
        ) from exc
    if freq.upper().startswith("M"):
        return _xls_to_monthly_csv(xls_path, csv_path)
    return _xls_to_daily_csv(xls_path, csv_path)


def load_gpr(
    *,
    freq: Literal["M", "D"] = "M",
    column: str = "GPR",
    download: bool = True,
    pub_lag_months: int = 1,
    pub_lag_days: int = 1,
    force: bool = False,
) -> pd.Series:
    """Load Caldara–Iacoviello GPR with publication lag.

    If CSV/XLS missing and download fails, raises with stub instructions.
    """
    csv_path = macro_dir() / ("gpr_monthly.csv" if freq.upper().startswith("M") else "gpr_daily.csv")
    if download or not csv_path.exists():
        try:
            download_gpr(freq=freq, force=force)
        except RuntimeError:
            if not csv_path.exists():
                raise
    df = pd.read_csv(csv_path)
    if freq.upper().startswith("M"):
        date_col = "month" if "month" in df.columns else df.columns[0]
        idx = pd.to_datetime(df[date_col], utc=True)
        if column not in df.columns:
            raise KeyError(f"GPR column '{column}' not in {list(df.columns)}")
        s = pd.Series(pd.to_numeric(df[column], errors="coerce").values, index=idx, name=column)
        s = s.dropna().sort_index()
        if pub_lag_months > 0:
            s.index = s.index + pd.DateOffset(months=int(pub_lag_months))
            s = s[~s.index.duplicated(keep="last")].sort_index()
        s.attrs["pub_lag_months"] = int(pub_lag_months)
    else:
        date_col = "date" if "date" in df.columns else df.columns[0]
        idx = pd.to_datetime(df[date_col], utc=True)
        col = column if column in df.columns else "GPR"
        s = pd.Series(pd.to_numeric(df[col], errors="coerce").values, index=idx, name=col)
        s = s.dropna().sort_index()
        if pub_lag_days > 0:
            s.index = s.index + pd.Timedelta(days=int(pub_lag_days))
            s = s[~s.index.duplicated(keep="last")].sort_index()
        s.attrs["pub_lag_days"] = int(pub_lag_days)
    s.attrs["source"] = "caldara_iacoviello_gpr"
    s.attrs["citation"] = "Caldara & Iacoviello (2022), Measuring Geopolitical Risk, AER"
    s.attrs["url"] = GPR_PAGE
    return s


class GprIndex:
    """Thin stub/interface when live download is unavailable.

    Usage::

        gpr = GprIndex.from_csv(path)  # or GprIndex.stub()
        s = gpr.series(pub_lag_months=1)
    """

    def __init__(self, series: pd.Series | None = None, *, note: str = ""):
        self._series = series
        self.note = note or (
            "Stub: place Caldara–Iacoviello GPR CSV at data/macro/gpr_monthly.csv "
            f"or download from {GPR_PAGE}"
        )

    @classmethod
    def stub(cls) -> "GprIndex":
        return cls(None)

    @classmethod
    def from_csv(cls, path: str | Path, column: str = "GPR") -> "GprIndex":
        df = pd.read_csv(path)
        date_col = "month" if "month" in df.columns else ("date" if "date" in df.columns else df.columns[0])
        idx = pd.to_datetime(df[date_col], utc=True)
        s = pd.Series(pd.to_numeric(df[column], errors="coerce").values, index=idx, name=column)
        return cls(s.dropna().sort_index(), note=f"loaded from {path}")

    @classmethod
    def try_load(cls, **kwargs) -> "GprIndex":
        try:
            return cls(load_gpr(**kwargs), note="live/cached GPR")
        except Exception as exc:  # noqa: BLE001 — intentional stub fallback
            return cls(None, note=f"stub after load failure: {exc}")

    @property
    def available(self) -> bool:
        return self._series is not None and len(self._series) > 0

    def series(self, *, pub_lag_months: int = 0, pub_lag_days: int = 0) -> pd.Series:
        if not self.available:
            raise RuntimeError(self.note)
        s = self._series.copy()
        if pub_lag_months > 0:
            s.index = s.index + pd.DateOffset(months=int(pub_lag_months))
        if pub_lag_days > 0:
            s.index = s.index + pd.Timedelta(days=int(pub_lag_days))
        return s[~s.index.duplicated(keep="last")].sort_index()



# ---------------------------------------------------------------------------
# Country-specific GPR (Caldara–Iacoviello GPRC_*)
# ---------------------------------------------------------------------------

# ISO3 country code in GPRC_* → ISO currency for FX majors we trade
# EUR uses equal-weight of core EZ countries (no single "EUR" column).
GPRC_TO_CURRENCY: dict[str, str] = {
    "AUS": "AUD",
    "CAN": "CAD",
    "CHE": "CHF",
    "GBR": "GBP",
    "JPN": "JPY",
    "USA": "USD",
    "NZL": "NZD",  # not in standard 44-country set — documented gap
}

# Euro-area constituents available in the country file (equal-weight proxy for EUR)
EUR_GPRC_ISO3: tuple[str, ...] = ("DEU", "FRA", "ITA", "ESP", "NLD", "BEL")

# Currency → Yahoo/FTMO USD-major pair and sign: +1 means pair return = foreign vs USD
CURRENCY_USD_PAIR: dict[str, tuple[str, int]] = {
    "EUR": ("EURUSD", +1),
    "GBP": ("GBPUSD", +1),
    "AUD": ("AUDUSD", +1),
    "NZD": ("NZDUSD", +1),
    "JPY": ("USDJPY", -1),  # pair up = JPY down; foreign return = -pair
    "CAD": ("USDCAD", -1),
    "CHF": ("USDCHF", -1),
}


def country_gpr_csv_path() -> Path:
    return macro_dir() / "gpr_country_monthly.csv"


def ensure_country_gpr_csv(*, force: bool = False) -> Path:
    """Ensure ``gpr_country_monthly.csv`` exists (from monthly XLS or re-download)."""
    path = country_gpr_csv_path()
    if path.exists() and not force and path.stat().st_size > 50:
        return path
    # Prefer local monthly XLS / CSV conversion
    xls = macro_dir() / "gpr_monthly.xls"
    monthly_csv = macro_dir() / "gpr_monthly.csv"
    if xls.exists():
        _xls_to_monthly_csv(xls, monthly_csv)
        if path.exists():
            return path
    download_gpr(freq="M", force=force)
    if not path.exists():
        raise RuntimeError(
            f"Country GPR CSV missing at {path}. Place Caldara–Iacoviello monthly "
            f"Excel (with GPRC_* columns) at {xls} or download from {GPR_PAGE}."
        )
    return path


def load_country_gpr(
    *,
    download: bool = True,
    pub_lag_months: int = 1,
    force: bool = False,
    currencies: list[str] | None = None,
) -> pd.DataFrame:
    """Monthly country GPR mapped to FX currencies (article-share units).

    Columns are ISO currencies (AUD, CAD, …, EUR aggregate, USD). Values are the
    Caldara–Iacoviello country index (share of GPR articles mentioning the country).
    EUR = equal-weight mean of ``EUR_GPRC_ISO3``. Publication lag shifts the
    availability index forward by ``pub_lag_months`` (default 1).

    Notes
    -----
    - No ``GPRC_NZL`` in the standard 44-country file → NZD omitted (gap).
    - Indices reflect a *US newspaper perspective* on risks involving that country.
    """
    if download or force or not country_gpr_csv_path().exists():
        ensure_country_gpr_csv(force=force)
    df = pd.read_csv(country_gpr_csv_path())
    date_col = "month" if "month" in df.columns else df.columns[0]
    idx = pd.to_datetime(df[date_col], utc=True)
    raw = df.copy()
    raw.index = idx
    cols: dict[str, pd.Series] = {}
    for iso3, ccy in GPRC_TO_CURRENCY.items():
        col = f"GPRC_{iso3}"
        if col not in raw.columns:
            continue
        cols[ccy] = pd.to_numeric(raw[col], errors="coerce")
    # EUR aggregate
    ez = []
    for iso3 in EUR_GPRC_ISO3:
        col = f"GPRC_{iso3}"
        if col in raw.columns:
            ez.append(pd.to_numeric(raw[col], errors="coerce"))
    if ez:
        cols["EUR"] = pd.concat(ez, axis=1).mean(axis=1)
    panel = pd.DataFrame(cols).sort_index()
    panel = panel.dropna(how="all")
    if pub_lag_months > 0:
        panel.index = panel.index + pd.DateOffset(months=int(pub_lag_months))
        panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    if currencies is not None:
        keep = [c.upper() for c in currencies if c.upper() in panel.columns]
        panel = panel[keep]
    panel.attrs["source"] = "caldara_iacoviello_gpr_country"
    panel.attrs["citation"] = "Caldara & Iacoviello (2022), Measuring Geopolitical Risk, AER"
    panel.attrs["url"] = GPR_PAGE
    panel.attrs["pub_lag_months"] = int(pub_lag_months)
    panel.attrs["eur_constituents"] = list(EUR_GPRC_ISO3)
    panel.attrs["missing_currencies"] = ["NZD"]  # no GPRC_NZL
    return panel


def load_epu_tpu_stub() -> pd.DataFrame:
    """Backward-compatible stub: prefer live FRED EPU, else local CSV, else empty."""
    path = macro_dir() / "epu_tpu.csv"
    try:
        panel = load_epu_panel(download=True, pub_lag_months=0)
        out = panel.rename(columns={"US_EPU": "EPU"})
        if "GEPU" in out.columns:
            out["TPU"] = out["GEPU"]  # placeholder col; true TPU is separate
        else:
            out["TPU"] = pd.NA
        out.attrs["available"] = True
        out.attrs["note"] = "FRED USEPUINDXM/GEPUCURRENT; TPU column is GEPU placeholder"
        return out[["EPU", "TPU"]].sort_index()
    except Exception:
        pass
    if not path.exists():
        empty = pd.DataFrame(columns=["EPU", "TPU"])
        empty.attrs["instructions"] = TPU_NOTES
        empty.attrs["available"] = False
        return empty
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.attrs["available"] = True
    return df.sort_index()


# ---------------------------------------------------------------------------
# Baker–Bloom–Davis EPU (free FRED + optional policyuncertainty.com workbook)
# ---------------------------------------------------------------------------

EPU_PAGE = "https://www.policyuncertainty.com/"
EPU_ALL_COUNTRY_XLSX = "https://www.policyuncertainty.com/media/All_Country_Data.xlsx"
# FRED mirrors (no API key)
FRED_US_EPU = "USEPUINDXM"
FRED_GEPU = "GEPUCURRENT"


def download_epu_fred(*, force: bool = False) -> dict[str, Path]:
    """Download US EPU + Global EPU from FRED into ``data/macro/``."""
    from mt5_swing.data.fred_rates import download_fred_series

    out = {
        "US_EPU": download_fred_series(FRED_US_EPU, force=force),
        "GEPU": download_fred_series(FRED_GEPU, force=force),
    }
    return out


def download_epu_all_country_xlsx(*, force: bool = False) -> Path | None:
    """Try policyuncertainty.com All_Country_Data.xlsx (may be blocked).

    Returns path on success, None if download fails (FRED GEPU remains available).
    """
    path = _ensure_macro() / "epu_all_country.xlsx"
    if path.exists() and not force and path.stat().st_size > 1000:
        return path
    try:
        urlretrieve(EPU_ALL_COUNTRY_XLSX, path)
        if path.stat().st_size < 1000:
            path.unlink(missing_ok=True)
            return None
        return path
    except (URLError, HTTPError, OSError):
        return None


def load_epu(
    *,
    series: str = "US",
    download: bool = True,
    pub_lag_months: int = 1,
    force: bool = False,
) -> pd.Series:
    """Load Baker–Bloom–Davis EPU with publication lag.

    Parameters
    ----------
    series
        ``US`` → FRED USEPUINDXM; ``GEPU`` / ``GLOBAL`` → FRED GEPUCURRENT.
    pub_lag_months
        Default 1 (monthly index for month t first usable early t+1).
    """
    from mt5_swing.data.fred_rates import load_fred_series

    key = series.upper().strip()
    if key in {"US", "USEPU", "USEPUINDXM", "EPU"}:
        sid = FRED_US_EPU
        name = "US_EPU"
    elif key in {"GEPU", "GLOBAL", "GEPUCURRENT", "WORLD"}:
        sid = FRED_GEPU
        name = "GEPU"
    else:
        raise KeyError(f"Unknown EPU series '{series}' (use US or GEPU)")
    if download or force:
        download_epu_fred(force=force)
    s = load_fred_series(sid, download=False)
    s = s.dropna().sort_index()
    s.name = name
    if pub_lag_months > 0:
        s.index = s.index + pd.DateOffset(months=int(pub_lag_months))
        s = s[~s.index.duplicated(keep="last")].sort_index()
    s.attrs["source"] = f"fred_{sid}"
    s.attrs["citation"] = "Baker, Bloom & Davis (2016), Measuring Economic Policy Uncertainty, QJE"
    s.attrs["url"] = EPU_PAGE
    s.attrs["pub_lag_months"] = int(pub_lag_months)
    return s


def load_epu_panel(
    *,
    download: bool = True,
    pub_lag_months: int = 1,
    force: bool = False,
) -> pd.DataFrame:
    """US EPU + GEPU columns with shared publication lag. Also tries xlsx cache."""
    if download or force:
        download_epu_fred(force=force)
        download_epu_all_country_xlsx(force=force)
    us = load_epu(series="US", download=False, pub_lag_months=pub_lag_months)
    ge = load_epu(series="GEPU", download=False, pub_lag_months=pub_lag_months)
    df = pd.concat([us.rename("US_EPU"), ge.rename("GEPU")], axis=1).sort_index()
    # Optional country workbook → cache a convenience CSV if readable
    xlsx = macro_dir() / "epu_all_country.xlsx"
    csv_path = macro_dir() / "epu_all_country.csv"
    if xlsx.exists() and (force or not csv_path.exists()):
        try:
            raw = pd.read_excel(xlsx, sheet_name=0)
            if {"Year", "Month"}.issubset(raw.columns):
                raw["date"] = pd.to_datetime(
                    dict(year=raw["Year"], month=raw["Month"], day=1), utc=True
                )
                raw = raw.drop(columns=["Year", "Month"]).set_index("date").sort_index()
                raw.to_csv(csv_path)
        except Exception:  # noqa: BLE001
            pass
    df.attrs["available"] = True
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["citation"] = "Baker, Bloom & Davis (2016); Davis (2016) GEPU"
    df.attrs["xlsx_cached"] = bool(xlsx.exists())
    return df
