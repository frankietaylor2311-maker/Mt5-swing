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
    """Backward-compatible stub: prefer live FRED EPU + true TPU, else local CSV."""
    path = macro_dir() / "epu_tpu.csv"
    try:
        panel = load_epu_panel(download=True, pub_lag_months=0)
        out = panel.rename(columns={"US_EPU": "EPU"})
        try:
            tpu = load_tpu(download=True, pub_lag_months=0)
            out = out.join(tpu.rename("TPU"), how="outer")
        except Exception:
            if "GEPU" in out.columns:
                out["TPU"] = out["GEPU"]
                out.attrs["note"] = "TPU fallback=GEPU (true TPU load failed)"
            else:
                out["TPU"] = pd.NA
                out.attrs["note"] = "TPU unavailable"
        else:
            out.attrs["note"] = "FRED USEPUINDXM/GEPUCURRENT + categorical Trade policy TPU"
        out.attrs["available"] = True
        cols = [c for c in ("EPU", "TPU", "GEPU") if c in out.columns]
        return out[cols].sort_index()
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


# ---------------------------------------------------------------------------
# Country EPU (Baker–Bloom–Davis All_Country_Data) + US TPU
# ---------------------------------------------------------------------------

EPU_CATEGORICAL_XLSX = "https://www.policyuncertainty.com/media/Categorical_EPU_Data.xlsx"
TPU_TRADE_XLSX = "https://www.policyuncertainty.com/media/Trade_Uncertainty_Data.xlsx"

# Country workbook column → ISO currency (G10 FX majors we trade)
COUNTRY_EPU_TO_CURRENCY: dict[str, str] = {
    "Australia": "AUD",
    "Canada": "CAD",
    "Japan": "JPY",
    "UK": "GBP",
    "US": "USD",
}
# Euro-area EPU proxy = equal-weight France + Germany (Italy optional if present)
EUR_EPU_COUNTRIES: tuple[str, ...] = ("France", "Germany")


def _read_year_month_workbook(path: Path, *, value_cols: list[str] | None = None) -> pd.DataFrame:
    """Parse policyuncertainty.com Year/Month Excel → UTC month-start index."""
    raw = pd.read_excel(path, sheet_name=0)
    # Drop trailing source footnote rows
    if "Year" in raw.columns:
        year_num = pd.to_numeric(raw["Year"], errors="coerce")
        raw = raw.loc[year_num.notna()].copy()
        raw["Year"] = year_num.loc[raw.index].astype(int)
        raw["Month"] = pd.to_numeric(raw["Month"], errors="coerce").astype("Int64")
        raw = raw.dropna(subset=["Month"])
        raw["date"] = pd.to_datetime(
            dict(year=raw["Year"].astype(int), month=raw["Month"].astype(int), day=1),
            utc=True,
        )
        raw = raw.drop(columns=["Year", "Month"]).set_index("date").sort_index()
    else:
        raise ValueError(f"Expected Year/Month columns in {path}")
    if value_cols is not None:
        keep = [c for c in value_cols if c in raw.columns]
        raw = raw[keep]
    # Coerce numeric
    for c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    return raw[~raw.index.duplicated(keep="last")].sort_index()


def download_tpu(*, force: bool = False) -> Path:
    """Download categorical EPU (Trade policy) + optional Trade_Uncertainty workbook.

    Returns the categorical path when available (updated through present);
    otherwise the trade workbook path.
    """
    cat = _ensure_macro() / "epu_categorical.xlsx"
    path = _ensure_macro() / "tpu_trade_uncertainty.xlsx"
    if force or not cat.exists() or cat.stat().st_size < 1000:
        try:
            urlretrieve(EPU_CATEGORICAL_XLSX, cat)
        except (URLError, HTTPError, OSError):
            pass
    if force or not path.exists() or path.stat().st_size < 1000:
        try:
            urlretrieve(TPU_TRADE_XLSX, path)
        except (URLError, HTTPError, OSError):
            pass
    if cat.exists() and cat.stat().st_size > 1000:
        return cat
    if path.exists() and path.stat().st_size > 1000:
        return path
    raise RuntimeError(
        f"TPU download failed. Place Categorical_EPU_Data.xlsx at {cat} "
        f"or Trade_Uncertainty_Data.xlsx at {path} from {EPU_PAGE}"
    )


def load_tpu(
    *,
    download: bool = True,
    pub_lag_months: int = 1,
    force: bool = False,
    prefer: str = "trade_workbook",
) -> pd.Series:
    """US Trade Policy Uncertainty (Baker–Bloom–Davis categorical Trade policy).

    PIT: monthly index for calendar month *t* first usable early *t+1*
    → default ``pub_lag_months=1``. Distinct from aggregate US EPU / GEPU / VIX / GPR.
    """
    trade_path = macro_dir() / "tpu_trade_uncertainty.xlsx"
    cat_path = macro_dir() / "epu_categorical.xlsx"
    if download or force:
        try:
            download_tpu(force=force)
        except RuntimeError:
            if not trade_path.exists() and not cat_path.exists():
                raise
    # Prefer categorical workbook: identical historically to Trade_Uncertainty_Data
    # but updated through the present (trade workbook freezes ~2019).
    s: pd.Series | None = None
    source = ""
    if cat_path.exists() and prefer in {"categorical", "trade_workbook", "auto"}:
        df = _read_year_month_workbook(cat_path)
        col = None
        for c in df.columns:
            if "trade policy" in str(c).lower():
                col = c
                break
        if col is not None:
            s = df[col].dropna()
            source = "policyuncertainty_Categorical_EPU_Data"
    if s is None and trade_path.exists():
        df = _read_year_month_workbook(trade_path)
        col = "US Trade Policy Uncertainty"
        if col in df.columns:
            s = df[col].dropna()
            source = "policyuncertainty_Trade_Uncertainty_Data"
    if s is None:
        raise RuntimeError(
            "TPU series unavailable. Download Trade_Uncertainty_Data.xlsx or "
            f"Categorical_EPU_Data.xlsx into {macro_dir()} from {EPU_PAGE}"
        )
    s = s.sort_index()
    s.name = "TPU"
    if pub_lag_months > 0:
        s.index = s.index + pd.DateOffset(months=int(pub_lag_months))
        s = s[~s.index.duplicated(keep="last")].sort_index()
    s.attrs["source"] = source
    s.attrs["citation"] = (
        "Baker, Bloom & Davis (2016), Measuring Economic Policy Uncertainty, QJE; "
        "US Categorical EPU — Trade policy"
    )
    s.attrs["url"] = "https://www.policyuncertainty.com/trade_uncertainty.html"
    s.attrs["pub_lag_months"] = int(pub_lag_months)
    # Cache cleaned CSV
    csv_path = macro_dir() / "tpu_us_monthly.csv"
    s.to_frame().to_csv(csv_path)
    return s


def ensure_country_epu_csv(*, force: bool = False) -> Path:
    """Ensure ``epu_country_monthly.csv`` from All_Country_Data.xlsx."""
    csv_path = macro_dir() / "epu_country_monthly.csv"
    xlsx = macro_dir() / "epu_all_country.xlsx"
    if csv_path.exists() and not force and csv_path.stat().st_size > 50:
        return csv_path
    if force or not xlsx.exists() or xlsx.stat().st_size < 1000:
        got = download_epu_all_country_xlsx(force=force)
        if got is None and not xlsx.exists():
            raise RuntimeError(
                f"Country EPU workbook missing. Download All_Country_Data.xlsx from "
                f"{EPU_PAGE}all_country_data.html into {xlsx}"
            )
    raw = _read_year_month_workbook(xlsx)
    raw.to_csv(csv_path)
    return csv_path


def load_country_epu(
    *,
    download: bool = True,
    pub_lag_months: int = 1,
    force: bool = False,
    currencies: list[str] | None = None,
) -> pd.DataFrame:
    """Monthly country EPU mapped to FX currencies (Baker–Bloom–Davis).

    Columns are ISO currencies (AUD, CAD, GBP, JPY, EUR aggregate, USD).
    EUR = equal-weight mean of France + Germany. Publication lag shifts the
    availability index forward by ``pub_lag_months`` (default 1).

    Notes
    -----
    - No Switzerland / New Zealand country EPU in the standard 22-country file
      → CHF and NZD omitted (documented gap).
    - Yellow-highlighted imputed cells in the source workbook are retained as
      published (Davis 2016 imputation) — we do not re-impute.
    """
    if download or force or not (macro_dir() / "epu_country_monthly.csv").exists():
        try:
            ensure_country_epu_csv(force=force)
        except RuntimeError:
            if not (macro_dir() / "epu_all_country.xlsx").exists():
                raise
            ensure_country_epu_csv(force=True)
    df = pd.read_csv(macro_dir() / "epu_country_monthly.csv", index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    cols: dict[str, pd.Series] = {}
    for country, ccy in COUNTRY_EPU_TO_CURRENCY.items():
        if country in df.columns:
            cols[ccy] = pd.to_numeric(df[country], errors="coerce")
    ez = []
    for country in EUR_EPU_COUNTRIES:
        if country in df.columns:
            ez.append(pd.to_numeric(df[country], errors="coerce"))
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
    panel.attrs["source"] = "baker_bloom_davis_all_country_epu"
    panel.attrs["citation"] = "Baker, Bloom & Davis (2016); Davis (2016) GEPU; country pages on policyuncertainty.com"
    panel.attrs["url"] = "https://www.policyuncertainty.com/all_country_data.html"
    panel.attrs["pub_lag_months"] = int(pub_lag_months)
    panel.attrs["eur_constituents"] = list(EUR_EPU_COUNTRIES)
    panel.attrs["missing_currencies"] = ["NZD", "CHF"]
    return panel


def load_epu_tpu_bundle(
    *,
    download: bool = True,
    pub_lag_months: int = 1,
    force: bool = False,
) -> dict[str, pd.Series | pd.DataFrame]:
    """Convenience: US EPU, GEPU, TPU, country EPU panel with shared pub lag."""
    us = load_epu(series="US", download=download, pub_lag_months=pub_lag_months, force=force)
    ge = load_epu(series="GEPU", download=download, pub_lag_months=pub_lag_months, force=force)
    tpu = load_tpu(download=download, pub_lag_months=pub_lag_months, force=force)
    country = load_country_epu(download=download, pub_lag_months=pub_lag_months, force=force)
    return {"US_EPU": us, "GEPU": ge, "TPU": tpu, "country_epu": country}
