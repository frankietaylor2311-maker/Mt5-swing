"""FRED macro differentials for Dahlquist-style FX sorts (free, PIT-lagged).

Literature framing: Dahlquist & Hasseltoft (and related macro–FX work) link
currency excess returns to cross-country macro differentials — inflation,
activity (industrial production), and labour-market slack — versus the US.

We use freely downloadable FRED series only. Where a clean G10 series is
missing (e.g. NZ unemployment, AU industrial production on FRED), that
currency is omitted from that factor and documented in coverage attrs.

Publication lags (conservative, fixed a priori — not holdout-tuned):
- CPI / HICP: 1 month
- Industrial production: 2 months
- Unemployment rate: 1 month

Differentials are foreign minus US. Sign conventions for FX sorts are applied
in ``strategies.macro_diff_fx`` (high relative inflation → short foreign, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# CPI / price indexes (YoY computed when series is an index level)
CPI_SERIES: dict[str, tuple[str, str]] = {
    # currency -> (fred_id, kind)  kind: yoy | index | quarterly_index
    "USD": ("CPIAUCSL", "index"),
    "EUR": ("CP0000EZ19M086NEST", "index"),
    "GBP": ("GBRCPIALLMINMEI", "index"),
    "JPY": ("JPNCPIALLMINMEI", "index"),
    "CAD": ("CANCPIALLMINMEI", "index"),
    "CHF": ("CHECPIALLMINMEI", "index"),
    "AUD": ("AUSCPIALLQINMEI", "quarterly_index"),
    "NZD": ("NZLCPIALLQINMEI", "quarterly_index"),
}

# Industrial production indexes (YoY from level)
IP_SERIES: dict[str, tuple[str, str]] = {
    "USD": ("INDPRO", "index"),
    "EUR": ("EA19PRMNTO01IXOBSAM", "index"),  # short / ends 2023 — documented
    "GBP": ("GBRPROINDMISMEI", "index"),
    "JPY": ("JPNPROINDMISMEI", "index"),
    "CAD": ("CANPROINDMISMEI", "index"),
    "CHF": ("DEUPROINDMISMEI", "proxy_deu"),  # no clean CHE IP on FRED — omit in loader
    # AUD / NZD: no reliable FRED IP → omitted
}

# Override: drop CHF proxy — better to omit than use DEU
IP_SERIES.pop("CHF", None)

UR_SERIES: dict[str, str] = {
    "USD": "LRHUTTTTUSM156S",
    "EUR": "LRHUTTTTEZM156S",
    "GBP": "LRHUTTTTGBM156S",
    "JPY": "LRHUTTTTJPM156S",
    "AUD": "LRHUTTTTAUM156S",
    "CAD": "LRHUTTTTCAM156S",
    "CHF": "LMUNRRTTCHM156S",  # registered unemployment rate, Switzerland
    # NZD missing on FRED
}

DEFAULT_PUB_LAGS = {
    "cpi": 1,
    "ip": 2,
    "ur": 1,
}


def _to_month_start(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = s.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC")
    else:
        idx = idx.tz_localize("UTC")
    # month start
    s = s.copy()
    s.index = pd.DatetimeIndex(idx).to_period("M").to_timestamp().tz_localize("UTC")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _yoy_from_index(s: pd.Series, *, periods: int = 12) -> pd.Series:
    s = _to_month_start(s)
    return s.pct_change(periods=periods) * 100.0


def _load_macro_level(
    series_id: str,
    *,
    kind: str,
    download: bool,
) -> pd.Series:
    if download:
        download_fred_series(series_id, force=False)
    raw = load_fred_series(series_id, download=False)
    if kind == "yoy":
        return _to_month_start(raw)
    if kind == "quarterly_index":
        # forward-fill quarters to monthly then YoY over 4 quarters ≈ 12 months
        s = _to_month_start(raw).resample("MS").ffill()
        s.index = s.index.tz_localize("UTC") if s.index.tz is None else s.index
        return s.pct_change(12) * 100.0
    # index → YoY
    return _yoy_from_index(raw, periods=12)


def load_cpi_yoy_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAGS["cpi"],
    download: bool = True,
) -> pd.DataFrame:
    """YoY inflation (%). Columns = currencies. Publication lag applied."""
    curs = [c.upper() for c in (currencies or CPI_SERIES.keys())]
    cols = {}
    notes = {}
    for c in curs:
        meta = CPI_SERIES.get(c)
        if not meta:
            notes[c] = "unmapped"
            continue
        sid, kind = meta
        try:
            cols[c] = _load_macro_level(sid, kind=kind, download=download)
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]
    df = pd.DataFrame(cols).sort_index()
    if pub_lag_months > 0:
        df.index = df.index + pd.DateOffset(months=int(pub_lag_months))
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df.attrs["factor"] = "cpi_yoy"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = {c: CPI_SERIES[c][0] for c in df.columns if c in CPI_SERIES}
    df.attrs["notes"] = notes
    return df


def load_ip_yoy_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAGS["ip"],
    download: bool = True,
) -> pd.DataFrame:
    """YoY industrial production (%). Sparse for AUD/NZD/CHF (documented)."""
    curs = [c.upper() for c in (currencies or list(IP_SERIES.keys()) + ["AUD", "NZD", "CHF"])]
    cols = {}
    notes = {}
    for c in curs:
        meta = IP_SERIES.get(c)
        if not meta:
            notes[c] = "unmapped_or_unavailable_on_fred"
            continue
        sid, kind = meta
        try:
            cols[c] = _load_macro_level(sid, kind=kind, download=download)
            if c == "EUR":
                notes[c] = "EA19 series ends ~2023-10 on FRED — sparse tail"
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]
    df = pd.DataFrame(cols).sort_index()
    if pub_lag_months > 0:
        df.index = df.index + pd.DateOffset(months=int(pub_lag_months))
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df.attrs["factor"] = "ip_yoy"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = {c: IP_SERIES[c][0] for c in df.columns if c in IP_SERIES}
    df.attrs["notes"] = notes
    df.attrs["missing"] = [c for c in ("AUD", "NZD", "CHF") if c not in df.columns]
    return df


def load_ur_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAGS["ur"],
    download: bool = True,
) -> pd.DataFrame:
    """Unemployment rate level (%). NZD omitted (no FRED series)."""
    curs = [c.upper() for c in (currencies or list(UR_SERIES.keys()) + ["NZD"])]
    cols = {}
    notes = {}
    for c in curs:
        sid = UR_SERIES.get(c)
        if not sid:
            notes[c] = "unmapped_or_unavailable_on_fred"
            continue
        try:
            if download:
                download_fred_series(sid, force=False)
            cols[c] = _to_month_start(load_fred_series(sid, download=False))
            if c == "CHF":
                notes[c] = "LMUNRRTTCHM156S registered UR (not LRHUTTTT*)"
            if c == "EUR":
                notes[c] = "EA UR ends ~2023-01 on FRED — sparse tail"
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]
    df = pd.DataFrame(cols).sort_index()
    if pub_lag_months > 0:
        df.index = df.index + pd.DateOffset(months=int(pub_lag_months))
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df.attrs["factor"] = "unemployment"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = {c: UR_SERIES[c] for c in df.columns if c in UR_SERIES}
    df.attrs["notes"] = notes
    df.attrs["missing"] = ["NZD"]
    return df


def macro_differentials_vs_usd(panel: pd.DataFrame) -> pd.DataFrame:
    """foreign − USD for each non-USD column."""
    if "USD" not in panel.columns:
        raise ValueError("panel must include USD")
    usd = panel["USD"]
    out = panel.drop(columns=["USD"]).sub(usd, axis=0)
    out.attrs.update(getattr(panel, "attrs", {}))
    out.attrs["definition"] = "foreign_minus_usd"
    return out


def macro_diff_coverage() -> pd.DataFrame:
    """One-row-per-series coverage table for the research doc."""
    rows = []
    for factor, mapping, loader in (
        ("cpi_yoy", CPI_SERIES, load_cpi_yoy_panel),
        ("ip_yoy", IP_SERIES, load_ip_yoy_panel),
        ("unemployment", {k: (v, "level") for k, v in UR_SERIES.items()}, load_ur_panel),
    ):
        try:
            panel = loader(download=False)
        except Exception:
            panel = loader(download=True)
        notes = dict(panel.attrs.get("notes", {}))
        for ccy in sorted(set(list(mapping.keys()) + ["AUD", "NZD", "CHF", "EUR"])):
            if ccy in panel.columns:
                s = panel[ccy].dropna()
                rows.append(
                    {
                        "factor": factor,
                        "currency": ccy,
                        "series_id": panel.attrs.get("series_map", {}).get(ccy),
                        "n_obs": int(len(s)),
                        "start": str(s.index.min().date()) if len(s) else None,
                        "end": str(s.index.max().date()) if len(s) else None,
                        "pub_lag_months": panel.attrs.get("pub_lag_months"),
                        "status": "ok",
                        "note": notes.get(ccy, ""),
                    }
                )
            else:
                rows.append(
                    {
                        "factor": factor,
                        "currency": ccy,
                        "series_id": None,
                        "n_obs": 0,
                        "start": None,
                        "end": None,
                        "pub_lag_months": DEFAULT_PUB_LAGS.get(factor.split("_")[0], 1),
                        "status": "missing",
                        "note": notes.get(ccy, "unavailable"),
                    }
                )
    return pd.DataFrame(rows)


def _load_cpi_index_level(
    series_id: str,
    *,
    kind: str,
    download: bool,
) -> pd.Series:
    """Load CPI as an index *level* (not YoY). Quarterly series are ffilled to MS."""
    if download:
        download_fred_series(series_id, force=False)
    raw = load_fred_series(series_id, download=False)
    if kind == "quarterly_index":
        s = _to_month_start(raw).resample("MS").ffill()
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        return s
    if kind == "yoy":
        raise ValueError(f"{series_id} is YoY-only; cannot build CPI level")
    return _to_month_start(raw)


def load_cpi_level_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAGS["cpi"],
    download: bool = True,
) -> pd.DataFrame:
    """CPI / HICP *index levels* for real FX construction. Publication lag applied.

    Rebased columns are **not** required — real FX uses ratios CPI_US / CPI_f,
    so any common index base cancels. Quarterly AU/NZ series are forward-filled
    to month-start (documented limitation vs true monthly CPI).
    """
    curs = [c.upper() for c in (currencies or CPI_SERIES.keys())]
    cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    for c in curs:
        meta = CPI_SERIES.get(c)
        if not meta:
            notes[c] = "unmapped"
            continue
        sid, kind = meta
        try:
            cols[c] = _load_cpi_index_level(sid, kind=kind, download=download)
            if kind == "quarterly_index":
                notes[c] = "quarterly index ffilled to monthly"
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]
    df = pd.DataFrame(cols).sort_index()
    if pub_lag_months > 0:
        df.index = df.index + pd.DateOffset(months=int(pub_lag_months))
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df.attrs["factor"] = "cpi_level"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = {c: CPI_SERIES[c][0] for c in df.columns if c in CPI_SERIES}
    df.attrs["notes"] = notes
    return df


# ---------------------------------------------------------------------------
# Industrial production *levels* for Balassa–Samuelson / productivity work
# ---------------------------------------------------------------------------

def _load_ip_index_level(
    series_id: str,
    *,
    kind: str,
    download: bool,
) -> pd.Series:
    """Load IP as an index *level* (not YoY)."""
    if download:
        download_fred_series(series_id, force=False)
    raw = load_fred_series(series_id, download=False)
    if kind == "quarterly_index":
        s = _to_month_start(raw).resample("MS").ffill()
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        return s
    return _to_month_start(raw)


def load_ip_level_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAGS["ip"],
    download: bool = True,
) -> pd.DataFrame:
    """Industrial production *index levels* for relative productivity (BS).

    Publication lag default = 2 months (same as IP YoY). AUD/NZD/CHF omitted
    when unmapped on FRED (documented in attrs).
    """
    curs = [c.upper() for c in (currencies or list(IP_SERIES.keys()) + ["AUD", "NZD", "CHF"])]
    cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    for c in curs:
        meta = IP_SERIES.get(c)
        if not meta:
            notes[c] = "unmapped_or_unavailable_on_fred"
            continue
        sid, kind = meta
        try:
            cols[c] = _load_ip_index_level(sid, kind=kind, download=download)
            if c == "EUR":
                notes[c] = "EA19 series ends ~2023-10 on FRED — sparse tail"
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]
    df = pd.DataFrame(cols).sort_index()
    if pub_lag_months > 0:
        df.index = df.index + pd.DateOffset(months=int(pub_lag_months))
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df.attrs["factor"] = "ip_level"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = {c: IP_SERIES[c][0] for c in df.columns if c in IP_SERIES}
    df.attrs["notes"] = notes
    df.attrs["missing"] = [c for c in ("AUD", "NZD", "CHF") if c not in df.columns]
    return df


def relative_productivity_vs_usd(ip_levels: pd.DataFrame) -> pd.DataFrame:
    """log(IP_f) − log(IP_US). Relative manufacturing/output productivity proxy."""
    if "USD" not in ip_levels.columns:
        raise ValueError("ip_levels must include USD")
    usd = np.log(ip_levels["USD"].replace(0.0, np.nan))
    out = pd.DataFrame(index=ip_levels.index)
    for c in ip_levels.columns:
        if c == "USD":
            continue
        out[c] = np.log(ip_levels[c].replace(0.0, np.nan)) - usd
    out.attrs.update(getattr(ip_levels, "attrs", {}))
    out.attrs["definition"] = "log_ip_f_minus_log_ip_usd"
    return out
