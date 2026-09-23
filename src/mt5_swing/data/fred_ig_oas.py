"""Point-in-time FRED ICE BofA OAS (credit risk-appetite) loaders.

Literature
----------
Corporate credit OAS as a risk-appetite / intermediary-balance-sheet state
variable for FX — related to Brunnermeier–Nagel–Pedersen (2008) carry crashes
and Menkhoff et al. (2012a) risk-off channels. ICE BofA IG OAS is a *corporate*
credit risk premium distinct from Moody's ``BAA10Y`` (already in funding-liq
§20), NFCI, VIX, GPR, FX-RV, and EPU.

Free FRED series
----------------
- ``BAMLC0A0CM`` — ICE BofA US Corporate Index Option-Adjusted Spread (IG OAS).
- ``BAMLH0A0HYM2`` — ICE BofA US High Yield Index OAS (HY OAS companion).
- ``BAMLC0A4CBBB`` — ICE BofA BBB US Corporate Index OAS (optional companion).

**Public CSV truncation:** FRED's free ``fredgraph.csv`` endpoint currently
returns only ~3 calendar years of ICE BofA OAS (ICE licensing). Longer history
requires a FRED API key / vendor. Wave §36 uses the free public window honestly.

PIT lags (frozen — not holdout-tuned)
-------------------------------------
Daily OAS: ``pub_lag_days=1`` (FRED typically posts prior business day).

Do **not** overlay these coolers onto the locked FTMO sleeve; evaluate as
standalone scholarly FX factors.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series, macro_dir

FRED_IG_OAS_DAILY: dict[str, str] = {
    "IG_OAS": "BAMLC0A0CM",
    "HY_OAS": "BAMLH0A0HYM2",
    "BBB_OAS": "BAMLC0A4CBBB",
}

DEFAULT_DAILY_PUB_LAG_DAYS = 1


def _apply_pub_lag(s: pd.Series, pub_lag_days: int) -> pd.Series:
    out = s.copy()
    if pub_lag_days > 0:
        out.index = out.index + pd.Timedelta(days=int(pub_lag_days))
        out = out[~out.index.duplicated(keep="last")].sort_index()
    out.attrs["pub_lag_days"] = int(pub_lag_days)
    return out


def load_oas_series(
    series_id: str = "BAMLC0A0CM",
    *,
    download: bool = True,
    pub_lag_days: int = DEFAULT_DAILY_PUB_LAG_DAYS,
    force: bool = False,
) -> pd.Series:
    """Load one ICE BofA OAS series with publication lag."""
    sid = series_id.upper()
    s = load_fred_series(sid, download=download, force=force)
    s = _apply_pub_lag(s, pub_lag_days)
    s.name = sid
    s.attrs["source"] = f"fred_{sid}"
    s.attrs["frequency"] = "daily"
    s.attrs["family"] = "ice_bofa_oas"
    return s


def oas_change(s: pd.Series) -> pd.Series:
    """First difference of OAS (native daily frequency after pub lag)."""
    out = s.diff()
    out.name = f"d_{s.name}" if s.name else "d_oas"
    out.attrs.update({k: v for k, v in s.attrs.items() if k != "source"})
    out.attrs["source"] = s.attrs.get("source", "") + "_chg"
    return out


def load_ig_oas_bundle(
    *,
    download: bool = True,
    daily_pub_lag_days: int = DEFAULT_DAILY_PUB_LAG_DAYS,
    force: bool = False,
    include_hy: bool = True,
    include_bbb: bool = True,
) -> dict[str, pd.Series]:
    """Download/load IG OAS (+ optional HY/BBB) and ΔIG into a named dict."""
    out: dict[str, pd.Series] = {}
    ig = load_oas_series(
        "BAMLC0A0CM",
        download=download,
        pub_lag_days=daily_pub_lag_days,
        force=force,
    )
    out["IG_OAS"] = ig
    out["dIG_OAS"] = oas_change(ig)

    companions: list[tuple[str, str, bool]] = [
        ("HY_OAS", "BAMLH0A0HYM2", include_hy),
        ("BBB_OAS", "BAMLC0A4CBBB", include_bbb),
    ]
    for key, sid, want in companions:
        if not want:
            continue
        try:
            out[key] = load_oas_series(
                sid,
                download=download,
                pub_lag_days=daily_pub_lag_days,
                force=force,
            )
        except Exception as exc:  # noqa: BLE001
            out[f"_{key}_error"] = pd.Series({"error": str(exc)})

    out["_meta"] = pd.Series(
        {
            "daily_pub_lag_days": daily_pub_lag_days,
            "macro_dir": str(macro_dir()),
            "ig_start": str(ig.index.min().date()) if len(ig) else None,
            "ig_end": str(ig.index.max().date()) if len(ig) else None,
            "ig_n": int(len(ig)),
            "public_csv_note": (
                "FRED fredgraph.csv ICE BofA OAS truncated to ~3y without API key"
            ),
        }
    )
    return out


def ensure_ig_oas_cached(
    *,
    force: bool = False,
    series_ids: Iterable[str] | None = None,
) -> dict[str, str]:
    """Ensure FRED CSVs exist under ``data/macro/``; return series_id → path."""
    ids = list(series_ids) if series_ids is not None else list(FRED_IG_OAS_DAILY.values())
    paths: dict[str, str] = {}
    for sid in ids:
        p = download_fred_series(sid, force=force)
        paths[sid] = str(p)
    return paths
