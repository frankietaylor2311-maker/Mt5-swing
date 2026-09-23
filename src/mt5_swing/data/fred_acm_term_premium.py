"""Point-in-time FRED NY Fed ACM Treasury term-premium loaders.

Literature
----------
Adrian, Crump & Moench (ACM) Treasury term premium: elevated term premium =
compensation for duration / interest-rate risk → USD risk-appetite /
risk-off channel (safe-haven USD when TP high). Related: Lustig–Stathopoulos–
Verdelhan term structure of carry; Hofmann–Shim–Shin bond risk premia / FX.

Distinct from §16 yield-curve slope (IRLTLT−IRSTCI), §23 real-rate/TIPS
(``DFII10``/``T10YIE``), §20 funding-liq, §36 IG OAS.

Free FRED series
----------------
- ``THREEFYTP10`` — ACM 10-year Treasury term premium (primary; ~1990–present).
- ``THREEFYTP5`` — ACM 5-year Treasury term premium (optional companion).
- ``ACMTP10`` — may 404 / HTML; skip if unavailable.

PIT lags (frozen — not holdout-tuned)
-------------------------------------
Daily ACM: ``pub_lag_days=1`` (FRED typically posts prior business day).

Do **not** overlay these coolers onto the locked FTMO sleeve; evaluate as
standalone scholarly FX factors.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series, macro_dir

FRED_ACM_TP_DAILY: dict[str, str] = {
    "ACM_TP10": "THREEFYTP10",
    "ACM_TP05": "THREEFYTP5",
}

# Optional alts that often 404 — attempted only when requested
FRED_ACM_TP_OPTIONAL: dict[str, str] = {
    "ACMTP10": "ACMTP10",
    "ACMTP05": "ACMTP05",
}

DEFAULT_DAILY_PUB_LAG_DAYS = 1


def _apply_pub_lag(s: pd.Series, pub_lag_days: int) -> pd.Series:
    out = s.copy()
    if pub_lag_days > 0:
        out.index = out.index + pd.Timedelta(days=int(pub_lag_days))
        out = out[~out.index.duplicated(keep="last")].sort_index()
    out.attrs["pub_lag_days"] = int(pub_lag_days)
    return out


def load_acm_tp_series(
    series_id: str = "THREEFYTP10",
    *,
    download: bool = True,
    pub_lag_days: int = DEFAULT_DAILY_PUB_LAG_DAYS,
    force: bool = False,
) -> pd.Series:
    """Load one ACM term-premium series with publication lag."""
    sid = series_id.upper()
    s = load_fred_series(sid, download=download, force=force)
    s = _apply_pub_lag(s, pub_lag_days)
    s.name = sid
    s.attrs["source"] = f"fred_{sid}"
    s.attrs["frequency"] = "daily"
    s.attrs["family"] = "acm_term_premium"
    return s


def tp_change(s: pd.Series) -> pd.Series:
    """First difference of term premium (native daily frequency after pub lag)."""
    out = s.diff()
    out.name = f"d_{s.name}" if s.name else "d_acm_tp"
    out.attrs.update({k: v for k, v in s.attrs.items() if k != "source"})
    out.attrs["source"] = s.attrs.get("source", "") + "_chg"
    return out


def load_acm_tp_bundle(
    *,
    download: bool = True,
    daily_pub_lag_days: int = DEFAULT_DAILY_PUB_LAG_DAYS,
    force: bool = False,
    include_tp5: bool = True,
    try_acmtp10: bool = False,
) -> dict[str, pd.Series]:
    """Download/load ACM 10y TP (+ optional 5y) and ΔTP into a named dict."""
    out: dict[str, pd.Series] = {}
    tp10 = load_acm_tp_series(
        "THREEFYTP10",
        download=download,
        pub_lag_days=daily_pub_lag_days,
        force=force,
    )
    out["ACM_TP10"] = tp10
    out["dACM_TP10"] = tp_change(tp10)

    if include_tp5:
        try:
            out["ACM_TP05"] = load_acm_tp_series(
                "THREEFYTP5",
                download=download,
                pub_lag_days=daily_pub_lag_days,
                force=force,
            )
        except Exception as exc:  # noqa: BLE001
            out["_ACM_TP05_error"] = pd.Series({"error": str(exc)})

    if try_acmtp10:
        try:
            out["ACMTP10"] = load_acm_tp_series(
                "ACMTP10",
                download=download,
                pub_lag_days=daily_pub_lag_days,
                force=force,
            )
        except Exception as exc:  # noqa: BLE001
            out["_ACMTP10_error"] = pd.Series({"error": str(exc)})

    out["_meta"] = pd.Series(
        {
            "daily_pub_lag_days": daily_pub_lag_days,
            "macro_dir": str(macro_dir()),
            "tp10_start": str(tp10.index.min().date()) if len(tp10) else None,
            "tp10_end": str(tp10.index.max().date()) if len(tp10) else None,
            "tp10_n": int(len(tp10)),
            "public_csv_note": (
                "FRED fredgraph.csv THREEFYTP10 ACM 10y term premium ~1990–present live"
            ),
        }
    )
    return out


def ensure_acm_tp_cached(
    *,
    force: bool = False,
    series_ids: Iterable[str] | None = None,
) -> dict[str, str]:
    """Ensure FRED CSVs exist under ``data/macro/``; return series_id → path."""
    ids = list(series_ids) if series_ids is not None else list(FRED_ACM_TP_DAILY.values())
    paths: dict[str, str] = {}
    for sid in ids:
        p = download_fred_series(sid, force=force)
        paths[sid] = str(p)
    return paths
