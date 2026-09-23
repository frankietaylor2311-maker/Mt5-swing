"""Point-in-time FRED funding-liquidity / financial-conditions loaders.

Literature
----------
Brunnermeier, Nagel & Pedersen (2008), "Carry Trades and Currency Crashes,"
*NBER / RFS*: funding-liquidity spirals and crowded carry unwind in stress.
Menkhoff, Sarno, Schmeling & Schrimpf (2012a) related FX-vol / risk-off channel.
Chicago Fed NFCI / ANFCI summarize U.S. money, debt, and equity market conditions;
TED / commercial-paper–Treasury spreads proxy interbank / CP funding stress.

Free FRED series (no paid terminals)
------------------------------------
- ``NFCI`` / ``ANFCI`` — weekly Chicago Fed National / Adjusted Financial
  Conditions Index (0 ≈ average; >0 tighter).
- ``NFCIRISK``, ``NFCICREDIT``, ``NFCILEVERAGE`` — NFCI subindices.
- ``TEDRATE`` — TED spread (daily; ends ~2022-01 after LIBOR sunset).
- ``CPFF`` — 3M AA financial CP − 3M T-bill (daily post-LIBOR funding proxy).
- ``BAA10Y`` — Moody's Baa − 10Y Treasury (credit / risk-premium proxy).

PIT lags (conservative, frozen — not holdout-tuned)
---------------------------------------------------
- Weekly NFCI family: Chicago Fed typically releases mid-week for a Friday-ending
  observation week → ``pub_lag_days=7`` (one calendar week).
- Daily TED / CPFF / BAA: ``pub_lag_days=1``.

Do **not** overlay these coolers onto the locked FTMO sleeve; they are evaluated
as standalone scholarly FX factors.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series, macro_dir

# Weekly financial-conditions / funding state
FRED_NFCI_WEEKLY: dict[str, str] = {
    "NFCI": "NFCI",
    "ANFCI": "ANFCI",
    "NFCIRISK": "NFCIRISK",
    "NFCICREDIT": "NFCICREDIT",
    "NFCILEVERAGE": "NFCILEVERAGE",
}

# Daily funding / credit spreads
FRED_FUNDING_DAILY: dict[str, str] = {
    "TEDRATE": "TEDRATE",  # ends 2022-01
    "CPFF": "CPFF",  # CP − Tbill; continues post-LIBOR
    "BAA10Y": "BAA10Y",  # Moody's Baa − 10Y
}

DEFAULT_WEEKLY_PUB_LAG_DAYS = 7
DEFAULT_DAILY_PUB_LAG_DAYS = 1


def _apply_pub_lag(s: pd.Series, pub_lag_days: int) -> pd.Series:
    out = s.copy()
    if pub_lag_days > 0:
        out.index = out.index + pd.Timedelta(days=int(pub_lag_days))
        out = out[~out.index.duplicated(keep="last")].sort_index()
    out.attrs["pub_lag_days"] = int(pub_lag_days)
    return out


def load_nfci_series(
    series_id: str = "NFCI",
    *,
    download: bool = True,
    pub_lag_days: int = DEFAULT_WEEKLY_PUB_LAG_DAYS,
    force: bool = False,
) -> pd.Series:
    """Load one NFCI-family weekly series with publication lag."""
    sid = series_id.upper()
    if sid not in FRED_NFCI_WEEKLY and sid not in set(FRED_NFCI_WEEKLY.values()):
        # Allow raw FRED id passthrough for documented family members
        pass
    s = load_fred_series(sid, download=download, force=force)
    s = _apply_pub_lag(s, pub_lag_days)
    s.name = sid
    s.attrs["source"] = f"fred_{sid}"
    s.attrs["frequency"] = "weekly"
    return s


def load_funding_spread(
    series_id: str = "CPFF",
    *,
    download: bool = True,
    pub_lag_days: int = DEFAULT_DAILY_PUB_LAG_DAYS,
    force: bool = False,
) -> pd.Series:
    """Load one daily funding / credit spread with publication lag."""
    sid = series_id.upper()
    s = load_fred_series(sid, download=download, force=force)
    s = _apply_pub_lag(s, pub_lag_days)
    s.name = sid
    s.attrs["source"] = f"fred_{sid}"
    s.attrs["frequency"] = "daily"
    return s


def load_ted_or_cpff(
    *,
    download: bool = True,
    pub_lag_days: int = DEFAULT_DAILY_PUB_LAG_DAYS,
    force: bool = False,
) -> pd.Series:
    """Composite funding spread: TED while available, else CPFF (post-LIBOR).

    Returns a single series named ``FUNDING_SPREAD``; attrs record splice date.
    """
    ted = load_funding_spread("TEDRATE", download=download, pub_lag_days=pub_lag_days, force=force)
    cpff = load_funding_spread("CPFF", download=download, pub_lag_days=pub_lag_days, force=force)
    # Prefer TED on overlapping dates (classic BNP funding proxy); CPFF fills after
    combined = ted.combine_first(cpff)
    combined.name = "FUNDING_SPREAD"
    combined.attrs["source"] = "fred_TEDRATE+CPFF"
    combined.attrs["pub_lag_days"] = int(pub_lag_days)
    combined.attrs["ted_end"] = str(ted.index.max().date()) if len(ted) else None
    combined.attrs["cpff_start"] = str(cpff.index.min().date()) if len(cpff) else None
    return combined


def weekly_change(s: pd.Series) -> pd.Series:
    """First difference on the series' native frequency (weekly for NFCI)."""
    out = s.diff()
    out.name = f"d_{s.name}" if s.name else "d_series"
    out.attrs.update({k: v for k, v in s.attrs.items() if k != "source"})
    out.attrs["source"] = s.attrs.get("source", "") + "_chg"
    return out


def load_funding_liquidity_bundle(
    *,
    download: bool = True,
    weekly_pub_lag_days: int = DEFAULT_WEEKLY_PUB_LAG_DAYS,
    daily_pub_lag_days: int = DEFAULT_DAILY_PUB_LAG_DAYS,
    force: bool = False,
    include_subindices: bool = True,
) -> dict[str, pd.Series]:
    """Download/load NFCI, ANFCI, spreads, and ΔNFCI into a named dict."""
    out: dict[str, pd.Series] = {}
    weekly_ids: Iterable[str] = ("NFCI", "ANFCI")
    if include_subindices:
        weekly_ids = ("NFCI", "ANFCI", "NFCIRISK", "NFCICREDIT", "NFCILEVERAGE")
    for sid in weekly_ids:
        out[sid] = load_nfci_series(
            sid, download=download, pub_lag_days=weekly_pub_lag_days, force=force
        )
    out["dNFCI"] = weekly_change(out["NFCI"])
    out["dANFCI"] = weekly_change(out["ANFCI"])
    out["TEDRATE"] = load_funding_spread(
        "TEDRATE", download=download, pub_lag_days=daily_pub_lag_days, force=force
    )
    out["CPFF"] = load_funding_spread(
        "CPFF", download=download, pub_lag_days=daily_pub_lag_days, force=force
    )
    out["BAA10Y"] = load_funding_spread(
        "BAA10Y", download=download, pub_lag_days=daily_pub_lag_days, force=force
    )
    out["FUNDING_SPREAD"] = load_ted_or_cpff(
        download=download, pub_lag_days=daily_pub_lag_days, force=force
    )
    out["_meta"] = pd.Series(
        {
            "weekly_pub_lag_days": weekly_pub_lag_days,
            "daily_pub_lag_days": daily_pub_lag_days,
            "macro_dir": str(macro_dir()),
        }
    )
    return out


def ensure_funding_liquidity_cached(*, force: bool = False) -> dict[str, str]:
    """Ensure FRED CSVs exist under ``data/macro/``; return series_id → path."""
    paths: dict[str, str] = {}
    for sid in set(FRED_NFCI_WEEKLY.values()) | set(FRED_FUNDING_DAILY.values()):
        p = download_fred_series(sid, force=force)
        paths[sid] = str(p)
    return paths
