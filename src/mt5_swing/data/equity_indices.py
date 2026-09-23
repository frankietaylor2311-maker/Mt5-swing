"""G10 equity index panel for Hau–Rey style equity–FX research.

Downloads Yahoo major equity indices via yfinance into ``data/macro/``:
- US: ``^GSPC`` (S&P 500)
- EUR: ``^GDAXI`` (DAX — euro-area equity proxy)
- GBP: ``^FTSE`` (FTSE 100)
- JPY: ``^N225`` (Nikkei 225)
- CAD: ``^GSPTSE`` (S&P/TSX Composite)
- AUD: ``^AXJO`` (ASX 200)
- CHF: ``^SSMI`` (SMI) — optional; skip if download fails
- NZD: no reliable free Yahoo broad index → omitted

Point-in-time: default ``pub_lag_days=1`` (equity close treated as known next session).
Yahoo session timestamps often differ from FX D1 bar times — strategies must
calendar-date align (see ``align_daily_to_index`` in commodity_fx / equity_diff_fx).

Limitations (document honestly):
- National indices ≠ Hau–Rey (2006) country equity *portfolio flow* data.
- DAX ≠ pan-euro STOXX; ASX/TSX free proxies only.
- Yahoo / approximate_non_ftmo — not FTMO MT5.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Currency → Yahoo equity index ticker
EQUITY_TICKERS: dict[str, str] = {
    "USD": "^GSPC",
    "EUR": "^GDAXI",
    "GBP": "^FTSE",
    "JPY": "^N225",
    "CAD": "^GSPTSE",
    "AUD": "^AXJO",
    "CHF": "^SSMI",
}

# Currencies with a mapped local equity index (USD is the benchmark, not traded XS long)
EQUITY_CCYS: tuple[str, ...] = ("EUR", "GBP", "JPY", "CAD", "AUD", "CHF")


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"


def _ensure_macro() -> Path:
    d = macro_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def equity_panel_path() -> Path:
    return macro_dir() / "equity_yahoo_panel.csv"


def _download_one(ticker: str) -> pd.Series | None:
    import yfinance as yf

    try:
        raw = yf.Ticker(ticker).history(period="max", auto_adjust=True)
    except Exception as exc:  # noqa: BLE001 — network/yfinance flaky
        print(f"WARN equity download failed for {ticker}: {exc}")
        return None
    if raw is None or raw.empty or "Close" not in raw.columns:
        print(f"WARN empty equity download for {ticker}")
        return None
    s = pd.to_numeric(raw["Close"], errors="coerce").dropna()
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC")
    else:
        idx = idx.tz_localize("UTC")
    s.index = idx
    s.name = ticker
    return s


def download_equity_panel(*, force: bool = False) -> Path:
    """Download/cache Yahoo equity closes → ``data/macro/equity_yahoo_panel.csv``.

    Columns = currency codes (USD, EUR, …). Skips tickers that fail; requires USD
    plus at least 3 foreign indices. Does not invent prices.
    """
    path = equity_panel_path()
    if path.exists() and not force and path.stat().st_size > 100:
        return path

    _ensure_macro()
    series: dict[str, pd.Series] = {}
    for ccy, ticker in EQUITY_TICKERS.items():
        s = _download_one(ticker)
        if s is not None and len(s) > 50:
            series[ccy] = s
            one = _ensure_macro() / f"equity_{ccy}_{ticker.replace('^', '')}.csv"
            s.to_frame("close").to_csv(one)

    if "USD" not in series or len([c for c in series if c != "USD"]) < 3:
        if path.exists() and path.stat().st_size > 100:
            print(f"WARN equity download incomplete; keeping existing cache at {path}")
            return path
        raise RuntimeError(
            "Equity Yahoo download failed (need USD + ≥3 foreign) and no cache at "
            f"{path}. Place ^GSPC/^GDAXI/^FTSE/^N225/^GSPTSE/^AXJO closes or retry."
        )

    # Normalize each series to calendar UTC date BEFORE outer-join.
    # Otherwise session-time collisions (e.g. 05:00 vs 14:00) create duplicate
    # day rows and drop_duplicates(keep="last") silently drops other markets.
    normed: dict[str, pd.Series] = {}
    for ccy, s in series.items():
        idx = pd.DatetimeIndex(s.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        s2 = pd.Series(s.to_numpy(), index=idx.normalize(), name=ccy)
        s2 = s2[~s2.index.duplicated(keep="last")].sort_index()
        normed[ccy] = s2
    panel = pd.DataFrame(normed).sort_index()
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    panel.index.name = "date"
    panel.to_csv(path)
    meta = path.with_suffix(".meta.json")
    meta.write_text(
        json.dumps(
            {
                "source": "yfinance",
                "data_tag": "approximate_non_ftmo",
                "tickers": {k: EQUITY_TICKERS[k] for k in series},
                "missing": [k for k in EQUITY_TICKERS if k not in series],
                "columns": list(panel.columns),
                "n_rows": int(len(panel)),
                "start": str(panel.index.min()),
                "end": str(panel.index.max()),
            },
            indent=2,
        )
    )
    return path


def load_equity_panel(
    *,
    download: bool = True,
    pub_lag_days: int = 1,
    force: bool = False,
) -> pd.DataFrame:
    """Load equity close panel with optional publication lag (default 1 day).

    Index shifted forward by ``pub_lag_days`` so close of day t is first known on t+lag.
    """
    path = equity_panel_path()
    if download or not path.exists():
        try:
            download_equity_panel(force=force)
        except RuntimeError:
            if not path.exists():
                raise
    if not path.exists():
        raise FileNotFoundError(f"Equity panel missing: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True).normalize()
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.duplicated(keep="last")]

    if pub_lag_days > 0:
        df.index = df.index + pd.Timedelta(days=int(pub_lag_days))
        df.index = pd.DatetimeIndex(df.index).normalize()
        df = df[~df.index.duplicated(keep="last")].sort_index()

    meta = path.with_suffix(".meta.json")
    tickers = dict(EQUITY_TICKERS)
    missing: list[str] = []
    if meta.exists():
        try:
            m = json.loads(meta.read_text())
            tickers = m.get("tickers", tickers)
            missing = list(m.get("missing", []))
        except Exception:  # noqa: BLE001
            pass

    df.attrs["pub_lag_days"] = int(pub_lag_days)
    df.attrs["source"] = "yfinance_equity"
    df.attrs["data_tag"] = "approximate_non_ftmo"
    df.attrs["tickers"] = tickers
    df.attrs["missing"] = missing
    return df


def equity_returns(
    panel: pd.DataFrame | None = None,
    *,
    pub_lag_days: int = 1,
    download: bool = True,
) -> pd.DataFrame:
    """Simple percent returns of equity closes (after publication lag)."""
    if panel is None:
        panel = load_equity_panel(download=download, pub_lag_days=pub_lag_days)
    ret = panel.pct_change()
    ret.attrs.update(getattr(panel, "attrs", {}) or {})
    return ret
