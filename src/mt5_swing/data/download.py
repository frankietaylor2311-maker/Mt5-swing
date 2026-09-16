"""
Real FX OHLC downloader (yfinance) with careful closed-bar resampling.

Source: Yahoo Finance via ``yfinance`` tickers ``EURUSD=X``, ``GBPUSD=X``, ``USDJPY=X``.

Limitations (document in README):
- Intraday (1h) history is capped by Yahoo (~2–3 years). H4 is resampled from 1h
  closed bars only; incomplete trailing buckets are dropped.
- Daily (D1) history can span 10–20+ years depending on the pair.
- Yahoo FX quotes are not broker ticks: spreads/gaps differ from MT5; use for
  research only. Volume may be sparse or zero for FX.\n- **Not FTMO-ready:** tag every report using this data as ``approximate_non_ftmo``. Prefer ``data/mt5_import.py`` + FTMO MT5 exports.
- No look-ahead: resampling uses only bars fully inside each bucket (right-closed /
  left-labeled convention that drops the incomplete final period).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.loader import save_ohlc_csv

# Map research symbols → Yahoo FX tickers
YFINANCE_FX = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X",
}

DEFAULT_SPREAD_PIPS = {
    "EURUSD": 1.2,
    "GBPUSD": 1.4,
    "USDJPY": 1.2,
}


def yahoo_ticker(symbol: str) -> str:
    key = symbol.upper().replace("/", "")
    if key not in YFINANCE_FX:
        raise ValueError(f"Unsupported symbol {symbol}; known: {sorted(YFINANCE_FX)}")
    return YFINANCE_FX[key]


def _pip_size(symbol: str) -> float:
    return 0.01 if symbol.upper().endswith("JPY") else 0.0001


def _normalize_yf(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Convert yfinance OHLCV to mt5_swing schema (UTC index, lower-case cols)."""
    if df is None or df.empty:
        raise ValueError(f"Empty download for {symbol}")
    out = df.copy()
    # yfinance may return MultiIndex columns for single ticker in some versions
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Adj Close": "adj_close",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    need = {"open", "high", "low", "close"}
    if not need.issubset(out.columns):
        raise ValueError(f"Unexpected columns from yfinance: {list(out.columns)}")
    if "volume" not in out.columns:
        out["volume"] = 0.0
    # Timezone → UTC
    idx = pd.to_datetime(out.index, utc=True)
    out.index = idx
    out.index.name = "time"
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    pip = _pip_size(symbol)
    spread_pips = DEFAULT_SPREAD_PIPS.get(symbol.upper(), 1.5)
    out["spread"] = spread_pips * pip
    out.attrs["symbol"] = symbol.upper()
    out.attrs["source"] = "yfinance"
    out.attrs["data_source"] = "approximate_non_ftmo"
    return out[["open", "high", "low", "close", "volume", "spread"]]


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample to a coarser timeframe using only **closed** buckets.

    Uses ``label='left', closed='left'`` then drops the trailing incomplete bucket
    if the last source timestamp does not reach the end of that bucket.
    """
    if df.empty:
        return df.copy()
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "spread": "mean",
    }
    cols = {k: v for k, v in agg.items() if k in df.columns}
    res = df.resample(rule, label="left", closed="left").agg(cols)
    res = res.dropna(subset=["open", "high", "low", "close"])
    # Drop incomplete final bar: bucket end must be <= last source ts + epsilon
    # With closed='left', bucket [t, t+freq) is complete iff last_ts >= t+freq - 1ns
    freq = pd.tseries.frequencies.to_offset(rule)
    last_src = df.index[-1]
    if len(res):
        last_bucket_start = res.index[-1]
        bucket_end = last_bucket_start + freq
        if last_src < bucket_end - pd.Timedelta(seconds=1):
            res = res.iloc[:-1]
    res.index.name = "time"
    for k, v in df.attrs.items():
        res.attrs[k] = v
    return res


def fetch_yf_history(
    symbol: str,
    *,
    interval: str = "1h",
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Download raw bars from yfinance and normalize."""
    import yfinance as yf

    ticker = yahoo_ticker(symbol)
    kwargs: dict = {"interval": interval, "auto_adjust": True}
    if start or end:
        kwargs["start"] = start
        kwargs["end"] = end
    else:
        kwargs["period"] = period or ("730d" if interval in {"60m", "1h", "90m"} else "max")
    raw = yf.Ticker(ticker).history(**kwargs)
    return _normalize_yf(raw, symbol)


def download_symbol_timeframe(
    symbol: str,
    timeframe: str,
    *,
    years_d1: int = 10,
) -> pd.DataFrame:
    """
    Fetch real OHLC for ``symbol`` / ``timeframe``.

    - H4: download 1h (max available), resample to 4h, drop incomplete last bar.
    - D1: download 1d (period=max), keep last ``years_d1`` years.
    """
    tf = timeframe.upper()
    sym = symbol.upper().replace("/", "")
    if tf == "H4":
        h1 = fetch_yf_history(sym, interval="1h", period="730d")
        out = resample_ohlc(h1, "4h")
        out.attrs["timeframe"] = "H4"
        out.attrs["source_interval"] = "1h"
        out.attrs["data_source"] = "approximate_non_ftmo"
        return out
    if tf == "D1":
        d1 = fetch_yf_history(sym, interval="1d", period="max")
        if years_d1 and years_d1 > 0 and len(d1):
            cutoff = d1.index.max() - pd.DateOffset(years=years_d1)
            d1 = d1.loc[d1.index >= cutoff]
        d1.attrs["timeframe"] = "D1"
        d1.attrs["source_interval"] = "1d"
        d1.attrs["data_source"] = "approximate_non_ftmo"
        return d1
    if tf == "H1":
        h1 = fetch_yf_history(sym, interval="1h", period="730d")
        h1.attrs["timeframe"] = "H1"
        return h1
    raise ValueError(f"Unsupported timeframe {timeframe}; use H4, D1, or H1")


def download_history(
    symbols: Iterable[str] = ("EURUSD", "GBPUSD", "USDJPY"),
    timeframes: Iterable[str] = ("H4", "D1"),
    out_dir: str | Path | None = None,
    *,
    years_d1: int = 10,
) -> dict[str, Path]:
    """Download and save CSVs under ``data/history/``. Returns path map."""
    if out_dir is None:
        here = Path(__file__).resolve()
        out_dir = here.parents[3] / "data" / "history"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for sym in symbols:
        for tf in timeframes:
            df = download_symbol_timeframe(sym, tf, years_d1=years_d1)
            key = f"{sym.upper()}_{tf.upper()}"
            path = out_dir / f"{key}.csv"
            save_ohlc_csv(df, path)
            meta = path.with_suffix(".meta.json")
            meta.write_text(
                '{"data_source":"approximate_non_ftmo","symbol":"%s","timeframe":"%s",'
                '"bars":%d,"start":"%s","end":"%s","provider":"yfinance"}\n'
                % (sym.upper(), tf.upper(), len(df), df.index.min(), df.index.max()),
                encoding="utf-8",
            )
            paths[key] = path
            print(
                f"Saved {key} [approximate_non_ftmo]: {len(df)} bars "
                f"[{df.index.min()} → {df.index.max()}] → {path}"
            )
    return paths


def history_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "history"


def resolve_ohlc_path(
    symbol: str,
    timeframe: str,
    *,
    prefer_history: bool = True,
) -> Path | None:
    """Return path to history CSV if present, else None."""
    key = f"{symbol.upper()}_{timeframe.upper()}.csv"
    hist = history_dir() / key
    if prefer_history and hist.exists():
        return hist
    return None
