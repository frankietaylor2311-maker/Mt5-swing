"""
Importer for MetaTrader 5 History Center / script-exported OHLC CSVs.

Production research for FTMO Challenge accounts should use **FTMO MT5 terminal
exports** (same symbols, session, and spread behaviour as the prop platform).
This module normalizes common MT5 CSV layouts into the mt5_swing OHLC schema.

Supported layouts (auto-detected):
- MT5 History Center export: ``<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>``
  (tab or comma/semicolon separated; date often ``YYYY.MM.DD``)
- Flat headers: ``time,open,high,low,close[,volume,spread,tick_volume]``
- Separate Date + Time columns

Binary ``.hst`` files are **not** parsed here — export to CSV from MT5 first
(Windows FTMO terminal). Do not attempt to run MT5 on Linux for research.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.loader import save_ohlc_csv

DATA_SOURCE_FTMO_MT5 = "ftmo_mt5_export"
DATA_SOURCE_APPROX = "approximate_non_ftmo"


def _detect_sep(sample: str) -> str:
    if "\t" in sample:
        return "\t"
    if sample.count(";") >= sample.count(","):
        return ";"
    return ","


def _normalize_columns(cols: Iterable[str]) -> list[str]:
    out = []
    for c in cols:
        s = str(c).strip().lower()
        s = s.strip("<>").strip()
        s = re.sub(r"[\s\-]+", "_", s)
        aliases = {
            "date": "date",
            "time": "time",
            "datetime": "datetime",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "tickvol": "tick_volume",
            "tick_volume": "tick_volume",
            "tick_vol": "tick_volume",
            "vol": "volume",
            "volume": "volume",
            "spread": "spread",
        }
        out.append(aliases.get(s, s))
    return out


def load_mt5_export_csv(
    path: str | Path,
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    tz: str = "UTC",
    default_spread_pips: float | None = None,
    pip_size: float | None = None,
) -> pd.DataFrame:
    """
    Load an MT5-exported OHLC CSV into a UTC-indexed DataFrame.

    Sets ``attrs['data_source'] = 'ftmo_mt5_export'`` when used for FTMO research.
    Caller should tag the path provenance; this loader marks source as FTMO-capable
    export format (not Yahoo).
    """
    path = Path(path)
    raw_head = path.read_text(encoding="utf-8", errors="replace")[:2048]
    sep = _detect_sep(raw_head.splitlines()[0] if raw_head else ",")

    # Try header row first
    df = pd.read_csv(path, sep=sep)
    cols = _normalize_columns(df.columns)
    df.columns = cols

    # Headerless MT5 dump: DATE TIME OPEN HIGH LOW CLOSE TICKVOL VOL SPREAD
    headerish = any(c in cols for c in ("open", "close", "high", "low"))
    if not headerish:
        df = pd.read_csv(
            path,
            sep=sep,
            header=None,
            names=[
                "date",
                "time",
                "open",
                "high",
                "low",
                "close",
                "tick_volume",
                "volume",
                "spread",
            ],
        )
        cols = list(df.columns)

    if "datetime" in df.columns:
        ts = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    elif "date" in df.columns and "time" in df.columns:
        combined = df["date"].astype(str).str.replace(".", "-", regex=False) + " " + df["time"].astype(str)
        ts = pd.to_datetime(combined, utc=True, errors="coerce")
    elif "time" in df.columns:
        # Single time column may be full datetime or unix
        tcol = df["time"]
        if pd.api.types.is_numeric_dtype(tcol):
            ts = pd.to_datetime(tcol, unit="s", utc=True, errors="coerce")
        else:
            cleaned = tcol.astype(str).str.replace(".", "-", regex=False)
            ts = pd.to_datetime(cleaned, utc=True, errors="coerce")
    elif "date" in df.columns:
        cleaned = df["date"].astype(str).str.replace(".", "-", regex=False)
        ts = pd.to_datetime(cleaned, utc=True, errors="coerce")
    else:
        raise ValueError(f"MT5 CSV {path} missing date/time columns; got {list(df.columns)}")

    need = {"open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"MT5 CSV {path} missing OHLC; columns={list(df.columns)}")

    # Use numpy values so constructing with a new DatetimeIndex does not reindex-align to NaN
    out = pd.DataFrame(
        {
            "open": pd.to_numeric(df["open"], errors="coerce").to_numpy(),
            "high": pd.to_numeric(df["high"], errors="coerce").to_numpy(),
            "low": pd.to_numeric(df["low"], errors="coerce").to_numpy(),
            "close": pd.to_numeric(df["close"], errors="coerce").to_numpy(),
        },
        index=pd.DatetimeIndex(ts),
    )
    if "volume" in df.columns:
        out["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).to_numpy()
    elif "tick_volume" in df.columns:
        out["volume"] = pd.to_numeric(df["tick_volume"], errors="coerce").fillna(0.0).to_numpy()
    else:
        out["volume"] = 0.0

    if "spread" in df.columns:
        # MT5 spread is usually in points (integer)
        spread_pts = pd.to_numeric(df["spread"], errors="coerce")
        sym = (symbol or "").upper()
        pip = pip_size
        if pip is None:
            pip = 0.01 if sym.endswith("JPY") or sym.startswith("XAU") else 0.0001
            if sym.startswith("XAU"):
                pip = 0.01
            if sym.startswith("XAG"):
                pip = 0.001
        # points → price: for 5-digit FX, 1 point = 0.1 pip = pip/10; MT5 digits vary.
        # Store price units as spread_points * point. Use pip/10 as point for majors.
        point = pip / 10.0 if pip <= 0.01 and not sym.startswith("XAU") else pip / 10.0
        if sym.startswith(("XAU", "XAG", "US30", "US100", "US500", "GER40", "UK100")):
            point = pip  # treat exported spread as already near price step
        out["spread"] = (spread_pts.fillna(0.0).to_numpy() * point).astype(float)
    else:
        pip = pip_size or (0.01 if (symbol or "").upper().endswith("JPY") else 0.0001)
        pips = default_spread_pips if default_spread_pips is not None else 1.2
        out["spread"] = float(pips) * float(pip)

    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[~out.index.isna()]
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "time"
    if out.index.tz is None:
        out.index = out.index.tz_localize(tz)
    else:
        out.index = out.index.tz_convert("UTC")

    if symbol:
        out.attrs["symbol"] = symbol.upper()
    if timeframe:
        out.attrs["timeframe"] = timeframe.upper()
    out.attrs["data_source"] = DATA_SOURCE_FTMO_MT5
    out.attrs["source"] = "mt5_export_csv"
    return out


def import_mt5_exports(
    files: dict[str, str | Path],
    out_dir: str | Path,
    *,
    default_spreads: dict[str, float] | None = None,
) -> dict[str, Path]:
    """
    Import a mapping of ``SYMBOL_TF -> csv_path`` into ``out_dir`` as normalized CSVs.

    Example keys: ``EURUSD_H4``, ``XAUUSD_D1``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    default_spreads = default_spreads or {}
    paths: dict[str, Path] = {}
    for key, src in files.items():
        parts = key.rsplit("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Key must be SYMBOL_TF, got {key}")
        sym, tf = parts[0], parts[1]
        df = load_mt5_export_csv(
            src,
            symbol=sym,
            timeframe=tf,
            default_spread_pips=default_spreads.get(sym),
        )
        dest = out_dir / f"{sym}_{tf}.csv"
        save_ohlc_csv(df, dest)
        # Sidecar metadata for reports
        meta = dest.with_suffix(".meta.json")
        meta.write_text(
            '{"data_source":"%s","symbol":"%s","timeframe":"%s","bars":%d,'
            '"start":"%s","end":"%s","source_file":"%s"}\n'
            % (
                DATA_SOURCE_FTMO_MT5,
                sym,
                tf,
                len(df),
                df.index.min(),
                df.index.max(),
                Path(src).name,
            ),
            encoding="utf-8",
        )
        paths[key] = dest
        print(f"Imported FTMO/MT5 {key}: {len(df)} bars → {dest}")
    return paths
