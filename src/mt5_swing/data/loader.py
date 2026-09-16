"""OHLC CSV loader and deterministic sample data generator."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLS = ("time", "open", "high", "low", "close")
OPTIONAL_COLS = ("volume", "spread")


def load_ohlc_csv(
    path: str | Path,
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> pd.DataFrame:
    """
    Load OHLC bars from CSV.

    Expected columns: time, open, high, low, close [, volume, spread].
    ``time`` is parsed to UTC-aware datetime and used as index (sorted ascending).
    No future bars are invented; rows with NaN OHLC are dropped.
    """
    path = Path(path)
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    rename = {}
    for need in REQUIRED_COLS:
        if need not in cols_lower:
            raise ValueError(f"CSV {path} missing required column '{need}'")
        rename[cols_lower[need]] = need
    for opt in OPTIONAL_COLS:
        if opt in cols_lower:
            rename[cols_lower[opt]] = opt
    df = df.rename(columns=rename)
    keep = [c for c in list(REQUIRED_COLS) + list(OPTIONAL_COLS) if c in df.columns]
    df = df[keep].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0
    if "spread" in df.columns:
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce").fillna(0.0)
    else:
        df["spread"] = 0.0
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    df = df.set_index("time")
    if symbol:
        df.attrs["symbol"] = symbol
    if timeframe:
        df.attrs["timeframe"] = timeframe
    return df


def generate_sample_ohlc(
    *,
    symbol: str = "EURUSD",
    timeframe: str = "H4",
    n_bars: int = 2000,
    start: str = "2018-01-01",
    seed: int = 42,
    start_price: float | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic OHLC with mild trend + mean-reversion noise.

    Deterministic given ``seed``. Suitable for offline tests and walk-forward demos.
    Not meant to mimic real market microstructure.
    """
    rng = np.random.default_rng(seed)
    freq = {"H1": "1h", "H4": "4h", "D1": "1D", "M15": "15min"}.get(timeframe.upper(), "4h")
    idx = pd.date_range(start=start, periods=n_bars, freq=freq, tz="UTC")
    if start_price is None:
        start_price = {"EURUSD": 1.10, "GBPUSD": 1.30, "USDJPY": 110.0}.get(
            symbol.upper(), 1.0
        )

    # Slow random walk with occasional regime shifts
    reversion = 0.02
    vol = 0.0015 if not symbol.upper().endswith("JPY") else 0.15
    shocks = rng.normal(0, vol, size=n_bars)
    # Inject a few drift regimes (still causal; no lookahead)
    regime = np.zeros(n_bars)
    for _ in range(5):
        a, b = sorted(rng.integers(0, n_bars, size=2))
        regime[a:b] += rng.choice([-1.0, 1.0]) * vol * 0.3

    prices = np.empty(n_bars)
    prices[0] = start_price
    for i in range(1, n_bars):
        prices[i] = prices[i - 1] + shocks[i] + regime[i] - reversion * (
            prices[i - 1] - start_price
        ) * 0.001

    # Build OHLC from close path
    noise = np.abs(rng.normal(0, vol * 0.5, size=n_bars))
    close = prices
    open_ = np.roll(close, 1)
    open_[0] = start_price
    high = np.maximum(open_, close) + noise
    low = np.minimum(open_, close) - noise
    volume = rng.integers(100, 5000, size=n_bars).astype(float)
    # Spread in price units (approx 1–2 pips for majors)
    pip = 0.01 if symbol.upper().endswith("JPY") else 0.0001
    spread = rng.uniform(1.0, 2.5, size=n_bars) * pip

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "spread": spread,
        },
        index=idx,
    )
    df.index.name = "time"
    df.attrs["symbol"] = symbol
    df.attrs["timeframe"] = timeframe
    return df


def save_ohlc_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.reset_index()
    out.to_csv(path, index=False)
    return path


def ensure_sample_data(
    data_dir: str | Path | None = None,
    symbols: Iterable[str] = ("EURUSD", "GBPUSD", "USDJPY"),
    timeframes: Iterable[str] = ("H4", "D1"),
) -> dict[str, Path]:
    """Write sample CSVs under data/sample if missing; return path map."""
    if data_dir is None:
        # Prefer package-relative then repo-relative
        here = Path(__file__).resolve()
        candidates = [
            here.parents[3] / "data" / "sample",  # repo root when editable
            Path.cwd() / "data" / "sample",
        ]
        data_dir = next((c for c in candidates if c.parent.exists() or True), candidates[0])
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for sym in symbols:
        for tf in timeframes:
            n = 2500 if tf.upper() == "H4" else 1200
            key = f"{sym}_{tf}"
            path = data_dir / f"{key}.csv"
            if not path.exists():
                seed = abs(hash(key)) % (2**31)
                df = generate_sample_ohlc(symbol=sym, timeframe=tf, n_bars=n, seed=seed)
                save_ohlc_csv(df, path)
            paths[key] = path
    return paths
