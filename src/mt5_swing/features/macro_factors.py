"""Causal macro / geopolitical factor loaders for literature-backed FX designs.

All series are aligned with an explicit publication lag so strategies never see
same-bar (or same-month) macro prints. Sources are public (FRED, Caldara–
Iacoviello GPR, Yahoo VIX). approximate_non_ftmo research only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MACRO = ROOT / "data" / "macro"

# ISO currency → FRED short-rate CSV (OECD / national overnight or policy proxies)
RATE_FILES: dict[str, str] = {
    "USD": "fred_IRSTCI01USM156N.csv",
    "AUD": "fred_IRSTCI01AUM156N.csv",
    "CAD": "fred_IRSTCI01CAM156N.csv",
    "CHF": "fred_IRSTCI01CHM156N.csv",
    "EUR": "fred_IRSTCI01EZM156N.csv",
    "GBP": "fred_IRSTCI01GBM156N.csv",
    "JPY": "fred_IRSTCI01JPM156N.csv",
    "NZD": "fred_IRSTCI01NZM156N.csv",
}

# Fallback daily policy rates when monthly OECD series lag
DAILY_RATE_FALLBACK: dict[str, str] = {
    "USD": "fred_DFF.csv",
    "EUR": "fred_ECBDFR.csv",
    "GBP": "fred_IUDSOIA.csv",
}


def _read_fred_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    date_col = df.columns[0]
    val_col = df.columns[1]
    s = df.set_index(pd.to_datetime(df[date_col], utc=True))[val_col]
    s = pd.to_numeric(s, errors="coerce").sort_index()
    s.name = path.stem
    return s.dropna()


def load_policy_rate(ccy: str, *, monthly_lag: int = 1) -> pd.Series:
    """Monthly policy / short rate with ``monthly_lag`` completed-month lag (causal)."""
    ccy = ccy.upper()
    path = MACRO / RATE_FILES[ccy]
    if not path.exists():
        raise FileNotFoundError(path)
    s = _read_fred_csv(path)
    # Month-end stamp then lag full months so August print is first usable in September
    s = s.resample("ME").last().dropna()
    return s.shift(int(monthly_lag))


def rate_differential(base: str, quote: str, *, monthly_lag: int = 1) -> pd.Series:
    """``rate(base) - rate(quote)`` with causal month lag (Lustig–Verdelhan carry sign)."""
    a = load_policy_rate(base, monthly_lag=monthly_lag)
    b = load_policy_rate(quote, monthly_lag=monthly_lag)
    return (a - b).dropna().rename(f"{base}_{quote}_diff")


def pair_currencies(symbol: str) -> tuple[str, str]:
    """Split FX symbol into (base, quote)."""
    sym = symbol.upper().replace("/", "")
    if len(sym) != 6:
        raise ValueError(f"Expected 6-letter FX symbol, got {symbol}")
    return sym[:3], sym[3:]


def load_vix(*, bar_lag: int = 1) -> pd.Series:
    """Daily VIX close with bar lag (Menkhoff et al. global FX vol risk proxy)."""
    path = MACRO / "vix_yahoo.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["Date"])
    s = pd.to_numeric(df.set_index(pd.to_datetime(df["Date"], utc=True))["close"], errors="coerce")
    return s.sort_index().dropna().shift(int(bar_lag)).rename("vix")


def load_gpr_daily(*, bar_lag: int = 1, col: str = "GPR") -> pd.Series:
    """Caldara–Iacoviello recent daily GPR (or AI-GPR) with bar lag."""
    # Prefer classic GPR daily CSV; fall back to AI-GPR
    p_classic = MACRO / "gpr_daily.csv"
    p_ai = MACRO / "ai_gpr_daily.csv"
    if p_classic.exists():
        df = pd.read_csv(p_classic)
        date_col = "date" if "date" in df.columns else df.columns[0]
        use = col if col in df.columns else ("GPR" if "GPR" in df.columns else df.columns[1])
        s = pd.to_numeric(df.set_index(pd.to_datetime(df[date_col], utc=True))[use], errors="coerce")
    elif p_ai.exists():
        df = pd.read_csv(p_ai, parse_dates=["Date"])
        use = "GPR_AI" if "GPR_AI" in df.columns else df.columns[1]
        s = pd.to_numeric(df.set_index(pd.to_datetime(df["Date"], utc=True))[use], errors="coerce")
    else:
        raise FileNotFoundError("No GPR daily CSV in data/macro/")
    return s.sort_index().dropna().shift(int(bar_lag)).rename("gpr")


def rolling_z(series: pd.Series, lookback: int = 252) -> pd.Series:
    """Causal rolling z-score (uses only past window including current lagged point)."""
    mu = series.rolling(lookback, min_periods=max(40, lookback // 5)).mean()
    sd = series.rolling(lookback, min_periods=max(40, lookback // 5)).std()
    return ((series - mu) / sd.replace(0, np.nan)).rename(f"{series.name}_z")


def align_to_index(factor: pd.Series, index: pd.DatetimeIndex, *, method: str = "ffill") -> pd.Series:
    """Reindex factor onto bar index with forward-fill (still causal if factor was lagged)."""
    f = factor.copy()
    if f.index.tz is None and index.tz is not None:
        f.index = f.index.tz_localize("UTC")
    elif f.index.tz is not None and index.tz is None:
        f.index = f.index.tz_convert("UTC").tz_localize(None)
    return f.reindex(index, method=method)
