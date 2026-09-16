"""
Point-in-time indicators.

All rolling calculations use only past and current bar data (no center=True).
``lag(n)`` shifts features by n bars so strategies never see the bar they trade on
when ``signal_lag=1`` (default): decision at close of bar t executes at open of t+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def lag(series: pd.Series, n: int = 1) -> pd.Series:
    """Shift forward in time by n bars (introduces NaN at start). Explicit anti-lookahead."""
    if n < 0:
        raise ValueError("lag must be >= 0; negative lag would look ahead")
    if n == 0:
        return series.copy()
    return series.shift(n)


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window, min_periods=window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average Directional Index (Wilder-style smoothing approx via EWM)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = atr(high, low, close, window=1)  # raw TR series via 1-bar ATR path
    # Rebuild TR properly
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    alpha = 1 / window
    atr_s = tr.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=alpha, adjust=False, min_periods=window
    ).mean() / atr_s.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=alpha, adjust=False, min_periods=window
    ).mean() / atr_s.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=alpha, adjust=False, min_periods=window).mean()


def donchian(
    high: pd.Series, low: pd.Series, window: int = 20
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Upper/lower/mid Donchian using *prior* ``window`` bars (excludes current).

    Classic Turtle / breakout definition: break of the previous N-bar high/low.
    Including the current bar made ``close > upper`` almost impossible.
    """
    prior_high = high.shift(1)
    prior_low = low.shift(1)
    upper = prior_high.rolling(window=window, min_periods=window).max()
    lower = prior_low.rolling(window=window, min_periods=window).min()
    mid = (upper + lower) / 2.0
    return upper, lower, mid


def apply_feature_pipeline(
    df: pd.DataFrame,
    *,
    signal_lag: int = 1,
    sma_fast: int = 20,
    sma_slow: int = 50,
    atr_window: int = 14,
    adx_window: int = 14,
    rsi_window: int = 14,
    donchian_window: int = 20,
) -> pd.DataFrame:
    """
    Compute common features and apply ``signal_lag`` so strategies only see lagged values.

    Raw OHLCV columns are preserved unlagged for fill simulation; feature columns
    are suffixed and lagged.
    """
    out = df.copy()
    close, high, low = out["close"], out["high"], out["low"]
    feats = {
        "sma_fast": sma(close, sma_fast),
        "sma_slow": sma(close, sma_slow),
        "ema_fast": ema(close, sma_fast),
        "atr": atr(high, low, close, atr_window),
        "adx": adx(high, low, close, adx_window),
        "rsi": rsi(close, rsi_window),
    }
    upper, lower, mid = donchian(high, low, donchian_window)
    feats["donchian_upper"] = upper
    feats["donchian_lower"] = lower
    feats["donchian_mid"] = mid
    # Bollinger (causal rolling mean/std on close)
    bb_mid = close.rolling(window=20, min_periods=20).mean()
    bb_std = close.rolling(window=20, min_periods=20).std(ddof=0)
    feats["bb_mid"] = bb_mid
    feats["bb_upper"] = bb_mid + 2.0 * bb_std
    feats["bb_lower"] = bb_mid - 2.0 * bb_std
    # MACD-ish momentum (EMA12 - EMA26), causal
    feats["ema_12"] = ema(close, 12)
    feats["ema_26"] = ema(close, 26)
    feats["macd"] = feats["ema_12"] - feats["ema_26"]
    feats["macd_signal"] = feats["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    # Higher-TF proxy from same bar series (e.g. ~20/50 D1 on H4≈6 bars/day)
    feats["htf_sma_fast"] = sma(close, 120)
    feats["htf_sma_slow"] = sma(close, 300)

    for name, series in feats.items():
        out[name] = lag(series, signal_lag) if signal_lag else series
    # Lagged close for strategies that must not peek same-bar close vs channels
    out["signal_close"] = lag(close, signal_lag) if signal_lag else close.copy()
    out.attrs["signal_lag"] = signal_lag
    return out
