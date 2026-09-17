"""Literature-style currency momentum (Menkhoff–Sarno–Schmeling–Schrimpf 2012).

Cross-sectional FX momentum: rank currencies by past excess return over a
formation window, long winners / short losers, hold one period.

Defaults follow common academic settings (research priors — **not** holdout-tuned):
- Formation: 3 months (approx 63 trading days) or 12–1 month style
- Skip most recent month for 12-1 (``skip_days``)
- Rebalance monthly; equal-weight top/bottom ``n_long`` / ``n_short``
- ``signal_lag=1`` on the daily return panel

USD-pair mapping matches ``carry_rank.USD_PAIRS`` for FTMO-tradable majors.
Dollar factor (Lustig–Roussanov–Verdelhan): average excess return of all foreign
currencies vs USD is available as ``dollar_factor_returns``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.strategies.base import Signal
from mt5_swing.strategies.carry_rank import (
    USD_PAIRS,
    currency_weights_to_pair_weights,
    expand_weights_to_daily,
    portfolio_returns_from_weights,
)


@dataclass
class FxMomentumConfig:
    formation_days: int = 63  # ~3 months
    skip_days: int = 21  # skip most recent month (12-1 style when formation=252)
    n_long: int = 2
    n_short: int = 2
    rebalance: str = "M"
    signal_lag: int = 1
    min_periods: int = 40


def pair_returns_to_currency_returns(pair_ret: pd.DataFrame) -> pd.DataFrame:
    """Convert USD-pair simple returns → currency-vs-USD returns.

    EURUSD up → EUR up vs USD. USDJPY up → JPY down vs USD.
    """
    cols = {}
    # Invert map: currency from pairs
    for ccy, (sym, sign) in USD_PAIRS.items():
        if sym not in pair_ret.columns:
            continue
        # currency return vs USD = sign * pair return
        cols[ccy] = pair_ret[sym] * sign
    # USD is numeraire → 0
    out = pd.DataFrame(cols, index=pair_ret.index)
    out["USD"] = 0.0
    return out


def formation_momentum(
    ccy_ret: pd.DataFrame,
    *,
    formation_days: int,
    skip_days: int,
    min_periods: int,
) -> pd.DataFrame:
    """Trailing cumulative return ending ``skip_days`` ago (causal)."""
    # r_{t-skip-f+1 : t-skip}
    rolled = (
        ccy_ret.shift(int(skip_days))
        .rolling(int(formation_days), min_periods=int(min_periods))
        .apply(lambda x: np.prod(1.0 + x) - 1.0, raw=True)
    )
    return rolled


def momentum_weights_from_returns(
    pair_ret: pd.DataFrame,
    *,
    cfg: FxMomentumConfig | None = None,
) -> pd.DataFrame:
    """Month-end currency weights from cross-sectional momentum scores."""
    cfg = cfg or FxMomentumConfig()
    ccy_ret = pair_returns_to_currency_returns(pair_ret)
    score = formation_momentum(
        ccy_ret.drop(columns=["USD"], errors="ignore"),
        formation_days=cfg.formation_days,
        skip_days=cfg.skip_days,
        min_periods=cfg.min_periods,
    )
    rebal = score.resample("ME").last().dropna(how="all")
    rows = []
    for dt, row in rebal.iterrows():
        s = row.dropna().sort_values(ascending=False)
        w = pd.Series(0.0, index=score.columns)
        if len(s) >= cfg.n_long + cfg.n_short:
            longs = list(s.index[: cfg.n_long])
            shorts = list(s.index[-cfg.n_short :])
            lw = 0.5 / max(cfg.n_long, 1)
            sw = 0.5 / max(cfg.n_short, 1)
            for c in longs:
                w[c] = lw
            for c in shorts:
                w[c] = -sw
        w.name = dt
        rows.append(w)
    if not rows:
        return pd.DataFrame(columns=score.columns)
    out = pd.DataFrame(rows)
    out.index = pd.DatetimeIndex(out.index, tz="UTC")
    out.attrs["strategy"] = "fx_momentum"
    out.attrs["formation_days"] = cfg.formation_days
    out.attrs["skip_days"] = cfg.skip_days
    return out


def dollar_factor_returns(pair_ret: pd.DataFrame) -> pd.Series:
    """Average foreign-currency return vs USD (LRV dollar factor proxy)."""
    ccy = pair_returns_to_currency_returns(pair_ret)
    foreign = ccy.drop(columns=["USD"], errors="ignore")
    s = foreign.mean(axis=1)
    s.name = "dollar_factor"
    return s


def ts_momentum_pair_signal(
    pair_px: pd.Series,
    *,
    lookback: int = 63,
    skip: int = 1,
) -> pd.Series:
    """Single-pair time-series momentum Signal (sign of lagged return)."""
    # Formation ending skip bars ago (causal)
    r = pair_px.shift(skip) / pair_px.shift(skip + lookback) - 1.0
    sig = pd.Series(int(Signal.FLAT), index=pair_px.index, dtype=int)
    sig = sig.mask(r > 0, int(Signal.LONG))
    sig = sig.mask(r < 0, int(Signal.SHORT))
    return sig.astype(int)


class FxMomentumBasket:
    name = "fx_momentum"

    def __init__(self, **kwargs):
        fields = FxMomentumConfig.__dataclass_fields__
        self.cfg = FxMomentumConfig(**{k: v for k, v in kwargs.items() if k in fields})

    def weights(self, pair_ret: pd.DataFrame) -> pd.DataFrame:
        return currency_weights_to_pair_weights(momentum_weights_from_returns(pair_ret, cfg=self.cfg))

    def portfolio_returns(self, pair_ret: pd.DataFrame) -> pd.Series:
        ccy_w = momentum_weights_from_returns(pair_ret, cfg=self.cfg)
        pw = currency_weights_to_pair_weights(ccy_w)
        daily_w = expand_weights_to_daily(pw, pair_ret.index, signal_lag=self.cfg.signal_lag)
        return portfolio_returns_from_weights(daily_w, pair_ret)


class FxMomentumPairSignal:
    """Per-symbol TS momentum using ``signal_close`` / ``close``."""

    name = "fx_momentum_pair"

    def __init__(self, lookback: int = 63, skip: int = 1):
        self.lookback = int(lookback)
        self.skip = int(skip)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        px = data["signal_close"] if "signal_close" in data.columns else data["close"]
        return ts_momentum_pair_signal(px, lookback=self.lookback, skip=self.skip)
