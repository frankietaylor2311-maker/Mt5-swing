"""Literature-style FX carry: rank currencies by short-rate differential.

Classic cross-sectional carry (Lustig–Roussanov–Verdelhan; Menkhoff–Sarno–Schmeling–Schrimpf):
long high-yielding currencies, short low-yielding. We implement a USD-centric
basket suitable for FTMO single-account FX (no multi-currency cash book):

- For each rebalance date, score currency ``c`` by ``rate_c - rate_USD``
  (already lagged via the FRED loader).
- Map to traded USD pairs:
  - High score (carry long foreign): **long** ``XXXUSD`` (or **short** ``USDXXX``)
  - Low score (carry short foreign): **short** ``XXXUSD`` (or **long** ``USDXXX``)

FTMO netting / hedging notes
----------------------------
- FTMO allows hedging on most account types but margin is charged on both legs;
  prefer **net USD exposure** via a small set of USD majors rather than offsetting
  crosses that double margin.
- Keep gross leverage modest (research default: equal-weight top/bottom ``n_long``
  each side, portfolio weights sum abs ≤ 1 after vol scale).
- Netting: opposing EURUSD long + short cancel; we emit **net weights per symbol**.
- Do **not** claim carry alone hits ~1%/month FTMO — literature Sharpe is typically
  ~0.4–0.8 pre-cost on diversified G10, with crash risk in risk-off.

Parameters are research priors (not holdout-tuned): ``n_long``, monthly rebalance,
``signal_lag`` bars after rate availability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from mt5_swing.strategies.base import Signal

# USD-quote vs USD-base pair map for G10
USD_PAIRS: dict[str, tuple[str, int]] = {
    # currency → (symbol, sign) where +1 means long symbol = long currency vs USD
    "EUR": ("EURUSD", +1),
    "GBP": ("GBPUSD", +1),
    "AUD": ("AUDUSD", +1),
    "NZD": ("NZDUSD", +1),
    "JPY": ("USDJPY", -1),  # long JPY = short USDJPY
    "CAD": ("USDCAD", -1),
    "CHF": ("USDCHF", -1),
}


@dataclass
class CarryRankConfig:
    n_long: int = 2
    n_short: int = 2
    rebalance: str = "M"  # month-end
    signal_lag: int = 1  # trading days after rate month availability
    equal_weight: bool = True
    min_history_months: int = 12


def _month_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Use month-end timestamps aligned to UTC
    s = pd.Series(1, index=idx)
    return s.resample("ME").last().index


def rank_currencies(rates_row: pd.Series, *, exclude: Iterable[str] = ("USD",)) -> pd.Series:
    """Descending rank by rate level / differential (higher = more attractive long)."""
    s = rates_row.drop(labels=[c for c in exclude if c in rates_row.index], errors="ignore")
    s = s.dropna()
    return s.sort_values(ascending=False)


def carry_weights_from_rates(
    rates: pd.DataFrame,
    *,
    cfg: CarryRankConfig | None = None,
) -> pd.DataFrame:
    """Month-end currency weights in currency space (long high / short low).

    Returns DataFrame indexed by rebalance dates, columns = currencies, values in
    {-w, 0, +w} with sum(abs) ≈ 1 when both sides filled.
    """
    cfg = cfg or CarryRankConfig()
    if "USD" in rates.columns:
        # Prefer differential vs USD when present
        scores = rates.drop(columns=["USD"]).sub(rates["USD"], axis=0)
    else:
        scores = rates.copy()
    # Rebalance on score index (already publication-lagged month starts/ends)
    rebal = scores.resample("ME").last().dropna(how="all")
    rows = []
    for dt, row in rebal.iterrows():
        ranked = rank_currencies(row)
        if len(ranked) < cfg.n_long + cfg.n_short:
            w = pd.Series(0.0, index=scores.columns)
        else:
            longs = list(ranked.index[: cfg.n_long])
            shorts = list(ranked.index[-cfg.n_short :])
            w = pd.Series(0.0, index=scores.columns)
            lw = 0.5 / max(cfg.n_long, 1)
            sw = 0.5 / max(cfg.n_short, 1)
            for c in longs:
                if c in w.index:
                    w[c] = lw
            for c in shorts:
                if c in w.index:
                    w[c] = -sw
        w.name = dt
        rows.append(w)
    if not rows:
        return pd.DataFrame(columns=scores.columns)
    out = pd.DataFrame(rows)
    out.index = pd.DatetimeIndex(out.index, tz="UTC")
    out.attrs["strategy"] = "carry_rank"
    out.attrs["signal_lag"] = cfg.signal_lag
    return out


def currency_weights_to_pair_weights(ccy_w: pd.DataFrame) -> pd.DataFrame:
    """Map currency weights → traded USD-pair weights (netted)."""
    pair_cols = sorted({USD_PAIRS[c][0] for c in ccy_w.columns if c in USD_PAIRS})
    rows = []
    for dt, row in ccy_w.iterrows():
        pw = {p: 0.0 for p in pair_cols}
        for ccy, w in row.items():
            if ccy not in USD_PAIRS or abs(float(w)) < 1e-15:
                continue
            sym, sign = USD_PAIRS[ccy]
            # + currency weight → +sign on symbol
            pw[sym] = pw.get(sym, 0.0) + float(w) * sign
        rows.append(pd.Series(pw, name=dt))
    out = pd.DataFrame(rows)
    out.index = pd.DatetimeIndex(out.index, tz="UTC")
    return out


def expand_weights_to_daily(
    weights: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    *,
    signal_lag: int = 1,
) -> pd.DataFrame:
    """Forward-fill month-end weights onto a daily index with ``signal_lag`` days."""
    if weights.empty:
        return pd.DataFrame(0.0, index=daily_index, columns=weights.columns)
    w = weights.copy()
    w.index = pd.DatetimeIndex(w.index).tz_convert("UTC") if w.index.tz else pd.DatetimeIndex(w.index, tz="UTC")
    # Shift availability by signal_lag calendar days
    if signal_lag > 0:
        w.index = w.index + pd.Timedelta(days=int(signal_lag))
    aligned = w.reindex(daily_index, method="ffill").fillna(0.0)
    return aligned


def portfolio_returns_from_weights(
    pair_weights: pd.DataFrame,
    pair_returns: pd.DataFrame,
) -> pd.Series:
    """``r_port_t = sum_i w_{i,t-1} * r_{i,t}`` (weights already lagged if desired).

    Caller should pass weights known before return realization (use expand with lag).
    """
    common = pair_weights.columns.intersection(pair_returns.columns)
    if len(common) == 0:
        return pd.Series(0.0, index=pair_returns.index, name="carry_rank")
    # Use prior weight for today's return
    w_lag = pair_weights[common].shift(1).fillna(0.0)
    r = pair_returns[common].fillna(0.0)
    port = (w_lag * r).sum(axis=1)
    port.name = "carry_rank"
    return port


class CarryRankBasket:
    """Cross-sectional carry basket (panel), not a single-symbol Signal strategy."""

    name = "carry_rank"

    def __init__(self, **kwargs):
        self.cfg = CarryRankConfig(**{k: v for k, v in kwargs.items() if k in CarryRankConfig.__dataclass_fields__})

    def weights(self, rates: pd.DataFrame) -> pd.DataFrame:
        return currency_weights_to_pair_weights(carry_weights_from_rates(rates, cfg=self.cfg))


class CarryRankPairSignal:
    """Single-pair Signal adapter: long/short/flat from panel carry weight sign.

    Expects ``data.attrs['carry_pair_weight']`` series aligned to index, or a
    column ``carry_w``. Used for unit tests / per-symbol backtest hooks.
    """

    name = "carry_rank_pair"

    def __init__(self, weight_col: str = "carry_w", flat_eps: float = 1e-8):
        self.weight_col = weight_col
        self.flat_eps = float(flat_eps)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if self.weight_col in data.columns:
            w = data[self.weight_col]
        else:
            w = data.attrs.get("carry_pair_weight")
            if w is None:
                return pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
            w = pd.Series(w, index=data.index)
        sig = pd.Series(int(Signal.FLAT), index=data.index, dtype=int)
        sig = sig.mask(w > self.flat_eps, int(Signal.LONG))
        sig = sig.mask(w < -self.flat_eps, int(Signal.SHORT))
        return sig.astype(int)
