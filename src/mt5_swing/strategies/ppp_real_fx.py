"""PPP / real exchange-rate value mean-reversion (Rogoff-style; free FRED CPI).

Literature framing
------------------
Relative PPP / real FX: currencies that are expensive in real terms (high real
exchange rate vs a long-run mean) tend to depreciate over long horizons
(Rogoff 1996 PPP puzzle — slow mean reversion, half-life often measured in
years). We implement a **tradable, point-in-time** cross-sectional sort on
real-FX *value* deviation, not a claim that we resolve the PPP puzzle.

Construction (USD numeraire)
----------------------------
For foreign currency *f*:
  S_f  = USD per 1 unit of f  (from USD-major closes; USDJPY → 1/USDJPY, …)
  q_f  = S_f * (CPI_US / CPI_f)     # real FX level; high = f expensive vs USD
  z_f  = (q_f − μ_L) / σ_L         # trailing L-month z (L∈{60,120} fixed priors)
Score for long-f = −z_f            # undervalued (cheap) → long; rich → short

Lags (fixed a priori — not holdout-tuned)
-----------------------------------------
- CPI publication lag: 1 month (loader)
- Extra signal_lag months on z (default 1)
- Daily pair weights: +1 trading-day lag on expanded monthly weights

No technical overlays; no paid NLP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.data.macro_uncertainty import CURRENCY_USD_PAIR
from mt5_swing.strategies.country_gpr_fx import (
    currency_weights_to_pair_weights_fx,
    expand_monthly_weights_to_daily,
    foreign_vs_usd_returns,
    portfolio_returns_from_pair_weights,
)


@dataclass
class PppRealFxConfig:
    """Fixed priors for PPP / real-FX value sorts."""

    signal_lag: int = 1  # months after pub-lagged CPI + month-end FX are known
    n_long: int = 2
    n_short: int = 2
    lookbacks: tuple[int, ...] = (60, 120)  # months for trailing real-FX mean/std
    min_periods: int = 36
    cost_bps_per_side: float = 0.0  # research default: zero; document if set


def _month_end(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    return pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="end"), tz="UTC")


def _month_start(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    return pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start"), tz="UTC")


def nominal_fx_usd_per_foreign(pair_close: pd.DataFrame) -> pd.DataFrame:
    """Daily USD-per-foreign levels from USD-major closes.

    EURUSD/GBPUSD/AUDUSD/NZDUSD: close is already USD per foreign.
    USDJPY/USDCAD/USDCHF: close is foreign per USD → invert.
    """
    out: dict[str, pd.Series] = {}
    for ccy, (pair, sign) in CURRENCY_USD_PAIR.items():
        if pair not in pair_close.columns:
            continue
        px = pair_close[pair].astype(float)
        if sign > 0:
            out[ccy] = px
        else:
            out[ccy] = 1.0 / px.replace(0.0, np.nan)
    return pd.DataFrame(out, index=pair_close.index).sort_index()


def monthly_nominal_fx(pair_close: pd.DataFrame) -> pd.DataFrame:
    """Month-end USD-per-foreign levels."""
    daily = nominal_fx_usd_per_foreign(pair_close)
    if daily.empty:
        return daily
    return daily.resample("ME").last()


def real_fx_panel(
    monthly_fx: pd.DataFrame,
    cpi_levels: pd.DataFrame,
) -> pd.DataFrame:
    """q_f = S_f * CPI_US / CPI_f. Aligns on month-start CPI × month-end FX month.

    CPI panel is expected publication-lagged already (index = month when known).
    We map both to calendar month keys then join on the FX month-end's month.
    """
    if "USD" not in cpi_levels.columns:
        raise ValueError("cpi_levels must include USD")
    fx = monthly_fx.copy()
    fx.index = _month_end(pd.DatetimeIndex(fx.index))
    fx = fx[~fx.index.duplicated(keep="last")].sort_index()

    cpi = cpi_levels.copy()
    cpi.index = _month_start(pd.DatetimeIndex(cpi.index))
    cpi = cpi[~cpi.index.duplicated(keep="last")].sort_index()

    # Map FX month-end → month-start key for CPI join (same calendar month)
    fx_ms = _month_start(pd.DatetimeIndex(fx.index))
    cpi_aligned = cpi.reindex(fx_ms)
    cpi_aligned.index = fx.index

    usd = cpi_aligned["USD"]
    cols = [c for c in fx.columns if c in cpi_aligned.columns and c != "USD"]
    q = pd.DataFrame(index=fx.index)
    for c in cols:
        q[c] = fx[c] * (usd / cpi_aligned[c].replace(0.0, np.nan))
    q.attrs["definition"] = "S_f * CPI_US / CPI_f (USD per foreign, real)"
    return q


def real_fx_zscore(
    real_fx: pd.DataFrame,
    *,
    lookback: int,
    min_periods: int,
) -> pd.DataFrame:
    """Trailing z of real FX (positive = foreign expensive vs its own history)."""
    mu = real_fx.rolling(int(lookback), min_periods=int(min_periods)).mean()
    sd = real_fx.rolling(int(lookback), min_periods=int(min_periods)).std()
    return (real_fx - mu) / sd.replace(0.0, np.nan)


def ppp_value_scores(
    real_fx: pd.DataFrame,
    *,
    cfg: PppRealFxConfig | None = None,
    lookback: int = 60,
) -> pd.DataFrame:
    """Score for long foreign = −z(q). High score = undervalued / cheap."""
    cfg = cfg or PppRealFxConfig()
    z = real_fx_zscore(real_fx, lookback=lookback, min_periods=cfg.min_periods)
    score = (-z).copy()
    score.index = _month_start(pd.DatetimeIndex(score.index))
    if cfg.signal_lag > 0:
        score = score.shift(int(cfg.signal_lag))
    score.attrs["lookback"] = int(lookback)
    score.attrs["signal_lag"] = int(cfg.signal_lag)
    return score


def _rank_sort_weights(
    score: pd.DataFrame,
    *,
    n_long: int,
    n_short: int,
) -> pd.DataFrame:
    rows = []
    cols = list(score.columns)
    for dt, row in score.iterrows():
        s = row.dropna()
        if len(s) < n_long + n_short:
            continue
        ranked = s.sort_values()
        shorts = ranked.index[:n_short]
        longs = ranked.index[-n_long:]
        w = pd.Series(0.0, index=cols)
        w.loc[list(longs)] = 0.5 / n_long
        w.loc[list(shorts)] = -0.5 / n_short
        w.name = dt
        rows.append(w)
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_index()


def _ts_sign_weights(score: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight long cheap / short rich among currencies with |score|>0."""
    rows = []
    cols = list(score.columns)
    for dt, row in score.iterrows():
        s = row.dropna()
        s = s[s != 0]
        if s.empty:
            continue
        pos = s[s > 0]
        neg = s[s < 0]
        w = pd.Series(0.0, index=cols)
        if len(pos):
            w.loc[pos.index] = 0.5 / len(pos)
        if len(neg):
            w.loc[neg.index] = -0.5 / len(neg)
        w.name = dt
        rows.append(w)
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_index()


def _ccy_weights_to_daily_returns(
    ccy_w: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    name: str,
    cost_bps_per_side: float = 0.0,
) -> pd.Series:
    if ccy_w.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    ccy_w = ccy_w.copy()
    ccy_w.index = (
        pd.DatetimeIndex(ccy_w.index)
        .tz_convert(None)
        .to_period("M")
        .to_timestamp(how="end")
        .tz_localize("UTC")
    )
    keep = [c for c in ccy_w.columns if c in CURRENCY_USD_PAIR]
    ccy_w = ccy_w[keep]
    pair_w = currency_weights_to_pair_weights_fx(ccy_w)
    daily_w = expand_monthly_weights_to_daily(pair_w, pair_ret.index, signal_lag_days=1)
    r = portfolio_returns_from_pair_weights(daily_w, pair_ret)
    r.name = name
    if cost_bps_per_side and cost_bps_per_side > 0:
        # Turnover proxy: L1 change in weights / 2 * bps
        turn = daily_w.diff().abs().sum(axis=1).fillna(0.0) * 0.5
        r = r - turn * (float(cost_bps_per_side) / 10_000.0)
        r.name = name
    return r


def ppp_factor_returns(
    pair_close: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cpi_levels: pd.DataFrame,
    *,
    cfg: PppRealFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Build XS sorts (60m / 120m) + optional TS sign blend and EW of XS.

    Returns dict of daily portfolio return series.
    """
    cfg = cfg or PppRealFxConfig()
    fx_m = monthly_nominal_fx(pair_close)
    q = real_fx_panel(fx_m, cpi_levels)
    factors: dict[str, pd.Series] = {}
    xs_list: list[pd.Series] = []
    for lb in cfg.lookbacks:
        score = ppp_value_scores(q, cfg=cfg, lookback=int(lb))
        ccy_w = _rank_sort_weights(score, n_long=cfg.n_long, n_short=cfg.n_short)
        name = f"ppp_xs_{int(lb)}m"
        r = _ccy_weights_to_daily_returns(
            ccy_w, pair_ret, name=name, cost_bps_per_side=cfg.cost_bps_per_side
        )
        factors[name] = r
        xs_list.append(r)
        # Time-series sign variant (same lookback)
        ts_w = _ts_sign_weights(score)
        ts_name = f"ppp_ts_{int(lb)}m"
        factors[ts_name] = _ccy_weights_to_daily_returns(
            ts_w, pair_ret, name=ts_name, cost_bps_per_side=cfg.cost_bps_per_side
        )
    if xs_list:
        blend = sum(xs_list) / len(xs_list)
        blend.name = "ppp_xs_ew"
        factors["ppp_xs_ew"] = blend
    # Attach diagnostics on a representative series
    if factors:
        primary = next(iter(factors.values()))
        primary.attrs["real_fx_last"] = {
            c: float(q[c].dropna().iloc[-1]) for c in q.columns if q[c].notna().any()
        }
        primary.attrs["n_real_fx_months"] = int(q.dropna(how="all").shape[0])
    return factors


def ppp_real_fx_snapshot(
    pair_close: pd.DataFrame,
    cpi_levels: pd.DataFrame,
    *,
    cfg: PppRealFxConfig | None = None,
    lookback: int = 60,
) -> pd.DataFrame:
    """Month-end panel: S, CPI ratio, q, z, score (for reports)."""
    cfg = cfg or PppRealFxConfig()
    fx_m = monthly_nominal_fx(pair_close)
    q = real_fx_panel(fx_m, cpi_levels)
    z = real_fx_zscore(q, lookback=lookback, min_periods=cfg.min_periods)
    score = -z
    # Flatten last overlapping rows into long form for CSV
    rows = []
    for c in q.columns:
        for dt in q.index:
            rows.append(
                {
                    "date": dt,
                    "currency": c,
                    "nominal_usd_per_f": float(fx_m.reindex(q.index).loc[dt, c])
                    if c in fx_m.columns and dt in fx_m.index
                    else float("nan"),
                    "real_fx_q": float(q.loc[dt, c]) if np.isfinite(q.loc[dt, c]) else float("nan"),
                    "z": float(z.loc[dt, c]) if dt in z.index and np.isfinite(z.loc[dt, c]) else float("nan"),
                    "score_long_f": float(score.loc[dt, c])
                    if dt in score.index and np.isfinite(score.loc[dt, c])
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "PppRealFxConfig",
    "nominal_fx_usd_per_foreign",
    "monthly_nominal_fx",
    "real_fx_panel",
    "real_fx_zscore",
    "ppp_value_scores",
    "ppp_factor_returns",
    "ppp_real_fx_snapshot",
    "foreign_vs_usd_returns",
]
