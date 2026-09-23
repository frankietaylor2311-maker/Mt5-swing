"""Balassa–Samuelson / productivity-adjusted real FX (free FRED IP + CPI).

Literature framing
------------------
Harrod–Balassa–Samuelson (HBS): countries with higher relative productivity in
tradables tend to have higher relative price levels (real appreciation).
Empirically, real exchange rates co-move with relative productivity; residuals
are a productivity-*adjusted* PPP / value signal (Chong–Jordà–Taylor 2012;
Ricci–Milesi-Ferretti–Lee / related BS–RER work).

We implement **tradable, point-in-time** factors — not a claim that we resolve
the Balassa–Samuelson hypothesis or earn 1%/mo.

Construction (USD numeraire)
----------------------------
1. Real FX: q_f = S_f · (CPI_US / CPI_f)   [reuse PPP helpers]
2. Relative productivity proxy: p_f = log(IP_f) − log(IP_US)
3. Factors (fixed priors):
   - bs_gap:     misalignment = z(log q) − z(p); score = −misalignment
                 (unit-coeff HBS co-movement residual after z-score)
   - bs_resid:   rolling OLS log(q) ~ a + b·p; score = −residual
   - bs_prod:    score = z(p)  (high relative productivity → long foreign)
   - bs_ew:      equal-weight blend of available XS sorts

Lags (fixed a priori — not holdout-tuned)
-----------------------------------------
- CPI publication lag: 1 month (loader)
- IP publication lag: 2 months (loader)
- Extra signal_lag months on scores (default 1)
- Daily pair weights: +1 trading-day lag on expanded monthly weights

Coverage: IP missing for AUD/NZD/CHF on FRED → those currencies drop from
productivity legs (documented). No technical overlays; no paid NLP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.data.macro_uncertainty import CURRENCY_USD_PAIR
from mt5_swing.strategies.country_gpr_fx import (
    currency_weights_to_pair_weights_fx,
    expand_monthly_weights_to_daily,
    portfolio_returns_from_pair_weights,
)
from mt5_swing.strategies.ppp_real_fx import (
    monthly_nominal_fx,
    real_fx_panel,
    real_fx_zscore,
)


@dataclass
class BalassaSamuelsonFxConfig:
    """Fixed priors for Balassa–Samuelson / productivity-adjusted value."""

    signal_lag: int = 1  # months after pub-lagged CPI/IP are known
    n_long: int = 2
    n_short: int = 2
    lookbacks: tuple[int, ...] = (60, 120)  # months for z / rolling OLS
    min_periods: int = 36
    cost_bps_per_side: float = 0.0


def _month_start(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    return pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start"), tz="UTC")


def _month_end(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    return pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="end"), tz="UTC")


def _align_monthly(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two month panels on shared month-start keys / common columns."""
    aa = a.copy()
    bb = b.copy()
    aa.index = _month_start(pd.DatetimeIndex(aa.index))
    bb.index = _month_start(pd.DatetimeIndex(bb.index))
    aa = aa[~aa.index.duplicated(keep="last")].sort_index()
    bb = bb[~bb.index.duplicated(keep="last")].sort_index()
    cols = [c for c in aa.columns if c in bb.columns]
    idx = aa.index.intersection(bb.index)
    return aa.loc[idx, cols], bb.loc[idx, cols]


def trailing_z(panel: pd.DataFrame, *, lookback: int, min_periods: int) -> pd.DataFrame:
    mu = panel.rolling(int(lookback), min_periods=int(min_periods)).mean()
    sd = panel.rolling(int(lookback), min_periods=int(min_periods)).std()
    return (panel - mu) / sd.replace(0.0, np.nan)


def bs_gap_scores(
    real_fx: pd.DataFrame,
    rel_prod: pd.DataFrame,
    *,
    cfg: BalassaSamuelsonFxConfig | None = None,
    lookback: int = 60,
) -> pd.DataFrame:
    """score = −(z(log q) − z(p)); cheap vs productivity → long foreign."""
    cfg = cfg or BalassaSamuelsonFxConfig()
    q, p = _align_monthly(real_fx, rel_prod)
    if q.empty:
        return pd.DataFrame()
    log_q = np.log(q.replace(0.0, np.nan))
    z_q = trailing_z(log_q, lookback=lookback, min_periods=cfg.min_periods)
    z_p = trailing_z(p, lookback=lookback, min_periods=cfg.min_periods)
    mis = z_q - z_p
    score = (-mis).copy()
    score.index = _month_start(pd.DatetimeIndex(score.index))
    if cfg.signal_lag > 0:
        score = score.shift(int(cfg.signal_lag))
    score.attrs["lookback"] = int(lookback)
    score.attrs["kind"] = "bs_gap"
    return score


def _rolling_ols_residual(
    y: pd.Series,
    x: pd.Series,
    *,
    lookback: int,
    min_periods: int,
) -> pd.Series:
    """Rolling OLS residual of y on [1, x]; NaN until min_periods."""
    y = y.astype(float)
    x = x.astype(float)
    out = pd.Series(np.nan, index=y.index, dtype=float)
    vals_y = y.to_numpy()
    vals_x = x.to_numpy()
    n = len(y)
    lb = int(lookback)
    mp = int(min_periods)
    for i in range(n):
        lo = max(0, i - lb + 1)
        yy = vals_y[lo : i + 1]
        xx = vals_x[lo : i + 1]
        mask = np.isfinite(yy) & np.isfinite(xx)
        if mask.sum() < mp:
            continue
        yy = yy[mask]
        xx = xx[mask]
        X = np.column_stack([np.ones(len(xx)), xx])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, yy, rcond=None)
        except np.linalg.LinAlgError:
            continue
        # residual at end of window (current observation)
        if not (np.isfinite(vals_y[i]) and np.isfinite(vals_x[i])):
            continue
        out.iloc[i] = float(vals_y[i] - (beta[0] + beta[1] * vals_x[i]))
    return out


def bs_residual_scores(
    real_fx: pd.DataFrame,
    rel_prod: pd.DataFrame,
    *,
    cfg: BalassaSamuelsonFxConfig | None = None,
    lookback: int = 60,
) -> pd.DataFrame:
    """score = −rolling OLS residual of log(q) on relative productivity."""
    cfg = cfg or BalassaSamuelsonFxConfig()
    q, p = _align_monthly(real_fx, rel_prod)
    if q.empty:
        return pd.DataFrame()
    log_q = np.log(q.replace(0.0, np.nan))
    resid = pd.DataFrame(index=log_q.index)
    for c in log_q.columns:
        resid[c] = _rolling_ols_residual(
            log_q[c], p[c], lookback=lookback, min_periods=cfg.min_periods
        )
    score = (-resid).copy()
    score.index = _month_start(pd.DatetimeIndex(score.index))
    if cfg.signal_lag > 0:
        score = score.shift(int(cfg.signal_lag))
    score.attrs["lookback"] = int(lookback)
    score.attrs["kind"] = "bs_resid"
    return score


def bs_productivity_scores(
    rel_prod: pd.DataFrame,
    *,
    cfg: BalassaSamuelsonFxConfig | None = None,
    lookback: int = 60,
) -> pd.DataFrame:
    """score = z(relative productivity); high prod → long foreign (HBS channel)."""
    cfg = cfg or BalassaSamuelsonFxConfig()
    p = rel_prod.copy()
    p.index = _month_start(pd.DatetimeIndex(p.index))
    p = p[~p.index.duplicated(keep="last")].sort_index()
    z = trailing_z(p, lookback=lookback, min_periods=cfg.min_periods)
    score = z.copy()
    if cfg.signal_lag > 0:
        score = score.shift(int(cfg.signal_lag))
    score.attrs["lookback"] = int(lookback)
    score.attrs["kind"] = "bs_prod"
    return score


def _rank_sort_weights(
    score: pd.DataFrame,
    *,
    n_long: int,
    n_short: int,
) -> pd.DataFrame:
    rows = []
    cols = list(score.columns)
    # Adaptive n when coverage is thin (e.g. only EUR/GBP/JPY/CAD have IP)
    for dt, row in score.iterrows():
        s = row.dropna()
        nl = min(n_long, max(1, len(s) // 2))
        ns = min(n_short, max(1, len(s) // 2))
        if len(s) < nl + ns:
            continue
        ranked = s.sort_values()
        shorts = ranked.index[:ns]
        longs = ranked.index[-nl:]
        w = pd.Series(0.0, index=cols)
        w.loc[list(longs)] = 0.5 / nl
        w.loc[list(shorts)] = -0.5 / ns
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
        turn = daily_w.diff().abs().sum(axis=1).fillna(0.0) * 0.5
        r = r - turn * (float(cost_bps_per_side) / 10_000.0)
        r.name = name
    return r


def balassa_samuelson_factor_returns(
    pair_close: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cpi_levels: pd.DataFrame,
    rel_prod: pd.DataFrame,
    *,
    cfg: BalassaSamuelsonFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Build BS gap / residual / productivity XS sorts + EW blend.

    Returns dict of daily portfolio return series.
    """
    cfg = cfg or BalassaSamuelsonFxConfig()
    fx_m = monthly_nominal_fx(pair_close)
    q = real_fx_panel(fx_m, cpi_levels)
    factors: dict[str, pd.Series] = {}
    xs_list: list[pd.Series] = []

    for lb in cfg.lookbacks:
        # Gap
        sc_gap = bs_gap_scores(q, rel_prod, cfg=cfg, lookback=int(lb))
        if not sc_gap.empty:
            w = _rank_sort_weights(sc_gap, n_long=cfg.n_long, n_short=cfg.n_short)
            name = f"bs_gap_{int(lb)}m"
            r = _ccy_weights_to_daily_returns(
                w, pair_ret, name=name, cost_bps_per_side=cfg.cost_bps_per_side
            )
            factors[name] = r
            xs_list.append(r)

        # Rolling OLS residual
        sc_res = bs_residual_scores(q, rel_prod, cfg=cfg, lookback=int(lb))
        if not sc_res.empty:
            w = _rank_sort_weights(sc_res, n_long=cfg.n_long, n_short=cfg.n_short)
            name = f"bs_resid_{int(lb)}m"
            r = _ccy_weights_to_daily_returns(
                w, pair_ret, name=name, cost_bps_per_side=cfg.cost_bps_per_side
            )
            factors[name] = r
            xs_list.append(r)

        # Relative productivity channel
        sc_prod = bs_productivity_scores(rel_prod, cfg=cfg, lookback=int(lb))
        if not sc_prod.empty:
            w = _rank_sort_weights(sc_prod, n_long=cfg.n_long, n_short=cfg.n_short)
            name = f"bs_prod_{int(lb)}m"
            r = _ccy_weights_to_daily_returns(
                w, pair_ret, name=name, cost_bps_per_side=cfg.cost_bps_per_side
            )
            factors[name] = r
            xs_list.append(r)

    if xs_list:
        # Prefer primary lookback gap+resid+prod for EW if both lookbacks present
        primary = [factors[k] for k in factors if k.endswith("_60m")]
        blend_src = primary if primary else xs_list
        blend = sum(blend_src) / len(blend_src)
        blend.name = "bs_ew"
        factors["bs_ew"] = blend

    if factors:
        primary = next(iter(factors.values()))
        primary.attrs["n_real_fx_months"] = int(q.dropna(how="all").shape[0])
        primary.attrs["rel_prod_cols"] = list(rel_prod.columns)
        primary.attrs["real_fx_cols"] = list(q.columns)
    return factors


__all__ = [
    "BalassaSamuelsonFxConfig",
    "trailing_z",
    "bs_gap_scores",
    "bs_residual_scores",
    "bs_productivity_scores",
    "balassa_samuelson_factor_returns",
    "real_fx_panel",
    "real_fx_zscore",
    "monthly_nominal_fx",
]
