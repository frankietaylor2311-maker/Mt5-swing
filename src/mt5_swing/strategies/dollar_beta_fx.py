"""Lustig–Verdelhan dollar-factor beta FX sorts.

Literature priors (frozen — not holdout-tuned)
---------------------------------------------
Lustig, Roussanov & Verdelhan (2011, 2014); Verdelhan dollar-factor work:
the *dollar factor* RX is the equal-weight average of foreign-currency excess
returns vs USD. Sort currencies by rolling β of currency returns on RX
(typically **60-month** window ending t−1). Long high-$β / short low-$β.

Optional conditioning: when average forward discount
(AFD = avg foreign−US short rate) > 0 keep the high−low $β HML; when AFD < 0
flip sign. Companion: unconditional 36m β window; dollar RX time-series
momentum (sign of trailing RX).

Distinct from Lustig carry (rate sort), Menkhoff FX-RV, Hau–Rey equity-diff,
BNP crash-skew, AI-GPR.

Implementation choice (documented)
----------------------------------
β estimated on **monthly** currency returns and monthly RX with rolling
window = 60 months (matches Verdelhan). Month-end scores → daily pair weights
via ``expand_weights_to_daily``. PIT: β window ends with ``skip_months≥1``
(decision month excluded) + ``signal_lag≥1`` month; costs ~1.5 bps/side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.data.fred_rates import rate_differentials_vs_usd
from mt5_swing.strategies.carry_rank import (
    USD_PAIRS,
    currency_weights_to_pair_weights,
    expand_weights_to_daily,
    portfolio_returns_from_weights,
)
from mt5_swing.strategies.commodity_fx import apply_turnover_costs
from mt5_swing.strategies.fx_momentum import (
    dollar_factor_returns,
    pair_returns_to_currency_returns,
)


@dataclass
class DollarBetaConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    beta_window_months: int = 60  # Verdelhan ~60m rolling
    beta_window_months_short: int = 36  # companion
    skip_months: int = 1  # exclude decision month from β window
    signal_lag: int = 1  # extra month lag after β formed
    min_periods: int = 36
    n_long: int = 2
    n_short: int = 2
    cost_bps_per_side: float = 1.5
    tsmom_lookback_months: int = 12  # trailing RX for dollar TSMOM
    tsmom_min_periods: int = 6


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def monthly_currency_returns(pair_ret: pd.DataFrame) -> pd.DataFrame:
    """Compound daily currency-vs-USD returns to month-end."""
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    ccy = pair_returns_to_currency_returns(pret).drop(columns=["USD"], errors="ignore")
    if ccy.empty:
        return ccy
    return (1.0 + ccy.fillna(0.0)).resample("ME").prod() - 1.0


def monthly_rx_dollar(pair_ret: pd.DataFrame) -> pd.Series:
    """EW foreign-currency monthly return vs USD (LRV dollar factor RX)."""
    m = monthly_currency_returns(pair_ret)
    if m.empty:
        return pd.Series(dtype=float, name="rx_dollar")
    s = m.mean(axis=1)
    s.name = "rx_dollar"
    return s


def average_forward_discount(rates: pd.DataFrame) -> pd.Series:
    """AFD = cross-sectional mean of (foreign short rate − US) at each date.

    Rates should already carry publication lag from ``load_currency_rates``.
    """
    if rates is None or rates.empty or "USD" not in rates.columns:
        return pd.Series(dtype=float, name="afd")
    diffs = rate_differentials_vs_usd(rates)
    afd = diffs.mean(axis=1)
    afd.name = "afd"
    # Month-end for alignment with β rebalance
    if len(afd):
        afd = afd.resample("ME").last()
    return afd


def _rolling_beta_vs_factor(
    y: pd.DataFrame,
    x: pd.Series,
    *,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    """Causal rolling OLS slope of each column of ``y`` on factor ``x``."""
    x = x.reindex(y.index).astype(float)
    out = pd.DataFrame(index=y.index, columns=list(y.columns), dtype=float)
    w = int(window)
    mp = int(min_periods)
    var = x.rolling(w, min_periods=mp).var().replace(0.0, np.nan)
    for col in y.columns:
        yi = y[col].astype(float)
        cov = yi.rolling(w, min_periods=mp).cov(x)
        out[col] = cov / var
    return out


def prepare_dollar_betas(
    pair_ret: pd.DataFrame,
    *,
    cfg: DollarBetaConfig | None = None,
    beta_window_months: int | None = None,
) -> pd.DataFrame:
    """Month-end $β of each currency on RX, post skip + signal_lag.

    Window ends ``skip_months`` before the score date (decision month excluded),
    then scores are shifted by ``signal_lag`` months.
    """
    cfg = cfg or DollarBetaConfig()
    win = int(beta_window_months if beta_window_months is not None else cfg.beta_window_months)
    m_ccy = monthly_currency_returns(pair_ret)
    if m_ccy.empty:
        return pd.DataFrame()
    rx = m_ccy.mean(axis=1)
    rx.name = "rx_dollar"

    # Exclude decision month from estimation: lag inputs by skip_months
    skip = int(cfg.skip_months)
    y = m_ccy.shift(skip) if skip > 0 else m_ccy
    x = rx.shift(skip) if skip > 0 else rx

    beta = _rolling_beta_vs_factor(
        y, x, window=win, min_periods=cfg.min_periods
    )
    if cfg.signal_lag > 0:
        beta = beta.shift(int(cfg.signal_lag))
    beta.attrs["beta_window_months"] = win
    beta.attrs["skip_months"] = skip
    beta.attrs["signal_lag"] = int(cfg.signal_lag)
    beta.attrs["estimation"] = "monthly_ols_60m_verdelhan"
    return beta


def _rank_sort_weights(
    score: pd.DataFrame,
    *,
    n_long: int,
    n_short: int,
) -> pd.DataFrame:
    """Long high score / short low. Each side |sum|=0.5 → gross≈1, net≈0."""
    cols = list(score.columns)
    rows = []
    for dt, row in score.iterrows():
        s = row.dropna()
        if len(s) < n_long + n_short:
            continue
        ranked = s.sort_values(ascending=False)
        longs = ranked.index[:n_long]
        shorts = ranked.index[-n_short:]
        w = pd.Series(0.0, index=cols)
        w.loc[list(longs)] = 0.5 / n_long
        w.loc[list(shorts)] = -0.5 / n_short
        w.name = dt
        rows.append(w)
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows).sort_index()
    out.index = _ensure_utc(pd.DatetimeIndex(out.index))
    return out


def _ccy_weights_to_daily_pair_port(
    ccy_w: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cost_bps_per_side: float = 1.5,
    name: str = "dollar_beta",
) -> pd.Series:
    if ccy_w.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    keep = [c for c in ccy_w.columns if c in USD_PAIRS]
    if not keep:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    cw = ccy_w[keep].copy()
    cw.index = _ensure_utc(pd.DatetimeIndex(cw.index))
    pair_w = currency_weights_to_pair_weights(cw)
    # Scores already month-lagged; expand without extra calendar lag;
    # portfolio_returns_from_weights still shifts weights by 1 for causality.
    daily_w = expand_weights_to_daily(pair_w, pair_ret.index, signal_lag=0)
    gross = portfolio_returns_from_weights(daily_w, pair_ret)
    gross.name = name
    net = apply_turnover_costs(gross, daily_w, cost_bps_per_side=cost_bps_per_side)
    net.name = name
    return net


def dollar_beta_xs_returns(
    pair_ret: pd.DataFrame,
    *,
    cfg: DollarBetaConfig | None = None,
    beta_window_months: int | None = None,
    name: str = "dollar_beta_xs",
) -> pd.Series:
    """Unconditional HML: long high-$β / short low-$β currencies."""
    cfg = cfg or DollarBetaConfig()
    beta = prepare_dollar_betas(
        pair_ret, cfg=cfg, beta_window_months=beta_window_months
    )
    if beta.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    ccy_w = _rank_sort_weights(beta, n_long=cfg.n_long, n_short=cfg.n_short)
    return _ccy_weights_to_daily_pair_port(
        ccy_w, pair_ret, cost_bps_per_side=cfg.cost_bps_per_side, name=name
    )


def dollar_beta_afd_returns(
    pair_ret: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    cfg: DollarBetaConfig | None = None,
    name: str = "dollar_beta_afd",
) -> pd.Series:
    """Same $β HML, signed by sign(AFD): long high-β when AFD>0, flip when AFD<0."""
    cfg = cfg or DollarBetaConfig()
    beta = prepare_dollar_betas(pair_ret, cfg=cfg)
    if beta.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    ccy_w = _rank_sort_weights(beta, n_long=cfg.n_long, n_short=cfg.n_short)
    if ccy_w.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)

    afd = average_forward_discount(rates)
    if afd.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    # Align AFD to rebalance dates; AFD already pub-lagged in rates loader.
    # Extra signal_lag months so AFD known before trade month.
    afd_lag = afd.shift(int(cfg.signal_lag)) if cfg.signal_lag > 0 else afd
    afd_aligned = afd_lag.reindex(ccy_w.index, method="ffill")
    sign = np.sign(afd_aligned.to_numpy(dtype=float))
    sign = np.where(np.isfinite(sign) & (sign != 0), sign, np.nan)
    # Drop rows with unknown AFD sign
    flip = pd.Series(sign, index=ccy_w.index)
    cw = ccy_w.mul(flip, axis=0)
    cw = cw.dropna(how="all")
    return _ccy_weights_to_daily_pair_port(
        cw, pair_ret, cost_bps_per_side=cfg.cost_bps_per_side, name=name
    )


def dollar_rx_tsmom_returns(
    pair_ret: pd.DataFrame,
    *,
    cfg: DollarBetaConfig | None = None,
    name: str = "dollar_rx_tsmom",
) -> pd.Series:
    """Long/flat/short all FX vs USD by sign of trailing monthly RX.

    Positive trailing RX → long equal-weight foreign currencies; negative → short.
    """
    cfg = cfg or DollarBetaConfig()
    m_ccy = monthly_currency_returns(pair_ret)
    if m_ccy.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    rx = m_ccy.mean(axis=1)
    # Trailing mean of RX ending skip_months ago, then signal_lag
    lb = int(cfg.tsmom_lookback_months)
    trail = (
        rx.shift(int(cfg.skip_months))
        .rolling(lb, min_periods=cfg.tsmom_min_periods)
        .mean()
    )
    if cfg.signal_lag > 0:
        trail = trail.shift(int(cfg.signal_lag))
    sig = np.sign(trail)
    cols = list(m_ccy.columns)
    rows = []
    for dt, s in sig.items():
        if not np.isfinite(s) or s == 0:
            continue
        w = pd.Series(float(s) / max(len(cols), 1), index=cols)
        w.name = dt
        rows.append(w)
    if not rows:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    ccy_w = pd.DataFrame(rows)
    ccy_w.index = _ensure_utc(pd.DatetimeIndex(ccy_w.index))
    return _ccy_weights_to_daily_pair_port(
        ccy_w, pair_ret, cost_bps_per_side=cfg.cost_bps_per_side, name=name
    )


def dollar_beta_factor_returns(
    pair_ret: pd.DataFrame,
    *,
    rates: pd.DataFrame | None = None,
    cfg: DollarBetaConfig | None = None,
) -> dict[str, pd.Series]:
    """Build Lustig–Verdelhan dollar-factor beta FX factor board (post costs)."""
    cfg = cfg or DollarBetaConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    out: dict[str, pd.Series] = {}

    out["dollar_beta_xs"] = dollar_beta_xs_returns(
        pret, cfg=cfg, beta_window_months=cfg.beta_window_months, name="dollar_beta_xs"
    )
    out["dollar_beta_xs_36m"] = dollar_beta_xs_returns(
        pret,
        cfg=cfg,
        beta_window_months=cfg.beta_window_months_short,
        name="dollar_beta_xs_36m",
    )

    if rates is not None and not rates.empty:
        out["dollar_beta_afd"] = dollar_beta_afd_returns(
            pret, rates, cfg=cfg, name="dollar_beta_afd"
        )

    out["dollar_rx_tsmom"] = dollar_rx_tsmom_returns(
        pret, cfg=cfg, name="dollar_rx_tsmom"
    )

    blend_keys = [
        k
        for k in (
            "dollar_beta_xs",
            "dollar_beta_xs_36m",
            "dollar_beta_afd",
            "dollar_rx_tsmom",
        )
        if k in out
    ]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "dollar_ew"
        out["dollar_ew"] = blend

    return out


__all__ = [
    "DollarBetaConfig",
    "monthly_currency_returns",
    "monthly_rx_dollar",
    "average_forward_discount",
    "prepare_dollar_betas",
    "dollar_beta_xs_returns",
    "dollar_beta_afd_returns",
    "dollar_rx_tsmom_returns",
    "dollar_beta_factor_returns",
    "dollar_factor_returns",  # re-export for convenience
]
