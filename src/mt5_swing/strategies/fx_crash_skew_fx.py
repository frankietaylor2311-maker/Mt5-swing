"""Brunnermeier–Nagel–Pedersen currency crash-risk / return-skewness FX factors.

Literature priors (frozen — not holdout-tuned)
---------------------------------------------
Brunnermeier, Nagel & Pedersen (2008), "Carry Trades and Currency Crashes,"
*Review of Financial Studies*. Currencies with more *negative* return skewness
(crash risk) earn a premium on average — distinct from Menkhoff FX realized-vol
(§15), Lustig carry, and AI-GPR roles (§25).

Free-data construction (OHLC only; no paid IV/RR / news NLP)
-----------------------------------------------------------
1. Map USD-pair returns → currency-vs-USD returns.
2. Trailing return skewness / left-tail shortfall on each FX (formation 63d /
   126d), with ``skip_days≥1`` so the score excludes the decision-day return,
   then ``signal_lag≥1``.
3. Cross-sectional HML: long high-crash-risk (more negative skew / deeper
   left-tail shortfall) / short low-crash-risk.
4. Companion: FX momentum gated to the crash-prone half of the skew cross-
   section; EW of crash legs; scholarly carry×crash cool (**not** for locked
   fx4plus overlay).

PIT: all scores use lagged returns only; costs ~1.5 bps/side on turnover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.strategies.carry_rank import (
    USD_PAIRS,
    CarryRankConfig,
    carry_weights_from_rates,
    currency_weights_to_pair_weights,
    expand_weights_to_daily,
    portfolio_returns_from_weights,
)
from mt5_swing.strategies.commodity_fx import apply_turnover_costs
from mt5_swing.strategies.fx_momentum import (
    formation_momentum,
    pair_returns_to_currency_returns,
)
from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z


@dataclass
class FxCrashSkewConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    formation_days: int = 63  # ~3m trailing skew (BNP-style window)
    formation_days_long: int = 126  # companion ~6m
    skip_days: int = 1  # exclude decision-day return from score
    signal_lag: int = 1  # extra trading-day lag after score formed
    min_periods: int = 40
    left_tail_q: float = 0.05  # 5th-percentile shortfall
    n_long: int = 2
    n_short: int = 2
    cost_bps_per_side: float = 1.5
    # Aggregate crash-risk z for carry cool (scholarly sleeve only)
    z_window: int = 252
    min_periods_z: int = 60
    z_high: float = 1.0  # |z| threshold via risk_scale_from_z (elevated crash)
    z_low: float = 0.0
    cool: float = 0.35
    carry_n_long: int = 2
    carry_n_short: int = 2
    # Mom gate: formation matches Menkhoff-ish 63/21
    mom_formation: int = 63
    mom_skip: int = 21
    mom_min_periods: int = 40


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def trailing_return_skewness(
    ccy_ret: pd.DataFrame,
    *,
    formation_days: int,
    skip_days: int,
    min_periods: int,
) -> pd.DataFrame:
    """Causal rolling sample skewness ending ``skip_days`` ago."""
    lagged = ccy_ret.shift(int(skip_days))
    out = lagged.rolling(int(formation_days), min_periods=int(min_periods)).skew()
    return out


def _left_tail_mean(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return float("nan")
    thr = float(np.nanpercentile(x, q * 100.0))
    left = x[x <= thr]
    if len(left) == 0:
        return float("nan")
    return float(left.mean())


def trailing_left_tail_shortfall(
    ccy_ret: pd.DataFrame,
    *,
    formation_days: int,
    skip_days: int,
    min_periods: int,
    q: float = 0.05,
) -> pd.DataFrame:
    """Mean of returns at/below the ``q`` quantile over the formation window.

    More negative → deeper crash / left-tail risk (BNP premium direction).
    """
    lagged = ccy_ret.shift(int(skip_days))
    q = float(q)

    def _apply(col: pd.Series) -> pd.Series:
        return col.rolling(int(formation_days), min_periods=int(min_periods)).apply(
            lambda x: _left_tail_mean(x, q), raw=True
        )

    return lagged.apply(_apply)


def _crash_score_from_skew(skew: pd.DataFrame) -> pd.DataFrame:
    """Higher score = more crash risk = more negative skew → ``-skew``."""
    return -skew


def _crash_score_from_shortfall(sf: pd.DataFrame) -> pd.DataFrame:
    """Higher score = deeper left tail = more negative shortfall → ``-sf``."""
    return -sf


def _rank_sort_weights(
    score: pd.DataFrame,
    *,
    n_long: int,
    n_short: int,
) -> pd.DataFrame:
    """Long high crash-risk score / short low. Each side |sum|=0.5."""
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


def _monthly_rebalance(score_daily: pd.DataFrame) -> pd.DataFrame:
    """Month-end snapshot of daily scores (academic monthly rebalance prior)."""
    if score_daily.empty:
        return score_daily
    return score_daily.resample("ME").last().dropna(how="all")


def _ccy_weights_to_daily_pair_port(
    ccy_w: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cost_bps_per_side: float = 1.5,
    name: str = "crash_skew",
) -> pd.Series:
    if ccy_w.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    keep = [c for c in ccy_w.columns if c in USD_PAIRS]
    if not keep:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    cw = ccy_w[keep].copy()
    cw.index = _ensure_utc(pd.DatetimeIndex(cw.index))
    pair_w = currency_weights_to_pair_weights(cw)
    # signal_lag already on scores; expand without extra calendar lag;
    # portfolio_returns_from_weights still shifts weights by 1 for causality.
    daily_w = expand_weights_to_daily(pair_w, pair_ret.index, signal_lag=0)
    gross = portfolio_returns_from_weights(daily_w, pair_ret)
    gross.name = name
    net = apply_turnover_costs(gross, daily_w, cost_bps_per_side=cost_bps_per_side)
    net.name = name
    return net


def prepare_crash_skew_scores(
    pair_ret: pd.DataFrame,
    *,
    cfg: FxCrashSkewConfig | None = None,
    formation_days: int | None = None,
) -> pd.DataFrame:
    """Daily crash-risk scores (−skew) per foreign currency, post signal_lag."""
    cfg = cfg or FxCrashSkewConfig()
    fdays = int(formation_days if formation_days is not None else cfg.formation_days)
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    ccy = pair_returns_to_currency_returns(pret).drop(columns=["USD"], errors="ignore")
    if ccy.empty:
        return pd.DataFrame()
    skew = trailing_return_skewness(
        ccy,
        formation_days=fdays,
        skip_days=cfg.skip_days,
        min_periods=cfg.min_periods,
    )
    score = _crash_score_from_skew(skew)
    if cfg.signal_lag > 0:
        score = score.shift(int(cfg.signal_lag))
    score.attrs["formation_days"] = fdays
    score.attrs["skip_days"] = int(cfg.skip_days)
    score.attrs["signal_lag"] = int(cfg.signal_lag)
    return score


def prepare_left_tail_scores(
    pair_ret: pd.DataFrame,
    *,
    cfg: FxCrashSkewConfig | None = None,
) -> pd.DataFrame:
    """Daily left-tail shortfall crash scores (−shortfall), post signal_lag."""
    cfg = cfg or FxCrashSkewConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    ccy = pair_returns_to_currency_returns(pret).drop(columns=["USD"], errors="ignore")
    if ccy.empty:
        return pd.DataFrame()
    sf = trailing_left_tail_shortfall(
        ccy,
        formation_days=cfg.formation_days,
        skip_days=cfg.skip_days,
        min_periods=cfg.min_periods,
        q=cfg.left_tail_q,
    )
    score = _crash_score_from_shortfall(sf)
    if cfg.signal_lag > 0:
        score = score.shift(int(cfg.signal_lag))
    score.attrs["left_tail_q"] = float(cfg.left_tail_q)
    score.attrs["signal_lag"] = int(cfg.signal_lag)
    return score


def crash_skew_xs_returns(
    pair_ret: pd.DataFrame,
    *,
    cfg: FxCrashSkewConfig | None = None,
    formation_days: int | None = None,
    name: str = "crash_skew_xs",
) -> pd.Series:
    cfg = cfg or FxCrashSkewConfig()
    score = prepare_crash_skew_scores(pair_ret, cfg=cfg, formation_days=formation_days)
    if score.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    rebal = _monthly_rebalance(score)
    ccy_w = _rank_sort_weights(rebal, n_long=cfg.n_long, n_short=cfg.n_short)
    return _ccy_weights_to_daily_pair_port(
        ccy_w, pair_ret, cost_bps_per_side=cfg.cost_bps_per_side, name=name
    )


def left_tail_xs_returns(
    pair_ret: pd.DataFrame,
    *,
    cfg: FxCrashSkewConfig | None = None,
    name: str = "left_tail_xs",
) -> pd.Series:
    cfg = cfg or FxCrashSkewConfig()
    score = prepare_left_tail_scores(pair_ret, cfg=cfg)
    if score.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    rebal = _monthly_rebalance(score)
    ccy_w = _rank_sort_weights(rebal, n_long=cfg.n_long, n_short=cfg.n_short)
    return _ccy_weights_to_daily_pair_port(
        ccy_w, pair_ret, cost_bps_per_side=cfg.cost_bps_per_side, name=name
    )


def mom_skew_regime_returns(
    pair_ret: pd.DataFrame,
    *,
    cfg: FxCrashSkewConfig | None = None,
    name: str = "mom_skew_regime",
) -> pd.Series:
    """FX momentum restricted to the high-crash-risk half of the skew cross-section.

    Prior: BNP crash premium interacts with momentum — only long/short mom among
    currencies whose lagged skew crash-score is above the cross-sectional median
    (more crash-prone). Monthly rebalance.
    """
    cfg = cfg or FxCrashSkewConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    ccy = pair_returns_to_currency_returns(pret).drop(columns=["USD"], errors="ignore")
    if ccy.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)

    crash_score = prepare_crash_skew_scores(pret, cfg=cfg)
    mom = formation_momentum(
        ccy,
        formation_days=cfg.mom_formation,
        skip_days=cfg.mom_skip,
        min_periods=cfg.mom_min_periods,
    )
    if cfg.signal_lag > 0:
        mom = mom.shift(int(cfg.signal_lag))

    # Align calendars
    common_cols = [c for c in crash_score.columns if c in mom.columns]
    if len(common_cols) < cfg.n_long + cfg.n_short:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    crash_m = _monthly_rebalance(crash_score[common_cols])
    mom_m = _monthly_rebalance(mom[common_cols])
    idx = crash_m.index.intersection(mom_m.index)
    rows = []
    for dt in idx:
        cs = crash_m.loc[dt].dropna()
        ms = mom_m.loc[dt].reindex(cs.index).dropna()
        cs = cs.reindex(ms.index).dropna()
        if len(cs) < cfg.n_long + cfg.n_short:
            continue
        med = float(cs.median())
        eligible = cs[cs >= med]  # high crash-score half
        if len(eligible) < max(2, cfg.n_long):
            continue
        # Mom sort within eligible crash-prone set
        m_elig = ms.reindex(eligible.index).dropna()
        if len(m_elig) < cfg.n_long + cfg.n_short:
            # If half is thin, take top/bottom within eligible with what we have
            if len(m_elig) < 2:
                continue
            ranked = m_elig.sort_values(ascending=False)
            n_l = min(cfg.n_long, max(1, len(ranked) // 2))
            n_s = min(cfg.n_short, max(1, len(ranked) - n_l))
        else:
            ranked = m_elig.sort_values(ascending=False)
            n_l, n_s = cfg.n_long, cfg.n_short
        longs = list(ranked.index[:n_l])
        shorts = list(ranked.index[-n_s:])
        w = pd.Series(0.0, index=common_cols)
        w.loc[longs] = 0.5 / max(n_l, 1)
        w.loc[shorts] = -0.5 / max(n_s, 1)
        w.name = dt
        rows.append(w)
    if not rows:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    ccy_w = pd.DataFrame(rows)
    ccy_w.index = _ensure_utc(pd.DatetimeIndex(ccy_w.index))
    return _ccy_weights_to_daily_pair_port(
        ccy_w, pret, cost_bps_per_side=cfg.cost_bps_per_side, name=name
    )


def aggregate_crash_skew_z(
    pair_ret: pd.DataFrame,
    *,
    cfg: FxCrashSkewConfig | None = None,
) -> pd.Series:
    """EW foreign-currency return skew → trailing z (elevated crash when z of −skew high).

    We z-score the *crash score* (−skew): high z ⇒ more negative aggregate skew
    ⇒ cool carry (BNP carry-crash channel).
    """
    cfg = cfg or FxCrashSkewConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    ccy = pair_returns_to_currency_returns(pret).drop(columns=["USD"], errors="ignore")
    skew = trailing_return_skewness(
        ccy,
        formation_days=cfg.formation_days,
        skip_days=cfg.skip_days,
        min_periods=cfg.min_periods,
    )
    crash = (-skew).mean(axis=1)
    mu = crash.rolling(cfg.z_window, min_periods=cfg.min_periods_z).mean()
    sd = crash.rolling(cfg.z_window, min_periods=cfg.min_periods_z).std()
    z = (crash - mu) / sd.replace(0.0, np.nan)
    if cfg.signal_lag > 0:
        z = z.shift(int(cfg.signal_lag))
    z.name = "agg_crash_skew_z"
    return z


def carry_crash_cool_returns(
    pair_ret: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    cfg: FxCrashSkewConfig | None = None,
    name: str = "carry_crash_cool",
) -> pd.Series:
    """Scholarly carry scaled down when aggregate crash-skew z is elevated.

    **Do not** apply this cooler to locked ``fx4plus_gbpcad_d1_voltarget_0025``.
    """
    cfg = cfg or FxCrashSkewConfig()
    if rates is None or rates.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    ccfg = CarryRankConfig(
        n_long=cfg.carry_n_long,
        n_short=cfg.carry_n_short,
        signal_lag=cfg.signal_lag,
    )
    ccy_w = carry_weights_from_rates(rates, cfg=ccfg)
    carry_pw = currency_weights_to_pair_weights(ccy_w)
    carry_daily = expand_weights_to_daily(
        carry_pw, pret.index, signal_lag=cfg.signal_lag
    )
    cols = list(pret.columns)
    for c in cols:
        if c not in carry_daily.columns:
            carry_daily[c] = 0.0
    carry_daily = carry_daily.reindex(columns=cols).fillna(0.0)

    z = aggregate_crash_skew_z(pret, cfg=cfg)
    gcfg = GprRegimeConfig(
        z_high=cfg.z_high,
        z_low=cfg.z_low,
        cool=cfg.cool,
        signal_lag=0,  # z already lagged
        usd_tilt=0.0,
    )
    scale = risk_scale_from_z(z, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
    cooled = carry_daily.mul(scale, axis=0)
    r = portfolio_returns_from_weights(cooled, pret)
    r = apply_turnover_costs(r, cooled, cost_bps_per_side=cfg.cost_bps_per_side)
    r.name = name
    return r


def crash_skew_factor_returns(
    pair_ret: pd.DataFrame,
    *,
    rates: pd.DataFrame | None = None,
    cfg: FxCrashSkewConfig | None = None,
) -> dict[str, pd.Series]:
    """Build BNP crash-skew FX factor board (post costs)."""
    cfg = cfg or FxCrashSkewConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    out: dict[str, pd.Series] = {}

    out["crash_skew_xs"] = crash_skew_xs_returns(
        pret, cfg=cfg, formation_days=cfg.formation_days, name="crash_skew_xs"
    )
    out["crash_skew_xs_126"] = crash_skew_xs_returns(
        pret,
        cfg=cfg,
        formation_days=cfg.formation_days_long,
        name="crash_skew_xs_126",
    )
    out["left_tail_xs"] = left_tail_xs_returns(pret, cfg=cfg, name="left_tail_xs")
    out["mom_skew_regime"] = mom_skew_regime_returns(
        pret, cfg=cfg, name="mom_skew_regime"
    )

    if rates is not None and not rates.empty:
        out["carry_crash_cool"] = carry_crash_cool_returns(
            pret, rates, cfg=cfg, name="carry_crash_cool"
        )

    blend_keys = [
        k
        for k in ("crash_skew_xs", "crash_skew_xs_126", "left_tail_xs", "mom_skew_regime")
        if k in out
    ]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "crash_ew"
        out["crash_ew"] = blend

    return out


__all__ = [
    "FxCrashSkewConfig",
    "trailing_return_skewness",
    "trailing_left_tail_shortfall",
    "prepare_crash_skew_scores",
    "prepare_left_tail_scores",
    "crash_skew_xs_returns",
    "left_tail_xs_returns",
    "mom_skew_regime_returns",
    "aggregate_crash_skew_z",
    "carry_crash_cool_returns",
    "crash_skew_factor_returns",
]
