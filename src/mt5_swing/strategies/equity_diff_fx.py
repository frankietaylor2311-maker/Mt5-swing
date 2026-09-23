"""Hau–Rey equity–FX channel (local vs US equity differentials → FX).

Literature prior (fixed — not holdout-tuned)
--------------------------------------------
Hau & Rey (2006): relative local equity *outperformance* vs US tends to
associate with local FX appreciation (portfolio / risk-appetite channel).
Related: equity–FX co-movement / order-flow literature.

Implementation
--------------
1. **eq_diff_xs** (primary): score = lagged (local − US) equity momentum over
   ``formation_days`` skipping most recent ``skip_days``; long high / short low
   (n_long=n_short=2).
2. **eq_diff_ts**: time-series sign of (local−US) mom per currency — long when
   positive, short when negative; equal-weight across active G10.
3. **eq_mom_xs** (control): same XS sort on *local* equity mom only (no US sub).
4. **carry_eq_cool** (optional): scholarly carry cooled when lagged US equity
   momentum < 0 (risk-off) — single binary cool, no grid.

All signals: equity ``pub_lag_days`` + ``signal_lag_days≥1``. Calendar-date
align Yahoo equity timestamps onto FX D1 (commodity wave fix).
Costs: ~1.5 bps/side on turnover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.data.equity_indices import EQUITY_CCYS
from mt5_swing.strategies.carry_rank import (
    USD_PAIRS,
    CarryRankConfig,
    carry_weights_from_rates,
    currency_weights_to_pair_weights,
    expand_weights_to_daily,
    portfolio_returns_from_weights,
)
from mt5_swing.strategies.commodity_fx import align_daily_to_index, apply_turnover_costs
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class EquityDiffFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    formation_days: int = 21  # ~1m equity differential (Hau–Rey co-movement horizon)
    skip_days: int = 0
    signal_lag_days: int = 1
    n_long: int = 2
    n_short: int = 2
    cost_bps_per_side: float = 1.5
    cool: float = 0.35  # carry scale when US equity mom < 0
    carry_n_long: int = 2
    carry_n_short: int = 2
    min_currencies: int = 4


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def trailing_equity_momentum(
    prices: pd.Series,
    *,
    formation_days: int = 21,
    skip_days: int = 0,
) -> pd.Series:
    """Cumulative return over [t-formation-skip, t-skip].

    Computed on the series' own non-null trading days (Yahoo G10 calendars
    outer-join into a sparse panel — row-wise shift would be wrong).
    """
    lag_near = int(skip_days)
    lag_far = int(formation_days + skip_days)
    s = prices.dropna()
    if s.empty:
        out = prices.copy() * np.nan
        out.name = prices.name or "equity_mom"
        return out
    near = s.shift(lag_near) if lag_near > 0 else s
    far = s.shift(lag_far)
    mom = near / far - 1.0
    mom.name = prices.name or "equity_mom"
    # Reindex to original panel index (NaN on foreign holidays)
    return mom.reindex(prices.index)


def prepare_equity_diff_scores(
    equity_panel: pd.DataFrame,
    *,
    cfg: EquityDiffFxConfig | None = None,
    relative: bool = True,
) -> pd.DataFrame:
    """Daily equity momentum scores per foreign currency (pre signal_lag).

    If ``relative``, score = local_mom − US_mom (Hau–Rey differential).
    Else score = local_mom only (control).
    """
    cfg = cfg or EquityDiffFxConfig()
    if "USD" not in equity_panel.columns:
        raise ValueError("equity panel requires USD column (^GSPC)")
    us_mom = trailing_equity_momentum(
        equity_panel["USD"],
        formation_days=cfg.formation_days,
        skip_days=cfg.skip_days,
    )
    # Light ffill across local holidays so G10 scores overlap on FX days
    us_mom = us_mom.ffill(limit=3)
    cols: dict[str, pd.Series] = {}
    for ccy in EQUITY_CCYS:
        if ccy not in equity_panel.columns:
            continue
        loc = trailing_equity_momentum(
            equity_panel[ccy],
            formation_days=cfg.formation_days,
            skip_days=cfg.skip_days,
        ).ffill(limit=3)
        cols[ccy] = (loc - us_mom) if relative else loc
    if not cols:
        return pd.DataFrame()
    sig = pd.DataFrame(cols).sort_index()
    if cfg.signal_lag_days > 0:
        sig = sig.shift(int(cfg.signal_lag_days))
    sig.attrs["signal_lag_days"] = int(cfg.signal_lag_days)
    sig.attrs["relative"] = bool(relative)
    sig.attrs["formation_days"] = int(cfg.formation_days)
    return sig


def _rank_sort_weights(
    score: pd.DataFrame,
    *,
    n_long: int,
    n_short: int,
) -> pd.DataFrame:
    """Long high score / short low score. Each side |sum|=0.5."""
    cols = list(score.columns)
    rows = []
    for dt, row in score.iterrows():
        s = row.dropna()
        if len(s) < n_long + n_short:
            continue
        ranked = s.sort_values(ascending=False)  # high relative equity first
        longs = ranked.index[:n_long]
        shorts = ranked.index[-n_short:]
        w = pd.Series(0.0, index=cols)
        w.loc[list(longs)] = 0.5 / n_long
        w.loc[list(shorts)] = -0.5 / n_short
        w.name = dt
        rows.append(w)
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_index()


def _ts_sign_weights(score: pd.DataFrame) -> pd.DataFrame:
    """Long positive / short negative (local−US) mom; equal |w| across actives."""
    cols = list(score.columns)
    rows = []
    for dt, row in score.iterrows():
        s = row.dropna()
        w = pd.Series(0.0, index=cols)
        pos = s[s > 0]
        neg = s[s < 0]
        if len(pos):
            w.loc[list(pos.index)] = 0.5 / len(pos)
        if len(neg):
            w.loc[list(neg.index)] = -0.5 / len(neg)
        w.name = dt
        rows.append(w)
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_index()


def _ccy_weights_to_daily_pair_port(
    ccy_w: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cost_bps_per_side: float = 1.5,
    name: str = "equity_diff",
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


def eq_diff_xs_returns(
    equity_panel: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: EquityDiffFxConfig | None = None,
) -> pd.Series:
    cfg = cfg or EquityDiffFxConfig()
    sig = prepare_equity_diff_scores(equity_panel, cfg=cfg, relative=True)
    if sig.empty:
        return pd.Series(0.0, index=pair_ret.index, name="eq_diff_xs")
    sig = align_daily_to_index(sig, pair_ret.index)
    ccy_w = _rank_sort_weights(sig, n_long=cfg.n_long, n_short=cfg.n_short)
    return _ccy_weights_to_daily_pair_port(
        ccy_w, pair_ret, cost_bps_per_side=cfg.cost_bps_per_side, name="eq_diff_xs"
    )


def eq_diff_ts_returns(
    equity_panel: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: EquityDiffFxConfig | None = None,
) -> pd.Series:
    cfg = cfg or EquityDiffFxConfig()
    sig = prepare_equity_diff_scores(equity_panel, cfg=cfg, relative=True)
    if sig.empty:
        return pd.Series(0.0, index=pair_ret.index, name="eq_diff_ts")
    sig = align_daily_to_index(sig, pair_ret.index)
    ccy_w = _ts_sign_weights(sig)
    return _ccy_weights_to_daily_pair_port(
        ccy_w, pair_ret, cost_bps_per_side=cfg.cost_bps_per_side, name="eq_diff_ts"
    )


def eq_mom_xs_returns(
    equity_panel: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: EquityDiffFxConfig | None = None,
) -> pd.Series:
    cfg = cfg or EquityDiffFxConfig()
    sig = prepare_equity_diff_scores(equity_panel, cfg=cfg, relative=False)
    if sig.empty:
        return pd.Series(0.0, index=pair_ret.index, name="eq_mom_xs")
    sig = align_daily_to_index(sig, pair_ret.index)
    ccy_w = _rank_sort_weights(sig, n_long=cfg.n_long, n_short=cfg.n_short)
    return _ccy_weights_to_daily_pair_port(
        ccy_w, pair_ret, cost_bps_per_side=cfg.cost_bps_per_side, name="eq_mom_xs"
    )


def carry_eq_cool_returns(
    equity_panel: pd.DataFrame,
    pair_ret: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    cfg: EquityDiffFxConfig | None = None,
) -> pd.Series:
    """Scholarly carry scaled to ``cool`` when lagged US equity mom < 0."""
    cfg = cfg or EquityDiffFxConfig()
    if "USD" not in equity_panel.columns or rates is None or rates.empty:
        return pd.Series(0.0, index=pair_ret.index, name="carry_eq_cool")
    us_mom = trailing_equity_momentum(
        equity_panel["USD"],
        formation_days=cfg.formation_days,
        skip_days=cfg.skip_days,
    )
    if cfg.signal_lag_days > 0:
        us_mom = us_mom.shift(int(cfg.signal_lag_days))
    us_mom = align_daily_to_index(us_mom, pair_ret.index)

    ccfg = CarryRankConfig(
        n_long=cfg.carry_n_long,
        n_short=cfg.carry_n_short,
        signal_lag=cfg.signal_lag_days,
    )
    ccy_w = carry_weights_from_rates(rates, cfg=ccfg)
    carry_pw = currency_weights_to_pair_weights(ccy_w)
    carry_daily = expand_weights_to_daily(
        carry_pw, pair_ret.index, signal_lag=cfg.signal_lag_days
    )
    cols = list(pair_ret.columns)
    for c in cols:
        if c not in carry_daily.columns:
            carry_daily[c] = 0.0
    carry_daily = carry_daily.reindex(columns=cols).fillna(0.0)

    # Binary cool: US equity down → scale to cool; else 1.0
    scale = pd.Series(1.0, index=pair_ret.index)
    down = us_mom.reindex(pair_ret.index).ffill() < 0
    scale = scale.where(~down.fillna(False), float(cfg.cool))
    cooled = carry_daily.mul(scale, axis=0)
    r = portfolio_returns_from_weights(cooled, pair_ret)
    r = apply_costs(r, cooled, bps_side=cfg.cost_bps_per_side)
    r.name = "carry_eq_cool"
    return r


def equity_diff_factor_returns(
    equity_panel: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    rates: pd.DataFrame | None = None,
    cfg: EquityDiffFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Build Hau–Rey equity–FX factor daily returns."""
    cfg = cfg or EquityDiffFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    out: dict[str, pd.Series] = {
        "eq_diff_xs": eq_diff_xs_returns(equity_panel, pret, cfg=cfg),
        "eq_diff_ts": eq_diff_ts_returns(equity_panel, pret, cfg=cfg),
        "eq_mom_xs": eq_mom_xs_returns(equity_panel, pret, cfg=cfg),
    }
    if rates is not None and not rates.empty:
        out["carry_eq_cool"] = carry_eq_cool_returns(
            equity_panel, pret, rates, cfg=cfg
        )
    return out


__all__ = [
    "EquityDiffFxConfig",
    "trailing_equity_momentum",
    "prepare_equity_diff_scores",
    "eq_diff_xs_returns",
    "eq_diff_ts_returns",
    "eq_mom_xs_returns",
    "carry_eq_cool_returns",
    "equity_diff_factor_returns",
    "align_daily_to_index",
]
