"""Term-structure / yield-curve FX factors (Chen–Tsang / Ang–Chen style).

Fixed priors (no holdout tuning)
--------------------------------
- ``curve_slope_xs``: long steep / short flat *slope differentials vs USD*
  (slope = OECD LT govt yield − OECD immediate short rate).
- ``curve_lt_xs``: long high / short low *long-rate differentials* (level / long-carry).
- ``curve_slope_z_xs``: same sort on trailing z-score of slope_diff (60m).
- ``slope_x_carry``: score = z(slope_diff) × z(st_diff); long high interaction.
- ``curve_ew``: equal-weight of ``curve_slope_xs`` and ``curve_lt_xs``.
- Secondary UIP: ``uip_st_xs`` (short-rate differential — carry baseline) and
  ``uip_ir3m_xs`` when OECD 3m panel is supplied.

All monthly scores use ``signal_lag`` months on top of loader ``pub_lag_months``.
Daily portfolio uses an extra 1-day weight lag. Optional turnover haircut.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mt5_swing.data.macro_uncertainty import CURRENCY_USD_PAIR
from mt5_swing.strategies.country_gpr_fx import (
    currency_weights_to_pair_weights_fx,
    expand_monthly_weights_to_daily,
    portfolio_returns_from_pair_weights,
)


@dataclass
class YieldCurveFxConfig:
    signal_lag: int = 1  # months after publication-lagged yields are known
    n_long: int = 2
    n_short: int = 2
    z_window: int = 60
    min_periods: int = 24
    cost_bps_side: float = 1.5
    # Factors to build (frozen set — not tuned on OOS)
    build_slope_xs: bool = True
    build_lt_xs: bool = True
    build_slope_z: bool = True
    build_slope_x_carry: bool = True
    build_uip_st: bool = True
    build_uip_ir3m: bool = True
    build_ew: bool = True


def _month_start(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    return pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start"), tz="UTC")


def trailing_z(panel: pd.DataFrame, *, lookback: int, min_periods: int) -> pd.DataFrame:
    mu = panel.rolling(lookback, min_periods=min_periods).mean()
    sd = panel.rolling(lookback, min_periods=min_periods).std()
    return (panel - mu) / sd.replace(0.0, np.nan)


def _rank_sort_weights(
    score: pd.DataFrame,
    *,
    n_long: int,
    n_short: int,
) -> pd.DataFrame:
    """Long high score / short low score; each side |sum|=0.5."""
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


def apply_signal_lag(panel: pd.DataFrame, lag_months: int) -> pd.DataFrame:
    s = panel.copy()
    s.index = _month_start(pd.DatetimeIndex(s.index))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if lag_months > 0:
        s = s.shift(int(lag_months))
    return s


def apply_costs(port: pd.Series, weights: pd.DataFrame, *, bps_side: float) -> pd.Series:
    if bps_side <= 0 or weights.empty:
        return port
    dw = weights.diff().abs().sum(axis=1)
    cost = dw * (float(bps_side) / 10_000.0)
    out = port - cost.reindex(port.index).fillna(0.0)
    out.name = port.name
    return out


def _scores_to_daily_returns(
    score: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cfg: YieldCurveFxConfig,
    *,
    name: str,
) -> tuple[pd.Series, pd.DataFrame]:
    ccy_w = _rank_sort_weights(score, n_long=cfg.n_long, n_short=cfg.n_short)
    if ccy_w.empty:
        empty = pd.Series(0.0, index=pair_ret.index, name=name)
        return empty, pd.DataFrame(0.0, index=pair_ret.index, columns=[])
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
    r = apply_costs(r, daily_w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r, daily_w


def prepare_curve_scores(
    slope_diff: pd.DataFrame,
    lt_diff: pd.DataFrame,
    st_diff: pd.DataFrame,
    *,
    ir3m_diff: pd.DataFrame | None = None,
    cfg: YieldCurveFxConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build signed monthly score panels (pub-lag already in loaders)."""
    cfg = cfg or YieldCurveFxConfig()
    out: dict[str, pd.DataFrame] = {}

    if cfg.build_slope_xs and slope_diff is not None and not slope_diff.empty:
        out["curve_slope_xs"] = apply_signal_lag(slope_diff, cfg.signal_lag)

    if cfg.build_lt_xs and lt_diff is not None and not lt_diff.empty:
        out["curve_lt_xs"] = apply_signal_lag(lt_diff, cfg.signal_lag)

    if cfg.build_slope_z and slope_diff is not None and not slope_diff.empty:
        z = trailing_z(slope_diff, lookback=cfg.z_window, min_periods=cfg.min_periods)
        out["curve_slope_z_xs"] = apply_signal_lag(z, cfg.signal_lag)

    if cfg.build_slope_x_carry and slope_diff is not None and st_diff is not None:
        # Align then interact z-scores (frozen construction — no threshold hunt)
        common_idx = slope_diff.index.intersection(st_diff.index)
        common_cols = sorted(set(slope_diff.columns) & set(st_diff.columns))
        if common_cols:
            zs = trailing_z(
                slope_diff.loc[common_idx, common_cols],
                lookback=cfg.z_window,
                min_periods=cfg.min_periods,
            )
            zc = trailing_z(
                st_diff.loc[common_idx, common_cols],
                lookback=cfg.z_window,
                min_periods=cfg.min_periods,
            )
            interact = zs * zc
            out["slope_x_carry"] = apply_signal_lag(interact, cfg.signal_lag)

    if cfg.build_uip_st and st_diff is not None and not st_diff.empty:
        out["uip_st_xs"] = apply_signal_lag(st_diff, cfg.signal_lag)

    if cfg.build_uip_ir3m and ir3m_diff is not None and not ir3m_diff.empty:
        out["uip_ir3m_xs"] = apply_signal_lag(ir3m_diff, cfg.signal_lag)

    return out


def yield_curve_factor_returns(
    pair_ret: pd.DataFrame,
    slope_diff: pd.DataFrame,
    lt_diff: pd.DataFrame,
    st_diff: pd.DataFrame,
    *,
    ir3m_diff: pd.DataFrame | None = None,
    cfg: YieldCurveFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Daily returns for each curve / UIP sort + optional EW blend."""
    cfg = cfg or YieldCurveFxConfig()
    scores = prepare_curve_scores(
        slope_diff, lt_diff, st_diff, ir3m_diff=ir3m_diff, cfg=cfg
    )
    factors: dict[str, pd.Series] = {}
    for name, sc in scores.items():
        r, _ = _scores_to_daily_returns(sc, pair_ret, cfg, name=name)
        factors[name] = r

    if cfg.build_ew:
        legs = [factors[k] for k in ("curve_slope_xs", "curve_lt_xs") if k in factors]
        if legs:
            blend = sum(legs) / len(legs)
            blend.name = "curve_ew"
            factors["curve_ew"] = blend

    return factors
