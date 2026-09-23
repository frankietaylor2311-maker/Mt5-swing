"""CFTC COT / speculative-positioning FX factors (frozen scholarly construction).

Literature priors (not holdout-tuned)
------------------------------------
- Speculative pressure / positioning: elevated non-commercial or leveraged-fund
  net long positions co-move with FX; weekly *changes* relate to exchange-rate
  moves (Klitgaard & Weir 2004 NY Fed EPR; related COT predictive-content work).
- Classic traders also treat *extremes* of net positioning as mean-reversion
  setups — we pre-specify a z-score contrarian leg separately from continuation.

Legs
----
- ``cot_lev_net_xs``: long high / short low TFF leveraged-money net/OI
  (continuation).
- ``cot_lev_chg_xs``: same sort on weekly Δ(lev_net/OI).
- ``cot_lev_z_xs``: sort on trailing z of lev_net/OI (continuation).
- ``cot_lev_z_mr_xs``: sort on −z (mean-reversion / fade extremes).
- ``cot_noncomm_net_xs``: Legacy NonComm net/OI continuation sort.
- ``cot_dx_usd``: DX leveraged net/OI → long USD (short EW foreign) when DX
  speculative net is elevated; flat when unavailable.
- ``cot_ew``: equal-weight of ``cot_lev_net_xs`` and ``cot_lev_chg_xs``.

PIT: panel index must already be *known_date* (report + release_lag). Extra
``signal_lag`` trading days applied on daily weight expansion (default 1).
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
from mt5_swing.strategies.yield_curve_fx import apply_costs


G10_FX: tuple[str, ...] = ("EUR", "GBP", "JPY", "CAD", "CHF", "AUD", "NZD")


@dataclass
class CotPositioningFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # trading days after known_date weights
    n_long: int = 2
    n_short: int = 2
    z_window: int = 52  # weeks
    min_periods: int = 26
    cost_bps_side: float = 1.5
    currencies: tuple[str, ...] = G10_FX
    build_lev_net: bool = True
    build_lev_chg: bool = True
    build_lev_z: bool = True
    build_lev_z_mr: bool = True
    build_noncomm: bool = True
    build_dx_usd: bool = True
    build_ew: bool = True


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


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _scores_to_daily_returns(
    score: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cfg: CotPositioningFxConfig,
    *,
    name: str,
) -> tuple[pd.Series, pd.DataFrame]:
    ccy_w = _rank_sort_weights(score, n_long=cfg.n_long, n_short=cfg.n_short)
    if ccy_w.empty:
        empty = pd.Series(0.0, index=pair_ret.index, name=name)
        return empty, pd.DataFrame(0.0, index=pair_ret.index, columns=[])
    keep = [c for c in ccy_w.columns if c in CURRENCY_USD_PAIR]
    ccy_w = ccy_w[keep]
    ccy_w.index = _ensure_utc(pd.DatetimeIndex(ccy_w.index))
    pair_w = currency_weights_to_pair_weights_fx(ccy_w)
    daily_w = expand_monthly_weights_to_daily(
        pair_w, pair_ret.index, signal_lag_days=int(cfg.signal_lag)
    )
    r = portfolio_returns_from_pair_weights(daily_w, pair_ret)
    r.name = name
    r = apply_costs(r, daily_w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r, daily_w


def dx_usd_tilt_returns(
    dx_score: pd.Series,
    pair_ret: pd.DataFrame,
    *,
    cfg: CotPositioningFxConfig | None = None,
) -> pd.Series:
    """Map DX speculative net to a USD tilt: long USD when DX score > 0.

    Implementation: equal-weight short all available foreign currencies when
    DX score positive; long foreign (short USD) when negative; flat at 0.
    Gross exposure |sum| ≈ 1 on the active side.
    """
    cfg = cfg or CotPositioningFxConfig()
    if dx_score is None or dx_score.dropna().empty:
        return pd.Series(0.0, index=pair_ret.index, name="cot_dx_usd")

    fx_cols = [c for c in cfg.currencies if c in CURRENCY_USD_PAIR]
    # Build sparse currency weights on known_dates
    rows = []
    for dt, val in dx_score.dropna().items():
        v = float(val)
        w = pd.Series(0.0, index=fx_cols)
        if abs(v) < 1e-15 or not np.isfinite(v):
            continue
        # High DX speculative long → long USD → short foreign
        sign = -1.0 if v > 0 else +1.0
        w.loc[fx_cols] = sign / len(fx_cols)
        w.name = dt
        rows.append(w)
    if not rows:
        return pd.Series(0.0, index=pair_ret.index, name="cot_dx_usd")
    ccy_w = pd.DataFrame(rows).sort_index()
    ccy_w.index = _ensure_utc(pd.DatetimeIndex(ccy_w.index))
    pair_w = currency_weights_to_pair_weights_fx(ccy_w)
    daily_w = expand_monthly_weights_to_daily(
        pair_w, pair_ret.index, signal_lag_days=int(cfg.signal_lag)
    )
    r = portfolio_returns_from_pair_weights(daily_w, pair_ret)
    r.name = "cot_dx_usd"
    r = apply_costs(r, daily_w, bps_side=cfg.cost_bps_side)
    r.name = "cot_dx_usd"
    return r


def prepare_cot_scores(
    lev_net_oi: pd.DataFrame,
    *,
    nc_net_oi: pd.DataFrame | None = None,
    cfg: CotPositioningFxConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build weekly score panels from lev_net_oi (and optional legacy nc)."""
    cfg = cfg or CotPositioningFxConfig()
    out: dict[str, pd.DataFrame] = {}
    panel = lev_net_oi.copy()
    panel.index = _ensure_utc(pd.DatetimeIndex(panel.index))
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    cols = [c for c in cfg.currencies if c in panel.columns]
    panel = panel[cols]

    if cfg.build_lev_net:
        out["cot_lev_net_xs"] = panel

    if cfg.build_lev_chg:
        out["cot_lev_chg_xs"] = panel.diff()

    z = trailing_z(panel, lookback=cfg.z_window, min_periods=cfg.min_periods)
    if cfg.build_lev_z:
        out["cot_lev_z_xs"] = z
    if cfg.build_lev_z_mr:
        out["cot_lev_z_mr_xs"] = -z

    if cfg.build_noncomm and nc_net_oi is not None and not nc_net_oi.empty:
        nc = nc_net_oi.copy()
        nc.index = _ensure_utc(pd.DatetimeIndex(nc.index))
        nc = nc[~nc.index.duplicated(keep="last")].sort_index()
        nc_cols = [c for c in cfg.currencies if c in nc.columns]
        if nc_cols:
            out["cot_noncomm_net_xs"] = nc[nc_cols]

    return out


def cot_positioning_factor_returns(
    pair_ret: pd.DataFrame,
    lev_net_oi: pd.DataFrame,
    *,
    nc_net_oi: pd.DataFrame | None = None,
    dx_lev_net_oi: pd.Series | None = None,
    cfg: CotPositioningFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Daily portfolio returns for each frozen COT positioning leg."""
    cfg = cfg or CotPositioningFxConfig()
    scores = prepare_cot_scores(lev_net_oi, nc_net_oi=nc_net_oi, cfg=cfg)
    factors: dict[str, pd.Series] = {}
    for name, sc in scores.items():
        r, _ = _scores_to_daily_returns(sc, pair_ret, cfg, name=name)
        factors[name] = r

    if cfg.build_dx_usd and dx_lev_net_oi is not None:
        factors["cot_dx_usd"] = dx_usd_tilt_returns(dx_lev_net_oi, pair_ret, cfg=cfg)

    if cfg.build_ew:
        legs = [factors[k] for k in ("cot_lev_net_xs", "cot_lev_chg_xs") if k in factors]
        if legs:
            blend = sum(legs) / len(legs)
            blend.name = "cot_ew"
            factors["cot_ew"] = blend

    return factors
