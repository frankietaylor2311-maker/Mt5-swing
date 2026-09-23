"""BIS real broad REER undervaluation / mean-reversion FX factors.

Literature framing (see also ``fred_bis_reer`` docstring)
---------------------------------------------------------
- Rogoff (1996) PPP puzzle / slow real-FX mean reversion.
- Taylor / REER misalignment: long undervalued (low REER z) / short overvalued.
- Optional secondary: REER *momentum* (high z → long) as documented contrast prior.
- Official BIS multilateral REER via FRED ``RB*BIS`` — **not** bilateral CPI DIY
  PPP (``ppp_real_fx.py``).

Fixed priors (no holdout tuning)
--------------------------------
- ``reer_cheap_xs`` (**primary**): long low 60m trailing REER z / short high z
  (undervalued → subsequent appreciation prior).
- ``reer_cheap_xs_36``: same sort on 36m z (board row; fixed a priori alt lookback).
- ``reer_mom_xs``: long high z / short low z (cheap→cheaper momentum contrast).
- ``reer_chg_xs``: long falling REER (Δ12 < 0) / short rising — change vs level.
- ``us_reer_strong_usd``: elevated US REER z → long USD (dollar-strength cooler).
- ``us_reer_meanrev_fx``: elevated US REER z → long foreign (dollar mean-reversion).
- ``reer_ew``: EW of ``reer_cheap_xs``, ``reer_chg_xs``, ``us_reer_meanrev_fx``.

PIT: loader ``pub_lag_months`` (default 2) + ``signal_lag`` months + 1 trading day
weight lag. Distinct from homemade PPP, TB (§30), debt (§29), fiscal (§28), CA,
CB-BS, macro-diff, equity-diff, AI-GPR, ToT, dollar-beta, crash-skew.

Explicit: do **not** overlay on the locked FTMO sleeve.
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
from mt5_swing.strategies.gpr_regime import USD_LONG_PAIRS
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class BisReerFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # months after publication-lagged REER is known
    n_long: int = 2
    n_short: int = 2
    z_window: int = 60  # primary trailing z lookback (months)
    z_window_alt: int = 36  # board-only alternate lookback
    min_periods: int = 24
    z_high: float = 1.0  # |z| threshold for US REER tilt
    usd_tilt: float = 0.5  # gross |w| when tilt is on
    cost_bps_side: float = 1.5
    chg_periods: int = 12  # YoY change on monthly REER


def _month_start(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    return pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start"), tz="UTC")


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


def trailing_z(s: pd.Series, *, lookback: int, min_periods: int) -> pd.Series:
    mu = s.rolling(lookback, min_periods=min_periods).mean()
    sd = s.rolling(lookback, min_periods=min_periods).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def _panel_trailing_z(
    panel: pd.DataFrame,
    *,
    lookback: int,
    min_periods: int,
) -> pd.DataFrame:
    cols = {}
    for c in panel.columns:
        cols[c] = trailing_z(panel[c], lookback=lookback, min_periods=min_periods)
    return pd.DataFrame(cols, index=panel.index)


def prepare_reer_scores(
    reer_panel: pd.DataFrame,
    *,
    cfg: BisReerFxConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build signed monthly score panels from PIT BIS REER (already pub-lagged).

    Scores exclude USD column when present (XS sorts are foreign vs USD).
    """
    cfg = cfg or BisReerFxConfig()
    panel = reer_panel.copy()
    panel.index = _month_start(pd.DatetimeIndex(panel.index))
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    foreign_cols = [c for c in panel.columns if c.upper() != "USD" and c in CURRENCY_USD_PAIR]
    foreign = panel[foreign_cols]

    out: dict[str, pd.DataFrame] = {}

    z60 = _panel_trailing_z(foreign, lookback=cfg.z_window, min_periods=cfg.min_periods)
    z36 = _panel_trailing_z(
        foreign, lookback=cfg.z_window_alt, min_periods=min(cfg.min_periods, cfg.z_window_alt)
    )

    # Cheap / undervalued: low REER z → high score → long
    cheap = (-z60).copy()
    if cfg.signal_lag > 0:
        cheap = cheap.shift(int(cfg.signal_lag))
    out["reer_cheap"] = cheap

    cheap36 = (-z36).copy()
    if cfg.signal_lag > 0:
        cheap36 = cheap36.shift(int(cfg.signal_lag))
    out["reer_cheap_36"] = cheap36

    # Momentum: high REER z → high score → long (contrast prior)
    mom = z60.copy()
    if cfg.signal_lag > 0:
        mom = mom.shift(int(cfg.signal_lag))
    out["reer_mom"] = mom

    # Change: falling REER (cheapening) → long
    chg = (-foreign.diff(int(cfg.chg_periods))).copy()
    if cfg.signal_lag > 0:
        chg = chg.shift(int(cfg.signal_lag))
    out["reer_chg"] = chg

    return out


def _scores_to_daily_returns(
    score: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cfg: BisReerFxConfig,
    *,
    name: str,
) -> pd.Series:
    ccy_w = _rank_sort_weights(score, n_long=cfg.n_long, n_short=cfg.n_short)
    if ccy_w.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)
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
    for c in pair_ret.columns:
        if c not in daily_w.columns:
            daily_w[c] = 0.0
    daily_w = daily_w.reindex(columns=list(pair_ret.columns)).fillna(0.0)
    r = portfolio_returns_from_pair_weights(daily_w, pair_ret)
    r = apply_costs(r, daily_w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


def _us_reer_tilt_returns(
    us_reer: pd.Series,
    pair_ret: pd.DataFrame,
    *,
    cfg: BisReerFxConfig,
    mode: str,
    name: str,
) -> pd.Series:
    """Binary USD / FX tilt from lagged US BIS REER z.

    mode='strong': z ≥ +z_high → long USD (dollar strength / overvalued cooler).
    mode='meanrev': z ≥ +z_high → long foreign (dollar mean-reversion prior).
    """
    s = us_reer.copy()
    s.index = _month_start(pd.DatetimeIndex(s.index))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    z = trailing_z(s, lookback=cfg.z_window, min_periods=cfg.min_periods)
    if cfg.signal_lag > 0:
        z = z.shift(int(cfg.signal_lag))

    z_me = z.copy()
    z_me.index = (
        pd.DatetimeIndex(z_me.index)
        .tz_convert(None)
        .to_period("M")
        .to_timestamp(how="end")
        .tz_localize("UTC")
    )
    z_daily = z_me.reindex(pair_ret.index, method="ffill").shift(1)

    on = (z_daily >= float(cfg.z_high)).astype(float)
    on = on.where(z_daily.notna(), 0.0)
    intensity = on * float(cfg.usd_tilt)

    cols = [c for c in pair_ret.columns if c.upper() in {p.upper() for p in USD_LONG_PAIRS}]
    if not cols:
        cols = list(pair_ret.columns)
    n = max(len(cols), 1)
    w = pd.DataFrame(0.0, index=pair_ret.index, columns=list(pair_ret.columns))
    for sym in cols:
        usd_sign = USD_LONG_PAIRS.get(sym.upper(), 0)
        if mode == "strong":
            w[sym] = (intensity / n) * usd_sign
        else:
            w[sym] = (intensity / n) * (-usd_sign)
    r = portfolio_returns_from_pair_weights(w, pair_ret)
    r = apply_costs(r, w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


def bis_reer_factor_returns(
    pair_ret: pd.DataFrame,
    reer_panel: pd.DataFrame,
    *,
    cfg: BisReerFxConfig | None = None,
    us_reer_override: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Daily returns for BIS REER factors + US tilts + EW."""
    cfg = cfg or BisReerFxConfig()
    scores = prepare_reer_scores(reer_panel, cfg=cfg)
    factors: dict[str, pd.Series] = {}

    if "reer_cheap" in scores:
        factors["reer_cheap_xs"] = _scores_to_daily_returns(
            scores["reer_cheap"], pair_ret, cfg, name="reer_cheap_xs"
        )
    if "reer_cheap_36" in scores:
        factors["reer_cheap_xs_36"] = _scores_to_daily_returns(
            scores["reer_cheap_36"], pair_ret, cfg, name="reer_cheap_xs_36"
        )
    if "reer_mom" in scores:
        factors["reer_mom_xs"] = _scores_to_daily_returns(
            scores["reer_mom"], pair_ret, cfg, name="reer_mom_xs"
        )
    if "reer_chg" in scores:
        factors["reer_chg_xs"] = _scores_to_daily_returns(
            scores["reer_chg"], pair_ret, cfg, name="reer_chg_xs"
        )

    us = us_reer_override
    if us is None and "USD" in reer_panel.columns:
        us = reer_panel["USD"]
    if us is not None and us.dropna().shape[0] >= cfg.min_periods:
        factors["us_reer_strong_usd"] = _us_reer_tilt_returns(
            us, pair_ret, cfg=cfg, mode="strong", name="us_reer_strong_usd"
        )
        factors["us_reer_meanrev_fx"] = _us_reer_tilt_returns(
            us, pair_ret, cfg=cfg, mode="meanrev", name="us_reer_meanrev_fx"
        )

    blend_keys = [
        k for k in ("reer_cheap_xs", "reer_chg_xs", "us_reer_meanrev_fx") if k in factors
    ]
    if blend_keys:
        ew = sum(factors[k] for k in blend_keys) / len(blend_keys)
        ew.name = "reer_ew"
        factors["reer_ew"] = ew

    return factors
