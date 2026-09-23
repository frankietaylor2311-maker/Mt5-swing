"""Real-rate / breakeven inflation → FX factors (frozen scholarly).

Literature priors (not holdout-tuned)
------------------------------------
- Frankel (1979) / Meese–Rogoff (1988): higher *real* interest differentials
  favor the high-real-rate currency (real interest differential / RID channel).
- Dahlquist–Hasseltoft / Lustig–Stathopoulos–Verdelhan: real / local bond
  premia and the term structure of carry matter for FX.
- US TIPS (DFII10) and breakeven (T10YIE) are the free market measures of US
  real rates and inflation expectations; foreign LT−CPI YoY is an honest proxy
  where linkers are unavailable on free FRED.

Legs
----
- ``us_real_usd`` (**primary**): high lagged z(DFII10) → long USD (RID / capital
  inflow prior for the dollar).
- ``us_real_chg_usd``: high z(Δ DFII10) → long USD (change channel).
- ``us_be_fx``: high z(T10YIE) → long foreign / short USD (growth / risk-on
  inflation-expectations prior).
- ``us_be_usd``: alternate — high BE → long USD (stagflation / Fed-hike fear).
- ``rr_xs``: long high / short low *foreign real-proxy − US TIPS* differentials.
- ``rr_z_xs``: same sort on trailing z of the differential (60m).
- ``rr_chg_xs``: sort on 3m change in differential.
- ``rr_ew``: EW of ``us_real_usd`` + ``rr_xs``.

PIT: loaders apply daily/monthly pub lags; this module adds ``signal_lag``
(trading days on daily tilts; months on XS scores) + daily weight lag on XS.
Costs 1.5 bps/side. Explicit: do **not** overlay on the locked FTMO sleeve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.data.macro_uncertainty import CURRENCY_USD_PAIR
from mt5_swing.strategies.carry_rank import portfolio_returns_from_weights
from mt5_swing.strategies.country_gpr_fx import (
    currency_weights_to_pair_weights_fx,
    expand_monthly_weights_to_daily,
)
from mt5_swing.strategies.gpr_regime import USD_LONG_PAIRS, align_macro_to_index
from mt5_swing.strategies.yield_curve_fx import apply_costs, apply_signal_lag, trailing_z


@dataclass
class RealRateFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag_days: int = 1  # trading days after pub-lagged daily macro
    signal_lag_months: int = 1  # months after pub-lagged monthly differentials
    z_window_daily: int = 252
    z_window_monthly: int = 60
    min_periods_daily: int = 60
    min_periods_monthly: int = 24
    z_high: float = 1.0
    usd_tilt: float = 0.5
    cost_bps_side: float = 1.5
    n_long: int = 2
    n_short: int = 2
    binary_tilt: bool = True
    chg_days: int = 21  # ~1m change on daily DFII10
    chg_months: int = 3  # 3m change on monthly differentials


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def trailing_z_series(s: pd.Series, *, lookback: int, min_periods: int) -> pd.Series:
    mu = s.rolling(lookback, min_periods=min_periods).mean()
    sd = s.rolling(lookback, min_periods=min_periods).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def align_and_z_daily(
    series: pd.Series,
    index: pd.DatetimeIndex,
    *,
    cfg: RealRateFxConfig,
) -> pd.Series:
    """As-of align → trailing z → signal_lag trading days."""
    aligned = align_macro_to_index(series, index)
    z = trailing_z_series(
        aligned, lookback=cfg.z_window_daily, min_periods=cfg.min_periods_daily
    )
    if cfg.signal_lag_days > 0:
        z = z.shift(int(cfg.signal_lag_days))
    z.name = f"{series.name or 'macro'}_z"
    return z


def _tilt_intensity(z: pd.Series, *, cfg: RealRateFxConfig) -> pd.Series:
    if cfg.binary_tilt:
        intensity = (z >= cfg.z_high).astype(float) * float(cfg.usd_tilt)
        return intensity.where(z.notna(), 0.0)
    intensity = (z.clip(lower=0.0) / max(float(cfg.z_high), 1e-6)).clip(0.0, 1.0)
    intensity = intensity * float(cfg.usd_tilt)
    return intensity.fillna(0.0)


def usd_or_fx_tilt_weights(
    z: pd.Series,
    pair_columns: list[str],
    *,
    cfg: RealRateFxConfig,
    mode: str,
) -> pd.DataFrame:
    """mode='usd': high z → long USD. mode='fx': high z → long foreign."""
    intensity = _tilt_intensity(z, cfg=cfg)
    cols = [c for c in pair_columns if c.upper() in {p.upper() for p in USD_LONG_PAIRS}]
    if not cols:
        cols = list(pair_columns)
    n = max(len(cols), 1)
    w = pd.DataFrame(0.0, index=z.index, columns=cols)
    for sym in cols:
        usd_sign = USD_LONG_PAIRS.get(sym.upper(), 0)
        if mode == "usd":
            w[sym] = (intensity / n) * usd_sign
        else:
            w[sym] = (intensity / n) * (-usd_sign)
    return w.fillna(0.0)


def _weights_to_returns(
    w: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: RealRateFxConfig,
    name: str,
) -> pd.Series:
    cols = list(pair_ret.columns)
    for c in cols:
        if c not in w.columns:
            w[c] = 0.0
    w = w.reindex(columns=cols).fillna(0.0)
    r = portfolio_returns_from_weights(w, pair_ret)
    r = apply_costs(r, w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


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


def _xs_score_to_returns(
    score: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: RealRateFxConfig,
    name: str,
) -> pd.Series:
    """Monthly score panel → daily pair returns via rank sort + weight lag."""
    keep = [c for c in score.columns if c in CURRENCY_USD_PAIR]
    if len(keep) < cfg.n_long + cfg.n_short:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    sc = apply_signal_lag(score[keep], cfg.signal_lag_months)
    ccy_w = _rank_sort_weights(sc, n_long=cfg.n_long, n_short=cfg.n_short)
    if ccy_w.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    # Month-start scores → month-end timestamps for expand helper
    ccy_w.index = (
        pd.DatetimeIndex(ccy_w.index)
        .tz_convert(None)
        .to_period("M")
        .to_timestamp(how="end")
        .tz_localize("UTC")
    )
    pair_w = currency_weights_to_pair_weights_fx(ccy_w)
    daily_w = expand_monthly_weights_to_daily(pair_w, pair_ret.index, signal_lag_days=1)
    for c in pair_ret.columns:
        if c not in daily_w.columns:
            daily_w[c] = 0.0
    daily_w = daily_w.reindex(columns=list(pair_ret.columns)).fillna(0.0)
    r = portfolio_returns_from_weights(daily_w, pair_ret)
    r = apply_costs(r, daily_w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


def real_rate_factor_returns(
    pair_ret: pd.DataFrame,
    *,
    us_real: pd.Series | None = None,
    us_be: pd.Series | None = None,
    rr_diff: pd.DataFrame | None = None,
    cfg: RealRateFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Daily returns for real-rate / breakeven FX factors."""
    cfg = cfg or RealRateFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    cols = list(pret.columns)
    out: dict[str, pd.Series] = {}

    if us_real is not None and len(us_real):
        z = align_and_z_daily(us_real, pret.index, cfg=cfg)
        out["us_real_usd"] = _weights_to_returns(
            usd_or_fx_tilt_weights(z, cols, cfg=cfg, mode="usd"),
            pret,
            cfg=cfg,
            name="us_real_usd",
        )
        chg = us_real.diff(int(cfg.chg_days))
        chg.name = "DFII10_chg"
        zc = align_and_z_daily(chg, pret.index, cfg=cfg)
        out["us_real_chg_usd"] = _weights_to_returns(
            usd_or_fx_tilt_weights(zc, cols, cfg=cfg, mode="usd"),
            pret,
            cfg=cfg,
            name="us_real_chg_usd",
        )

    if us_be is not None and len(us_be):
        z = align_and_z_daily(us_be, pret.index, cfg=cfg)
        out["us_be_fx"] = _weights_to_returns(
            usd_or_fx_tilt_weights(z, cols, cfg=cfg, mode="fx"),
            pret,
            cfg=cfg,
            name="us_be_fx",
        )
        out["us_be_usd"] = _weights_to_returns(
            usd_or_fx_tilt_weights(z, cols, cfg=cfg, mode="usd"),
            pret,
            cfg=cfg,
            name="us_be_usd",
        )

    if rr_diff is not None and not rr_diff.empty:
        out["rr_xs"] = _xs_score_to_returns(rr_diff, pret, cfg=cfg, name="rr_xs")
        z_panel = trailing_z(
            rr_diff,
            lookback=cfg.z_window_monthly,
            min_periods=cfg.min_periods_monthly,
        )
        out["rr_z_xs"] = _xs_score_to_returns(z_panel, pret, cfg=cfg, name="rr_z_xs")
        chg_m = rr_diff.diff(int(cfg.chg_months))
        out["rr_chg_xs"] = _xs_score_to_returns(chg_m, pret, cfg=cfg, name="rr_chg_xs")

    blend_keys = [k for k in ("us_real_usd", "rr_xs") if k in out]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "rr_ew"
        out["rr_ew"] = blend

    return out
