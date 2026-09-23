"""Central-bank balance-sheet / QE differential → FX factors (frozen scholarly).

Literature priors (not holdout-tuned)
------------------------------------
- Neely (2015) / Bauer–Neely (2014): Fed LSAP / QE depreciates the USD via
  portfolio-balance and signalling (primary channel here).
- Gagnon et al. (2011): LSAPs compress term premia — related portfolio-balance
  channel for FX when CB balance sheets expand relatively.
- Cross-CB: currencies whose central bank expands assets faster tend to
  *depreciate* on the portfolio-balance prior (long low-growth / short
  high-growth foreign BS YoY).

Legs
----
- ``walcl_pb_fx`` (**primary**): high lagged z(WALCL YoY) → long foreign / short
  USD (portfolio-balance depreciation of expanding Fed).
- ``walcl_haven_usd``: alternate — high Fed BS growth often coincides with
  crisis / risk-off → long USD (reported for honesty).
- ``walcl_chg_pb_fx``: same PB tilt using z(Δ WALCL) / short-horizon change.
- ``bs_diff_pb_fx``: high (US − peer) YoY differential → long foreign / short USD.
- ``bs_peer_xs``: among EUR/JPY, long low BS YoY / short high BS YoY.
- ``bs_gdp_pb_fx``: high z(US BS/GDP) → long foreign (level stock prior).
- ``bs_ew``: EW of ``walcl_pb_fx`` + ``bs_diff_pb_fx`` + ``bs_peer_xs``.

PIT: loaders apply weekly/monthly/GDP pub lags; this module adds ``signal_lag``
trading days on aligned z before weights. ``portfolio_returns_from_weights``
applies an extra weight lag.

Explicit: do **not** overlay on the locked FTMO sleeve.
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
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class CbBalanceSheetFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # trading days after pub-lagged macro is known
    z_window: int = 252  # trading days for trailing z on daily-aligned series
    min_periods: int = 60
    z_high: float = 1.0
    usd_tilt: float = 0.5  # gross |w| when tilt is on
    cost_bps_side: float = 1.5
    n_long: int = 1  # peer XS: only EUR/JPY available
    n_short: int = 1
    binary_tilt: bool = True
    chg_periods_weekly: int = 13  # ~quarterly change on weekly WALCL


def trailing_z(s: pd.Series, *, lookback: int, min_periods: int) -> pd.Series:
    mu = s.rolling(lookback, min_periods=min_periods).mean()
    sd = s.rolling(lookback, min_periods=min_periods).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def align_and_z(
    series: pd.Series,
    index: pd.DatetimeIndex,
    *,
    cfg: CbBalanceSheetFxConfig,
) -> pd.Series:
    """As-of align → trailing z → signal_lag trading days."""
    aligned = align_macro_to_index(series, index)
    z = trailing_z(aligned, lookback=cfg.z_window, min_periods=cfg.min_periods)
    if cfg.signal_lag > 0:
        z = z.shift(int(cfg.signal_lag))
    z.name = f"{series.name or 'macro'}_z"
    return z


def _tilt_intensity(z: pd.Series, *, cfg: CbBalanceSheetFxConfig) -> pd.Series:
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
    cfg: CbBalanceSheetFxConfig,
    mode: str,
) -> pd.DataFrame:
    """mode='pb': high z → long foreign (short USD). mode='haven': high z → long USD."""
    intensity = _tilt_intensity(z, cfg=cfg)
    cols = [c for c in pair_columns if c.upper() in {p.upper() for p in USD_LONG_PAIRS}]
    if not cols:
        cols = list(pair_columns)
    n = max(len(cols), 1)
    w = pd.DataFrame(0.0, index=z.index, columns=cols)
    for sym in cols:
        usd_sign = USD_LONG_PAIRS.get(sym.upper(), 0)
        if mode == "pb":
            w[sym] = (intensity / n) * (-usd_sign)
        else:
            w[sym] = (intensity / n) * usd_sign
    return w.fillna(0.0)


def _weights_to_returns(
    w: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: CbBalanceSheetFxConfig,
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


def _peer_xs_returns(
    yoy_panel: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: CbBalanceSheetFxConfig,
    name: str,
) -> pd.Series:
    """Long low BS YoY / short high BS YoY among foreign CBs with USD pairs."""
    foreign = [c for c in yoy_panel.columns if c.upper() != "USD" and c in CURRENCY_USD_PAIR]
    if len(foreign) < cfg.n_long + cfg.n_short:
        return pd.Series(0.0, index=pair_ret.index, name=name)

    # Score = −YoY so low expansion ranks high → long
    score = (-yoy_panel[foreign]).copy()
    # Align to month-end for weight expansion (weekly panel → month-end last)
    score.index = _ensure_utc(pd.DatetimeIndex(score.index))
    # Resample to month-end (last known PIT value in month)
    score_m = score.resample("ME").last()
    # Extra month signal lag on top of loader pub lag (conservative)
    score_m = score_m.shift(1)

    ccy_w = _rank_sort_weights(score_m, n_long=cfg.n_long, n_short=cfg.n_short)
    if ccy_w.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)
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


def cb_balance_sheet_factor_returns(
    pair_ret: pd.DataFrame,
    *,
    walcl_yoy: pd.Series | None = None,
    walcl_level: pd.Series | None = None,
    us_peer_diff: pd.Series | None = None,
    yoy_panel: pd.DataFrame | None = None,
    us_bs_gdp: pd.Series | None = None,
    cfg: CbBalanceSheetFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Daily returns for CB balance-sheet / QE differential FX factors."""
    cfg = cfg or CbBalanceSheetFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    cols = list(pret.columns)
    out: dict[str, pd.Series] = {}

    if walcl_yoy is not None and len(walcl_yoy):
        z = align_and_z(walcl_yoy, pret.index, cfg=cfg)
        out["walcl_pb_fx"] = _weights_to_returns(
            usd_or_fx_tilt_weights(z, cols, cfg=cfg, mode="pb"),
            pret,
            cfg=cfg,
            name="walcl_pb_fx",
        )
        out["walcl_haven_usd"] = _weights_to_returns(
            usd_or_fx_tilt_weights(z, cols, cfg=cfg, mode="haven"),
            pret,
            cfg=cfg,
            name="walcl_haven_usd",
        )

    if walcl_level is not None and len(walcl_level):
        # Short-horizon change on weekly assets → align → z
        chg = walcl_level.diff(int(cfg.chg_periods_weekly))
        chg.name = "walcl_chg"
        z = align_and_z(chg, pret.index, cfg=cfg)
        out["walcl_chg_pb_fx"] = _weights_to_returns(
            usd_or_fx_tilt_weights(z, cols, cfg=cfg, mode="pb"),
            pret,
            cfg=cfg,
            name="walcl_chg_pb_fx",
        )

    if us_peer_diff is not None and len(us_peer_diff):
        z = align_and_z(us_peer_diff, pret.index, cfg=cfg)
        out["bs_diff_pb_fx"] = _weights_to_returns(
            usd_or_fx_tilt_weights(z, cols, cfg=cfg, mode="pb"),
            pret,
            cfg=cfg,
            name="bs_diff_pb_fx",
        )

    if yoy_panel is not None and not yoy_panel.empty:
        out["bs_peer_xs"] = _peer_xs_returns(
            yoy_panel, pret, cfg=cfg, name="bs_peer_xs"
        )

    if us_bs_gdp is not None and len(us_bs_gdp):
        z = align_and_z(us_bs_gdp, pret.index, cfg=cfg)
        out["bs_gdp_pb_fx"] = _weights_to_returns(
            usd_or_fx_tilt_weights(z, cols, cfg=cfg, mode="pb"),
            pret,
            cfg=cfg,
            name="bs_gdp_pb_fx",
        )

    blend_keys = [k for k in ("walcl_pb_fx", "bs_diff_pb_fx", "bs_peer_xs") if k in out]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "bs_ew"
        out["bs_ew"] = blend

    return out
