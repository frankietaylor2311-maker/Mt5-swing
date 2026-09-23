"""Labour productivity (LP) differential FX factors (OECD MEI ULQELP01).

Literature framing (see also ``fred_ulqelp`` docstring)
------------------------------------------------------
- Relative labour-productivity growth → competitiveness / Balassa–Samuelson-
  adjacent FX (OECD MEI ULQELP01; output-per-worker channel).
- Primary: long **high** relative LP YoY growth / short low (productivity).
- Honesty alternate: long **low** LP growth (productivity-stress debtor).
- Companion to ULC §48 (competitiveness = ULC vs productivity). Distinct from
  employment §41, IP §42, ULC §48, Balassa–Samuelson §14 (IP levels), retail
  §46, house-price §35, REER §31, money §33, credit §32, DSR §47, CLI/CCI/BCI,
  WUI, EPU/TPU.

Fixed priors (no holdout tuning)
--------------------------------
- ``high_lp_xs`` (**primary**): long high YoY LP growth / short low
  (n_long=n_short=2 on full G10 foreign panel).
- ``low_lp_xs``: opposite honesty alternate.
- ``high_lp_z_xs``: long high 60m trailing z of LP YoY growth.
- ``lp_chg_xs``: long accelerating LP (positive Δ12 of YoY) / short decelerating
  — same productivity sign as primary on *changes*.
- ``us_lp_stress_fx``: depressed US LP YoY z → long foreign (USD soft /
  lost US productivity).
- ``us_lp_haven_usd``: depressed US LP YoY z → long USD (haven alt).
- ``lp_ew``: EW of ``high_lp_xs``, ``lp_chg_xs``, ``us_lp_stress_fx``.

Score basis (fixed a priori): **YoY growth as reported** (Q657S panel values;
note free FRED ULQELP panel ends ~2023-04..07 — stale).
PIT: loader ``pub_lag_months`` (default 3) + ``signal_lag`` months (default 1)
+ **1 trading-day** weight lag. Explicit: do **not** overlay on the locked
FTMO sleeve.
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
class LabourProdFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # months; quarterly-ish after pub_lag
    n_long: int = 2  # full G10 foreign panel
    n_short: int = 2
    z_window: int = 60  # months for trailing z (5y)
    min_periods: int = 24
    z_low: float = -1.0  # US LP YoY z ≤ z_low → stress / haven tilt
    usd_tilt: float = 0.5  # gross |w| when tilt is on
    cost_bps_side: float = 1.5
    chg_periods: int = 12  # Δ12 of YoY growth (acceleration)


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


def prepare_labour_prod_scores(
    lp_panel: pd.DataFrame,
    *,
    cfg: LabourProdFxConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build signed monthly score panels from PIT LP YoY (already pub-lagged).

    Scores exclude USD column when present (XS sorts are foreign vs USD).
    Panel values are already YoY growth % — used directly (no pct_change).
    """
    cfg = cfg or LabourProdFxConfig()
    panel = lp_panel.copy()
    panel.index = _month_start(pd.DatetimeIndex(panel.index))
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    foreign_cols = [c for c in panel.columns if c.upper() != "USD" and c in CURRENCY_USD_PAIR]
    foreign = panel[foreign_cols]

    out: dict[str, pd.DataFrame] = {}

    # High LP YoY → high score → long (productivity → appreciate)
    high = foreign.copy()
    if cfg.signal_lag > 0:
        high = high.shift(int(cfg.signal_lag))
    out["high_lp"] = high

    # Low LP honesty alternate (productivity-stress debtor premium)
    low = (-foreign).copy()
    if cfg.signal_lag > 0:
        low = low.shift(int(cfg.signal_lag))
    out["low_lp"] = low

    # 5y trailing z of YoY: high z → long (productivity)
    z = pd.DataFrame(
        {
            c: trailing_z(foreign[c], lookback=cfg.z_window, min_periods=cfg.min_periods)
            for c in foreign.columns
        },
        index=foreign.index,
    )
    high_z = z.copy()
    if cfg.signal_lag > 0:
        high_z = high_z.shift(int(cfg.signal_lag))
    out["high_lp_z"] = high_z

    # Acceleration: rising YoY → long (score = +Δ12 of YoY)
    chg = foreign.diff(int(cfg.chg_periods)).copy()
    if cfg.signal_lag > 0:
        chg = chg.shift(int(cfg.signal_lag))
    out["lp_chg"] = chg

    return out


def _scores_to_daily_returns(
    score: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cfg: LabourProdFxConfig,
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


def _us_lp_tilt_returns(
    us_lp: pd.Series,
    pair_ret: pd.DataFrame,
    *,
    cfg: LabourProdFxConfig,
    mode: str,
    name: str,
) -> pd.Series:
    """Binary USD / FX tilt from lagged US LP YoY z.

    mode='stress': z ≤ z_low → long foreign (short USD) — US LP depressed /
    lost productivity.
    mode='haven': z ≤ z_low → long USD — safe-haven alternate.
    """
    s = us_lp.copy()
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

    on = (z_daily <= float(cfg.z_low)).astype(float)
    on = on.where(z_daily.notna(), 0.0)
    intensity = on * float(cfg.usd_tilt)

    cols = [c for c in pair_ret.columns if c.upper() in {p.upper() for p in USD_LONG_PAIRS}]
    if not cols:
        cols = list(pair_ret.columns)
    n = max(len(cols), 1)
    w = pd.DataFrame(0.0, index=pair_ret.index, columns=list(pair_ret.columns))
    for sym in cols:
        usd_sign = USD_LONG_PAIRS.get(sym.upper(), 0)
        if mode == "stress":
            w[sym] = (intensity / n) * (-usd_sign)
        else:
            w[sym] = (intensity / n) * usd_sign
    r = portfolio_returns_from_pair_weights(w, pair_ret)
    r = apply_costs(r, w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


def labour_prod_factor_returns(
    pair_ret: pd.DataFrame,
    lp_panel: pd.DataFrame,
    *,
    cfg: LabourProdFxConfig | None = None,
    us_lp_override: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Daily returns for LP-growth factors + US tilts + EW."""
    cfg = cfg or LabourProdFxConfig()
    scores = prepare_labour_prod_scores(lp_panel, cfg=cfg)
    factors: dict[str, pd.Series] = {}

    if "high_lp" in scores:
        factors["high_lp_xs"] = _scores_to_daily_returns(
            scores["high_lp"], pair_ret, cfg, name="high_lp_xs"
        )
    if "low_lp" in scores:
        factors["low_lp_xs"] = _scores_to_daily_returns(
            scores["low_lp"], pair_ret, cfg, name="low_lp_xs"
        )
    if "high_lp_z" in scores:
        factors["high_lp_z_xs"] = _scores_to_daily_returns(
            scores["high_lp_z"], pair_ret, cfg, name="high_lp_z_xs"
        )
    if "lp_chg" in scores:
        factors["lp_chg_xs"] = _scores_to_daily_returns(
            scores["lp_chg"], pair_ret, cfg, name="lp_chg_xs"
        )

    us = us_lp_override
    if us is None and "USD" in lp_panel.columns:
        us = lp_panel["USD"]
    if us is not None and us.dropna().shape[0] >= cfg.min_periods:
        factors["us_lp_stress_fx"] = _us_lp_tilt_returns(
            us, pair_ret, cfg=cfg, mode="stress", name="us_lp_stress_fx"
        )
        factors["us_lp_haven_usd"] = _us_lp_tilt_returns(
            us, pair_ret, cfg=cfg, mode="haven", name="us_lp_haven_usd"
        )

    blend_keys = [
        k
        for k in ("high_lp_xs", "lp_chg_xs", "us_lp_stress_fx")
        if k in factors
    ]
    if blend_keys:
        ew = sum(factors[k] for k in blend_keys) / len(blend_keys)
        ew.name = "lp_ew"
        factors["lp_ew"] = ew

    return factors
