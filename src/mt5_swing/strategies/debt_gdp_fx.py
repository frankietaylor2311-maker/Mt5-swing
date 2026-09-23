"""Government debt/GDP differential FX factors (fiscal sustainability / debt overhang).

Fixed priors (no holdout tuning)
--------------------------------
- ``low_debt_xs`` (**primary**, fiscal sustainability / debt overhang):
  long low debt/GDP / short high debt/GDP → stronger relative sustainability
  → subsequent appreciation prior.
- ``high_debt_xs``: opposite debtor-premium sort (long high debt / short low)
  — Della Corte–Riddiough–Sarno-style honesty alternate.
- ``debt_chg_xs``: long improving Δ(debt/GDP) i.e. falling debt / short rising.
- ``us_debt_twin_fx``: elevated US debt/GDP z → long foreign / short USD
  (debt-overhang USD depreciation / external-adjustment prior).
- ``us_debt_haven_usd``: elevated US debt z → long USD (safe-haven alternate).
- ``debt_fiscal_blend``: scholarly EW of ``low_debt_xs`` + ``fiscal_surplus_xs``
  when fiscal surplus returns are supplied (joint stock–flow channel) — **not**
  an overlay on the locked FTMO sleeve.
- ``debt_ew``: EW of ``low_debt_xs``, ``debt_chg_xs``, ``us_debt_twin_fx``.

PIT: loader ``pub_lag_months`` (default 15 for annual WEO) + ``signal_lag`` months
+ 1 trading day weight lag. Distinct from GGNLBA fiscal (§28), CA/GDP (§21),
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
class DebtGdpFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # months after publication-lagged debt is known
    n_long: int = 2
    n_short: int = 2
    z_window: int = 60  # months for trailing z (US debt state)
    min_periods: int = 24
    z_high: float = 1.0  # |z| threshold for US debt tilt
    usd_tilt: float = 0.5  # gross |w| when tilt is on
    cost_bps_side: float = 1.5
    chg_periods: int = 12  # YoY change on monthly-ffilled annual debt


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


def prepare_debt_scores(
    debt_panel: pd.DataFrame,
    *,
    cfg: DebtGdpFxConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build signed monthly score panels from PIT debt/GDP (already pub-lagged).

    Scores exclude USD column when present (XS sorts are foreign vs USD).
    """
    cfg = cfg or DebtGdpFxConfig()
    panel = debt_panel.copy()
    panel.index = _month_start(pd.DatetimeIndex(panel.index))
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    foreign_cols = [c for c in panel.columns if c.upper() != "USD" and c in CURRENCY_USD_PAIR]
    foreign = panel[foreign_cols]

    out: dict[str, pd.DataFrame] = {}

    # Low debt / sustainability: low debt/GDP → high score → long
    low = (-foreign).copy()
    if cfg.signal_lag > 0:
        low = low.shift(int(cfg.signal_lag))
    out["low_debt"] = low

    # High debt risk premium: high debt → high score → long
    high = foreign.copy()
    if cfg.signal_lag > 0:
        high = high.shift(int(cfg.signal_lag))
    out["high_debt"] = high

    # Change: improving = falling debt → long (score = −Δ debt)
    chg = (-foreign.diff(int(cfg.chg_periods))).copy()
    if cfg.signal_lag > 0:
        chg = chg.shift(int(cfg.signal_lag))
    out["debt_chg"] = chg

    return out


def _scores_to_daily_returns(
    score: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cfg: DebtGdpFxConfig,
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


def _us_debt_tilt_returns(
    us_debt: pd.Series,
    pair_ret: pd.DataFrame,
    *,
    cfg: DebtGdpFxConfig,
    mode: str,
    name: str,
) -> pd.Series:
    """Binary USD / FX tilt from lagged US debt/GDP z.

    mode='twin': z ≥ +z_high → long foreign (short USD) — debt-overhang adjustment.
    mode='haven': z ≥ +z_high → long USD — safe-haven alternate.
    """
    s = us_debt.copy()
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
        if mode == "twin":
            w[sym] = (intensity / n) * (-usd_sign)
        else:
            # haven → long USD
            w[sym] = (intensity / n) * usd_sign
    r = portfolio_returns_from_pair_weights(w, pair_ret)
    r = apply_costs(r, w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


def debt_gdp_factor_returns(
    pair_ret: pd.DataFrame,
    debt_panel: pd.DataFrame,
    *,
    cfg: DebtGdpFxConfig | None = None,
    us_debt_override: pd.Series | None = None,
    fiscal_surplus_returns: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Daily returns for debt/GDP factors + optional fiscal blend + EW."""
    cfg = cfg or DebtGdpFxConfig()
    scores = prepare_debt_scores(debt_panel, cfg=cfg)
    factors: dict[str, pd.Series] = {}

    if "low_debt" in scores:
        factors["low_debt_xs"] = _scores_to_daily_returns(
            scores["low_debt"], pair_ret, cfg, name="low_debt_xs"
        )
    if "high_debt" in scores:
        factors["high_debt_xs"] = _scores_to_daily_returns(
            scores["high_debt"], pair_ret, cfg, name="high_debt_xs"
        )
    if "debt_chg" in scores:
        factors["debt_chg_xs"] = _scores_to_daily_returns(
            scores["debt_chg"], pair_ret, cfg, name="debt_chg_xs"
        )

    us = us_debt_override
    if us is None and "USD" in debt_panel.columns:
        us = debt_panel["USD"]
    if us is not None and us.dropna().shape[0] >= cfg.min_periods:
        factors["us_debt_twin_fx"] = _us_debt_tilt_returns(
            us, pair_ret, cfg=cfg, mode="twin", name="us_debt_twin_fx"
        )
        factors["us_debt_haven_usd"] = _us_debt_tilt_returns(
            us, pair_ret, cfg=cfg, mode="haven", name="us_debt_haven_usd"
        )

    if fiscal_surplus_returns is not None and "low_debt_xs" in factors:
        a = factors["low_debt_xs"]
        b = fiscal_surplus_returns.reindex(a.index).fillna(0.0)
        blend = 0.5 * a + 0.5 * b
        blend.name = "debt_fiscal_blend"
        factors["debt_fiscal_blend"] = blend

    blend_keys = [
        k for k in ("low_debt_xs", "debt_chg_xs", "us_debt_twin_fx") if k in factors
    ]
    if blend_keys:
        ew = sum(factors[k] for k in blend_keys) / len(blend_keys)
        ew.name = "debt_ew"
        factors["debt_ew"] = ew

    return factors
