"""Monetary-approach / money-growth differential FX factors (Frenkel–Bilson).

Literature framing (see also ``fred_money_growth`` docstring)
-------------------------------------------------------------
- Frenkel (1976) / Bilson (1978) monetary approach; Dornbusch (1976) sticky-price
  overshooting: higher *relative* money growth → depreciation.
- Primary: long **low** relative money growth (tight) / short high growth (loose).
- Honesty alternate: long **high** money growth (liquidity / risk-premium sort).

Fixed priors (no holdout tuning)
--------------------------------
- ``low_money_growth_xs`` (**primary**): long low OECD broad-money growth /
  short high growth (Frenkel–Bilson sign).
- ``high_money_growth_xs``: opposite honesty alternate.
- ``low_money_growth_z_xs``: long low 60m trailing z of money growth.
- ``money_growth_chg_xs``: long decelerating growth (Δ12 of growth < 0) /
  short accelerating.
- ``us_money_stress_fx``: elevated US money-growth z → long foreign
  (USD-weak / monetary-expansion prior).
- ``us_money_haven_usd``: elevated US money-growth z → long USD (haven alt).
- ``money_ew``: EW of ``low_money_growth_xs``, ``money_growth_chg_xs``,
  ``us_money_stress_fx``.

PIT: loader ``pub_lag_months`` (default 2) + ``signal_lag`` months + 1 trading
day weight lag. Distinct from CB-BS/QE (§22), real-rate (§23), debt (§29),
fiscal (§28), TB (§30), REER (§31), BIS credit (§32), funding-liq, macro-diff.

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
class MoneyGrowthFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # months after publication-lagged growth is known
    n_long: int = 2
    n_short: int = 2
    z_window: int = 60  # months for trailing z (5y)
    min_periods: int = 24
    z_high: float = 1.0  # |z| threshold for US money-growth tilt
    usd_tilt: float = 0.5  # gross |w| when tilt is on
    cost_bps_side: float = 1.5
    chg_periods: int = 12  # YoY change of the growth rate itself


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


def prepare_money_growth_scores(
    growth_panel: pd.DataFrame,
    *,
    cfg: MoneyGrowthFxConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build signed monthly score panels from PIT money growth (already pub-lagged).

    Scores exclude USD column when present (XS sorts are foreign vs USD).
    """
    cfg = cfg or MoneyGrowthFxConfig()
    panel = growth_panel.copy()
    panel.index = _month_start(pd.DatetimeIndex(panel.index))
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    foreign_cols = [c for c in panel.columns if c.upper() != "USD" and c in CURRENCY_USD_PAIR]
    foreign = panel[foreign_cols]

    out: dict[str, pd.DataFrame] = {}

    # Low growth → high score → long (Frenkel–Bilson tightness → appreciate)
    low = (-foreign).copy()
    if cfg.signal_lag > 0:
        low = low.shift(int(cfg.signal_lag))
    out["low_money_growth"] = low

    # High growth honesty alternate
    high = foreign.copy()
    if cfg.signal_lag > 0:
        high = high.shift(int(cfg.signal_lag))
    out["high_money_growth"] = high

    # 5y trailing z: low z → long
    z = pd.DataFrame(
        {
            c: trailing_z(foreign[c], lookback=cfg.z_window, min_periods=cfg.min_periods)
            for c in foreign.columns
        },
        index=foreign.index,
    )
    low_z = (-z).copy()
    if cfg.signal_lag > 0:
        low_z = low_z.shift(int(cfg.signal_lag))
    out["low_money_growth_z"] = low_z

    # Change: decelerating growth → long (score = −Δ growth)
    chg = (-foreign.diff(int(cfg.chg_periods))).copy()
    if cfg.signal_lag > 0:
        chg = chg.shift(int(cfg.signal_lag))
    out["money_growth_chg"] = chg

    return out


def _scores_to_daily_returns(
    score: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cfg: MoneyGrowthFxConfig,
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


def _us_money_tilt_returns(
    us_growth: pd.Series,
    pair_ret: pd.DataFrame,
    *,
    cfg: MoneyGrowthFxConfig,
    mode: str,
    name: str,
) -> pd.Series:
    """Binary USD / FX tilt from lagged US money-growth z.

    mode='stress': z ≥ +z_high → long foreign (short USD) — monetary expansion.
    mode='haven': z ≥ +z_high → long USD — safe-haven alternate.
    """
    s = us_growth.copy()
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
        if mode == "stress":
            w[sym] = (intensity / n) * (-usd_sign)
        else:
            w[sym] = (intensity / n) * usd_sign
    r = portfolio_returns_from_pair_weights(w, pair_ret)
    r = apply_costs(r, w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


def money_growth_factor_returns(
    pair_ret: pd.DataFrame,
    growth_panel: pd.DataFrame,
    *,
    cfg: MoneyGrowthFxConfig | None = None,
    us_growth_override: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Daily returns for money-growth factors + US tilts + EW."""
    cfg = cfg or MoneyGrowthFxConfig()
    scores = prepare_money_growth_scores(growth_panel, cfg=cfg)
    factors: dict[str, pd.Series] = {}

    if "low_money_growth" in scores:
        factors["low_money_growth_xs"] = _scores_to_daily_returns(
            scores["low_money_growth"], pair_ret, cfg, name="low_money_growth_xs"
        )
    if "high_money_growth" in scores:
        factors["high_money_growth_xs"] = _scores_to_daily_returns(
            scores["high_money_growth"], pair_ret, cfg, name="high_money_growth_xs"
        )
    if "low_money_growth_z" in scores:
        factors["low_money_growth_z_xs"] = _scores_to_daily_returns(
            scores["low_money_growth_z"], pair_ret, cfg, name="low_money_growth_z_xs"
        )
    if "money_growth_chg" in scores:
        factors["money_growth_chg_xs"] = _scores_to_daily_returns(
            scores["money_growth_chg"], pair_ret, cfg, name="money_growth_chg_xs"
        )

    us = us_growth_override
    if us is None and "USD" in growth_panel.columns:
        us = growth_panel["USD"]
    if us is not None and us.dropna().shape[0] >= cfg.min_periods:
        factors["us_money_stress_fx"] = _us_money_tilt_returns(
            us, pair_ret, cfg=cfg, mode="stress", name="us_money_stress_fx"
        )
        factors["us_money_haven_usd"] = _us_money_tilt_returns(
            us, pair_ret, cfg=cfg, mode="haven", name="us_money_haven_usd"
        )

    blend_keys = [
        k
        for k in ("low_money_growth_xs", "money_growth_chg_xs", "us_money_stress_fx")
        if k in factors
    ]
    if blend_keys:
        ew = sum(factors[k] for k in blend_keys) / len(blend_keys)
        ew.name = "money_ew"
        factors["money_ew"] = ew

    return factors
