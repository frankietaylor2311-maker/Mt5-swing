"""Housing wealth / collateral channel FX factors (BIS residential HPI).

Literature framing (see also ``fred_house_prices`` docstring)
-------------------------------------------------------------
- Aoki–Proudman–Vlieghe / housing–credit / collateral: high relative
  house-price *momentum* → wealth / collateral → subsequent appreciation.
- Primary: long **high** relative HPI YoY growth / short low.
- Honesty alternate: long **low** HPI momentum (housing-stress premium).

Fixed priors (no holdout tuning)
--------------------------------
- ``high_hpi_xs`` (**primary**): long high YoY log-diff HPI / short low.
- ``low_hpi_xs``: opposite honesty alternate.
- ``high_hpi_z_xs``: long high 60m trailing z of HPI YoY growth.
- ``hpi_chg_xs``: long accelerating HPI (Δ12 of YoY) / short decelerating.
- ``us_hpi_stress_fx``: depressed US HPI z → long foreign (USD-weak /
  housing-stress prior).
- ``us_hpi_haven_usd``: depressed US HPI z → long USD (haven alt).
- ``hpi_ew``: EW of ``high_hpi_xs``, ``hpi_chg_xs``, ``us_hpi_stress_fx``.

PIT: loader ``pub_lag_months`` (default 4) + ``signal_lag`` months + 1 trading
day weight lag. Distinct from BIS credit (§32), debt (§29), REER (§31),
money-growth (§33), reserves (§34), CA/TB, equity-diff, funding-liq, IG OAS.

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
class HousePriceFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # months after publication-lagged HPI is known
    n_long: int = 2
    n_short: int = 2
    z_window: int = 60  # months for trailing z (5y)
    min_periods: int = 24
    z_low: float = -1.0  # US HPI growth z ≤ z_low → stress / haven tilt
    usd_tilt: float = 0.5  # gross |w| when tilt is on
    cost_bps_side: float = 1.5
    yoy_periods: int = 12  # YoY log-diff of index
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


def hpi_yoy_log_diff(panel: pd.DataFrame, *, periods: int = 12) -> pd.DataFrame:
    """YoY log-diff of HPI index levels (≈ YoY % change)."""
    log_p = np.log(panel.replace(0.0, np.nan))
    return log_p.diff(int(periods))


def prepare_house_price_scores(
    hpi_panel: pd.DataFrame,
    *,
    cfg: HousePriceFxConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build signed monthly score panels from PIT HPI (already pub-lagged).

    Scores exclude USD column when present (XS sorts are foreign vs USD).
    Unit: YoY log-diff of index (not raw levels).
    """
    cfg = cfg or HousePriceFxConfig()
    panel = hpi_panel.copy()
    panel.index = _month_start(pd.DatetimeIndex(panel.index))
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    foreign_cols = [c for c in panel.columns if c.upper() != "USD" and c in CURRENCY_USD_PAIR]
    foreign = panel[foreign_cols]

    yoy = hpi_yoy_log_diff(foreign, periods=cfg.yoy_periods)
    out: dict[str, pd.DataFrame] = {}

    # High HPI momentum → high score → long (wealth/collateral → appreciate)
    high = yoy.copy()
    if cfg.signal_lag > 0:
        high = high.shift(int(cfg.signal_lag))
    out["high_hpi"] = high

    # Low HPI honesty alternate
    low = (-yoy).copy()
    if cfg.signal_lag > 0:
        low = low.shift(int(cfg.signal_lag))
    out["low_hpi"] = low

    # 5y trailing z of YoY growth: high z → long
    z = pd.DataFrame(
        {
            c: trailing_z(yoy[c], lookback=cfg.z_window, min_periods=cfg.min_periods)
            for c in yoy.columns
        },
        index=yoy.index,
    )
    high_z = z.copy()
    if cfg.signal_lag > 0:
        high_z = high_z.shift(int(cfg.signal_lag))
    out["high_hpi_z"] = high_z

    # Acceleration: rising YoY → long (score = Δ12 of YoY)
    chg = yoy.diff(int(cfg.chg_periods)).copy()
    if cfg.signal_lag > 0:
        chg = chg.shift(int(cfg.signal_lag))
    out["hpi_chg"] = chg

    return out


def _scores_to_daily_returns(
    score: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cfg: HousePriceFxConfig,
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


def _us_hpi_tilt_returns(
    us_hpi: pd.Series,
    pair_ret: pd.DataFrame,
    *,
    cfg: HousePriceFxConfig,
    mode: str,
    name: str,
) -> pd.Series:
    """Binary USD / FX tilt from lagged US HPI YoY growth z.

    mode='stress': z ≤ z_low → long foreign (short USD) — US housing stress.
    mode='haven': z ≤ z_low → long USD — safe-haven alternate.
    """
    s = us_hpi.copy()
    s.index = _month_start(pd.DatetimeIndex(s.index))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    # YoY of US index, then trailing z
    log_s = np.log(s.replace(0.0, np.nan))
    yoy = log_s.diff(int(cfg.yoy_periods))
    z = trailing_z(yoy, lookback=cfg.z_window, min_periods=cfg.min_periods)
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


def house_price_factor_returns(
    pair_ret: pd.DataFrame,
    hpi_panel: pd.DataFrame,
    *,
    cfg: HousePriceFxConfig | None = None,
    us_hpi_override: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Daily returns for house-price factors + US tilts + EW."""
    cfg = cfg or HousePriceFxConfig()
    scores = prepare_house_price_scores(hpi_panel, cfg=cfg)
    factors: dict[str, pd.Series] = {}

    if "high_hpi" in scores:
        factors["high_hpi_xs"] = _scores_to_daily_returns(
            scores["high_hpi"], pair_ret, cfg, name="high_hpi_xs"
        )
    if "low_hpi" in scores:
        factors["low_hpi_xs"] = _scores_to_daily_returns(
            scores["low_hpi"], pair_ret, cfg, name="low_hpi_xs"
        )
    if "high_hpi_z" in scores:
        factors["high_hpi_z_xs"] = _scores_to_daily_returns(
            scores["high_hpi_z"], pair_ret, cfg, name="high_hpi_z_xs"
        )
    if "hpi_chg" in scores:
        factors["hpi_chg_xs"] = _scores_to_daily_returns(
            scores["hpi_chg"], pair_ret, cfg, name="hpi_chg_xs"
        )

    us = us_hpi_override
    if us is None and "USD" in hpi_panel.columns:
        us = hpi_panel["USD"]
    if us is not None and us.dropna().shape[0] >= cfg.min_periods + cfg.yoy_periods:
        factors["us_hpi_stress_fx"] = _us_hpi_tilt_returns(
            us, pair_ret, cfg=cfg, mode="stress", name="us_hpi_stress_fx"
        )
        factors["us_hpi_haven_usd"] = _us_hpi_tilt_returns(
            us, pair_ret, cfg=cfg, mode="haven", name="us_hpi_haven_usd"
        )

    blend_keys = [
        k
        for k in ("high_hpi_xs", "hpi_chg_xs", "us_hpi_stress_fx")
        if k in factors
    ]
    if blend_keys:
        ew = sum(factors[k] for k in blend_keys) / len(blend_keys)
        ew.name = "hpi_ew"
        factors["hpi_ew"] = ew

    return factors
