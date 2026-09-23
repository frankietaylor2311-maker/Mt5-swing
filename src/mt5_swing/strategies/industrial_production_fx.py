"""Industrial-production differential FX factors (OECD MEI PRINTO01 / PRMNTO01).

Literature framing (see also ``fred_industrial_production`` docstring)
---------------------------------------------------------------------
- Real-activity / growth differentials → FX (Molodtsova–Papell /
  growth-channel; Dahlquist–Hasseltoft-style applied to coincident IP).
- Primary: long **high** relative IP YoY growth / short low.
- Honesty alternate: long **low** IP growth (activity-stress debtor premium).
- Distinct from macro-diff EW CPI/IP/UR blend, OECD CLI/CCI/BCI, employment
  §41, WUI, EPU/TPU, CA/TB, fiscal/debt, BIS, reserves, house-price, money,
  equity, IG OAS, commodity.

Fixed priors (no holdout tuning)
--------------------------------
- ``high_ip_xs`` (**primary**): long high YoY IP growth / short low
  (n_long=n_short=2 on full G10 foreign panel).
- ``low_ip_xs``: opposite honesty alternate.
- ``high_ip_z_xs``: long high 60m trailing z of IP YoY growth.
- ``ip_chg_xs``: long accelerating IP (Δ12 of YoY) / short decelerating.
- ``us_ip_stress_fx``: depressed US IP YoY z → long foreign (USD soft).
- ``us_ip_haven_usd``: depressed US IP YoY z → long USD (haven alt).
- ``ip_ew``: EW of ``high_ip_xs``, ``ip_chg_xs``, ``us_ip_stress_fx``.

Score basis (fixed a priori): **YoY growth as reported** (GYSAM/GYSAQ panel
values — not derived from levels). PIT: loader ``pub_lag_months`` (default 2)
+ **1 trading-day** weight lag (no extra month signal lag — fixed a priori
for §42). Explicit: do **not** overlay on the locked FTMO sleeve.
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
class IndustrialProductionFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 0  # months; §42 uses 1 trading-day weight lag only
    n_long: int = 2  # full G10 foreign panel
    n_short: int = 2
    z_window: int = 60  # months for trailing z (5y)
    min_periods: int = 24
    z_low: float = -1.0  # US IP YoY z ≤ z_low → stress / haven tilt
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


def prepare_industrial_production_scores(
    ip_panel: pd.DataFrame,
    *,
    cfg: IndustrialProductionFxConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build signed monthly score panels from PIT IP YoY (already pub-lagged).

    Scores exclude USD column when present (XS sorts are foreign vs USD).
    Panel values are already YoY growth % — used directly (no pct_change).
    """
    cfg = cfg or IndustrialProductionFxConfig()
    panel = ip_panel.copy()
    panel.index = _month_start(pd.DatetimeIndex(panel.index))
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    foreign_cols = [c for c in panel.columns if c.upper() != "USD" and c in CURRENCY_USD_PAIR]
    foreign = panel[foreign_cols]

    out: dict[str, pd.DataFrame] = {}

    # High IP YoY → high score → long (growth-channel → appreciate)
    high = foreign.copy()
    if cfg.signal_lag > 0:
        high = high.shift(int(cfg.signal_lag))
    out["high_ip"] = high

    # Low IP honesty alternate (activity-stress debtor premium)
    low = (-foreign).copy()
    if cfg.signal_lag > 0:
        low = low.shift(int(cfg.signal_lag))
    out["low_ip"] = low

    # 5y trailing z of YoY: high z → long
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
    out["high_ip_z"] = high_z

    # Acceleration: rising YoY → long (score = Δ12 of YoY)
    chg = foreign.diff(int(cfg.chg_periods)).copy()
    if cfg.signal_lag > 0:
        chg = chg.shift(int(cfg.signal_lag))
    out["ip_chg"] = chg

    return out


def _scores_to_daily_returns(
    score: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cfg: IndustrialProductionFxConfig,
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


def _us_ip_tilt_returns(
    us_ip: pd.Series,
    pair_ret: pd.DataFrame,
    *,
    cfg: IndustrialProductionFxConfig,
    mode: str,
    name: str,
) -> pd.Series:
    """Binary USD / FX tilt from lagged US IP YoY z.

    mode='stress': z ≤ z_low → long foreign (short USD) — US IP soft.
    mode='haven': z ≤ z_low → long USD — safe-haven alternate.
    """
    s = us_ip.copy()
    s.index = _month_start(pd.DatetimeIndex(s.index))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    # Panel already YoY % — use directly
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


def industrial_production_factor_returns(
    pair_ret: pd.DataFrame,
    ip_panel: pd.DataFrame,
    *,
    cfg: IndustrialProductionFxConfig | None = None,
    us_ip_override: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Daily returns for IP-growth factors + US tilts + EW."""
    cfg = cfg or IndustrialProductionFxConfig()
    scores = prepare_industrial_production_scores(ip_panel, cfg=cfg)
    factors: dict[str, pd.Series] = {}

    if "high_ip" in scores:
        factors["high_ip_xs"] = _scores_to_daily_returns(
            scores["high_ip"], pair_ret, cfg, name="high_ip_xs"
        )
    if "low_ip" in scores:
        factors["low_ip_xs"] = _scores_to_daily_returns(
            scores["low_ip"], pair_ret, cfg, name="low_ip_xs"
        )
    if "high_ip_z" in scores:
        factors["high_ip_z_xs"] = _scores_to_daily_returns(
            scores["high_ip_z"], pair_ret, cfg, name="high_ip_z_xs"
        )
    if "ip_chg" in scores:
        factors["ip_chg_xs"] = _scores_to_daily_returns(
            scores["ip_chg"], pair_ret, cfg, name="ip_chg_xs"
        )

    us = us_ip_override
    if us is None and "USD" in ip_panel.columns:
        us = ip_panel["USD"]
    if us is not None and us.dropna().shape[0] >= cfg.min_periods:
        factors["us_ip_stress_fx"] = _us_ip_tilt_returns(
            us, pair_ret, cfg=cfg, mode="stress", name="us_ip_stress_fx"
        )
        factors["us_ip_haven_usd"] = _us_ip_tilt_returns(
            us, pair_ret, cfg=cfg, mode="haven", name="us_ip_haven_usd"
        )

    blend_keys = [
        k
        for k in ("high_ip_xs", "ip_chg_xs", "us_ip_stress_fx")
        if k in factors
    ]
    if blend_keys:
        ew = sum(factors[k] for k in blend_keys) / len(blend_keys)
        ew.name = "ip_ew"
        factors["ip_ew"] = ew

    return factors
