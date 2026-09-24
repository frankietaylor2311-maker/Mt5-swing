"""Unemployment-rate differential FX factors (OECD MEI / national UR levels).

Literature framing (see also ``fred_unemployment`` docstring)
-------------------------------------------------------------
- Dahlquist–Hasseltoft (2020) economic momentum — labour channel via
  **unemployment rates** (distinct from employment persons §41).
- Primary: long **low** relative UR / short high (strong labour → appreciate).
- Honesty alternate: long **high** UR (labour-stress debtor premium).
- Distinct from employment §41, macro-diff §12 EW CPI+IP+UR blend, OECD
  CLI/CCI/BCI, WUI, EPU/TPU, PPI/CPI, ULC/LP/CU.

Fixed priors (no holdout tuning)
--------------------------------
- ``low_ur_xs`` (**primary**): long low relative UR level / short high
  (n_long=n_short=2 on available foreign panel; NZD unmapped).
- ``high_ur_xs``: honesty reverse.
- ``low_ur_z_xs``: long low 60m trailing z of UR level / short high z.
- ``ur_chg_xs``: long **falling** UR (−Δ12 of level) / short rising.
- ``us_ur_stress_fx``: elevated US UR z ≥ +1.0 → long foreign (USD soft).
- ``us_ur_haven_usd``: elevated US UR z ≥ +1.0 → long USD (haven alt).
- ``ur_ew``: EW of ``low_ur_xs``, ``ur_chg_xs``, ``us_ur_stress_fx``.

Score basis (fixed a priori): **UR level %** (not YoY of persons).
PIT: loader ``pub_lag_months`` (default 1) + ``signal_lag`` months (default 1)
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
class UnemploymentFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # months; a priori after pub_lag (UR §54)
    n_long: int = 2  # available foreign panel (NZD unmapped)
    n_short: int = 2
    z_window: int = 60  # months for trailing z (5y)
    min_periods: int = 24
    z_high: float = 1.0  # US UR level z ≥ z_high → stress / haven tilt
    usd_tilt: float = 0.5  # gross |w| when tilt is on
    cost_bps_side: float = 1.5
    chg_periods: int = 12  # Δ12 of UR level (falling = improving labour)


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


def prepare_unemployment_scores(
    ur_panel: pd.DataFrame,
    *,
    cfg: UnemploymentFxConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build signed monthly score panels from PIT UR **levels** (already pub-lagged).

    Scores exclude USD column when present (XS sorts are foreign vs USD).
    Score basis = unemployment rate levels (%) — not YoY.
    """
    cfg = cfg or UnemploymentFxConfig()
    panel = ur_panel.copy()
    panel.index = _month_start(pd.DatetimeIndex(panel.index))
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    foreign_cols = [c for c in panel.columns if c.upper() != "USD" and c in CURRENCY_USD_PAIR]
    foreign = panel[foreign_cols]

    out: dict[str, pd.DataFrame] = {}

    # Low UR → high score → long (strong labour → appreciate)
    low = (-foreign).copy()
    if cfg.signal_lag > 0:
        low = low.shift(int(cfg.signal_lag))
    out["low_ur"] = low

    # High UR honesty alternate (labour-stress debtor premium)
    high = foreign.copy()
    if cfg.signal_lag > 0:
        high = high.shift(int(cfg.signal_lag))
    out["high_ur"] = high

    # 5y trailing z of *levels*: low z → high score → long
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
    out["low_ur_z"] = low_z

    # Falling UR → long (score = −Δ12 of level)
    chg = (-foreign.diff(int(cfg.chg_periods))).copy()
    if cfg.signal_lag > 0:
        chg = chg.shift(int(cfg.signal_lag))
    out["ur_chg"] = chg

    return out


def _scores_to_daily_returns(
    score: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cfg: UnemploymentFxConfig,
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


def _us_ur_tilt_returns(
    us_ur: pd.Series,
    pair_ret: pd.DataFrame,
    *,
    cfg: UnemploymentFxConfig,
    mode: str,
    name: str,
) -> pd.Series:
    """Binary USD / FX tilt from lagged US UR **level** z.

    mode='stress': z ≥ z_high → long foreign (short USD) — US labour soft.
    mode='haven': z ≥ z_high → long USD — safe-haven alternate.
    """
    s = us_ur.copy()
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


def unemployment_factor_returns(
    pair_ret: pd.DataFrame,
    ur_panel: pd.DataFrame,
    *,
    cfg: UnemploymentFxConfig | None = None,
    us_ur_override: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Daily returns for UR-level factors + US tilts + EW."""
    cfg = cfg or UnemploymentFxConfig()
    scores = prepare_unemployment_scores(ur_panel, cfg=cfg)
    factors: dict[str, pd.Series] = {}

    if "low_ur" in scores:
        factors["low_ur_xs"] = _scores_to_daily_returns(
            scores["low_ur"], pair_ret, cfg, name="low_ur_xs"
        )
    if "high_ur" in scores:
        factors["high_ur_xs"] = _scores_to_daily_returns(
            scores["high_ur"], pair_ret, cfg, name="high_ur_xs"
        )
    if "low_ur_z" in scores:
        factors["low_ur_z_xs"] = _scores_to_daily_returns(
            scores["low_ur_z"], pair_ret, cfg, name="low_ur_z_xs"
        )
    if "ur_chg" in scores:
        factors["ur_chg_xs"] = _scores_to_daily_returns(
            scores["ur_chg"], pair_ret, cfg, name="ur_chg_xs"
        )

    us = us_ur_override
    if us is None and "USD" in ur_panel.columns:
        us = ur_panel["USD"]
    if us is not None and us.dropna().shape[0] >= cfg.min_periods:
        factors["us_ur_stress_fx"] = _us_ur_tilt_returns(
            us, pair_ret, cfg=cfg, mode="stress", name="us_ur_stress_fx"
        )
        factors["us_ur_haven_usd"] = _us_ur_tilt_returns(
            us, pair_ret, cfg=cfg, mode="haven", name="us_ur_haven_usd"
        )

    blend_keys = [
        k
        for k in ("low_ur_xs", "ur_chg_xs", "us_ur_stress_fx")
        if k in factors
    ]
    if blend_keys:
        ew = sum(factors[k] for k in blend_keys) / len(blend_keys)
        ew.name = "ur_ew"
        factors["ur_ew"] = ew

    return factors
