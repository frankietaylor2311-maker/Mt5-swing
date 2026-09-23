"""Country-GPR bilateral (relative-to-US) FX factors — scholarly wave §45.

Literature framing
------------------
- Liu–Zhang / Caldara–Iacoviello: currencies with *low relative* geopolitical
  risk (home − US) subsequently appreciate vs high-rel-GPR peers.
- Distinct from §9 absolute home GPRC_* sort + LP; §8 aggregate GPR regime;
  §25 AI-GPR threats/acts/oil roles.

Fixed priors (no holdout tuning)
--------------------------------
- ``low_rel_gpr_xs`` (**primary**): long low (home−US) GPR trailing z / short high
  (n_long=n_short=2 on EUR/GBP/JPY/CAD/AUD/CHF; NZD absent).
- ``high_rel_gpr_xs``: honesty opposite.
- ``low_rel_gpr_lvl_xs``: same sort on raw (home−US) levels (not z).
- ``rel_gpr_chg_xs``: long falling relative GPR (score = −Δ12 of rel levels).
- ``us_gpr_stress_fx``: elevated US GPR z → long foreign FX basket (USD soft).
- ``us_gpr_haven_usd``: elevated US GPR z → long USD haven alt.
- ``gpr_bilat_ew``: EW of primary + chg + us_gpr_stress_fx.

PIT (a priori): loader ``pub_lag_months=1`` (same as §9) + **1 trading-day**
weight lag; ``signal_lag`` months = **0** (modern XS waves §43-style — pub_lag
already provides PIT; do not add a second month lag). z_window=60m.
Costs 1.5 bps/side. Explicit: do **not** overlay on the locked FTMO sleeve.
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
class CountryGprBilateralFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 0  # months; §45 uses pub_lag=1m + 1d weight lag only
    n_long: int = 2
    n_short: int = 2
    z_window: int = 60  # months for trailing z of (home−US)
    min_periods: int = 24
    z_high: float = 1.0  # US GPR z ≥ z_high → stress / haven tilt
    usd_tilt: float = 0.5  # gross |w| when tilt is on
    cost_bps_side: float = 1.5
    chg_periods: int = 12  # Δ12 of relative levels


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


def prepare_rel_gpr_scores(
    rel_panel: pd.DataFrame,
    *,
    cfg: CountryGprBilateralFxConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build signed monthly score panels from (home−US) GPR levels.

    Scores are foreign-only. Primary prior: low relative GPR → appreciate →
    long low-rel → score = −z(rel) so high score = low relative risk.
    """
    cfg = cfg or CountryGprBilateralFxConfig()
    panel = rel_panel.copy()
    panel.index = _month_start(pd.DatetimeIndex(panel.index))
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    foreign_cols = [c for c in panel.columns if c.upper() != "USD" and c in CURRENCY_USD_PAIR]
    foreign = panel[foreign_cols]

    out: dict[str, pd.DataFrame] = {}

    # Trailing z of (home−US): low z → long (primary)
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
    out["low_rel_gpr"] = low_z

    high_z = z.copy()
    if cfg.signal_lag > 0:
        high_z = high_z.shift(int(cfg.signal_lag))
    out["high_rel_gpr"] = high_z

    # Raw levels: low (home−US) → long
    low_lvl = (-foreign).copy()
    if cfg.signal_lag > 0:
        low_lvl = low_lvl.shift(int(cfg.signal_lag))
    out["low_rel_gpr_lvl"] = low_lvl

    # Falling relative GPR (Δ12 < 0) → appreciate → score = −Δ12
    chg = (-foreign.diff(int(cfg.chg_periods))).copy()
    if cfg.signal_lag > 0:
        chg = chg.shift(int(cfg.signal_lag))
    out["rel_gpr_chg"] = chg

    return out


def _scores_to_daily_returns(
    score: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cfg: CountryGprBilateralFxConfig,
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


def _us_gpr_tilt_returns(
    us_gpr: pd.Series,
    pair_ret: pd.DataFrame,
    *,
    cfg: CountryGprBilateralFxConfig,
    mode: str,
    name: str,
) -> pd.Series:
    """Binary USD / FX tilt from lagged US GPR z.

    mode='stress': z ≥ z_high → long foreign (short USD) — US geopolitics soft USD.
    mode='haven': z ≥ z_high → long USD — safe-haven alternate.
    """
    s = us_gpr.copy()
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


def country_gpr_bilateral_factor_returns(
    pair_ret: pd.DataFrame,
    rel_panel: pd.DataFrame,
    *,
    cfg: CountryGprBilateralFxConfig | None = None,
    us_gpr: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Daily returns for relative-to-US GPR factors + US tilts + EW."""
    cfg = cfg or CountryGprBilateralFxConfig()
    scores = prepare_rel_gpr_scores(rel_panel, cfg=cfg)
    factors: dict[str, pd.Series] = {}

    if "low_rel_gpr" in scores:
        factors["low_rel_gpr_xs"] = _scores_to_daily_returns(
            scores["low_rel_gpr"], pair_ret, cfg, name="low_rel_gpr_xs"
        )
    if "high_rel_gpr" in scores:
        factors["high_rel_gpr_xs"] = _scores_to_daily_returns(
            scores["high_rel_gpr"], pair_ret, cfg, name="high_rel_gpr_xs"
        )
    if "low_rel_gpr_lvl" in scores:
        factors["low_rel_gpr_lvl_xs"] = _scores_to_daily_returns(
            scores["low_rel_gpr_lvl"], pair_ret, cfg, name="low_rel_gpr_lvl_xs"
        )
    if "rel_gpr_chg" in scores:
        factors["rel_gpr_chg_xs"] = _scores_to_daily_returns(
            scores["rel_gpr_chg"], pair_ret, cfg, name="rel_gpr_chg_xs"
        )

    if us_gpr is not None and us_gpr.dropna().shape[0] >= cfg.min_periods:
        factors["us_gpr_stress_fx"] = _us_gpr_tilt_returns(
            us_gpr, pair_ret, cfg=cfg, mode="stress", name="us_gpr_stress_fx"
        )
        factors["us_gpr_haven_usd"] = _us_gpr_tilt_returns(
            us_gpr, pair_ret, cfg=cfg, mode="haven", name="us_gpr_haven_usd"
        )

    blend_keys = [
        k
        for k in ("low_rel_gpr_xs", "rel_gpr_chg_xs", "us_gpr_stress_fx")
        if k in factors
    ]
    if blend_keys:
        ew = sum(factors[k] for k in blend_keys) / len(blend_keys)
        ew.name = "gpr_bilat_ew"
        factors["gpr_bilat_ew"] = ew

    return factors
