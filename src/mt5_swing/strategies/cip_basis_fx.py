"""Du–Schreger government-bond CIP / U.S. Treasury premium → FX factors.

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Du, Tepper & Verdelhan (2018), *JF*: CIP deviations reflect intermediary
  constraints / dollar funding stress.
- Du, Im & Schreger (2018) / Du, Keerati & Schreger (2025): government-bond CIP
  ``x = y_i − ρ − y_USD`` (bps) measures synthetic-dollar sovereign cost vs UST;
  cross-sectional mean ≈ −**U.S. Treasury Premium** (convenience yield).

Sign map (document carefully)
-----------------------------
Dataset ``cip_govt`` (bps): higher / more positive ↔ foreign gov looks *worse*
vs UST after FX hedge, and (via ρ = irs_i + bs − irs_USD) also embeds a *more
negative* cross-currency basis (dollar scarcity in that currency's hedge market).

**Primary ``low_cip_xs``:** long currencies with **low / more negative**
``cip_govt`` / short high ``cip_govt``.
Rationale (Du–Schreger convenience + DTV funding channel): low CIP = stronger
relative foreign convenience / less dollar-funding stress embedded → FX
outperformance; high CIP = funding-stress / convenience discount → underperform.

**Honesty ``high_cip_xs``:** reverse sort (long high CIP / short low).

Companions
----------
- ``low_cip_z_xs`` — XS on 60m trailing z of CIP (long low z / short high z).
- ``cip_chg_xs`` — long **improving** CIP (falling Δ12 of cip_govt) / short
  worsening (score = −Δ12).
- ``ust_premium_haven_usd`` — high UST premium (−mean CIP) z ≥ +1 → long USD.
- ``ust_premium_stress_fx`` — honesty: high UST premium → long foreign / short USD.
- ``cip_ew`` — EW of ``low_cip_xs``, ``cip_chg_xs``, ``ust_premium_haven_usd``.

PIT: loader ``pub_lag_days`` (default 1) on daily → month-end panel;
``signal_lag`` months (default 1) + **1 trading-day** weight lag.
``n_long=n_short=2`` on available foreign panel; costs 1.5 bps/side.

Distinct from funding_liq §20, fwd_carry §19, IG OAS §36, CB-BS §22.
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
class CipBasisFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # months after pub-lagged month-end CIP is known
    n_long: int = 2
    n_short: int = 2
    z_window: int = 60  # months (~5y)
    min_periods: int = 24
    z_high: float = 1.0  # UST premium z ≥ z_high → haven / stress tilt
    usd_tilt: float = 0.5
    cost_bps_side: float = 1.5
    chg_periods: int = 12  # Δ12 of CIP level (bps)


def _month_start(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    return pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start"), tz="UTC")


def _month_end(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    return pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="end"), tz="UTC")


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


def prepare_cip_scores(
    cip_panel: pd.DataFrame,
    *,
    cfg: CipBasisFxConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build signed monthly score panels from PIT ``cip_govt`` (bps).

    Scores exclude USD when present. Primary score ``low_cip`` = −cip_govt so
    that rank-sort (long high score) longs low CIP.
    """
    cfg = cfg or CipBasisFxConfig()
    panel = cip_panel.copy()
    # Normalise to month-start for lag math; values are month-end samples
    panel.index = _month_start(pd.DatetimeIndex(panel.index))
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    foreign_cols = [c for c in panel.columns if c.upper() != "USD" and c in CURRENCY_USD_PAIR]
    foreign = panel[foreign_cols]

    out: dict[str, pd.DataFrame] = {}

    # Primary: low CIP → high score → long (convenience / less funding stress)
    low = (-foreign).copy()
    if cfg.signal_lag > 0:
        low = low.shift(int(cfg.signal_lag))
    out["low_cip"] = low

    # Honesty reverse: high CIP → high score → long
    high = foreign.copy()
    if cfg.signal_lag > 0:
        high = high.shift(int(cfg.signal_lag))
    out["high_cip"] = high

    # Trailing z: low z (more negative CIP vs own history) → long
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
    out["low_cip_z"] = low_z

    # Improving CIP (falling level) → long; score = −Δ12
    chg = (-foreign.diff(int(cfg.chg_periods))).copy()
    if cfg.signal_lag > 0:
        chg = chg.shift(int(cfg.signal_lag))
    out["cip_chg"] = chg

    return out


def _scores_to_daily_returns(
    score: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cfg: CipBasisFxConfig,
    *,
    name: str,
) -> pd.Series:
    ccy_w = _rank_sort_weights(score, n_long=cfg.n_long, n_short=cfg.n_short)
    if ccy_w.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    ccy_w.index = _month_end(pd.DatetimeIndex(ccy_w.index))
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


def _ust_premium_tilt_returns(
    ust_premium: pd.Series,
    pair_ret: pd.DataFrame,
    *,
    cfg: CipBasisFxConfig,
    mode: str,
    name: str,
) -> pd.Series:
    """Binary USD / FX tilt from lagged UST-premium z.

    mode='haven': z ≥ z_high → long USD (Treasury specialness / safe-haven).
    mode='stress': z ≥ z_high → long foreign (honesty alternate).
    """
    s = ust_premium.copy()
    s.index = _month_start(pd.DatetimeIndex(s.index))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    z = trailing_z(s, lookback=cfg.z_window, min_periods=cfg.min_periods)
    if cfg.signal_lag > 0:
        z = z.shift(int(cfg.signal_lag))

    z_me = z.copy()
    z_me.index = _month_end(pd.DatetimeIndex(z_me.index))
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


def cip_basis_factor_returns(
    pair_ret: pd.DataFrame,
    cip_panel: pd.DataFrame,
    *,
    cfg: CipBasisFxConfig | None = None,
    ust_premium_override: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Daily returns for CIP/basis XS factors + UST-premium tilts + EW."""
    cfg = cfg or CipBasisFxConfig()
    scores = prepare_cip_scores(cip_panel, cfg=cfg)
    factors: dict[str, pd.Series] = {}

    if "low_cip" in scores:
        factors["low_cip_xs"] = _scores_to_daily_returns(
            scores["low_cip"], pair_ret, cfg, name="low_cip_xs"
        )
    if "high_cip" in scores:
        factors["high_cip_xs"] = _scores_to_daily_returns(
            scores["high_cip"], pair_ret, cfg, name="high_cip_xs"
        )
    if "low_cip_z" in scores:
        factors["low_cip_z_xs"] = _scores_to_daily_returns(
            scores["low_cip_z"], pair_ret, cfg, name="low_cip_z_xs"
        )
    if "cip_chg" in scores:
        factors["cip_chg_xs"] = _scores_to_daily_returns(
            scores["cip_chg"], pair_ret, cfg, name="cip_chg_xs"
        )

    prem = ust_premium_override
    if prem is None:
        foreign = cip_panel[[c for c in cip_panel.columns if c.upper() != "USD"]]
        prem = (-foreign.mean(axis=1, skipna=True)).rename("ust_premium")
    if prem is not None and prem.dropna().shape[0] >= cfg.min_periods:
        factors["ust_premium_haven_usd"] = _ust_premium_tilt_returns(
            prem, pair_ret, cfg=cfg, mode="haven", name="ust_premium_haven_usd"
        )
        factors["ust_premium_stress_fx"] = _ust_premium_tilt_returns(
            prem, pair_ret, cfg=cfg, mode="stress", name="ust_premium_stress_fx"
        )

    blend_keys = [
        k
        for k in ("low_cip_xs", "cip_chg_xs", "ust_premium_haven_usd")
        if k in factors
    ]
    if blend_keys:
        ew = sum(factors[k] for k in blend_keys) / len(blend_keys)
        ew.name = "cip_ew"
        factors["cip_ew"] = ew

    return factors


__all__ = [
    "CipBasisFxConfig",
    "cip_basis_factor_returns",
    "prepare_cip_scores",
    "trailing_z",
]
