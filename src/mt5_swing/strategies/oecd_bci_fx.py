"""OECD Business Confidence (BCI) / manufacturing-confidence differential FX factors.

Literature framing (see also ``fred_oecd_bci`` docstring)
---------------------------------------------------------
- OECD BCI / manufacturing business confidence → activity & FX risk appetite
  (PMI-style); distinct from *consumer* CCI (§38) and *leading* OECD CLI (§37).
- ISM manufacturing literature; Dahlquist–Hasseltoft-style macro–FX
  differentials applied to *business* confidence.
- Primary: long **high** relative BCI YoY change / short low.
- Honesty alternate: long **low** BCI change.

Fixed priors (no holdout tuning)
--------------------------------
- ``high_bci_xs`` (**primary**): long high YoY BCI change / short low.
- ``low_bci_xs``: opposite honesty alternate.
- ``high_bci_z_xs``: long high 60m trailing z of BCI YoY change.
- ``bci_chg_xs``: long accelerating BCI (Δ12 of YoY) / short decelerating.
- ``us_bci_weak_fx``: depressed US BCI z → long foreign (USD-weak / US soft).
- ``us_bci_haven_usd``: depressed US BCI z → long USD (haven alt).
- ``bci_ew``: EW of ``high_bci_xs``, ``bci_chg_xs``, ``us_bci_weak_fx``.

Coverage constraint (fixed a priori): foreign BSCICP02 panel is EUR/GBP/CHF
only (AUD/CAD/NZD/JPY 404; amplitude BSCICP03 stale). Therefore
``n_long=n_short=1`` (need ≥2 names) — not the 2/2 used on thicker CCI/CLI
panels. Documented, not holdout-tuned.

PIT: loader ``pub_lag_months`` (default 2) + **1 trading-day** weight lag
(no extra month signal lag — fixed a priori for §39). Distinct from OECD CLI
(§37), OECD CCI (§38), macro-diff CPI/IP/UR, house-price, money-growth,
CA/TB, BIS, reserves, IG OAS, equity-diff.

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
class OecdBciFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 0  # months; §39 uses 1 trading-day weight lag only
    # EUR/GBP/CHF foreign panel only → 1/1 a priori (not 2/2)
    n_long: int = 1
    n_short: int = 1
    z_window: int = 60  # months for trailing z (5y)
    min_periods: int = 24
    z_low: float = -1.0  # US BCI YoY z ≤ z_low → weak / haven tilt
    usd_tilt: float = 0.5  # gross |w| when tilt is on
    cost_bps_side: float = 1.5
    yoy_periods: int = 12  # YoY diff of BCI level
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


def bci_yoy_diff(panel: pd.DataFrame, *, periods: int = 12) -> pd.DataFrame:
    """YoY difference of BCI levels (≈ YoY change in business confidence)."""
    return panel.diff(int(periods))


def prepare_oecd_bci_scores(
    bci_panel: pd.DataFrame,
    *,
    cfg: OecdBciFxConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build signed monthly score panels from PIT BCI levels (already pub-lagged).

    Scores exclude USD column when present (XS sorts are foreign vs USD).
    Foreign columns share BSCICP02 balance units; USD is tilt-only.
    """
    cfg = cfg or OecdBciFxConfig()
    panel = bci_panel.copy()
    panel.index = _month_start(pd.DatetimeIndex(panel.index))
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    foreign_cols = [c for c in panel.columns if c.upper() != "USD" and c in CURRENCY_USD_PAIR]
    foreign_lvl = panel[foreign_cols]
    foreign = bci_yoy_diff(foreign_lvl, periods=cfg.yoy_periods)

    out: dict[str, pd.DataFrame] = {}

    # High BCI YoY → high score → long (business-confidence strength → appreciate)
    high = foreign.copy()
    if cfg.signal_lag > 0:
        high = high.shift(int(cfg.signal_lag))
    out["high_bci"] = high

    # Low BCI honesty alternate
    low = (-foreign).copy()
    if cfg.signal_lag > 0:
        low = low.shift(int(cfg.signal_lag))
    out["low_bci"] = low

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
    out["high_bci_z"] = high_z

    # Acceleration: rising YoY → long (score = Δ12 of YoY)
    chg = foreign.diff(int(cfg.chg_periods)).copy()
    if cfg.signal_lag > 0:
        chg = chg.shift(int(cfg.signal_lag))
    out["bci_chg"] = chg

    return out


def _scores_to_daily_returns(
    score: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cfg: OecdBciFxConfig,
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


def _us_bci_tilt_returns(
    us_bci: pd.Series,
    pair_ret: pd.DataFrame,
    *,
    cfg: OecdBciFxConfig,
    mode: str,
    name: str,
) -> pd.Series:
    """Binary USD / FX tilt from lagged US BCI YoY z.

    mode='weak': z ≤ z_low → long foreign (short USD) — US soft business sentiment.
    mode='haven': z ≤ z_low → long USD — safe-haven alternate.
    """
    s = us_bci.copy()
    s.index = _month_start(pd.DatetimeIndex(s.index))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    yoy = s.diff(int(cfg.yoy_periods))
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
        if mode == "weak":
            w[sym] = (intensity / n) * (-usd_sign)
        else:
            w[sym] = (intensity / n) * usd_sign
    r = portfolio_returns_from_pair_weights(w, pair_ret)
    r = apply_costs(r, w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


def oecd_bci_factor_returns(
    pair_ret: pd.DataFrame,
    bci_panel: pd.DataFrame,
    *,
    cfg: OecdBciFxConfig | None = None,
    us_bci_override: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Daily returns for OECD BCI factors + US tilts + EW."""
    cfg = cfg or OecdBciFxConfig()
    scores = prepare_oecd_bci_scores(bci_panel, cfg=cfg)
    factors: dict[str, pd.Series] = {}

    if "high_bci" in scores:
        factors["high_bci_xs"] = _scores_to_daily_returns(
            scores["high_bci"], pair_ret, cfg, name="high_bci_xs"
        )
    if "low_bci" in scores:
        factors["low_bci_xs"] = _scores_to_daily_returns(
            scores["low_bci"], pair_ret, cfg, name="low_bci_xs"
        )
    if "high_bci_z" in scores:
        factors["high_bci_z_xs"] = _scores_to_daily_returns(
            scores["high_bci_z"], pair_ret, cfg, name="high_bci_z_xs"
        )
    if "bci_chg" in scores:
        factors["bci_chg_xs"] = _scores_to_daily_returns(
            scores["bci_chg"], pair_ret, cfg, name="bci_chg_xs"
        )

    us = us_bci_override
    if us is None and "USD" in bci_panel.columns:
        us = bci_panel["USD"]
    if us is not None and us.dropna().shape[0] >= cfg.min_periods + cfg.yoy_periods:
        factors["us_bci_weak_fx"] = _us_bci_tilt_returns(
            us, pair_ret, cfg=cfg, mode="weak", name="us_bci_weak_fx"
        )
        factors["us_bci_haven_usd"] = _us_bci_tilt_returns(
            us, pair_ret, cfg=cfg, mode="haven", name="us_bci_haven_usd"
        )

    blend_keys = [
        k
        for k in ("high_bci_xs", "bci_chg_xs", "us_bci_weak_fx")
        if k in factors
    ]
    if blend_keys:
        ew = sum(factors[k] for k in blend_keys) / len(blend_keys)
        ew.name = "bci_ew"
        factors["bci_ew"] = ew

    return factors
