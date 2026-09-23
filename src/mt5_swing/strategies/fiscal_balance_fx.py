"""Fiscal-balance / government-budget differential FX factors (twin deficits).

Fixed priors (no holdout tuning)
--------------------------------
- ``fiscal_surplus_xs`` (**primary**, fiscal sustainability / twin-deficits flow):
  long high fiscal balance (surplus / less deficit) / short deep deficit.
  Stronger relative fiscal stance → subsequent appreciation prior.
- ``fiscal_deficit_xs``: opposite risk-premium sort (long deficit / short surplus)
  — mirrors Della Corte–Riddiough–Sarno debtor premium for honesty.
- ``fiscal_chg_xs``: long improving Δ(fiscal/GDP) / short deteriorating.
- ``us_fiscal_twin_fx``: deep US fiscal deficit z → long foreign / short USD
  (twin-deficits USD depreciation / external-adjustment prior).
- ``us_fiscal_haven_usd``: deep US deficit z → long USD (safe-haven alternate).
- ``us_mts_chg_usd``: Monthly Treasury Statement Δ surplus z → USD tilt
  (improving MTS → long USD). Higher-frequency US-only overlay.
- ``fiscal_ca_blend``: scholarly EW of ``fiscal_surplus_xs`` + ``ca_surplus_xs``
  when a CA panel is supplied (twin-deficits joint channel) — **not** an overlay
  on the locked FTMO sleeve.
- ``fiscal_ew``: EW of ``fiscal_surplus_xs``, ``fiscal_chg_xs``, ``us_fiscal_twin_fx``.

PIT: loader ``pub_lag_months`` (default 15 for annual WEO) + ``signal_lag`` months
+ 1 trading day weight lag. Distinct from CA/GDP (§21), CB-BS (§22), macro-diff.

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
class FiscalBalanceFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # months after publication-lagged fiscal is known
    n_long: int = 2
    n_short: int = 2
    z_window: int = 60  # months for trailing z (US fiscal state)
    min_periods: int = 24
    z_high: float = 1.0  # |z| threshold for US fiscal tilt
    usd_tilt: float = 0.5  # gross |w| when tilt is on
    cost_bps_side: float = 1.5
    chg_periods: int = 12  # YoY change on monthly-ffilled annual fiscal


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


def prepare_fiscal_scores(
    fiscal_panel: pd.DataFrame,
    *,
    cfg: FiscalBalanceFxConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build signed monthly score panels from PIT fiscal/GDP (already pub-lagged).

    Scores exclude USD column when present (XS sorts are foreign vs USD).
    """
    cfg = cfg or FiscalBalanceFxConfig()
    panel = fiscal_panel.copy()
    panel.index = _month_start(pd.DatetimeIndex(panel.index))
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    foreign_cols = [c for c in panel.columns if c.upper() != "USD" and c in CURRENCY_USD_PAIR]
    foreign = panel[foreign_cols]

    out: dict[str, pd.DataFrame] = {}

    # Surplus / sustainability: high fiscal balance → high score → long
    surplus = foreign.copy()
    if cfg.signal_lag > 0:
        surplus = surplus.shift(int(cfg.signal_lag))
    out["fiscal_surplus"] = surplus

    # Deficit risk premium: low fiscal → high score → long
    deficit = (-foreign).copy()
    if cfg.signal_lag > 0:
        deficit = deficit.shift(int(cfg.signal_lag))
    out["fiscal_deficit"] = deficit

    # Change: improving fiscal → long
    chg = foreign.diff(int(cfg.chg_periods))
    if cfg.signal_lag > 0:
        chg = chg.shift(int(cfg.signal_lag))
    out["fiscal_chg"] = chg

    return out


def _scores_to_daily_returns(
    score: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cfg: FiscalBalanceFxConfig,
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


def _us_fiscal_tilt_returns(
    us_fiscal: pd.Series,
    pair_ret: pd.DataFrame,
    *,
    cfg: FiscalBalanceFxConfig,
    mode: str,
    name: str,
) -> pd.Series:
    """Binary USD / FX tilt from lagged US fiscal/GDP z.

    mode='twin': z ≤ −z_high → long foreign (short USD) — twin-deficits adjustment.
    mode='haven': z ≤ −z_high → long USD — safe-haven alternate.
    mode='mts_improve': z of MTS level ≥ +z_high → long USD (improving surplus).
    """
    s = us_fiscal.copy()
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

    if mode in ("twin", "haven"):
        on = (z_daily <= -float(cfg.z_high)).astype(float)
    else:  # mts_improve
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
            # haven or mts_improve → long USD
            w[sym] = (intensity / n) * usd_sign
    r = portfolio_returns_from_pair_weights(w, pair_ret)
    r = apply_costs(r, w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


def fiscal_balance_factor_returns(
    pair_ret: pd.DataFrame,
    fiscal_panel: pd.DataFrame,
    *,
    cfg: FiscalBalanceFxConfig | None = None,
    us_mts: pd.Series | None = None,
    ca_surplus_returns: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Daily returns for fiscal-balance factors + optional CA blend + EW."""
    cfg = cfg or FiscalBalanceFxConfig()
    scores = prepare_fiscal_scores(fiscal_panel, cfg=cfg)
    factors: dict[str, pd.Series] = {}

    if "fiscal_surplus" in scores:
        factors["fiscal_surplus_xs"] = _scores_to_daily_returns(
            scores["fiscal_surplus"], pair_ret, cfg, name="fiscal_surplus_xs"
        )
    if "fiscal_deficit" in scores:
        factors["fiscal_deficit_xs"] = _scores_to_daily_returns(
            scores["fiscal_deficit"], pair_ret, cfg, name="fiscal_deficit_xs"
        )
    if "fiscal_chg" in scores:
        factors["fiscal_chg_xs"] = _scores_to_daily_returns(
            scores["fiscal_chg"], pair_ret, cfg, name="fiscal_chg_xs"
        )

    us = None
    if "USD" in fiscal_panel.columns:
        us = fiscal_panel["USD"]
    if us is not None and us.dropna().shape[0] >= cfg.min_periods:
        factors["us_fiscal_twin_fx"] = _us_fiscal_tilt_returns(
            us, pair_ret, cfg=cfg, mode="twin", name="us_fiscal_twin_fx"
        )
        factors["us_fiscal_haven_usd"] = _us_fiscal_tilt_returns(
            us, pair_ret, cfg=cfg, mode="haven", name="us_fiscal_haven_usd"
        )

    if us_mts is not None and us_mts.dropna().shape[0] >= cfg.min_periods:
        factors["us_mts_chg_usd"] = _us_fiscal_tilt_returns(
            us_mts, pair_ret, cfg=cfg, mode="mts_improve", name="us_mts_chg_usd"
        )

    if ca_surplus_returns is not None and "fiscal_surplus_xs" in factors:
        # Align and EW-blend (scholarly twin-deficits joint channel)
        a = factors["fiscal_surplus_xs"]
        b = ca_surplus_returns.reindex(a.index).fillna(0.0)
        blend = 0.5 * a + 0.5 * b
        blend.name = "fiscal_ca_blend"
        factors["fiscal_ca_blend"] = blend

    blend_keys = [
        k
        for k in ("fiscal_surplus_xs", "fiscal_chg_xs", "us_fiscal_twin_fx")
        if k in factors
    ]
    if blend_keys:
        ew = sum(factors[k] for k in blend_keys) / len(blend_keys)
        ew.name = "fiscal_ew"
        factors["fiscal_ew"] = ew

    return factors
