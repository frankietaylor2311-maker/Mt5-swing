"""Global imbalances / current-account FX factors (Gourinchas–Rey / DCRS style).

Fixed priors (no holdout tuning)
--------------------------------
- ``ca_debtor_xs`` (**primary**, Della Corte–Riddiough–Sarno): long low CA/GDP
  (deficit / debtor) currencies / short high CA/GDP (surplus / creditor).
  Debtor currencies earn a risk premium in the imbalances literature.
- ``ca_surplus_xs``: opposite flow / external-adjustment sort (long surplus).
- ``ca_chg_xs``: long improving Δ(CA/GDP) / short deteriorating (flow change).
- ``us_ca_gr_fx``: Gourinchas–Rey USD adjustment — when lagged US CA/GDP z is
  deeply negative (z ≤ −z_high), tilt **long foreign / short USD** (eventual
  external adjustment / USD depreciation prior).
- ``us_ca_haven_usd``: alternate risk prior — deep US deficit → safe-haven USD
  demand (long USD). Reported for honesty; not the primary GR channel.
- ``ca_ew``: equal-weight of ``ca_debtor_xs``, ``ca_chg_xs``, ``us_ca_gr_fx``.

PIT: loader ``pub_lag_quarters`` (default 2) + ``signal_lag`` months + 1 trading
day weight lag. Distinct from PPP/BS/macro-diff waves (those use CPI/IP/UR).

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
class CurrentAccountFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # months after publication-lagged CA is known
    n_long: int = 2
    n_short: int = 2
    z_window: int = 60  # months for trailing z (US CA state)
    min_periods: int = 24
    z_high: float = 1.0  # |z| threshold for US CA tilt
    usd_tilt: float = 0.5  # gross |w| when tilt is on
    cost_bps_side: float = 1.5
    chg_periods: int = 4  # YoY-ish change on monthly-ffilled quarterly CA (= 4 months)


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


def prepare_ca_scores(
    ca_panel: pd.DataFrame,
    *,
    cfg: CurrentAccountFxConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build signed monthly score panels from PIT CA/GDP (already pub-lagged).

    Scores exclude USD column when present (XS sorts are foreign vs USD).
    """
    cfg = cfg or CurrentAccountFxConfig()
    panel = ca_panel.copy()
    panel.index = _month_start(pd.DatetimeIndex(panel.index))
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    foreign_cols = [c for c in panel.columns if c.upper() != "USD" and c in CURRENCY_USD_PAIR]
    foreign = panel[foreign_cols]

    out: dict[str, pd.DataFrame] = {}

    # Debtor premium: low CA → high score → long
    debtor = (-foreign).copy()
    if cfg.signal_lag > 0:
        debtor = debtor.shift(int(cfg.signal_lag))
    out["ca_debtor"] = debtor

    # Surplus / flow: high CA → high score → long
    surplus = foreign.copy()
    if cfg.signal_lag > 0:
        surplus = surplus.shift(int(cfg.signal_lag))
    out["ca_surplus"] = surplus

    # Change: improving CA → long
    chg = foreign.diff(int(cfg.chg_periods))
    if cfg.signal_lag > 0:
        chg = chg.shift(int(cfg.signal_lag))
    out["ca_chg"] = chg

    return out


def _scores_to_daily_returns(
    score: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cfg: CurrentAccountFxConfig,
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
    # align columns
    for c in pair_ret.columns:
        if c not in daily_w.columns:
            daily_w[c] = 0.0
    daily_w = daily_w.reindex(columns=list(pair_ret.columns)).fillna(0.0)
    r = portfolio_returns_from_pair_weights(daily_w, pair_ret)
    r = apply_costs(r, daily_w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


def _us_ca_tilt_returns(
    us_ca: pd.Series,
    pair_ret: pd.DataFrame,
    *,
    cfg: CurrentAccountFxConfig,
    mode: str,
    name: str,
) -> pd.Series:
    """Binary USD / FX tilt from lagged US CA/GDP z.

    mode='gr': z ≤ −z_high → long foreign (short USD) — Gourinchas–Rey adjustment.
    mode='haven': z ≤ −z_high → long USD — safe-haven alternate.
    """
    s = us_ca.copy()
    s.index = _month_start(pd.DatetimeIndex(s.index))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    z = trailing_z(s, lookback=cfg.z_window, min_periods=cfg.min_periods)
    if cfg.signal_lag > 0:
        z = z.shift(int(cfg.signal_lag))

    # Expand monthly z to daily via month-end then ffill + 1d lag
    z_me = z.copy()
    z_me.index = (
        pd.DatetimeIndex(z_me.index)
        .tz_convert(None)
        .to_period("M")
        .to_timestamp(how="end")
        .tz_localize("UTC")
    )
    z_daily = z_me.reindex(pair_ret.index, method="ffill").shift(1)

    on = (z_daily <= -float(cfg.z_high)).astype(float)
    on = on.where(z_daily.notna(), 0.0)
    intensity = on * float(cfg.usd_tilt)

    cols = [c for c in pair_ret.columns if c.upper() in {p.upper() for p in USD_LONG_PAIRS}]
    if not cols:
        cols = list(pair_ret.columns)
    n = max(len(cols), 1)
    w = pd.DataFrame(0.0, index=pair_ret.index, columns=list(pair_ret.columns))
    for sym in cols:
        usd_sign = USD_LONG_PAIRS.get(sym.upper(), 0)
        if mode == "gr":
            # long foreign = opposite of long-USD sign
            w[sym] = (intensity / n) * (-usd_sign)
        else:
            w[sym] = (intensity / n) * usd_sign
    r = portfolio_returns_from_pair_weights(w, pair_ret)
    r = apply_costs(r, w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


def current_account_factor_returns(
    pair_ret: pd.DataFrame,
    ca_panel: pd.DataFrame,
    *,
    cfg: CurrentAccountFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Daily returns for CA imbalance factors + EW blend."""
    cfg = cfg or CurrentAccountFxConfig()
    scores = prepare_ca_scores(ca_panel, cfg=cfg)
    factors: dict[str, pd.Series] = {}

    if "ca_debtor" in scores:
        factors["ca_debtor_xs"] = _scores_to_daily_returns(
            scores["ca_debtor"], pair_ret, cfg, name="ca_debtor_xs"
        )
    if "ca_surplus" in scores:
        factors["ca_surplus_xs"] = _scores_to_daily_returns(
            scores["ca_surplus"], pair_ret, cfg, name="ca_surplus_xs"
        )
    if "ca_chg" in scores:
        factors["ca_chg_xs"] = _scores_to_daily_returns(
            scores["ca_chg"], pair_ret, cfg, name="ca_chg_xs"
        )

    us = None
    if "USD" in ca_panel.columns:
        us = ca_panel["USD"]
    if us is not None and us.dropna().shape[0] >= cfg.min_periods:
        factors["us_ca_gr_fx"] = _us_ca_tilt_returns(
            us, pair_ret, cfg=cfg, mode="gr", name="us_ca_gr_fx"
        )
        factors["us_ca_haven_usd"] = _us_ca_tilt_returns(
            us, pair_ret, cfg=cfg, mode="haven", name="us_ca_haven_usd"
        )

    blend_keys = [k for k in ("ca_debtor_xs", "ca_chg_xs", "us_ca_gr_fx") if k in factors]
    if blend_keys:
        blend = sum(factors[k] for k in blend_keys) / len(blend_keys)
        blend.name = "ca_ew"
        factors["ca_ew"] = blend

    return factors
