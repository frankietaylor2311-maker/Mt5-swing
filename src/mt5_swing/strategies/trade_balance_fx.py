"""Monthly trade-balance FX factors (high-frequency external imbalance).

Fixed priors (no holdout tuning)
--------------------------------
- ``tb_deficit_xs`` (**primary**, Della Corte–Riddiough–Sarno / GR-style
  debtor premium): long low TB/exports (deficit) / short high surplus.
- ``tb_surplus_xs``: opposite flow / external-adjustment sort (long surplus).
- ``tb_chg_xs``: long improving Δ(TB/exports) / short deteriorating.
- ``us_tb_gr_fx``: elevated US TB deficit z (z ≤ −z_high) → long foreign /
  short USD (Gourinchas–Rey trade-channel adjustment prior).
- ``us_tb_haven_usd``: elevated US TB deficit z → long USD (safe-haven alt).
- ``tb_ca_blend``: scholarly EW of ``tb_deficit_xs`` + ``ca_debtor_xs`` when
  CA debtor returns are supplied — **not** an overlay on the locked sleeve.
- ``tb_ew``: EW of ``tb_deficit_xs``, ``tb_chg_xs``, ``us_tb_gr_fx``.

PIT: loader ``pub_lag_months`` (default 2 for monthly OECD/BEA) + ``signal_lag``
months + 1 trading day weight lag. Distinct from quarterly CA/GDP (§21),
debt/GDP (§29), fiscal (§28), CB-BS, macro-diff, equity-diff, AI-GPR, ToT,
dollar-beta, crash-skew.

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
class TradeBalanceFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # months after publication-lagged TB is known
    n_long: int = 2
    n_short: int = 2
    z_window: int = 60  # months for trailing z (US TB state)
    min_periods: int = 24
    z_high: float = 1.0  # |z| threshold for US TB tilt
    usd_tilt: float = 0.5  # gross |w| when tilt is on
    cost_bps_side: float = 1.5
    chg_periods: int = 12  # YoY change on monthly TB/exports


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


def prepare_tb_scores(
    tb_panel: pd.DataFrame,
    *,
    cfg: TradeBalanceFxConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build signed monthly score panels from PIT TB/exports (already pub-lagged).

    Scores exclude USD column when present (XS sorts are foreign vs USD).
    """
    cfg = cfg or TradeBalanceFxConfig()
    panel = tb_panel.copy()
    panel.index = _month_start(pd.DatetimeIndex(panel.index))
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    foreign_cols = [c for c in panel.columns if c.upper() != "USD" and c in CURRENCY_USD_PAIR]
    foreign = panel[foreign_cols]

    out: dict[str, pd.DataFrame] = {}

    # Debtor premium: low TB (deficit) → high score → long
    deficit = (-foreign).copy()
    if cfg.signal_lag > 0:
        deficit = deficit.shift(int(cfg.signal_lag))
    out["tb_deficit"] = deficit

    # Surplus / flow: high TB → high score → long
    surplus = foreign.copy()
    if cfg.signal_lag > 0:
        surplus = surplus.shift(int(cfg.signal_lag))
    out["tb_surplus"] = surplus

    # Change: improving = rising TB → long
    chg = foreign.diff(int(cfg.chg_periods)).copy()
    if cfg.signal_lag > 0:
        chg = chg.shift(int(cfg.signal_lag))
    out["tb_chg"] = chg

    return out


def _scores_to_daily_returns(
    score: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cfg: TradeBalanceFxConfig,
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


def _us_tb_tilt_returns(
    us_tb: pd.Series,
    pair_ret: pd.DataFrame,
    *,
    cfg: TradeBalanceFxConfig,
    mode: str,
    name: str,
) -> pd.Series:
    """Binary USD / FX tilt from lagged US TB z.

    mode='gr': z ≤ −z_high → long foreign (short USD) — GR trade adjustment.
    mode='haven': z ≤ −z_high → long USD — safe-haven alternate.
    """
    s = us_tb.copy()
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
            w[sym] = (intensity / n) * (-usd_sign)
        else:
            w[sym] = (intensity / n) * usd_sign
    r = portfolio_returns_from_pair_weights(w, pair_ret)
    r = apply_costs(r, w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


def trade_balance_factor_returns(
    pair_ret: pd.DataFrame,
    tb_panel: pd.DataFrame,
    *,
    cfg: TradeBalanceFxConfig | None = None,
    us_tb_override: pd.Series | None = None,
    ca_debtor_returns: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Daily returns for trade-balance factors + optional CA blend + EW."""
    cfg = cfg or TradeBalanceFxConfig()
    scores = prepare_tb_scores(tb_panel, cfg=cfg)
    factors: dict[str, pd.Series] = {}

    if "tb_deficit" in scores:
        factors["tb_deficit_xs"] = _scores_to_daily_returns(
            scores["tb_deficit"], pair_ret, cfg, name="tb_deficit_xs"
        )
    if "tb_surplus" in scores:
        factors["tb_surplus_xs"] = _scores_to_daily_returns(
            scores["tb_surplus"], pair_ret, cfg, name="tb_surplus_xs"
        )
    if "tb_chg" in scores:
        factors["tb_chg_xs"] = _scores_to_daily_returns(
            scores["tb_chg"], pair_ret, cfg, name="tb_chg_xs"
        )

    us = us_tb_override
    if us is None and "USD" in tb_panel.columns:
        us = tb_panel["USD"]
    if us is not None and us.dropna().shape[0] >= cfg.min_periods:
        factors["us_tb_gr_fx"] = _us_tb_tilt_returns(
            us, pair_ret, cfg=cfg, mode="gr", name="us_tb_gr_fx"
        )
        factors["us_tb_haven_usd"] = _us_tb_tilt_returns(
            us, pair_ret, cfg=cfg, mode="haven", name="us_tb_haven_usd"
        )

    if ca_debtor_returns is not None and "tb_deficit_xs" in factors:
        a = factors["tb_deficit_xs"]
        b = ca_debtor_returns.reindex(a.index).fillna(0.0)
        blend = 0.5 * a + 0.5 * b
        blend.name = "tb_ca_blend"
        factors["tb_ca_blend"] = blend

    blend_keys = [
        k for k in ("tb_deficit_xs", "tb_chg_xs", "us_tb_gr_fx") if k in factors
    ]
    if blend_keys:
        ew = sum(factors[k] for k in blend_keys) / len(blend_keys)
        ew.name = "tb_ew"
        factors["tb_ew"] = ew

    return factors
