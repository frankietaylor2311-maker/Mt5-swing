"""WUI-conditioned Rogoff PPP / real-FX value factors (§88).

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Rogoff (1996) PPP puzzle / real exchange-rate mean reversion —
  long undervalued / short overvalued cross-sectional real-FX value
  (``ppp_real_fx``: trailing 60m z of q = S·CPI_US/CPI_f; honesty 120m).
- Ahir, Bloom & Furceri World Uncertainty Index (WUI): EIU-text country
  uncertainty (quarterly → monthly). We use **US WUI** (FRED WUIUSA) as the
  free aggregate stress series via ``mt5_swing.data.fred_wui.load_us_wui``
  (pub_lag_months=4 already applied in the loader).

Claim (a priori)
----------------
Rogoff PPP / real-FX cross-sectional value earns more when US / world-
uncertainty stress is **low**. Trade value only when lagged US WUI z is
subdued; cool or sit out when WUI stress is elevated. Single-stress design
(like CIP×PPP §83), not dual EPU+TPU.

Legs
----
- ``value_low_wui`` (**PRIMARY**): PPP 60m XS value only when lagged
  US WUI z ≤ 0 (flat in elevated stress).
- ``value_wui_cool``: value × risk_scale ∈ [cool, 1] from WUI z
  (cool when z ≥ z_high).
- ``value_raw``: always-on PPP 60m XS (honesty baseline).
- ``value_high_wui``: honesty inverse — value only when WUI z ≥ z_high.
- ``us_wui_haven_usd``: long USD when lagged WUI z ≥ z_high
  (§40 / CIP-stress haven companion pattern / §85–§87).
- ``value_wui_stack``: value_low_wui × cool scale (sequential gate then cool —
  WUI mom §87 / WUI carry §86 mirror, not CIP value EW cool⊕haven).
- ``value_wui_ew``: EW of value_low_wui ⊕ value_wui_cool ⊕ us_wui_haven_usd.
- ``value_wui_regime``: value_low_wui + us_wui_haven_usd (exclusive regimes:
  value when z≤0, haven when z≥z_high).

PIT
---
``load_us_wui`` applies ``pub_lag_months=4`` (quarterly EIU). This module adds
``signal_lag_months`` (default 1) on **monthly** trailing z (z_window=60m,
min_periods=24 — mirror CIP §71 / EPU §76 / WUI soft §85 / WUI carry §86 /
WUI mom §87, **not** 252d daily) + ``weight_lag_days=1`` after month-end ffill
to the trading calendar. PPP: CPI publication lag (loader) + ``ppp_signal_lag``
months on z (default 1) + 1 trading-day lag on expanded monthly weights.
``portfolio_returns_from_weights`` applies an extra weight lag.
Costs 1.5 bps/side.

Gate / cool thresholds: fixed z (z_high=1.0 / gate ≤0) — mirror §71–§87.
Literature often uses other cutoffs; documented symmetry, **not** HO-tuned.

Distinct from: raw WUI XS §40, WUI soft §85, WUI carry §86, WUI mom §87,
CIP×PPP §83, VIX/GPR×PPP §75, EPU×PPP §80, raw PPP, BIS REER §31,
soft–carry–mom–value–REER §68–§84, capital-sleeve §53/§70, combo §8.
Explicit: do **not** overlay coolers on locked ``fx4plus_gbpcad_d1_voltarget_0025``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mt5_swing.data.fred_wui import DEFAULT_PUB_LAG_MONTHS, load_us_wui
from mt5_swing.data.macro_uncertainty import CURRENCY_USD_PAIR
from mt5_swing.strategies.carry_rank import portfolio_returns_from_weights
from mt5_swing.strategies.country_gpr_fx import (
    currency_weights_to_pair_weights_fx,
    expand_monthly_weights_to_daily,
)
from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z
from mt5_swing.strategies.ppp_real_fx import (
    PppRealFxConfig,
    monthly_nominal_fx,
    ppp_value_scores,
    real_fx_panel,
)
from mt5_swing.strategies.wui_conditioned_soft_fx import (
    WuiConditionedSoftFxConfig,
    align_monthly_stress_z_daily,
    usd_tilt_from_wui_z,
)
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class WuiConditionedValueFxConfig:
    """Fixed research priors — do not grid / holdout-tune.

    WUI stress fields mirror §85–§87 (monthly z_window=60). PPP fields
    mirror §75/§80/§83 Rogoff construction.
    """

    # WUI monthly z (§85–§87 mirror)
    signal_lag_months: int = 1  # months after pub-lagged monthly index known
    weight_lag_days: int = 1  # trading days after month-end ffill
    z_window: int = 60  # months (~5y) — mirror CIP §71 / EPU §76 / WUI §85–§87
    min_periods: int = 24
    z_high: float = 1.0  # cool / haven thresh — §71–§87 symmetry
    z_low: float = 0.0
    cool: float = 0.35
    usd_tilt: float = 0.5  # gross |w| when haven tilt is on
    cost_bps_side: float = 1.5
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS  # informational (loader default 4)
    # Documented: lit often uses other cutoffs; this wave keeps 1.0 for symmetry
    lit_z_high_note: float = 1.5
    # Rogoff / PPP real-FX value (ppp_real_fx / §75/§80/§83 priors — not HO-tuned)
    lookback: int = 60  # primary months; honesty 120m available in raw ppp wave
    honesty_lookback: int = 120
    n_long: int = 2
    n_short: int = 2
    ppp_signal_lag: int = 1  # months after pub-lagged CPI
    ppp_min_periods: int = 36
    ppp_weight_lag_days: int = 1  # expand monthly → daily


PRIMARY = "value_low_wui"


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _soft_cfg_from_value(cfg: WuiConditionedValueFxConfig) -> WuiConditionedSoftFxConfig:
    """Reuse §85 align_monthly_stress_z_daily / usd_tilt config shape."""
    return WuiConditionedSoftFxConfig(
        signal_lag_months=cfg.signal_lag_months,
        weight_lag_days=cfg.weight_lag_days,
        z_window=cfg.z_window,
        min_periods=cfg.min_periods,
        z_high=cfg.z_high,
        z_low=cfg.z_low,
        cool=cfg.cool,
        usd_tilt=cfg.usd_tilt,
        cost_bps_side=cfg.cost_bps_side,
        pub_lag_months=cfg.pub_lag_months,
    )


def load_us_wui_series(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """Thin wrapper — Ahir–Bloom–Furceri US WUI (WUIUSA)."""
    return load_us_wui(
        pub_lag_months=pub_lag_months, download=download, force=force
    )


def _rank_sort_weights(
    score: pd.DataFrame,
    *,
    n_long: int,
    n_short: int,
) -> pd.DataFrame:
    """Mirror ppp_real_fx / cip_conditioned_value_fx rank-sort (local)."""
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


def _value_daily_weights(
    pair_close: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cpi_levels: pd.DataFrame,
    *,
    cfg: WuiConditionedValueFxConfig,
    lookback: int | None = None,
) -> pd.DataFrame:
    """Monthly PPP XS value → daily pair weights (PIT lags as §75/§80/§83)."""
    lb = int(lookback if lookback is not None else cfg.lookback)
    pcfg = PppRealFxConfig(
        signal_lag=cfg.ppp_signal_lag,
        n_long=cfg.n_long,
        n_short=cfg.n_short,
        lookbacks=(lb,),
        min_periods=cfg.ppp_min_periods,
        cost_bps_per_side=0.0,
    )
    fx_m = monthly_nominal_fx(pair_close)
    if fx_m.empty or cpi_levels is None or cpi_levels.empty:
        return pd.DataFrame(0.0, index=pair_ret.index, columns=list(pair_ret.columns))
    q = real_fx_panel(fx_m, cpi_levels)
    score = ppp_value_scores(q, cfg=pcfg, lookback=lb)
    ccy_w = _rank_sort_weights(score, n_long=cfg.n_long, n_short=cfg.n_short)
    if ccy_w.empty:
        return pd.DataFrame(0.0, index=pair_ret.index, columns=list(pair_ret.columns))
    ccy_w = ccy_w.copy()
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
    daily = expand_monthly_weights_to_daily(
        pair_w, pair_ret.index, signal_lag_days=cfg.ppp_weight_lag_days
    )
    cols = list(pair_ret.columns)
    for c in cols:
        if c not in daily.columns:
            daily[c] = 0.0
    return daily.reindex(columns=cols).fillna(0.0)


def _weights_to_returns(
    w: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: WuiConditionedValueFxConfig,
    name: str,
) -> pd.Series:
    cols = list(pair_ret.columns)
    for c in cols:
        if c not in w.columns:
            w[c] = 0.0
    w = w.reindex(columns=cols).fillna(0.0)
    r = portfolio_returns_from_weights(w, pair_ret)
    r = apply_costs(r, w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


def _haven_returns(
    z_daily: pd.Series,
    pair_ret: pd.DataFrame,
    *,
    cfg: WuiConditionedValueFxConfig,
) -> pd.Series:
    soft_cfg = _soft_cfg_from_value(cfg)
    cols = list(pair_ret.columns)
    w = usd_tilt_from_wui_z(z_daily, cols, cfg=soft_cfg)
    return _weights_to_returns(w, pair_ret, cfg=cfg, name="us_wui_haven_usd")


def wui_conditioned_value_factor_returns(
    pair_close: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cpi_levels: pd.DataFrame,
    *,
    us_wui: pd.Series | None = None,
    cfg: WuiConditionedValueFxConfig | None = None,
    include_haven: bool = True,
) -> dict[str, pd.Series]:
    """Build WUI-conditioned Rogoff PPP real-FX value factor daily returns.

    Distinct from §83 (CIP×PPP), §75 (VIX/GPR×PPP), §80 (EPU/TPU×PPP),
    §85 (WUI soft), §86 (WUI carry), §87 (WUI mom). Explicit: do **not**
    overlay coolers on locked fx4plus.
    """
    cfg = cfg or WuiConditionedValueFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    pclose = pair_close.copy()
    pclose.index = _ensure_utc(pd.DatetimeIndex(pclose.index))
    out: dict[str, pd.Series] = {}

    if pret.empty or pret.shape[1] < 3 or cpi_levels is None or cpi_levels.empty:
        return out

    if us_wui is None:
        us_wui = load_us_wui_series(pub_lag_months=cfg.pub_lag_months)

    soft_cfg = _soft_cfg_from_value(cfg)
    z_daily = align_monthly_stress_z_daily(us_wui, pret.index, cfg=soft_cfg)

    value_daily = _value_daily_weights(pclose, pret, cpi_levels, cfg=cfg)

    raw = portfolio_returns_from_weights(value_daily, pret)
    raw = apply_costs(raw, value_daily, bps_side=cfg.cost_bps_side)
    raw.name = "value_raw"
    out["value_raw"] = raw

    gcfg = GprRegimeConfig(
        z_high=cfg.z_high,
        z_low=cfg.z_low,
        cool=cfg.cool,
        signal_lag=0,  # z already lagged
        usd_tilt=0.0,
    )
    cool_scale = (
        risk_scale_from_z(z_daily, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
    )
    out["value_wui_cool"] = _weights_to_returns(
        value_daily.mul(cool_scale, axis=0), pret, cfg=cfg, name="value_wui_cool"
    )

    # Primary: trade value only when US WUI z ≤ 0
    gate_lo = (z_daily <= 0.0).astype(float)
    gate_lo = gate_lo.where(z_daily.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["value_low_wui"] = _weights_to_returns(
        value_daily.mul(gate_lo, axis=0), pret, cfg=cfg, name="value_low_wui"
    )

    # Honesty inverse: value only when WUI z ≥ z_high
    gate_hi = (z_daily >= float(cfg.z_high)).astype(float)
    gate_hi = gate_hi.where(z_daily.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["value_high_wui"] = _weights_to_returns(
        value_daily.mul(gate_hi, axis=0), pret, cfg=cfg, name="value_high_wui"
    )

    if include_haven:
        out["us_wui_haven_usd"] = _haven_returns(z_daily, pret, cfg=cfg)

    # Stack: sequential gate then cool (value_low weights × cool scale) — §86/§87 mirror
    stack_scale = (gate_lo * cool_scale).clip(lower=0.0, upper=1.0)
    out["value_wui_stack"] = _weights_to_returns(
        value_daily.mul(stack_scale, axis=0), pret, cfg=cfg, name="value_wui_stack"
    )

    # EW of primary + cool + haven (§68 / §85 / §86 / §87 mirror)
    blend_keys = [
        k
        for k in ("value_low_wui", "value_wui_cool", "us_wui_haven_usd")
        if k in out
    ]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "value_wui_ew"
        out["value_wui_ew"] = blend

    # Regime switch: value when low-WUI + haven when high-WUI (exclusive)
    if "value_low_wui" in out and "us_wui_haven_usd" in out:
        regime = out["value_low_wui"].fillna(0.0) + out["us_wui_haven_usd"].fillna(0.0)
        both_nan = out["value_low_wui"].isna() & out["us_wui_haven_usd"].isna()
        regime = regime.where(~both_nan)
        regime.name = "value_wui_regime"
        out["value_wui_regime"] = regime

    return out


__all__ = [
    "PRIMARY",
    "WuiConditionedValueFxConfig",
    "align_monthly_stress_z_daily",
    "load_us_wui_series",
    "usd_tilt_from_wui_z",
    "wui_conditioned_value_factor_returns",
]
