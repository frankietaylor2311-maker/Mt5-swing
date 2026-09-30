"""Du–Schreger CIP-conditioned Rogoff PPP / real-FX value factors (§83).

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Rogoff (1996) PPP puzzle / real exchange-rate mean reversion —
  long undervalued / short overvalued cross-sectional real-FX value
  (``ppp_real_fx``: trailing 60m z of q = S·CPI_US/CPI_f; honesty 120m).
- Du, Tepper & Verdelhan (2018), *JF*: observed CIP / cross-currency basis
  deviations reflect intermediary / dollar-funding stress.
- Du, Im & Schreger (2018) / Du, Keerati & Schreger (2025): government-bond CIP
  ``cip_govt`` (bps) and U.S. Treasury Premium (−cross-sectional mean CIP)
  measure synthetic-dollar sovereign cost / Treasury convenience.

Claim (a priori)
----------------
Rogoff PPP / real-FX cross-sectional value earns more when aggregate CIP /
dollar-funding stress is **low**. Scale down or gate value when stress is
elevated: ``cip_stress ≡ UST premium = −mean(G10 cip_govt)``. Parallel to
CIP×carry §68 / CIP soft §71 / CIP×mom §82 and to VIX/GPR×PPP §75 /
EPU/TPU×PPP §80 — **not** a cooler overlay on locked fx4plus.

Legs
----
- ``value_low_cip`` (**PRIMARY**): PPP 60m XS value only when lagged
  CIP-stress z ≤ 0 (flat in elevated stress).
- ``value_cip_cool``: value × risk_scale ∈ [cool, 1] from CIP-stress z
  (cool when z ≥ z_high).
- ``value_raw``: always-on PPP 60m XS (honesty baseline).
- ``value_high_cip``: inverse gate — value only when stress z ≥ z_high.
- ``cip_stress_haven_usd``: long USD when lagged CIP-stress z ≥ z_high
  (same construction as §68/§71/§82).
- ``value_cip_stack``: EW of value_cip_cool ⊕ cip_stress_haven_usd
  (continuous cool stacked with haven tilt).
- ``value_cip_ew``: EW of value_low_cip ⊕ value_cip_cool ⊕ cip_stress_haven_usd.
- ``value_cip_regime``: value_low_cip + cip_stress_haven_usd (exclusive regimes:
  value when z≤0, haven when z≥z_high).

PIT
---
CIP panel: loader ``pub_lag_days=1`` → month-end; this module adds
``signal_lag_months`` (default 1) on stress z + **1 trading-day** weight lag
(reuse ``align_cip_stress_z_daily`` from §68 — same z_window=60m as §68/§71/§82).
PPP: CPI publication lag (loader) + ``ppp_signal_lag`` months on z
(default 1) + 1 trading-day lag on expanded monthly weights.
``portfolio_returns_from_weights`` applies an extra weight lag.

Gate / cool thresholds: fixed z (z_high=1.0 / gate ≤0) — mirror §68/§71/§82.
**Do not** grid on holdout.

Distinct from: raw PPP, BIS REER §31, VIX/GPR×PPP §75, EPU×PPP §80,
EPU×REER §77, VIX×REER §81, CIP soft §71, CIP×carry §68, CIP×mom §82,
soft–carry–value–REER §72–§81, capital-sleeve §53/§70, combo §8.
Explicit: do **not** overlay coolers on locked ``fx4plus_gbpcad_d1_voltarget_0025``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mt5_swing.data.macro_uncertainty import CURRENCY_USD_PAIR
from mt5_swing.strategies.carry_rank import portfolio_returns_from_weights
from mt5_swing.strategies.cip_conditioned_carry_fx import (
    CipConditionedCarryFxConfig,
    align_cip_stress_z_daily,
    cip_stress_from_panel,
    usd_tilt_from_cip_stress_z,
)
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
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class CipConditionedValueFxConfig:
    """Fixed research priors — do not grid / holdout-tune.

    CIP stress fields mirror §68/§71/§82 (monthly z_window=60). PPP fields
    mirror §75/§80 Rogoff construction.
    """

    # CIP monthly z (§68/§71/§82 mirror)
    signal_lag_months: int = 1
    weight_lag_days: int = 1  # after month-end ffill of stress z
    z_window: int = 60  # months (~5y)
    min_periods: int = 24
    z_high: float = 1.0
    z_low: float = 0.0
    cool: float = 0.35
    usd_tilt: float = 0.5
    cost_bps_side: float = 1.5
    # Rogoff / PPP real-FX value (ppp_real_fx / §75/§80 priors — not HO-tuned)
    lookback: int = 60  # primary months; honesty 120m available in raw ppp wave
    honesty_lookback: int = 120
    n_long: int = 2
    n_short: int = 2
    ppp_signal_lag: int = 1  # months after pub-lagged CPI
    ppp_min_periods: int = 36
    ppp_weight_lag_days: int = 1  # expand monthly → daily


PRIMARY = "value_low_cip"


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _cip_cfg_from_value(cfg: CipConditionedValueFxConfig) -> CipConditionedCarryFxConfig:
    """Reuse §68 align_cip_stress_z_daily / usd_tilt config shape."""
    return CipConditionedCarryFxConfig(
        signal_lag_months=cfg.signal_lag_months,
        weight_lag_days=cfg.weight_lag_days,
        z_window=cfg.z_window,
        min_periods=cfg.min_periods,
        z_high=cfg.z_high,
        z_low=cfg.z_low,
        cool=cfg.cool,
        usd_tilt=cfg.usd_tilt,
        cost_bps_side=cfg.cost_bps_side,
    )


def _rank_sort_weights(
    score: pd.DataFrame,
    *,
    n_long: int,
    n_short: int,
) -> pd.DataFrame:
    """Mirror ppp_real_fx / epu_tpu_conditioned_value_fx rank-sort (local)."""
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
    cfg: CipConditionedValueFxConfig,
    lookback: int | None = None,
) -> pd.DataFrame:
    """Monthly PPP XS value → daily pair weights (PIT lags as §75/§80)."""
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
    cfg: CipConditionedValueFxConfig,
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


def cip_conditioned_value_factor_returns(
    pair_close: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cpi_levels: pd.DataFrame,
    cip_panel: pd.DataFrame,
    *,
    cfg: CipConditionedValueFxConfig | None = None,
    cip_stress_override: pd.Series | None = None,
    include_haven: bool = True,
) -> dict[str, pd.Series]:
    """Build CIP-conditioned Rogoff PPP real-FX value factor daily returns.

    Distinct from §68 (CIP×carry), §71 (CIP×soft), §75 (VIX/GPR×PPP),
    §80 (EPU/TPU×PPP), §82 (CIP×mom). Explicit: do **not** overlay coolers
    on locked fx4plus.
    """
    cfg = cfg or CipConditionedValueFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    pclose = pair_close.copy()
    pclose.index = _ensure_utc(pd.DatetimeIndex(pclose.index))
    out: dict[str, pd.Series] = {}

    if pret.empty or pret.shape[1] < 3 or cpi_levels is None or cpi_levels.empty:
        return out

    cip_cfg = _cip_cfg_from_value(cfg)
    stress = cip_stress_override
    if stress is None:
        stress = cip_stress_from_panel(cip_panel)
    z_daily = align_cip_stress_z_daily(stress, pret.index, cfg=cip_cfg)

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
    scale = risk_scale_from_z(z_daily, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
    out["value_cip_cool"] = _weights_to_returns(
        value_daily.mul(scale, axis=0), pret, cfg=cfg, name="value_cip_cool"
    )

    # Primary: trade value only when CIP stress z ≤ 0
    gate_lo = (z_daily <= 0.0).astype(float)
    gate_lo = gate_lo.where(z_daily.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["value_low_cip"] = _weights_to_returns(
        value_daily.mul(gate_lo, axis=0), pret, cfg=cfg, name="value_low_cip"
    )

    # Honesty inverse: value only when stress z ≥ z_high
    gate_hi = (z_daily >= float(cfg.z_high)).astype(float)
    gate_hi = gate_hi.where(z_daily.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["value_high_cip"] = _weights_to_returns(
        value_daily.mul(gate_hi, axis=0), pret, cfg=cfg, name="value_high_cip"
    )

    if include_haven:
        cols = list(pret.columns)
        haven_w = usd_tilt_from_cip_stress_z(z_daily, cols, cfg=cip_cfg)
        out["cip_stress_haven_usd"] = _weights_to_returns(
            haven_w, pret, cfg=cfg, name="cip_stress_haven_usd"
        )

    # Stack: continuous cool ⊕ haven tilt
    stack_keys = [k for k in ("value_cip_cool", "cip_stress_haven_usd") if k in out]
    if len(stack_keys) >= 2:
        stack = pd.concat([out[k] for k in stack_keys], axis=1).mean(axis=1)
        stack.name = "value_cip_stack"
        out["value_cip_stack"] = stack

    # EW of primary + cool + haven (§68/§71/§82 mirror)
    blend_keys = [
        k
        for k in ("value_low_cip", "value_cip_cool", "cip_stress_haven_usd")
        if k in out
    ]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "value_cip_ew"
        out["value_cip_ew"] = blend

    # Regime switch: value when low-stress + haven when high-stress (exclusive)
    if "value_low_cip" in out and "cip_stress_haven_usd" in out:
        regime = out["value_low_cip"].fillna(0.0) + out["cip_stress_haven_usd"].fillna(
            0.0
        )
        both_nan = out["value_low_cip"].isna() & out["cip_stress_haven_usd"].isna()
        regime = regime.where(~both_nan)
        regime.name = "value_cip_regime"
        out["value_cip_regime"] = regime

    return out


__all__ = [
    "PRIMARY",
    "CipConditionedValueFxConfig",
    "cip_conditioned_value_factor_returns",
    "cip_stress_from_panel",
    "align_cip_stress_z_daily",
]
