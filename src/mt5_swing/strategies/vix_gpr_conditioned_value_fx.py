"""VIX/GPR-conditioned Rogoff PPP / real-FX value factors (§75).

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Rogoff (1996) PPP puzzle / real exchange-rate mean reversion —
  long undervalued / short overvalued cross-sectional real-FX value
  (``ppp_real_fx``: trailing 60m z of q = S·CPI_US/CPI_f; honesty 120m).
- Menkhoff, Sarno, Schmeling & Schrimpf (2012), *JF*: high global FX /
  equity vol → risk-off / crash states. Proxy with **VIX**.
- Caldara & Iacoviello (2022), *AER* / Liu & Zhang (2024), *JBF*: elevated
  aggregate GPR → risk-off. Free series ``gpr_daily.csv``.

Claim (a priori)
----------------
PPP / real-FX value earns more when global FX-vol / geopolitical stress is
**low**. Scale down or gate value when VIX or GPR stress is elevated.
Completes the classic FX triad under the same stress gates as §72–§74
(soft EW / carry / mom × VIX/GPR) — **not** a cooler overlay on locked fx4plus.

Legs
----
- ``value_low_vix`` (**PRIMARY**): PPP 60m XS value only when lagged VIX z ≤ 0.
- ``value_vix_cool``: value × risk_scale ∈ [cool, 1] from VIX z.
- ``value_low_gpr``: value only when lagged GPR z ≤ 0.
- ``value_gpr_cool``: value × cool from GPR z.
- ``value_raw``: always-on PPP 60m XS (honesty).
- ``value_high_vix``: honesty inverse — value only when VIX z ≥ z_high.
- ``value_vix_gpr_stack``: sequential VIX then GPR cool (product of scales).
- ``value_vix_gpr_ew``: EW of low_vix ⊕ vix_cool ⊕ low_gpr ⊕ gpr_cool.

PIT
---
CPI: publication lag (loader) + ``ppp_signal_lag`` months on z (default 1) +
1 trading-day lag on expanded monthly weights.
VIX / GPR: loader ``bar_lag=1`` + ``signal_lag`` (default 1d) on trailing z
(``z_window=252``, ``min_periods=60``) — same dual lag as §72–§74.
``portfolio_returns_from_weights`` applies an extra weight lag (mom/carry
conditioning approach).

Gate / cool: fixed z (z_high=1.0 / gate ≤0) — §71–§74 symmetry. Literature
often uses **1.5** for pure GPR risk-off; documented, not HO-tuned.

Distinct from: raw ppp_wave / bis_reer, soft §66/§72, carry §73, mom §74,
CIP waves, capital-sleeve, combo §8. Explicit: do **not** overlay coolers on
locked ``fx4plus_gbpcad_d1_voltarget_0025``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

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
from mt5_swing.strategies.vix_gpr_conditioned_soft_fx import (
    align_stress_z_daily,
    load_gpr_series,
    load_vix_series,
)
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class VixGprConditionedValueFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # trading days after loader bar_lag=1 (VIX/GPR z)
    z_window: int = 252  # daily, like §72–§74 — NOT monthly CIP 60
    min_periods: int = 60
    z_high: float = 1.0  # cool thresh — §71–§74 symmetry (lit GPR often 1.5)
    z_low: float = 0.0
    cool: float = 0.35
    cost_bps_side: float = 1.5
    # Rogoff / PPP real-FX value (ppp_real_fx priors — not HO-tuned)
    lookback: int = 60  # primary months; honesty 120m available in raw ppp wave
    honesty_lookback: int = 120
    n_long: int = 2
    n_short: int = 2
    ppp_signal_lag: int = 1  # months after pub-lagged CPI
    ppp_min_periods: int = 36
    ppp_weight_lag_days: int = 1  # expand monthly → daily
    # Documented: literature GPR cool often z≥1.5; this wave keeps 1.0 for symmetry
    gpr_lit_z_high_note: float = 1.5


PRIMARY = "value_low_vix"


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _rank_sort_weights(
    score: pd.DataFrame,
    *,
    n_long: int,
    n_short: int,
) -> pd.DataFrame:
    """Mirror ppp_real_fx._rank_sort_weights (local to avoid private import)."""
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
    cfg: VixGprConditionedValueFxConfig,
    lookback: int | None = None,
) -> pd.DataFrame:
    """Monthly PPP XS value → daily pair weights (PIT lags)."""
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
    cfg: VixGprConditionedValueFxConfig,
    name: str,
) -> pd.Series:
    cols = list(pair_ret.columns)
    for c in cols:
        if c not in w.columns:
            w[c] = 0.0
    w = w.reindex(columns=cols).fillna(0.0)
    # Mom/carry conditioning approach: extra weight lag inside helper
    r = portfolio_returns_from_weights(w, pair_ret)
    r = apply_costs(r, w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


def _soft_cfg_from_value(cfg: VixGprConditionedValueFxConfig):
    """Reuse §72–§74 align_stress_z_daily config shape."""
    from mt5_swing.strategies.vix_gpr_conditioned_soft_fx import (
        VixGprConditionedSoftFxConfig,
    )

    return VixGprConditionedSoftFxConfig(
        signal_lag=cfg.signal_lag,
        z_window=cfg.z_window,
        min_periods=cfg.min_periods,
        z_high=cfg.z_high,
        z_low=cfg.z_low,
        cool=cfg.cool,
        cost_bps_side=cfg.cost_bps_side,
    )


def vix_gpr_conditioned_value_factor_returns(
    pair_close: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cpi_levels: pd.DataFrame,
    *,
    vix: pd.Series | None = None,
    gpr: pd.Series | None = None,
    cfg: VixGprConditionedValueFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Build VIX/GPR-conditioned Rogoff PPP real-FX value factor daily returns.

    Distinct from raw ppp_wave, §72 soft, §73 carry, §74 mom. Explicit: do
    **not** overlay coolers on locked fx4plus.
    """
    cfg = cfg or VixGprConditionedValueFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    pclose = pair_close.copy()
    pclose.index = _ensure_utc(pd.DatetimeIndex(pclose.index))
    out: dict[str, pd.Series] = {}

    if pret.empty or pret.shape[1] < 3 or cpi_levels is None or cpi_levels.empty:
        return out

    if vix is None:
        vix = load_vix_series(bar_lag=1)
    if gpr is None:
        gpr = load_gpr_series(bar_lag=1)

    soft_cfg = _soft_cfg_from_value(cfg)
    z_vix = align_stress_z_daily(vix, pret.index, cfg=soft_cfg)
    z_gpr = align_stress_z_daily(gpr, pret.index, cfg=soft_cfg)

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
    vix_scale = risk_scale_from_z(z_vix, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
    gpr_scale = risk_scale_from_z(z_gpr, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)

    cooled_vix = value_daily.mul(vix_scale, axis=0)
    out["value_vix_cool"] = _weights_to_returns(
        cooled_vix, pret, cfg=cfg, name="value_vix_cool"
    )

    cooled_gpr = value_daily.mul(gpr_scale, axis=0)
    out["value_gpr_cool"] = _weights_to_returns(
        cooled_gpr, pret, cfg=cfg, name="value_gpr_cool"
    )

    # Primary: trade value only when VIX z ≤ 0
    gate_vix_lo = (z_vix <= 0.0).astype(float)
    gate_vix_lo = gate_vix_lo.where(z_vix.notna(), 0.0).reindex(pret.index).fillna(0.0)
    gated_lo = value_daily.mul(gate_vix_lo, axis=0)
    out["value_low_vix"] = _weights_to_returns(
        gated_lo, pret, cfg=cfg, name="value_low_vix"
    )

    gate_gpr_lo = (z_gpr <= 0.0).astype(float)
    gate_gpr_lo = gate_gpr_lo.where(z_gpr.notna(), 0.0).reindex(pret.index).fillna(0.0)
    gated_gpr = value_daily.mul(gate_gpr_lo, axis=0)
    out["value_low_gpr"] = _weights_to_returns(
        gated_gpr, pret, cfg=cfg, name="value_low_gpr"
    )

    # Honesty inverse: value only when VIX z ≥ z_high
    gate_vix_hi = (z_vix >= float(cfg.z_high)).astype(float)
    gate_vix_hi = gate_vix_hi.where(z_vix.notna(), 0.0).reindex(pret.index).fillna(0.0)
    gated_hi = value_daily.mul(gate_vix_hi, axis=0)
    out["value_high_vix"] = _weights_to_returns(
        gated_hi, pret, cfg=cfg, name="value_high_vix"
    )

    # Sequential VIX then GPR cool (product of scales)
    stack_scale = (vix_scale * gpr_scale).clip(lower=float(cfg.cool) ** 2, upper=1.0)
    cooled_stack = value_daily.mul(stack_scale, axis=0)
    out["value_vix_gpr_stack"] = _weights_to_returns(
        cooled_stack, pret, cfg=cfg, name="value_vix_gpr_stack"
    )

    blend_keys = [
        k
        for k in ("value_low_vix", "value_vix_cool", "value_low_gpr", "value_gpr_cool")
        if k in out
    ]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "value_vix_gpr_ew"
        out["value_vix_gpr_ew"] = blend

    return out


__all__ = [
    "PRIMARY",
    "VixGprConditionedValueFxConfig",
    "vix_gpr_conditioned_value_factor_returns",
    "align_stress_z_daily",
    "load_vix_series",
    "load_gpr_series",
]
