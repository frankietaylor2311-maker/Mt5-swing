"""EPU/TPU-conditioned Rogoff PPP / real-FX value factors (§80).

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Rogoff (1996) PPP puzzle / real exchange-rate mean reversion —
  long undervalued / short overvalued cross-sectional real-FX value
  (``ppp_real_fx``: trailing 60m z of q = S·CPI_US/CPI_f; honesty 120m).
- Baker, Bloom & Davis (2016), *QJE*: news-based Economic Policy Uncertainty
  (EPU). We use **US EPU** (FRED USEPUINDXM) as primary Stress A; GEPU as
  documented aggregate fall-through if US missing.
- Baker–Bloom–Davis categorical *Trade policy* EPU = US Trade Policy
  Uncertainty (TPU) on policyuncertainty.com — Stress B.

Claim (a priori)
----------------
Rogoff PPP / real-FX cross-sectional value earns more when economic /
trade-policy uncertainty is **low**. Scale down or gate value when EPU or
TPU stress is elevated. Parallel to §75 (VIX/GPR×PPP value) but with free
EPU/TPU **monthly** stress series (same as §76/§77/§78/§79) — **not** a
cooler overlay on locked fx4plus.

Legs
----
- ``value_low_epu`` (**PRIMARY**): PPP 60m XS value only when
  lagged EPU z ≤ 0 (flat when elevated).
- ``value_epu_cool``: value × risk_scale ∈ [cool, 1] from EPU z
  (cool when z ≥ z_high).
- ``value_low_tpu``: value only when lagged TPU z ≤ 0.
- ``value_tpu_cool``: value × cool from TPU z.
- ``value_raw``: always-on PPP 60m XS (honesty).
- ``value_high_epu``: honesty inverse — value only when EPU z ≥ z_high.
- ``value_epu_tpu_stack``: sequential EPU then TPU cool (product of scales).
- ``value_epu_tpu_ew``: EW of low_epu ⊕ epu_cool ⊕ low_tpu ⊕ tpu_cool.

PIT
---
EPU / TPU loaders apply ``pub_lag_months=1``; this module adds
``signal_lag_months`` (default 1) on **monthly** trailing z (z_window=60m,
CIP §71 / EPU soft §76 / REER §77 / carry §78 / mom §79 mirror) +
``weight_lag_days=1``.
PPP: CPI publication lag (loader) + ``ppp_signal_lag`` months on z
(default 1) + 1 trading-day lag on expanded monthly weights.
``portfolio_returns_from_weights`` applies an extra weight lag (mom/carry
conditioning approach — same as §75).

Gate / cool thresholds: fixed z (z_high=1.0 / gate ≤0) — mirror §71–§79.
Literature often uses other cutoffs; documented symmetry, **not** HO-tuned.

Distinct from: raw PPP, BIS REER §31, §75 VIX/GPR×PPP, EPU soft §76,
EPU×REER §77, EPU×carry §78, EPU×mom §79, capital-sleeve §53/§70, combo §8.
Explicit: do **not** overlay coolers on locked ``fx4plus_gbpcad_d1_voltarget_0025``.
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
from mt5_swing.strategies.epu_tpu_conditioned_soft_fx import (
    EpuTpuConditionedSoftFxConfig,
    align_monthly_stress_z_daily,
    load_epu_series,
    load_tpu_series,
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
class EpuTpuConditionedValueFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    # EPU/TPU monthly z (CIP §71 / EPU soft §76 / REER §77 / carry §78 / mom §79)
    signal_lag_months: int = 1
    weight_lag_days: int = 1  # after month-end ffill of stress z
    z_window: int = 60  # months
    min_periods: int = 24
    z_high: float = 1.0  # cool thresh — §71–§79 symmetry
    z_low: float = 0.0
    cool: float = 0.35
    cost_bps_side: float = 1.5
    # Rogoff / PPP real-FX value (ppp_real_fx / §75 priors — not HO-tuned)
    lookback: int = 60  # primary months; honesty 120m available in raw ppp wave
    honesty_lookback: int = 120
    n_long: int = 2
    n_short: int = 2
    ppp_signal_lag: int = 1  # months after pub-lagged CPI
    ppp_min_periods: int = 36
    ppp_weight_lag_days: int = 1  # expand monthly → daily
    # Documented: lit often uses other cutoffs; this wave keeps 1.0 for symmetry
    lit_z_high_note: float = 1.5


PRIMARY = "value_low_epu"

_TPU_AUTO = object()


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
    cfg: EpuTpuConditionedValueFxConfig,
    lookback: int | None = None,
) -> pd.DataFrame:
    """Monthly PPP XS value → daily pair weights (PIT lags as §75)."""
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
    cfg: EpuTpuConditionedValueFxConfig,
    name: str,
) -> pd.Series:
    cols = list(pair_ret.columns)
    for c in cols:
        if c not in w.columns:
            w[c] = 0.0
    w = w.reindex(columns=cols).fillna(0.0)
    # Mom/carry conditioning approach: extra weight lag inside helper (§75)
    r = portfolio_returns_from_weights(w, pair_ret)
    r = apply_costs(r, w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


def _soft_cfg_from_value(
    cfg: EpuTpuConditionedValueFxConfig,
) -> EpuTpuConditionedSoftFxConfig:
    """Reuse §76 align_monthly_stress_z_daily config shape."""
    return EpuTpuConditionedSoftFxConfig(
        signal_lag_months=cfg.signal_lag_months,
        weight_lag_days=cfg.weight_lag_days,
        z_window=cfg.z_window,
        min_periods=cfg.min_periods,
        z_high=cfg.z_high,
        z_low=cfg.z_low,
        cool=cfg.cool,
        cost_bps_side=cfg.cost_bps_side,
    )


def epu_tpu_conditioned_value_factor_returns(
    pair_close: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cpi_levels: pd.DataFrame,
    *,
    epu: pd.Series | None = None,
    tpu: pd.Series | None | object = _TPU_AUTO,
    cfg: EpuTpuConditionedValueFxConfig | None = None,
    allow_tpu_missing: bool = True,
) -> dict[str, pd.Series]:
    """Build EPU/TPU-conditioned Rogoff PPP real-FX value factor daily returns.

    Distinct from raw ppp_wave, §75 VIX/GPR×PPP, §76 soft EPU, §77 REER EPU,
    §78 carry EPU, §79 mom EPU. Explicit: do **not** overlay coolers on locked
    fx4plus.

    ``tpu`` semantics: default auto-load; pass ``None`` for explicit EPU-only
    fall-through (TPU companions omitted). If auto-load fails and
    ``allow_tpu_missing``, same EPU-only honesty path.
    """
    cfg = cfg or EpuTpuConditionedValueFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    pclose = pair_close.copy()
    pclose.index = _ensure_utc(pd.DatetimeIndex(pclose.index))
    out: dict[str, pd.Series] = {}

    if pret.empty or pret.shape[1] < 3 or cpi_levels is None or cpi_levels.empty:
        return out

    if epu is None:
        epu = load_epu_series(series="US", pub_lag_months=1)

    tpu_available = True
    if tpu is _TPU_AUTO:
        try:
            tpu = load_tpu_series(pub_lag_months=1)
        except Exception:
            if not allow_tpu_missing:
                raise
            tpu = None
            tpu_available = False
    elif tpu is None:
        tpu_available = False

    soft_cfg = _soft_cfg_from_value(cfg)
    z_epu = align_monthly_stress_z_daily(epu, pret.index, cfg=soft_cfg)
    z_tpu = (
        align_monthly_stress_z_daily(tpu, pret.index, cfg=soft_cfg)
        if tpu is not None
        else None
    )

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
    epu_scale = (
        risk_scale_from_z(z_epu, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
    )
    out["value_epu_cool"] = _weights_to_returns(
        value_daily.mul(epu_scale, axis=0), pret, cfg=cfg, name="value_epu_cool"
    )

    # Primary: trade value only when EPU z ≤ 0
    gate_epu_lo = (z_epu <= 0.0).astype(float)
    gate_epu_lo = gate_epu_lo.where(z_epu.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["value_low_epu"] = _weights_to_returns(
        value_daily.mul(gate_epu_lo, axis=0), pret, cfg=cfg, name="value_low_epu"
    )

    # Honesty inverse: value only when EPU z ≥ z_high
    gate_epu_hi = (z_epu >= float(cfg.z_high)).astype(float)
    gate_epu_hi = gate_epu_hi.where(z_epu.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["value_high_epu"] = _weights_to_returns(
        value_daily.mul(gate_epu_hi, axis=0), pret, cfg=cfg, name="value_high_epu"
    )

    if tpu_available and z_tpu is not None:
        tpu_scale = (
            risk_scale_from_z(z_tpu, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
        )
        out["value_tpu_cool"] = _weights_to_returns(
            value_daily.mul(tpu_scale, axis=0), pret, cfg=cfg, name="value_tpu_cool"
        )

        gate_tpu_lo = (z_tpu <= 0.0).astype(float)
        gate_tpu_lo = (
            gate_tpu_lo.where(z_tpu.notna(), 0.0).reindex(pret.index).fillna(0.0)
        )
        out["value_low_tpu"] = _weights_to_returns(
            value_daily.mul(gate_tpu_lo, axis=0), pret, cfg=cfg, name="value_low_tpu"
        )

        stack_scale = (epu_scale * tpu_scale).clip(
            lower=float(cfg.cool) ** 2, upper=1.0
        )
        out["value_epu_tpu_stack"] = _weights_to_returns(
            value_daily.mul(stack_scale, axis=0),
            pret,
            cfg=cfg,
            name="value_epu_tpu_stack",
        )

        blend_keys = [
            k
            for k in (
                "value_low_epu",
                "value_epu_cool",
                "value_low_tpu",
                "value_tpu_cool",
            )
            if k in out
        ]
        if len(blend_keys) >= 2:
            blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
            blend.name = "value_epu_tpu_ew"
            out["value_epu_tpu_ew"] = blend
    else:
        out["value_epu_tpu_stack"] = out["value_epu_cool"].copy()
        out["value_epu_tpu_stack"].name = "value_epu_tpu_stack"
        blend = pd.concat(
            [out["value_low_epu"], out["value_epu_cool"]], axis=1
        ).mean(axis=1)
        blend.name = "value_epu_tpu_ew"
        out["value_epu_tpu_ew"] = blend

    return out


__all__ = [
    "PRIMARY",
    "EpuTpuConditionedValueFxConfig",
    "epu_tpu_conditioned_value_factor_returns",
    "load_epu_series",
    "load_tpu_series",
    "align_monthly_stress_z_daily",
]
