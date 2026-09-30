"""VIX/GPR-conditioned BIS REER HML-FX value factors (§81).

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Rogoff (1996) PPP / Taylor REER misalignment / Asness–Moskowitz–Pedersen
  value spirit on real FX — long undervalued (low REER z) / short overvalued
  (``reer_cheap_xs`` from §31 ``bis_reer_fx`` / FRED ``RB*BIS``).
- Menkhoff, Sarno, Schmeling & Schrimpf (2012), *JF*: high global FX /
  equity vol → risk-off / crash states. Proxy with **VIX**.
- Caldara & Iacoviello (2022), *AER* / Liu & Zhang (2024), *JBF*: elevated
  aggregate GPR → risk-off. Free series ``gpr_daily.csv``.

Claim (a priori)
----------------
BIS multilateral REER undervaluation / HML-FX value (§31) earns more when
global FX-vol / geopolitical stress is **low**. Trade REER-value only when
lagged VIX (primary) or GPR stress is subdued; cool or sit out when stress
is elevated. Closes the missing VIX/GPR × REER parallel (VIX/GPR has
soft/carry/mom/PPP §72–§75 but not REER; EPU has REER §77) — **not** a
cooler overlay on locked fx4plus.

Legs
----
- ``reer_low_vix`` (**PRIMARY**): ``reer_cheap_xs`` × binary gate
  (on when lagged VIX z ≤ 0).
- ``reer_vix_cool``: reer_cheap × continuous cool ∈ [cool, 1] from VIX z.
- ``reer_low_gpr``: reer_cheap × binary gate (on when lagged GPR z ≤ 0).
- ``reer_gpr_cool``: reer_cheap × continuous cool from GPR z.
- ``reer_raw``: ungated ``reer_cheap_xs`` honesty baseline (§31).
- ``reer_high_vix``: honesty inverse — reer only when VIX z ≥ z_high.
- ``reer_vix_gpr_stack``: sequential VIX then GPR cool (product of scales).
- ``reer_vix_gpr_ew``: EW of reer_low_vix ⊕ reer_vix_cool ⊕ reer_low_gpr ⊕
  reer_gpr_cool.

PIT
---
VIX / GPR: loader ``bar_lag=1`` + ``signal_lag`` (default 1d) on trailing z
(``z_window=252``, ``min_periods=60``) — same dual lag as §72–§75.
REER: loader ``pub_lag_months`` (default 2) + ``reer_signal_lag`` months
(§31/§77) + 1 trading-day weight lag on expanded monthly weights.
Costs 1.5 bps/side inside factors.

Gate / cool thresholds: fixed z (z_high=1.0 / gate ≤0) — mirror §71–§75.
Literature often uses other cutoffs (GPR cool often 1.5); documented
symmetry, **not** HO-tuned.

Distinct from: raw bis_reer §31, raw ppp, §75 VIX/GPR×PPP value, EPU×REER
§77, EPU soft/carry/mom/value §76–§80, CIP waves, capital-sleeve §53/§70,
combo §8. Explicit: do **not** overlay coolers on locked
``fx4plus_gbpcad_d1_voltarget_0025``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mt5_swing.data.macro_uncertainty import CURRENCY_USD_PAIR
from mt5_swing.strategies.bis_reer_fx import BisReerFxConfig, prepare_reer_scores
from mt5_swing.strategies.country_gpr_fx import (
    currency_weights_to_pair_weights_fx,
    expand_monthly_weights_to_daily,
    portfolio_returns_from_pair_weights,
)
from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z
from mt5_swing.strategies.vix_gpr_conditioned_soft_fx import (
    align_stress_z_daily,
    load_gpr_series,
    load_vix_series,
)
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class VixGprConditionedReerValueFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    # VIX/GPR daily z (§72–§75 mirror — not monthly CIP/EPU 60m)
    signal_lag: int = 1  # trading days after loader bar_lag=1
    z_window: int = 252
    min_periods: int = 60
    z_high: float = 1.0  # cool thresh — §71–§75 symmetry (lit GPR often 1.5)
    z_low: float = 0.0
    cool: float = 0.35
    cost_bps_side: float = 1.5
    # BIS REER §31 / §77 priors
    reer_signal_lag: int = 1  # months after pub-lagged REER
    reer_z_window: int = 60
    reer_min_periods: int = 24
    n_long: int = 2
    n_short: int = 2
    reer_weight_lag_days: int = 1  # expand monthly → daily
    # Documented: literature GPR cool often z≥1.5; this wave keeps 1.0 for symmetry
    gpr_lit_z_high_note: float = 1.5


PRIMARY = "reer_low_vix"


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


def _reer_cheap_daily_weights(
    reer_panel: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: VixGprConditionedReerValueFxConfig,
) -> pd.DataFrame:
    """Monthly REER cheap XS → daily pair weights (PIT lags as §31/§77)."""
    bcfg = BisReerFxConfig(
        signal_lag=cfg.reer_signal_lag,
        n_long=cfg.n_long,
        n_short=cfg.n_short,
        z_window=cfg.reer_z_window,
        min_periods=cfg.reer_min_periods,
        cost_bps_side=0.0,
    )
    scores = prepare_reer_scores(reer_panel, cfg=bcfg)
    if "reer_cheap" not in scores:
        return pd.DataFrame(0.0, index=pair_ret.index, columns=list(pair_ret.columns))
    ccy_w = _rank_sort_weights(
        scores["reer_cheap"], n_long=cfg.n_long, n_short=cfg.n_short
    )
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
        pair_w, pair_ret.index, signal_lag_days=cfg.reer_weight_lag_days
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
    cfg: VixGprConditionedReerValueFxConfig,
    name: str,
) -> pd.Series:
    cols = list(pair_ret.columns)
    for c in cols:
        if c not in w.columns:
            w[c] = 0.0
    w = w.reindex(columns=cols).fillna(0.0)
    r = portfolio_returns_from_pair_weights(w, pair_ret)
    r = apply_costs(r, w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


def _soft_cfg_from_reer(cfg: VixGprConditionedReerValueFxConfig):
    """Reuse §72–§75 align_stress_z_daily config shape."""
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


def vix_gpr_conditioned_reer_value_factor_returns(
    pair_ret: pd.DataFrame,
    reer_panel: pd.DataFrame,
    *,
    vix: pd.Series | None = None,
    gpr: pd.Series | None = None,
    cfg: VixGprConditionedReerValueFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Build VIX/GPR-conditioned BIS REER HML-FX value factor daily returns.

    Distinct from raw bis_reer §31, §75 VIX/GPR×PPP, EPU×REER §77. Explicit:
    do **not** overlay coolers on locked fx4plus.
    """
    cfg = cfg or VixGprConditionedReerValueFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    out: dict[str, pd.Series] = {}

    if pret.empty or pret.shape[1] < 3 or reer_panel is None or reer_panel.empty:
        return out

    if vix is None:
        vix = load_vix_series(bar_lag=1)
    if gpr is None:
        gpr = load_gpr_series(bar_lag=1)

    soft_cfg = _soft_cfg_from_reer(cfg)
    z_vix = align_stress_z_daily(vix, pret.index, cfg=soft_cfg)
    z_gpr = align_stress_z_daily(gpr, pret.index, cfg=soft_cfg)

    value_daily = _reer_cheap_daily_weights(reer_panel, pret, cfg=cfg)

    raw = portfolio_returns_from_pair_weights(value_daily, pret)
    raw = apply_costs(raw, value_daily, bps_side=cfg.cost_bps_side)
    raw.name = "reer_raw"
    out["reer_raw"] = raw

    gcfg = GprRegimeConfig(
        z_high=cfg.z_high,
        z_low=cfg.z_low,
        cool=cfg.cool,
        signal_lag=0,  # z already lagged
        usd_tilt=0.0,
    )
    vix_scale = (
        risk_scale_from_z(z_vix, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
    )
    gpr_scale = (
        risk_scale_from_z(z_gpr, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
    )

    out["reer_vix_cool"] = _weights_to_returns(
        value_daily.mul(vix_scale, axis=0), pret, cfg=cfg, name="reer_vix_cool"
    )
    out["reer_gpr_cool"] = _weights_to_returns(
        value_daily.mul(gpr_scale, axis=0), pret, cfg=cfg, name="reer_gpr_cool"
    )

    # Primary: trade REER-value only when VIX z ≤ 0
    gate_vix_lo = (z_vix <= 0.0).astype(float)
    gate_vix_lo = gate_vix_lo.where(z_vix.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["reer_low_vix"] = _weights_to_returns(
        value_daily.mul(gate_vix_lo, axis=0), pret, cfg=cfg, name="reer_low_vix"
    )

    gate_gpr_lo = (z_gpr <= 0.0).astype(float)
    gate_gpr_lo = gate_gpr_lo.where(z_gpr.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["reer_low_gpr"] = _weights_to_returns(
        value_daily.mul(gate_gpr_lo, axis=0), pret, cfg=cfg, name="reer_low_gpr"
    )

    # Honesty inverse: value only when VIX z ≥ z_high
    gate_vix_hi = (z_vix >= float(cfg.z_high)).astype(float)
    gate_vix_hi = gate_vix_hi.where(z_vix.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["reer_high_vix"] = _weights_to_returns(
        value_daily.mul(gate_vix_hi, axis=0), pret, cfg=cfg, name="reer_high_vix"
    )

    # Sequential VIX then GPR cool (product of scales)
    stack_scale = (vix_scale * gpr_scale).clip(lower=float(cfg.cool) ** 2, upper=1.0)
    out["reer_vix_gpr_stack"] = _weights_to_returns(
        value_daily.mul(stack_scale, axis=0),
        pret,
        cfg=cfg,
        name="reer_vix_gpr_stack",
    )

    blend_keys = [
        k
        for k in (
            "reer_low_vix",
            "reer_vix_cool",
            "reer_low_gpr",
            "reer_gpr_cool",
        )
        if k in out
    ]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "reer_vix_gpr_ew"
        out["reer_vix_gpr_ew"] = blend

    return out


__all__ = [
    "PRIMARY",
    "VixGprConditionedReerValueFxConfig",
    "vix_gpr_conditioned_reer_value_factor_returns",
    "align_stress_z_daily",
    "load_vix_series",
    "load_gpr_series",
]
