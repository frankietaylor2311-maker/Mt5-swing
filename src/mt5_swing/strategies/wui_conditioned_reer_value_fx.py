"""WUI-conditioned BIS REER HML-FX value factors (§89).

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Rogoff (1996) PPP / Taylor REER misalignment / Asness–Moskowitz–Pedersen
  value spirit on real FX — long undervalued (low REER z) / short overvalued
  (``reer_cheap_xs`` from §31 ``bis_reer_fx`` / FRED ``RB*BIS``).
- Ahir, Bloom & Furceri World Uncertainty Index (WUI): EIU-text country
  uncertainty (quarterly → monthly). We use **US WUI** (FRED WUIUSA) as the
  free aggregate stress series via ``mt5_swing.data.fred_wui.load_us_wui``
  (pub_lag_months=4 already applied in the loader).

Claim (a priori)
----------------
BIS multilateral REER undervaluation / HML-FX value (§31) earns more when
US / world-uncertainty stress is **low**. Trade REER-value only when lagged
US WUI z is subdued; cool or sit out when WUI stress is elevated.
Single-stress design (like CIP×REER §84), not dual EPU+TPU.
Closes the WUI triad after §85 soft / §86 carry / §87 mom / §88 PPP.

Legs
----
- ``reer_low_wui`` (**PRIMARY**): REER cheap XS only when lagged
  US WUI z ≤ 0 (flat in elevated stress).
- ``reer_wui_cool``: reer × risk_scale ∈ [cool, 1] from WUI z
  (cool when z ≥ z_high).
- ``reer_raw``: always-on ``reer_cheap_xs`` honesty baseline (§31).
- ``reer_high_wui``: honesty inverse — reer only when WUI z ≥ z_high.
- ``us_wui_haven_usd``: long USD when lagged WUI z ≥ z_high
  (§40 / CIP-stress haven companion pattern / §85–§88).
- ``reer_wui_stack``: reer_low_wui × cool scale (sequential gate then cool —
  WUI soft/carry/mom/value §85–§88 mirror, not CIP REER EW cool⊕haven).
- ``reer_wui_ew``: EW of reer_low_wui ⊕ reer_wui_cool ⊕ us_wui_haven_usd.
- ``reer_wui_regime``: reer_low_wui + us_wui_haven_usd (exclusive regimes:
  reer when z≤0, haven when z≥z_high).

PIT
---
``load_us_wui`` applies ``pub_lag_months=4`` (quarterly EIU). This module adds
``signal_lag_months`` (default 1) on **monthly** trailing z (z_window=60m,
min_periods=24 — mirror CIP §71 / EPU §76 / WUI soft §85 / WUI carry §86 /
WUI mom §87 / WUI value §88, **not** 252d daily) + ``weight_lag_days=1`` after
month-end ffill to the trading calendar. REER: loader ``pub_lag_months``
(default 2) + ``reer_signal_lag`` months (§31/§77/§81/§84) + 1 trading-day
weight lag on expanded monthly weights.
``portfolio_returns_from_pair_weights`` applies an extra weight lag.
Costs 1.5 bps/side.

Gate / cool thresholds: fixed z (z_high=1.0 / gate ≤0) — mirror §71–§88.
Literature often uses other cutoffs; documented symmetry, **not** HO-tuned.

Distinct from: raw WUI XS §40, WUI soft §85, WUI carry §86, WUI mom §87,
WUI PPP value §88, raw bis_reer §31, CIP×REER §84, VIX/GPR×REER §81,
EPU×REER §77, CIP/VIX/EPU×PPP, soft–carry–mom–value stacks,
capital-sleeve §53/§70, combo §8.
Explicit: do **not** overlay coolers on locked ``fx4plus_gbpcad_d1_voltarget_0025``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mt5_swing.data.fred_wui import DEFAULT_PUB_LAG_MONTHS, load_us_wui
from mt5_swing.data.macro_uncertainty import CURRENCY_USD_PAIR
from mt5_swing.strategies.bis_reer_fx import BisReerFxConfig, prepare_reer_scores
from mt5_swing.strategies.country_gpr_fx import (
    currency_weights_to_pair_weights_fx,
    expand_monthly_weights_to_daily,
    portfolio_returns_from_pair_weights,
)
from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z
from mt5_swing.strategies.wui_conditioned_soft_fx import (
    WuiConditionedSoftFxConfig,
    align_monthly_stress_z_daily,
    usd_tilt_from_wui_z,
)
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class WuiConditionedReerValueFxConfig:
    """Fixed research priors — do not grid / holdout-tune.

    WUI stress fields mirror §85–§88 (monthly z_window=60). REER fields
    mirror §31/§77/§81/§84 BIS construction.
    """

    # WUI monthly z (§85–§88 mirror)
    signal_lag_months: int = 1  # months after pub-lagged monthly index known
    weight_lag_days: int = 1  # trading days after month-end ffill
    z_window: int = 60  # months (~5y) — mirror CIP §71 / EPU §76 / WUI §85–§88
    min_periods: int = 24
    z_high: float = 1.0  # cool / haven thresh — §71–§88 symmetry
    z_low: float = 0.0
    cool: float = 0.35
    usd_tilt: float = 0.5  # gross |w| when haven tilt is on
    cost_bps_side: float = 1.5
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS  # informational (loader default 4)
    # Documented: lit often uses other cutoffs; this wave keeps 1.0 for symmetry
    lit_z_high_note: float = 1.5
    # BIS REER §31 / §77 / §81 / §84 priors — not HO-tuned
    reer_signal_lag: int = 1  # months after pub-lagged REER
    reer_z_window: int = 60
    reer_min_periods: int = 24
    n_long: int = 2
    n_short: int = 2
    reer_weight_lag_days: int = 1  # expand monthly → daily


PRIMARY = "reer_low_wui"


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _soft_cfg_from_reer(
    cfg: WuiConditionedReerValueFxConfig,
) -> WuiConditionedSoftFxConfig:
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
    cfg: WuiConditionedReerValueFxConfig,
) -> pd.DataFrame:
    """Monthly REER cheap XS → daily pair weights (PIT lags as §31/§77/§81/§84)."""
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
    cfg: WuiConditionedReerValueFxConfig,
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


def _haven_returns(
    z_daily: pd.Series,
    pair_ret: pd.DataFrame,
    *,
    cfg: WuiConditionedReerValueFxConfig,
) -> pd.Series:
    soft_cfg = _soft_cfg_from_reer(cfg)
    cols = list(pair_ret.columns)
    w = usd_tilt_from_wui_z(z_daily, cols, cfg=soft_cfg)
    return _weights_to_returns(w, pair_ret, cfg=cfg, name="us_wui_haven_usd")


def wui_conditioned_reer_value_factor_returns(
    pair_ret: pd.DataFrame,
    reer_panel: pd.DataFrame,
    *,
    us_wui: pd.Series | None = None,
    cfg: WuiConditionedReerValueFxConfig | None = None,
    include_haven: bool = True,
) -> dict[str, pd.Series]:
    """Build WUI-conditioned BIS REER HML-FX value factor daily returns.

    Distinct from §84 (CIP×REER), §81 (VIX/GPR×REER), §77 (EPU/TPU×REER),
    §88 (WUI×PPP), §85–§87 (WUI soft/carry/mom). Explicit: do **not**
    overlay coolers on locked fx4plus.
    """
    cfg = cfg or WuiConditionedReerValueFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    out: dict[str, pd.Series] = {}

    if pret.empty or pret.shape[1] < 3 or reer_panel is None or reer_panel.empty:
        return out

    if us_wui is None:
        us_wui = load_us_wui_series(pub_lag_months=cfg.pub_lag_months)

    soft_cfg = _soft_cfg_from_reer(cfg)
    z_daily = align_monthly_stress_z_daily(us_wui, pret.index, cfg=soft_cfg)

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
    cool_scale = (
        risk_scale_from_z(z_daily, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
    )
    out["reer_wui_cool"] = _weights_to_returns(
        value_daily.mul(cool_scale, axis=0), pret, cfg=cfg, name="reer_wui_cool"
    )

    # Primary: trade REER-value only when US WUI z ≤ 0
    gate_lo = (z_daily <= 0.0).astype(float)
    gate_lo = gate_lo.where(z_daily.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["reer_low_wui"] = _weights_to_returns(
        value_daily.mul(gate_lo, axis=0), pret, cfg=cfg, name="reer_low_wui"
    )

    # Honesty inverse: value only when WUI z ≥ z_high
    gate_hi = (z_daily >= float(cfg.z_high)).astype(float)
    gate_hi = gate_hi.where(z_daily.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["reer_high_wui"] = _weights_to_returns(
        value_daily.mul(gate_hi, axis=0), pret, cfg=cfg, name="reer_high_wui"
    )

    if include_haven:
        out["us_wui_haven_usd"] = _haven_returns(z_daily, pret, cfg=cfg)

    # Stack: sequential gate then cool (reer_low weights × cool scale) — §85–§88 mirror
    stack_scale = (gate_lo * cool_scale).clip(lower=0.0, upper=1.0)
    out["reer_wui_stack"] = _weights_to_returns(
        value_daily.mul(stack_scale, axis=0), pret, cfg=cfg, name="reer_wui_stack"
    )

    # EW of primary + cool + haven (§68 / §85–§88 mirror)
    blend_keys = [
        k
        for k in ("reer_low_wui", "reer_wui_cool", "us_wui_haven_usd")
        if k in out
    ]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "reer_wui_ew"
        out["reer_wui_ew"] = blend

    # Regime switch: reer when low-WUI + haven when high-WUI (exclusive)
    if "reer_low_wui" in out and "us_wui_haven_usd" in out:
        regime = out["reer_low_wui"].fillna(0.0) + out["us_wui_haven_usd"].fillna(0.0)
        both_nan = out["reer_low_wui"].isna() & out["us_wui_haven_usd"].isna()
        regime = regime.where(~both_nan)
        regime.name = "reer_wui_regime"
        out["reer_wui_regime"] = regime

    return out


__all__ = [
    "PRIMARY",
    "WuiConditionedReerValueFxConfig",
    "align_monthly_stress_z_daily",
    "load_us_wui_series",
    "usd_tilt_from_wui_z",
    "wui_conditioned_reer_value_factor_returns",
]
