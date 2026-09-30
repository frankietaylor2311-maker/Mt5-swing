"""KCFSI-conditioned BIS REER HML-FX value factors (§109).

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Rogoff (1996) PPP / Taylor REER misalignment / Asness–Moskowitz–Pedersen
  value spirit on real FX — long undervalued (low REER z) / short overvalued
  (``reer_cheap_xs`` from §31 ``bis_reer_fx`` / FRED ``RB*BIS``).
- Brunnermeier, Nagel & Pedersen (2008), "Carry Trades and Currency Crashes":
  funding-liquidity spirals / tight financial conditions coincide with FX
  risk-off. Kansas City Fed **KCFSI** (Financial Stress Index; FRED ``KCFSI``;
  monthly; 0 ≈ average; >0 elevated stress; Hakkio & Keeton 2009) via
  ``mt5_swing.data.fred_funding_liquidity.load_kcfsi_series`` /
  ``load_us_kcfsi_series`` (``pub_lag_months=1`` already applied in the loader) —
  same stress path as KCFSI soft §105 / KCFSI carry §106 / KCFSI mom §107 /
  KCFSI PPP §108.

Claim (a priori)
----------------
BIS multilateral REER undervaluation / HML-FX value (§31) earns more when
US financial-stress is **low** (loose KCFSI). Trade REER-value only when
lagged KCFSI z is subdued; cool or sit out when Kansas City Fed stress is
elevated. Single-stress design (like CIP×REER §84 / WUI×REER §89 /
NFCI×REER §94 / ANFCI×REER §99 / STLFSI×REER §104 / KCFSI×PPP §108), not dual EPU+TPU.
KCFSI is a **distinct** conditioning series from NFCI/ANFCI (§90–§99) and
STLFSI4 (§100–§104) — Kansas City Fed construction / panel, not Chicago or
St. Louis Fed. Closes the KCFSI soft–carry–mom–value–REER stack after
§105 soft / §106 carry / §107 mom / §108 PPP.

Legs
----
- ``reer_low_kcfsi`` (**PRIMARY**): REER cheap XS only when lagged
  US KCFSI z ≤ 0 (flat in elevated stress).
- ``reer_kcfsi_cool``: reer × risk_scale ∈ [cool, 1] from KCFSI z
  (cool when z ≥ z_high).
- ``reer_raw``: always-on ``reer_cheap_xs`` honesty baseline (§31).
- ``reer_high_kcfsi``: honesty inverse — reer only when KCFSI z ≥ z_high.
- ``us_kcfsi_haven_usd``: long USD when lagged KCFSI z ≥ z_high
  (§20 / CIP-stress haven companion pattern / KCFSI soft–carry–mom–PPP §105–§108).
- ``reer_kcfsi_stack``: reer_low_kcfsi × cool scale (sequential gate then cool —
  KCFSI soft/carry/mom/PPP §105–§108 / STLFSI REER §104 mirror).
- ``reer_kcfsi_ew``: EW of reer_low_kcfsi ⊕ reer_kcfsi_cool ⊕ us_kcfsi_haven_usd.
- ``reer_kcfsi_regime``: reer_low_kcfsi + us_kcfsi_haven_usd (exclusive regimes:
  reer when z≤0, haven when z≥z_high).

PIT
---
``load_kcfsi_series`` / ``load_us_kcfsi_series`` apply ``pub_lag_months=1``
(monthly KC Fed release; like EPU USEPUINDXM). This module adds
``signal_lag_months`` (default 1) on **monthly** trailing z (z_window=60m,
min_periods=24 — mirror CIP §71 / EPU §76 / WUI §85 / NFCI soft §90 /
NFCI REER §94 / ANFCI soft §95 / ANFCI REER §99 / STLFSI soft §100 /
STLFSI REER §104 / KCFSI soft §105 / KCFSI carry §106 / KCFSI mom §107 /
KCFSI PPP §108, **not** 252d daily) + ``weight_lag_days=1`` after month-end
ffill to the trading calendar via ``align_monthly_stress_z_daily`` from
KCFSI soft §105. REER: loader ``pub_lag_months`` (default 2) +
``reer_signal_lag`` months (§31/§77/§81/§84/§89/§94/§99/§104) + 1 trading-day
weight lag on expanded monthly weights.
``portfolio_returns_from_pair_weights`` applies an extra weight lag.
Costs 1.5 bps/side.

Gate / cool thresholds: fixed z (z_high=1.0 / gate ≤0) — mirror §71–§108.
Literature often uses other cutoffs; documented symmetry, **not** HO-tuned.

Distinct from: funding_liquidity_fx §20, raw bis_reer §31,
NFCI soft–carry–mom–value–REER §90–§94, ANFCI soft–carry–mom–value–REER §95–§99,
STLFSI soft–carry–mom–value–REER §100–§104, KCFSI soft §105, KCFSI carry §106,
KCFSI mom §107, KCFSI PPP §108, CIP×REER §84, VIX/GPR×REER §81, EPU×REER §77,
WUI×REER §89, CIP/VIX/EPU/WUI × soft/carry/mom/PPP, soft–carry–mom–value–REER
stacks, capital-sleeve §53/§70, combo §8.
Explicit: do **not** overlay coolers on locked ``fx4plus_gbpcad_d1_voltarget_0025``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mt5_swing.data.fred_funding_liquidity import DEFAULT_MONTHLY_PUB_LAG_MONTHS
from mt5_swing.data.macro_uncertainty import CURRENCY_USD_PAIR
from mt5_swing.strategies.bis_reer_fx import BisReerFxConfig, prepare_reer_scores
from mt5_swing.strategies.country_gpr_fx import (
    currency_weights_to_pair_weights_fx,
    expand_monthly_weights_to_daily,
    portfolio_returns_from_pair_weights,
)
from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z
from mt5_swing.strategies.kcfsi_conditioned_soft_fx import (
    KcfsiConditionedSoftFxConfig,
    align_monthly_stress_z_daily,
    load_us_kcfsi_series,
    usd_tilt_from_kcfsi_z,
)
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class KcfsiConditionedReerValueFxConfig:
    """Fixed research priors — do not grid / holdout-tune.

    KCFSI stress fields mirror §105–§108 (monthly z_window=60). REER fields
    mirror §31/§77/§81/§84/§89/§94/§99/§104 BIS construction.
    """

    # KCFSI monthly z (§105–§108 mirror)
    signal_lag_months: int = 1  # months after pub-lagged monthly KCFSI known
    weight_lag_days: int = 1  # trading days after month-end ffill
    z_window: int = 60  # months (~5y) — mirror CIP §71 / EPU §76 / WUI / NFCI / ANFCI / STLFSI / KCFSI
    min_periods: int = 24
    z_high: float = 1.0  # cool / haven thresh — §71–§108 symmetry
    z_low: float = 0.0
    cool: float = 0.35
    usd_tilt: float = 0.5  # gross |w| when haven tilt is on
    cost_bps_side: float = 1.5
    pub_lag_months: int = DEFAULT_MONTHLY_PUB_LAG_MONTHS  # informational (loader default 1)
    # Documented: lit often uses other cutoffs; this wave keeps 1.0 for symmetry
    lit_z_high_note: float = 1.5
    # BIS REER §31 / §77 / §81 / §84 / §89 / §94 / §99 / §104 priors — not HO-tuned
    reer_signal_lag: int = 1  # months after pub-lagged REER
    reer_z_window: int = 60
    reer_min_periods: int = 24
    n_long: int = 2
    n_short: int = 2
    reer_weight_lag_days: int = 1  # expand monthly → daily


PRIMARY = "reer_low_kcfsi"


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _soft_cfg_from_reer(
    cfg: KcfsiConditionedReerValueFxConfig,
) -> KcfsiConditionedSoftFxConfig:
    """Reuse §105 align_monthly_stress_z_daily / usd_tilt config shape."""
    return KcfsiConditionedSoftFxConfig(
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
    cfg: KcfsiConditionedReerValueFxConfig,
) -> pd.DataFrame:
    """Monthly REER cheap XS → daily pair weights (PIT lags as §31/§77/§81/§84/§89/§94/§99/§104)."""
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
    cfg: KcfsiConditionedReerValueFxConfig,
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
    cfg: KcfsiConditionedReerValueFxConfig,
) -> pd.Series:
    soft_cfg = _soft_cfg_from_reer(cfg)
    cols = list(pair_ret.columns)
    w = usd_tilt_from_kcfsi_z(z_daily, cols, cfg=soft_cfg)
    return _weights_to_returns(w, pair_ret, cfg=cfg, name="us_kcfsi_haven_usd")


def kcfsi_conditioned_reer_value_factor_returns(
    pair_ret: pd.DataFrame,
    reer_panel: pd.DataFrame,
    *,
    us_kcfsi: pd.Series | None = None,
    cfg: KcfsiConditionedReerValueFxConfig | None = None,
    include_haven: bool = True,
) -> dict[str, pd.Series]:
    """Build KCFSI-conditioned BIS REER HML-FX value factor daily returns.

    Distinct from §84 (CIP×REER), §81 (VIX/GPR×REER), §77 (EPU/TPU×REER),
    §89 (WUI×REER), §90–§94 (NFCI), §95–§99 (ANFCI), §100–§104 (STLFSI),
    §105–§108 (KCFSI soft/carry/mom/PPP), §20 (funding_liq). Explicit: do **not** overlay coolers on locked fx4plus.
    """
    cfg = cfg or KcfsiConditionedReerValueFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    out: dict[str, pd.Series] = {}

    if pret.empty or pret.shape[1] < 3 or reer_panel is None or reer_panel.empty:
        return out

    if us_kcfsi is None:
        us_kcfsi = load_us_kcfsi_series(pub_lag_months=cfg.pub_lag_months)

    soft_cfg = _soft_cfg_from_reer(cfg)
    z_daily = align_monthly_stress_z_daily(us_kcfsi, pret.index, cfg=soft_cfg)

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
    out["reer_kcfsi_cool"] = _weights_to_returns(
        value_daily.mul(cool_scale, axis=0), pret, cfg=cfg, name="reer_kcfsi_cool"
    )

    # Primary: trade REER-value only when US KCFSI z ≤ 0
    gate_lo = (z_daily <= 0.0).astype(float)
    gate_lo = gate_lo.where(z_daily.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["reer_low_kcfsi"] = _weights_to_returns(
        value_daily.mul(gate_lo, axis=0), pret, cfg=cfg, name="reer_low_kcfsi"
    )

    # Honesty inverse: value only when KCFSI z ≥ z_high
    gate_hi = (z_daily >= float(cfg.z_high)).astype(float)
    gate_hi = gate_hi.where(z_daily.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["reer_high_kcfsi"] = _weights_to_returns(
        value_daily.mul(gate_hi, axis=0), pret, cfg=cfg, name="reer_high_kcfsi"
    )

    if include_haven:
        out["us_kcfsi_haven_usd"] = _haven_returns(z_daily, pret, cfg=cfg)

    # Stack: sequential gate then cool (reer_low weights × cool scale) — §105–§108 / §104 mirror
    stack_scale = (gate_lo * cool_scale).clip(lower=0.0, upper=1.0)
    out["reer_kcfsi_stack"] = _weights_to_returns(
        value_daily.mul(stack_scale, axis=0), pret, cfg=cfg, name="reer_kcfsi_stack"
    )

    # EW of primary + cool + haven (§68 / §89 / §94 / §99 / §104 / §105–§108 mirror)
    blend_keys = [
        k
        for k in ("reer_low_kcfsi", "reer_kcfsi_cool", "us_kcfsi_haven_usd")
        if k in out
    ]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "reer_kcfsi_ew"
        out["reer_kcfsi_ew"] = blend

    # Regime switch: reer when low-KCFSI + haven when high-KCFSI (exclusive)
    if "reer_low_kcfsi" in out and "us_kcfsi_haven_usd" in out:
        regime = out["reer_low_kcfsi"].fillna(0.0) + out["us_kcfsi_haven_usd"].fillna(0.0)
        both_nan = out["reer_low_kcfsi"].isna() & out["us_kcfsi_haven_usd"].isna()
        regime = regime.where(~both_nan)
        regime.name = "reer_kcfsi_regime"
        out["reer_kcfsi_regime"] = regime

    return out


__all__ = [
    "PRIMARY",
    "KcfsiConditionedReerValueFxConfig",
    "align_monthly_stress_z_daily",
    "load_us_kcfsi_series",
    "usd_tilt_from_kcfsi_z",
    "kcfsi_conditioned_reer_value_factor_returns",
]
