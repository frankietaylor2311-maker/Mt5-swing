"""KCFSI-conditioned Rogoff PPP / real-FX value factors (§108).

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Rogoff (1996) PPP puzzle / real exchange-rate mean reversion —
  long undervalued / short overvalued cross-sectional real-FX value
  (``ppp_real_fx``: trailing 60m z of q = S·CPI_US/CPI_f; honesty 120m).
- Brunnermeier, Nagel & Pedersen (2008), "Carry Trades and Currency Crashes":
  funding-liquidity spirals / tight financial conditions coincide with FX
  risk-off. Kansas City Fed **KCFSI** (Financial Stress Index; FRED ``KCFSI``;
  monthly; 0 ≈ average; >0 elevated stress; Hakkio & Keeton 2009) via
  ``mt5_swing.data.fred_funding_liquidity.load_kcfsi_series`` /
  ``load_us_kcfsi_series`` (``pub_lag_months=1`` already applied in the loader) —
  same stress path as KCFSI soft §105 / KCFSI carry §106 / KCFSI mom §107.

Claim (a priori)
----------------
Rogoff PPP / real-FX cross-sectional value earns more when US financial-stress
is **low** (loose KCFSI). Trade value only when lagged KCFSI z is subdued;
cool or sit out when Kansas City Fed stress is elevated. Single-stress design
(like CIP×PPP §83 / WUI×PPP §88 / NFCI×PPP §93 / ANFCI×PPP §98 /
STLFSI×PPP §103 / KCFSI×mom §107), not dual EPU+TPU. KCFSI is a **distinct**
conditioning series from NFCI/ANFCI (§90–§99) and STLFSI4 (§100–§104) —
Kansas City Fed construction / panel, not Chicago or St. Louis Fed.
Continues the KCFSI soft–carry–mom–value path after §105–§107.

Legs
----
- ``value_low_kcfsi`` (**PRIMARY**): PPP 60m XS value only when lagged
  US KCFSI z ≤ 0 (flat in elevated stress).
- ``value_kcfsi_cool``: value × risk_scale ∈ [cool, 1] from KCFSI z
  (cool when z ≥ z_high).
- ``value_raw``: always-on PPP 60m XS (honesty baseline).
- ``value_high_kcfsi``: honesty inverse — value only when KCFSI z ≥ z_high.
- ``us_kcfsi_haven_usd``: long USD when lagged KCFSI z ≥ z_high
  (§20 / CIP-stress haven companion pattern / KCFSI soft §105 / KCFSI carry §106 / KCFSI mom §107).
- ``value_kcfsi_stack``: value_low_kcfsi × cool scale (sequential gate then cool —
  KCFSI mom §107 / KCFSI carry §106 / STLFSI value §103 / ANFCI value §98 / NFCI value §93 mirror).
- ``value_kcfsi_ew``: EW of value_low_kcfsi ⊕ value_kcfsi_cool ⊕ us_kcfsi_haven_usd.
- ``value_kcfsi_regime``: value_low_kcfsi + us_kcfsi_haven_usd (exclusive regimes:
  value when z≤0, haven when z≥z_high).

PIT
---
``load_kcfsi_series`` / ``load_us_kcfsi_series`` apply ``pub_lag_months=1``
(monthly KC Fed release; like EPU USEPUINDXM). This module adds
``signal_lag_months`` (default 1) on **monthly** trailing z (z_window=60m,
min_periods=24 — mirror CIP §71 / EPU §76 / WUI §85 / NFCI soft §90 /
NFCI value §93 / ANFCI soft §95 / ANFCI value §98 / STLFSI soft §100 /
STLFSI value §103 / KCFSI soft §105 / KCFSI carry §106 / KCFSI mom §107,
**not** 252d daily) + ``weight_lag_days=1`` after month-end ffill to the
trading calendar via ``align_monthly_stress_z_daily`` from KCFSI soft §105.
PPP: CPI publication lag (loader) + ``ppp_signal_lag`` months on z (default 1)
+ 1 trading-day lag on expanded monthly weights.
``portfolio_returns_from_weights`` applies an extra weight lag. Costs 1.5 bps/side.

Gate / cool thresholds: fixed z (z_high=1.0 / gate ≤0) — mirror §71–§107.
Literature often uses other cutoffs; documented symmetry, **not** HO-tuned.

Distinct from: funding_liquidity_fx §20, NFCI value §93, ANFCI value §98,
STLFSI value §103, KCFSI soft §105, KCFSI carry §106, KCFSI mom §107,
WUI×PPP §88, CIP×PPP §83, VIX/GPR×PPP §75, EPU×PPP §80, raw PPP, BIS REER §31,
soft–carry–mom–value–REER stacks, capital-sleeve §53/§70, combo §8.
Explicit: do **not** overlay coolers on locked ``fx4plus_gbpcad_d1_voltarget_0025``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mt5_swing.data.fred_funding_liquidity import DEFAULT_MONTHLY_PUB_LAG_MONTHS
from mt5_swing.data.macro_uncertainty import CURRENCY_USD_PAIR
from mt5_swing.strategies.carry_rank import portfolio_returns_from_weights
from mt5_swing.strategies.country_gpr_fx import (
    currency_weights_to_pair_weights_fx,
    expand_monthly_weights_to_daily,
)
from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z
from mt5_swing.strategies.kcfsi_conditioned_soft_fx import (
    KcfsiConditionedSoftFxConfig,
    align_monthly_stress_z_daily,
    load_us_kcfsi_series,
    usd_tilt_from_kcfsi_z,
)
from mt5_swing.strategies.ppp_real_fx import (
    PppRealFxConfig,
    monthly_nominal_fx,
    ppp_value_scores,
    real_fx_panel,
)
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class KcfsiConditionedValueFxConfig:
    """Fixed research priors — do not grid / holdout-tune.

    KCFSI stress fields mirror §105–§107 (monthly z_window=60). PPP fields
    mirror §75/§80/§83/§88/§93/§98/§103 Rogoff construction.
    """

    # KCFSI monthly z (§105–§107 mirror)
    signal_lag_months: int = 1  # months after pub-lagged monthly KCFSI known
    weight_lag_days: int = 1  # trading days after month-end ffill
    z_window: int = 60  # months (~5y) — mirror CIP §71 / EPU §76 / WUI / NFCI / ANFCI / STLFSI / KCFSI
    min_periods: int = 24
    z_high: float = 1.0  # cool / haven thresh — §71–§107 symmetry
    z_low: float = 0.0
    cool: float = 0.35
    usd_tilt: float = 0.5  # gross |w| when haven tilt is on
    cost_bps_side: float = 1.5
    pub_lag_months: int = DEFAULT_MONTHLY_PUB_LAG_MONTHS  # informational (loader default 1)
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


PRIMARY = "value_low_kcfsi"


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _soft_cfg_from_value(cfg: KcfsiConditionedValueFxConfig) -> KcfsiConditionedSoftFxConfig:
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
    cfg: KcfsiConditionedValueFxConfig,
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
    cfg: KcfsiConditionedValueFxConfig,
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
    cfg: KcfsiConditionedValueFxConfig,
) -> pd.Series:
    soft_cfg = _soft_cfg_from_value(cfg)
    cols = list(pair_ret.columns)
    w = usd_tilt_from_kcfsi_z(z_daily, cols, cfg=soft_cfg)
    return _weights_to_returns(w, pair_ret, cfg=cfg, name="us_kcfsi_haven_usd")


def kcfsi_conditioned_value_factor_returns(
    pair_close: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cpi_levels: pd.DataFrame,
    *,
    us_kcfsi: pd.Series | None = None,
    cfg: KcfsiConditionedValueFxConfig | None = None,
    include_haven: bool = True,
) -> dict[str, pd.Series]:
    """Build KCFSI-conditioned Rogoff PPP real-FX value factor daily returns.

    Distinct from §83 (CIP×PPP), §75 (VIX/GPR×PPP), §80 (EPU/TPU×PPP),
    §88 (WUI×PPP), §93 (NFCI×PPP), §98 (ANFCI×PPP), §103 (STLFSI×PPP),
    §105 (KCFSI soft), §106 (KCFSI carry), §107 (KCFSI mom), §20 (funding_liq).
    Explicit: do **not** overlay coolers on locked fx4plus.
    """
    cfg = cfg or KcfsiConditionedValueFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    pclose = pair_close.copy()
    pclose.index = _ensure_utc(pd.DatetimeIndex(pclose.index))
    out: dict[str, pd.Series] = {}

    if pret.empty or pret.shape[1] < 3 or cpi_levels is None or cpi_levels.empty:
        return out

    if us_kcfsi is None:
        us_kcfsi = load_us_kcfsi_series(pub_lag_months=cfg.pub_lag_months)

    soft_cfg = _soft_cfg_from_value(cfg)
    z_daily = align_monthly_stress_z_daily(us_kcfsi, pret.index, cfg=soft_cfg)

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
    out["value_kcfsi_cool"] = _weights_to_returns(
        value_daily.mul(cool_scale, axis=0), pret, cfg=cfg, name="value_kcfsi_cool"
    )

    # Primary: trade value only when US KCFSI z ≤ 0
    gate_lo = (z_daily <= 0.0).astype(float)
    gate_lo = gate_lo.where(z_daily.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["value_low_kcfsi"] = _weights_to_returns(
        value_daily.mul(gate_lo, axis=0), pret, cfg=cfg, name="value_low_kcfsi"
    )

    # Honesty inverse: value only when KCFSI z ≥ z_high
    gate_hi = (z_daily >= float(cfg.z_high)).astype(float)
    gate_hi = gate_hi.where(z_daily.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["value_high_kcfsi"] = _weights_to_returns(
        value_daily.mul(gate_hi, axis=0), pret, cfg=cfg, name="value_high_kcfsi"
    )

    if include_haven:
        out["us_kcfsi_haven_usd"] = _haven_returns(z_daily, pret, cfg=cfg)

    # Stack: sequential gate then cool (value_low weights × cool scale) — §93/§98/§103/§106/§107 mirror
    stack_scale = (gate_lo * cool_scale).clip(lower=0.0, upper=1.0)
    out["value_kcfsi_stack"] = _weights_to_returns(
        value_daily.mul(stack_scale, axis=0), pret, cfg=cfg, name="value_kcfsi_stack"
    )

    # EW of primary + cool + haven (§68 / §88 / §93 / §98 / §103 / §105 / §106 / §107 mirror)
    blend_keys = [
        k
        for k in ("value_low_kcfsi", "value_kcfsi_cool", "us_kcfsi_haven_usd")
        if k in out
    ]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "value_kcfsi_ew"
        out["value_kcfsi_ew"] = blend

    # Regime switch: value when low-KCFSI + haven when high-KCFSI (exclusive)
    if "value_low_kcfsi" in out and "us_kcfsi_haven_usd" in out:
        regime = out["value_low_kcfsi"].fillna(0.0) + out["us_kcfsi_haven_usd"].fillna(0.0)
        both_nan = out["value_low_kcfsi"].isna() & out["us_kcfsi_haven_usd"].isna()
        regime = regime.where(~both_nan)
        regime.name = "value_kcfsi_regime"
        out["value_kcfsi_regime"] = regime

    return out


__all__ = [
    "PRIMARY",
    "KcfsiConditionedValueFxConfig",
    "align_monthly_stress_z_daily",
    "load_us_kcfsi_series",
    "usd_tilt_from_kcfsi_z",
    "kcfsi_conditioned_value_factor_returns",
]
