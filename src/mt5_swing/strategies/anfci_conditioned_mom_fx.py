"""ANFCI-conditioned Menkhoff FX momentum factors (§97).

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Menkhoff, Sarno, Schmeling & Schrimpf (2012), *JFE*: Currency Momentum Strategies
  — cross-sectional FX momentum (formation ~3m, skip ~1m, long winners / short losers).
- Brunnermeier, Nagel & Pedersen (2008), "Carry Trades and Currency Crashes":
  funding-liquidity spirals / tight financial conditions coincide with FX
  risk-off. Chicago Fed **ANFCI** (Adjusted National Financial Conditions
  Index) removes the Fed-funds-rate / business-cycle component from NFCI
  (FRED ``ANFCI``; weekly; 0 ≈ average; >0 tighter) via
  ``mt5_swing.data.fred_funding_liquidity.load_nfci_series`` with
  ``series_id="ANFCI"`` (``pub_lag_days=7`` already applied in the loader) —
  same stress path as ANFCI soft §95 / ANFCI carry §96.

Claim (a priori)
----------------
Menkhoff cross-sectional FX momentum earns more when US *adjusted*
financial-conditions stress is **low** (loose ANFCI). Trade mom only when
lagged ANFCI z is subdued; cool or sit out when ANFCI stress is elevated.
Single-stress design (like CIP×mom §82 / WUI×mom §87 / NFCI×mom §92 /
ANFCI×carry §96), not dual EPU+TPU. ANFCI is a **distinct** conditioning
series from NFCI (§90–§94) — residual financial stress after business-cycle /
policy rate.

Legs
----
- ``mom_low_anfci`` (**PRIMARY**): scholarly FX momentum only when lagged
  US ANFCI z ≤ 0 (flat in elevated stress).
- ``mom_anfci_cool``: mom × risk_scale ∈ [cool, 1] from ANFCI z
  (cool when z ≥ z_high).
- ``mom_raw``: always-on Menkhoff momentum (honesty baseline).
- ``mom_high_anfci``: honesty inverse — mom only when ANFCI z ≥ z_high.
- ``us_anfci_haven_usd``: long USD when lagged ANFCI z ≥ z_high
  (§20 / CIP-stress haven companion pattern / ANFCI soft §95 / ANFCI carry §96).
- ``mom_anfci_stack``: mom_low_anfci × cool scale (sequential gate then cool).
- ``mom_anfci_ew``: EW of mom_low_anfci ⊕ mom_anfci_cool ⊕ us_anfci_haven_usd.
- ``mom_anfci_regime``: mom_low_anfci + us_anfci_haven_usd (exclusive regimes:
  mom when z≤0, haven when z≥z_high).

PIT
---
``load_nfci_series(..., series_id="ANFCI")`` applies ``pub_lag_days=7``
(weekly Chicago Fed release). This module collapses pub-lagged weekly ANFCI
to **month-end** (last obs/month) via ``align_monthly_stress_z_daily`` from
ANFCI soft §95, then adds ``signal_lag_months`` (default 1) on **monthly**
trailing z (z_window=60m, min_periods=24 — mirror CIP §71 / EPU §76 / WUI §85 /
NFCI soft §90 / NFCI mom §92 / ANFCI soft §95 / ANFCI carry §96, **not** 252d
daily) + ``weight_lag_days=1`` after month-end ffill to the trading calendar.
Momentum: ``FxMomentumConfig`` defaults (formation=63d, skip=21d,
n_long=n_short=2, signal_lag=1) via ``expand_weights_to_daily`` — identical
to NFCI×mom §92. ``portfolio_returns_from_weights`` applies an extra weight
lag. Costs 1.5 bps/side.

Gate / cool thresholds: fixed z (z_high=1.0 / gate ≤0) — mirror §71–§96.
Literature often uses other cutoffs; documented symmetry, **not** HO-tuned.

Distinct from: funding_liquidity_fx §20, NFCI mom §92, ANFCI soft §95,
ANFCI carry §96, WUI mom §87, CIP×mom §82, VIX/GPR×mom §74, EPU/TPU×mom §79,
soft–carry–value–REER stacks, capital-sleeve §53/§70, combo §8.
Explicit: do **not** overlay coolers on locked ``fx4plus_gbpcad_d1_voltarget_0025``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mt5_swing.data.fred_funding_liquidity import DEFAULT_WEEKLY_PUB_LAG_DAYS
from mt5_swing.strategies.anfci_conditioned_soft_fx import (
    AnfciConditionedSoftFxConfig,
    align_monthly_stress_z_daily,
    load_us_anfci_series,
    usd_tilt_from_anfci_z,
)
from mt5_swing.strategies.carry_rank import (
    currency_weights_to_pair_weights,
    expand_weights_to_daily,
    portfolio_returns_from_weights,
)
from mt5_swing.strategies.fx_momentum import FxMomentumConfig, momentum_weights_from_returns
from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class AnfciConditionedMomFxConfig:
    """Fixed research priors — do not grid / holdout-tune.

    ANFCI stress fields mirror §95/§96 (monthly z_window=60). Mom fields mirror
    §74/§79/§82/§87/§92 Menkhoff construction.
    """

    # ANFCI monthly z (§95/§96 mirror)
    signal_lag_months: int = 1  # months after month-end ANFCI known
    weight_lag_days: int = 1  # trading days after month-end ffill
    z_window: int = 60  # months (~5y) — mirror CIP §71 / EPU §76 / WUI / NFCI / ANFCI
    min_periods: int = 24
    z_high: float = 1.0  # cool / haven thresh — §71–§96 symmetry
    z_low: float = 0.0
    cool: float = 0.35
    usd_tilt: float = 0.5  # gross |w| when haven tilt is on
    cost_bps_side: float = 1.5
    pub_lag_days: int = DEFAULT_WEEKLY_PUB_LAG_DAYS  # informational (loader default 7)
    # Documented: lit often uses other cutoffs; this wave keeps 1.0 for symmetry
    lit_z_high_note: float = 1.5
    # Menkhoff FX momentum defaults (fx_momentum.FxMomentumConfig — not HO-tuned)
    # Identical to NFCI×mom §92
    formation_days: int = 63
    skip_days: int = 21
    n_long: int = 2
    n_short: int = 2
    mom_signal_lag: int = 1
    mom_min_periods: int = 40


PRIMARY = "mom_low_anfci"


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _soft_cfg_from_mom(cfg: AnfciConditionedMomFxConfig) -> AnfciConditionedSoftFxConfig:
    """Reuse §95 align_monthly_stress_z_daily / usd_tilt config shape."""
    return AnfciConditionedSoftFxConfig(
        signal_lag_months=cfg.signal_lag_months,
        weight_lag_days=cfg.weight_lag_days,
        z_window=cfg.z_window,
        min_periods=cfg.min_periods,
        z_high=cfg.z_high,
        z_low=cfg.z_low,
        cool=cfg.cool,
        usd_tilt=cfg.usd_tilt,
        cost_bps_side=cfg.cost_bps_side,
        pub_lag_days=cfg.pub_lag_days,
    )


def _mom_daily_weights(
    pair_ret: pd.DataFrame,
    *,
    cfg: AnfciConditionedMomFxConfig,
) -> pd.DataFrame:
    mcfg = FxMomentumConfig(
        formation_days=cfg.formation_days,
        skip_days=cfg.skip_days,
        n_long=cfg.n_long,
        n_short=cfg.n_short,
        signal_lag=cfg.mom_signal_lag,
        min_periods=cfg.mom_min_periods,
    )
    ccy_w = momentum_weights_from_returns(pair_ret, cfg=mcfg)
    mom_pw = currency_weights_to_pair_weights(ccy_w)
    mom_daily = expand_weights_to_daily(
        mom_pw, pair_ret.index, signal_lag=cfg.mom_signal_lag
    )
    cols = list(pair_ret.columns)
    for c in cols:
        if c not in mom_daily.columns:
            mom_daily[c] = 0.0
    return mom_daily.reindex(columns=cols).fillna(0.0)


def _weights_to_returns(
    w: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: AnfciConditionedMomFxConfig,
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
    cfg: AnfciConditionedMomFxConfig,
) -> pd.Series:
    soft_cfg = _soft_cfg_from_mom(cfg)
    cols = list(pair_ret.columns)
    w = usd_tilt_from_anfci_z(z_daily, cols, cfg=soft_cfg)
    return _weights_to_returns(w, pair_ret, cfg=cfg, name="us_anfci_haven_usd")


def anfci_conditioned_mom_factor_returns(
    pair_ret: pd.DataFrame,
    *,
    us_anfci: pd.Series | None = None,
    cfg: AnfciConditionedMomFxConfig | None = None,
    include_haven: bool = True,
) -> dict[str, pd.Series]:
    """Build ANFCI-conditioned Menkhoff FX momentum factor daily returns.

    Distinct from §82 (CIP×mom), §74 (VIX/GPR×mom), §79 (EPU/TPU×mom),
    §87 (WUI×mom), §92 (NFCI×mom), §95 (ANFCI soft), §96 (ANFCI carry),
    §20 (funding_liq). Explicit: do **not** overlay coolers on locked fx4plus.
    """
    cfg = cfg or AnfciConditionedMomFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    out: dict[str, pd.Series] = {}

    if pret.empty or pret.shape[1] < 3:
        return out

    if us_anfci is None:
        us_anfci = load_us_anfci_series(pub_lag_days=cfg.pub_lag_days)

    soft_cfg = _soft_cfg_from_mom(cfg)
    z_daily = align_monthly_stress_z_daily(us_anfci, pret.index, cfg=soft_cfg)

    mom_daily = _mom_daily_weights(pret, cfg=cfg)

    raw = portfolio_returns_from_weights(mom_daily, pret)
    raw = apply_costs(raw, mom_daily, bps_side=cfg.cost_bps_side)
    raw.name = "mom_raw"
    out["mom_raw"] = raw

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
    out["mom_anfci_cool"] = _weights_to_returns(
        mom_daily.mul(cool_scale, axis=0), pret, cfg=cfg, name="mom_anfci_cool"
    )

    # Primary: trade mom only when US ANFCI z ≤ 0
    gate_lo = (z_daily <= 0.0).astype(float)
    gate_lo = gate_lo.where(z_daily.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["mom_low_anfci"] = _weights_to_returns(
        mom_daily.mul(gate_lo, axis=0), pret, cfg=cfg, name="mom_low_anfci"
    )

    # Honesty inverse: mom only when ANFCI z ≥ z_high
    gate_hi = (z_daily >= float(cfg.z_high)).astype(float)
    gate_hi = gate_hi.where(z_daily.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["mom_high_anfci"] = _weights_to_returns(
        mom_daily.mul(gate_hi, axis=0), pret, cfg=cfg, name="mom_high_anfci"
    )

    if include_haven:
        out["us_anfci_haven_usd"] = _haven_returns(z_daily, pret, cfg=cfg)

    # Stack: sequential gate then cool (mom_low weights × cool scale)
    stack_scale = (gate_lo * cool_scale).clip(lower=0.0, upper=1.0)
    out["mom_anfci_stack"] = _weights_to_returns(
        mom_daily.mul(stack_scale, axis=0), pret, cfg=cfg, name="mom_anfci_stack"
    )

    # EW of primary + cool + haven (§68 / §87 / §92 / §95 / §96 mirror)
    blend_keys = [
        k
        for k in ("mom_low_anfci", "mom_anfci_cool", "us_anfci_haven_usd")
        if k in out
    ]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "mom_anfci_ew"
        out["mom_anfci_ew"] = blend

    # Regime switch: mom when low-ANFCI + haven when high-ANFCI (exclusive)
    if "mom_low_anfci" in out and "us_anfci_haven_usd" in out:
        regime = out["mom_low_anfci"].fillna(0.0) + out["us_anfci_haven_usd"].fillna(0.0)
        both_nan = out["mom_low_anfci"].isna() & out["us_anfci_haven_usd"].isna()
        regime = regime.where(~both_nan)
        regime.name = "mom_anfci_regime"
        out["mom_anfci_regime"] = regime

    return out


__all__ = [
    "PRIMARY",
    "AnfciConditionedMomFxConfig",
    "align_monthly_stress_z_daily",
    "load_us_anfci_series",
    "usd_tilt_from_anfci_z",
    "anfci_conditioned_mom_factor_returns",
]
