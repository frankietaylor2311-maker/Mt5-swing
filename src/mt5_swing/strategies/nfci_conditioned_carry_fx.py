"""NFCI-conditioned Lustig–Verdelhan carry FX factors (§91).

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Lustig, Roussanov & Verdelhan (2011): FX carry sorted by interest-rate
  differentials (high-minus-low).
- Brunnermeier, Nagel & Pedersen (2008), "Carry Trades and Currency Crashes":
  funding-liquidity spirals / tight financial conditions coincide with FX
  risk-off. Chicago Fed **NFCI** (National Financial Conditions Index) is the
  free aggregate U.S. financial-conditions stress series (FRED ``NFCI``;
  weekly; 0 ≈ average; >0 tighter) via
  ``mt5_swing.data.fred_funding_liquidity.load_nfci_series``
  (``pub_lag_days=7`` already applied in the loader) — same stress path as
  NFCI soft §90.

Claim (a priori)
----------------
IR3M / policy-rate Lustig–Verdelhan carry earns more when US financial-
conditions stress is **low** (loose NFCI). Trade carry only when lagged NFCI
z is subdued; cool or sit out when NFCI stress is elevated. Single-stress
design (like CIP×carry §68 / WUI×carry §86), not dual EPU+TPU.

Legs
----
- ``carry_low_nfci`` (**PRIMARY**): scholarly cash-rate carry only when
  lagged US NFCI z ≤ 0 (flat in elevated stress).
- ``carry_nfci_cool``: carry × risk_scale ∈ [cool, 1] from NFCI z
  (cool when z ≥ z_high).
- ``carry_raw``: always-on Lustig–Verdelhan carry (honesty baseline).
- ``carry_high_nfci``: honesty inverse — carry only when NFCI z ≥ z_high.
- ``us_nfci_haven_usd``: long USD when lagged NFCI z ≥ z_high
  (§20 / CIP-stress haven companion pattern / NFCI soft §90).
- ``carry_nfci_stack``: carry_low_nfci × cool scale (sequential gate then cool).
- ``carry_nfci_ew``: EW of carry_low_nfci ⊕ carry_nfci_cool ⊕ us_nfci_haven_usd.
- ``carry_nfci_regime``: carry_low_nfci + us_nfci_haven_usd (exclusive regimes:
  carry when z≤0, haven when z≥z_high).

PIT
---
``load_nfci_series`` applies ``pub_lag_days=7`` (weekly Chicago Fed release).
This module collapses pub-lagged weekly NFCI to **month-end** (last obs/month)
via ``align_monthly_stress_z_daily`` from NFCI soft §90, then adds
``signal_lag_months`` (default 1) on **monthly** trailing z
(z_window=60m, min_periods=24 — mirror CIP §71 / EPU §76 / WUI §85 / NFCI soft
§90, **not** 252d daily) + ``weight_lag_days=1`` after month-end ffill to the
trading calendar. Rates: existing FRED carry loaders' PIT lags +
``carry_signal_lag`` trading days via ``CarryRankConfig`` /
``expand_weights_to_daily``. ``portfolio_returns_from_weights`` applies an
extra weight lag. Costs 1.5 bps/side.

Gate / cool thresholds: fixed z (z_high=1.0 / gate ≤0) — mirror §71–§90.
Literature often uses other cutoffs; documented symmetry, **not** HO-tuned.

Distinct from: funding_liquidity_fx §20 (USD tilts + carry×NFCI cool as
*standalone* factors — **not** IR3M Lustig–Verdelhan gate), NFCI soft §90,
WUI carry §86, CIP×carry §68, VIX/GPR×carry §73, EPU/TPU×carry §78,
CIP/VIX/EPU/WUI soft/mom/value/REER §71–§77/§79–§89, capital-sleeve §53/§70,
combo §8.
Explicit: do **not** overlay coolers on locked ``fx4plus_gbpcad_d1_voltarget_0025``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mt5_swing.data.fred_funding_liquidity import DEFAULT_WEEKLY_PUB_LAG_DAYS
from mt5_swing.strategies.carry_rank import (
    CarryRankConfig,
    carry_weights_from_rates,
    currency_weights_to_pair_weights,
    expand_weights_to_daily,
    portfolio_returns_from_weights,
)
from mt5_swing.strategies.gpr_regime import GprRegimeConfig, risk_scale_from_z
from mt5_swing.strategies.nfci_conditioned_soft_fx import (
    NfciConditionedSoftFxConfig,
    align_monthly_stress_z_daily,
    load_us_nfci_series,
    usd_tilt_from_nfci_z,
)
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class NfciConditionedCarryFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag_months: int = 1  # months after month-end NFCI known
    weight_lag_days: int = 1  # trading days after month-end ffill
    z_window: int = 60  # months (~5y) — mirror CIP §71 / EPU §76 / WUI / NFCI soft
    min_periods: int = 24
    z_high: float = 1.0  # cool / haven thresh — §71–§90 symmetry
    z_low: float = 0.0
    cool: float = 0.35
    usd_tilt: float = 0.5  # gross |w| when haven tilt is on
    cost_bps_side: float = 1.5
    carry_n_long: int = 2
    carry_n_short: int = 2
    carry_signal_lag: int = 1
    pub_lag_days: int = DEFAULT_WEEKLY_PUB_LAG_DAYS  # informational (loader default 7)
    # Documented: lit often uses other cutoffs; this wave keeps 1.0 for symmetry
    lit_z_high_note: float = 1.5


PRIMARY = "carry_low_nfci"


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _soft_cfg_from_carry(
    cfg: NfciConditionedCarryFxConfig,
) -> NfciConditionedSoftFxConfig:
    """Reuse §90 align_monthly_stress_z_daily / usd_tilt config shape."""
    return NfciConditionedSoftFxConfig(
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


def _carry_daily_weights(
    rates: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: NfciConditionedCarryFxConfig,
) -> pd.DataFrame:
    ccfg = CarryRankConfig(
        n_long=cfg.carry_n_long,
        n_short=cfg.carry_n_short,
        signal_lag=cfg.carry_signal_lag,
    )
    ccy_w = carry_weights_from_rates(rates, cfg=ccfg)
    carry_pw = currency_weights_to_pair_weights(ccy_w)
    carry_daily = expand_weights_to_daily(
        carry_pw, pair_ret.index, signal_lag=cfg.carry_signal_lag
    )
    cols = list(pair_ret.columns)
    for c in cols:
        if c not in carry_daily.columns:
            carry_daily[c] = 0.0
    return carry_daily.reindex(columns=cols).fillna(0.0)


def _weights_to_returns(
    w: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: NfciConditionedCarryFxConfig,
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
    cfg: NfciConditionedCarryFxConfig,
) -> pd.Series:
    soft_cfg = _soft_cfg_from_carry(cfg)
    cols = list(pair_ret.columns)
    w = usd_tilt_from_nfci_z(z_daily, cols, cfg=soft_cfg)
    return _weights_to_returns(w, pair_ret, cfg=cfg, name="us_nfci_haven_usd")


def nfci_conditioned_carry_factor_returns(
    pair_ret: pd.DataFrame,
    *,
    rates: pd.DataFrame,
    us_nfci: pd.Series | None = None,
    cfg: NfciConditionedCarryFxConfig | None = None,
    include_haven: bool = True,
) -> dict[str, pd.Series]:
    """Build NFCI-conditioned Lustig–Verdelhan carry FX factor daily returns.

    Distinct from §68 (CIP×carry), §73 (VIX/GPR×carry), §78 (EPU/TPU×carry),
    §86 (WUI×carry), §90 (NFCI soft), §20 (funding_liq standalone NFCI).
    Explicit: do **not** overlay coolers on locked fx4plus.
    """
    cfg = cfg or NfciConditionedCarryFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    out: dict[str, pd.Series] = {}

    if rates is None or rates.empty:
        return out

    if us_nfci is None:
        us_nfci = load_us_nfci_series(pub_lag_days=cfg.pub_lag_days)

    soft_cfg = _soft_cfg_from_carry(cfg)
    z_daily = align_monthly_stress_z_daily(us_nfci, pret.index, cfg=soft_cfg)

    carry_daily = _carry_daily_weights(rates, pret, cfg=cfg)

    raw = portfolio_returns_from_weights(carry_daily, pret)
    raw = apply_costs(raw, carry_daily, bps_side=cfg.cost_bps_side)
    raw.name = "carry_raw"
    out["carry_raw"] = raw

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
    out["carry_nfci_cool"] = _weights_to_returns(
        carry_daily.mul(cool_scale, axis=0), pret, cfg=cfg, name="carry_nfci_cool"
    )

    # Primary: trade carry only when US NFCI z ≤ 0
    gate_lo = (z_daily <= 0.0).astype(float)
    gate_lo = gate_lo.where(z_daily.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["carry_low_nfci"] = _weights_to_returns(
        carry_daily.mul(gate_lo, axis=0), pret, cfg=cfg, name="carry_low_nfci"
    )

    # Honesty inverse: carry only when NFCI z ≥ z_high
    gate_hi = (z_daily >= float(cfg.z_high)).astype(float)
    gate_hi = gate_hi.where(z_daily.notna(), 0.0).reindex(pret.index).fillna(0.0)
    out["carry_high_nfci"] = _weights_to_returns(
        carry_daily.mul(gate_hi, axis=0), pret, cfg=cfg, name="carry_high_nfci"
    )

    if include_haven:
        out["us_nfci_haven_usd"] = _haven_returns(z_daily, pret, cfg=cfg)

    # Stack: sequential gate then cool (carry_low weights × cool scale)
    stack_scale = (gate_lo * cool_scale).clip(lower=0.0, upper=1.0)
    out["carry_nfci_stack"] = _weights_to_returns(
        carry_daily.mul(stack_scale, axis=0), pret, cfg=cfg, name="carry_nfci_stack"
    )

    # EW of primary + cool + haven (§68 / §86 / §90 mirror)
    blend_keys = [
        k
        for k in ("carry_low_nfci", "carry_nfci_cool", "us_nfci_haven_usd")
        if k in out
    ]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "carry_nfci_ew"
        out["carry_nfci_ew"] = blend

    # Regime switch: carry when low-NFCI + haven when high-NFCI (exclusive)
    if "carry_low_nfci" in out and "us_nfci_haven_usd" in out:
        regime = out["carry_low_nfci"].fillna(0.0) + out["us_nfci_haven_usd"].fillna(0.0)
        both_nan = out["carry_low_nfci"].isna() & out["us_nfci_haven_usd"].isna()
        regime = regime.where(~both_nan)
        regime.name = "carry_nfci_regime"
        out["carry_nfci_regime"] = regime

    return out


__all__ = [
    "PRIMARY",
    "NfciConditionedCarryFxConfig",
    "align_monthly_stress_z_daily",
    "load_us_nfci_series",
    "usd_tilt_from_nfci_z",
    "nfci_conditioned_carry_factor_returns",
]
