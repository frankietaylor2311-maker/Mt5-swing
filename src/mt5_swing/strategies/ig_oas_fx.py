"""ICE BofA IG OAS / credit risk-appetite → FX factors (frozen scholarly).

Literature priors (not holdout-tuned)
------------------------------------
- Brunnermeier, Nagel & Pedersen (2008): carry crashes / USD demand in funding
  and risk-appetite stress; corporate credit spreads are a related state var.
- Menkhoff et al. (2012a): global FX risk-off channel.
- ICE BofA IG OAS (``BAMLC0A0CM``) is a *corporate* credit risk premium —
  distinct from Moody's ``BAA10Y`` (funding-liq §20), NFCI, VIX, GPR, FX-RV,
  EPU, crash-skew, house-price, credit-gap.

Legs (USD-tilt / carry-conditioned — NOT country XS)
----------------------------------------------------
- ``ig_oas_usd`` (PRIMARY): long USD when lagged z(IG OAS) ≥ z_high.
- ``hy_oas_usd``: same with HY OAS if available.
- ``ig_oas_chg_usd``: long USD when lagged z(Δ IG OAS) ≥ z_high.
- ``ig_oas_lvl_usd``: continuous intensity USD tilt from clipped z (non-binary).
- ``oas_stress_fx``: honesty alternate — long risk FX / short USD when OAS
  elevated (wrong-signed check vs primary haven prior).
- ``carry_ig_oas_cool``: scholarly cash-rate carry × OAS risk_scale ∈ [cool, 1].
- ``carry_ig_oas_loose``: carry only when OAS z ≤ 0 (flat in stress).
- ``oas_ew``: EW of primary + change + HY (if HY present).

PIT: loaders apply daily pub lag; this module adds ``signal_lag`` trading days
on aligned z before forming weights. ``portfolio_returns_from_weights`` applies
an extra weight lag.

Explicit: do **not** apply these coolers to the locked FTMO sleeve config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.strategies.carry_rank import (
    CarryRankConfig,
    carry_weights_from_rates,
    currency_weights_to_pair_weights,
    expand_weights_to_daily,
    portfolio_returns_from_weights,
)
from mt5_swing.strategies.funding_liquidity_fx import (
    FundingLiquidityFxConfig,
    align_and_z,
    trailing_z,
    usd_tilt_weights_from_z,
)
from mt5_swing.strategies.gpr_regime import (
    GprRegimeConfig,
    USD_LONG_PAIRS,
    align_macro_to_index,
    risk_scale_from_z,
)
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class IgOasFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # trading days after pub-lagged macro is known
    z_window: int = 252
    min_periods: int = 60
    z_high: float = 1.0
    z_low: float = 0.0
    cool: float = 0.35
    usd_tilt: float = 0.5
    cost_bps_side: float = 1.5
    carry_n_long: int = 2
    carry_n_short: int = 2
    binary_usd: bool = True


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _to_funding_cfg(cfg: IgOasFxConfig, *, binary_usd: bool | None = None) -> FundingLiquidityFxConfig:
    return FundingLiquidityFxConfig(
        signal_lag=cfg.signal_lag,
        z_window=cfg.z_window,
        min_periods=cfg.min_periods,
        z_high=cfg.z_high,
        z_low=cfg.z_low,
        cool=cfg.cool,
        usd_tilt=cfg.usd_tilt,
        cost_bps_side=cfg.cost_bps_side,
        carry_n_long=cfg.carry_n_long,
        carry_n_short=cfg.carry_n_short,
        binary_usd=cfg.binary_usd if binary_usd is None else binary_usd,
    )


def _weights_to_returns(
    w: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: IgOasFxConfig,
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


def risk_fx_tilt_weights_from_z(
    z: pd.Series,
    pair_columns: list[str],
    *,
    cfg: IgOasFxConfig,
) -> pd.DataFrame:
    """Honesty alternate: long risk FX / short USD when OAS z elevated.

    Sign-flips ``USD_LONG_PAIRS`` so elevated credit stress buys (not sells)
    risk currencies — wrong-signed check vs the haven prior.
    """
    fcfg = _to_funding_cfg(cfg)
    if fcfg.binary_usd:
        intensity = (z >= fcfg.z_high).astype(float) * float(fcfg.usd_tilt)
        intensity = intensity.where(z.notna(), 0.0)
    else:
        denom = max(float(fcfg.z_high) - float(fcfg.z_low), 1e-6)
        intensity = ((z - fcfg.z_low) / denom).clip(0.0, 1.0) * float(fcfg.usd_tilt)
        intensity = intensity.fillna(0.0)
    cols = [c for c in pair_columns if c.upper() in {p.upper() for p in USD_LONG_PAIRS}]
    if not cols:
        cols = list(pair_columns)
    n = max(len(cols), 1)
    w = pd.DataFrame(0.0, index=z.index, columns=cols)
    for sym in cols:
        # Flip haven sign → risk-on when OAS high
        sign = -USD_LONG_PAIRS.get(sym.upper(), 0)
        w[sym] = (intensity / n) * sign
    return w.fillna(0.0)


def ig_oas_factor_returns(
    pair_ret: pd.DataFrame,
    *,
    ig_oas: pd.Series | None = None,
    hy_oas: pd.Series | None = None,
    d_ig_oas: pd.Series | None = None,
    rates: pd.DataFrame | None = None,
    cfg: IgOasFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Build IG OAS / credit risk-appetite FX factor daily returns."""
    cfg = cfg or IgOasFxConfig()
    fcfg = _to_funding_cfg(cfg)
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    cols = list(pret.columns)
    out: dict[str, pd.Series] = {}

    if ig_oas is not None and len(ig_oas):
        z = align_and_z(ig_oas, pret.index, cfg=fcfg)
        out["ig_oas_usd"] = _weights_to_returns(
            usd_tilt_weights_from_z(z, cols, cfg=fcfg), pret, cfg=cfg, name="ig_oas_usd"
        )
        # Continuous intensity variant (non-binary)
        fcfg_cont = _to_funding_cfg(cfg, binary_usd=False)
        z_cont = align_and_z(ig_oas, pret.index, cfg=fcfg_cont)
        out["ig_oas_lvl_usd"] = _weights_to_returns(
            usd_tilt_weights_from_z(z_cont, cols, cfg=fcfg_cont),
            pret,
            cfg=cfg,
            name="ig_oas_lvl_usd",
        )
        # Honesty wrong-signed alternate
        out["oas_stress_fx"] = _weights_to_returns(
            risk_fx_tilt_weights_from_z(z, cols, cfg=cfg),
            pret,
            cfg=cfg,
            name="oas_stress_fx",
        )

    if hy_oas is not None and len(hy_oas):
        z = align_and_z(hy_oas, pret.index, cfg=fcfg)
        out["hy_oas_usd"] = _weights_to_returns(
            usd_tilt_weights_from_z(z, cols, cfg=fcfg), pret, cfg=cfg, name="hy_oas_usd"
        )

    chg = d_ig_oas
    if chg is None and ig_oas is not None and len(ig_oas):
        chg = ig_oas.diff()
        chg.name = "dIG_OAS"
    if chg is not None and len(chg):
        z = align_and_z(chg, pret.index, cfg=fcfg)
        out["ig_oas_chg_usd"] = _weights_to_returns(
            usd_tilt_weights_from_z(z, cols, cfg=fcfg),
            pret,
            cfg=cfg,
            name="ig_oas_chg_usd",
        )

    # Carry conditioned by IG OAS stress (scholarly sleeve only)
    if rates is not None and not rates.empty and ig_oas is not None and len(ig_oas):
        ccfg = CarryRankConfig(
            n_long=cfg.carry_n_long,
            n_short=cfg.carry_n_short,
            signal_lag=cfg.signal_lag,
        )
        ccy_w = carry_weights_from_rates(rates, cfg=ccfg)
        carry_pw = currency_weights_to_pair_weights(ccy_w)
        carry_daily = expand_weights_to_daily(carry_pw, pret.index, signal_lag=cfg.signal_lag)
        for c in cols:
            if c not in carry_daily.columns:
                carry_daily[c] = 0.0
        carry_daily = carry_daily.reindex(columns=cols).fillna(0.0)
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
        z = align_and_z(ig_oas, pret.index, cfg=fcfg)
        scale = risk_scale_from_z(z, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
        cooled = carry_daily.mul(scale, axis=0)
        r = portfolio_returns_from_weights(cooled, pret)
        r = apply_costs(r, cooled, bps_side=cfg.cost_bps_side)
        r.name = "carry_ig_oas_cool"
        out[r.name] = r

        # Trade carry only when OAS z ≤ 0 (loose / risk-on)
        gate = (z <= 0.0).astype(float)
        gate = gate.where(z.notna(), 0.0).reindex(pret.index).fillna(0.0)
        gated = carry_daily.mul(gate, axis=0)
        rg = portfolio_returns_from_weights(gated, pret)
        rg = apply_costs(rg, gated, bps_side=cfg.cost_bps_side)
        rg.name = "carry_ig_oas_loose"
        out[rg.name] = rg

    blend_keys = [k for k in ("ig_oas_usd", "ig_oas_chg_usd", "hy_oas_usd") if k in out]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "oas_ew"
        out["oas_ew"] = blend

    return out


# Re-export helpers used by tests
__all__ = [
    "IgOasFxConfig",
    "ig_oas_factor_returns",
    "risk_fx_tilt_weights_from_z",
    "trailing_z",
    "align_and_z",
    "usd_tilt_weights_from_z",
    "align_macro_to_index",
]
