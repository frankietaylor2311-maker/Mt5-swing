"""NY Fed ACM Treasury term premium → FX factors (frozen scholarly).

Literature priors (not holdout-tuned)
------------------------------------
- Adrian, Crump & Moench (ACM): elevated 10y Treasury term premium =
  compensation for duration / rate risk → USD risk-appetite / risk-off
  (safe-haven USD when TP high).
- Lustig–Stathopoulos–Verdelhan: term structure of currency carry.
- Hofmann–Shim–Shin: bond risk premia and FX.
- Distinct from §16 yield-curve slope (IRLTLT−IRSTCI), §23 real-rate/TIPS
  (DFII10/T10YIE), §20 funding-liq, §36 IG OAS.

Legs (USD-tilt / carry-conditioned — NOT country XS)
----------------------------------------------------
- ``acm_tp_usd`` (PRIMARY): long USD when lagged z(THREEFYTP10) ≥ z_high.
- ``acm_tp_chg_usd``: long USD when lagged z(Δ TP) ≥ z_high.
- ``acm_tp_lvl_usd``: continuous intensity USD tilt from clipped z.
- ``acm_tp_stress_fx``: honesty alternate — long risk FX / short USD when TP
  elevated (wrong-signed check vs primary haven prior).
- ``carry_acm_tp_cool``: scholarly cash-rate carry × TP risk_scale ∈ [cool, 1].
- ``carry_acm_tp_loose``: carry only when TP z ≤ 0 (flat in stress).
- ``acm_tp_ew``: EW of primary + change (+ 5y companion if present).
- ``acm_tp5_usd`` (optional): same as primary with THREEFYTP5.

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
class AcmTermPremiumFxConfig:
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


def _to_funding_cfg(
    cfg: AcmTermPremiumFxConfig, *, binary_usd: bool | None = None
) -> FundingLiquidityFxConfig:
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
    cfg: AcmTermPremiumFxConfig,
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
    cfg: AcmTermPremiumFxConfig,
) -> pd.DataFrame:
    """Honesty alternate: long risk FX / short USD when TP z elevated.

    Sign-flips ``USD_LONG_PAIRS`` so elevated term premium buys (not sells)
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
        # Flip haven sign → risk-on when TP high
        sign = -USD_LONG_PAIRS.get(sym.upper(), 0)
        w[sym] = (intensity / n) * sign
    return w.fillna(0.0)


def acm_tp_factor_returns(
    pair_ret: pd.DataFrame,
    *,
    acm_tp10: pd.Series | None = None,
    acm_tp5: pd.Series | None = None,
    d_acm_tp10: pd.Series | None = None,
    rates: pd.DataFrame | None = None,
    cfg: AcmTermPremiumFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Build ACM term-premium / USD risk-appetite FX factor daily returns."""
    cfg = cfg or AcmTermPremiumFxConfig()
    fcfg = _to_funding_cfg(cfg)
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    cols = list(pret.columns)
    out: dict[str, pd.Series] = {}

    if acm_tp10 is not None and len(acm_tp10):
        z = align_and_z(acm_tp10, pret.index, cfg=fcfg)
        out["acm_tp_usd"] = _weights_to_returns(
            usd_tilt_weights_from_z(z, cols, cfg=fcfg), pret, cfg=cfg, name="acm_tp_usd"
        )
        # Continuous intensity variant (non-binary)
        fcfg_cont = _to_funding_cfg(cfg, binary_usd=False)
        z_cont = align_and_z(acm_tp10, pret.index, cfg=fcfg_cont)
        out["acm_tp_lvl_usd"] = _weights_to_returns(
            usd_tilt_weights_from_z(z_cont, cols, cfg=fcfg_cont),
            pret,
            cfg=cfg,
            name="acm_tp_lvl_usd",
        )
        # Honesty wrong-signed alternate
        out["acm_tp_stress_fx"] = _weights_to_returns(
            risk_fx_tilt_weights_from_z(z, cols, cfg=cfg),
            pret,
            cfg=cfg,
            name="acm_tp_stress_fx",
        )

    if acm_tp5 is not None and len(acm_tp5):
        z = align_and_z(acm_tp5, pret.index, cfg=fcfg)
        out["acm_tp5_usd"] = _weights_to_returns(
            usd_tilt_weights_from_z(z, cols, cfg=fcfg), pret, cfg=cfg, name="acm_tp5_usd"
        )

    chg = d_acm_tp10
    if chg is None and acm_tp10 is not None and len(acm_tp10):
        chg = acm_tp10.diff()
        chg.name = "dACM_TP10"
    if chg is not None and len(chg):
        z = align_and_z(chg, pret.index, cfg=fcfg)
        out["acm_tp_chg_usd"] = _weights_to_returns(
            usd_tilt_weights_from_z(z, cols, cfg=fcfg),
            pret,
            cfg=cfg,
            name="acm_tp_chg_usd",
        )

    # Carry conditioned by ACM TP stress (scholarly sleeve only)
    if rates is not None and not rates.empty and acm_tp10 is not None and len(acm_tp10):
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
        z = align_and_z(acm_tp10, pret.index, cfg=fcfg)
        scale = risk_scale_from_z(z, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
        cooled = carry_daily.mul(scale, axis=0)
        r = portfolio_returns_from_weights(cooled, pret)
        r = apply_costs(r, cooled, bps_side=cfg.cost_bps_side)
        r.name = "carry_acm_tp_cool"
        out[r.name] = r

        # Trade carry only when TP z ≤ 0 (loose / risk-on)
        gate = (z <= 0.0).astype(float)
        gate = gate.where(z.notna(), 0.0).reindex(pret.index).fillna(0.0)
        gated = carry_daily.mul(gate, axis=0)
        rg = portfolio_returns_from_weights(gated, pret)
        rg = apply_costs(rg, gated, bps_side=cfg.cost_bps_side)
        rg.name = "carry_acm_tp_loose"
        out[rg.name] = rg

    blend_keys = [k for k in ("acm_tp_usd", "acm_tp_chg_usd", "acm_tp5_usd") if k in out]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "acm_tp_ew"
        out["acm_tp_ew"] = blend

    return out


# Re-export helpers used by tests
__all__ = [
    "AcmTermPremiumFxConfig",
    "acm_tp_factor_returns",
    "risk_fx_tilt_weights_from_z",
    "trailing_z",
    "align_and_z",
    "usd_tilt_weights_from_z",
    "align_macro_to_index",
]
