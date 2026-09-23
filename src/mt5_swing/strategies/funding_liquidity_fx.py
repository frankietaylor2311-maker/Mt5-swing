"""Funding-liquidity / financial-conditions → FX factors (frozen scholarly).

Literature priors (not holdout-tuned)
------------------------------------
- Brunnermeier, Nagel & Pedersen (2008): carry crashes when funding liquidity
  dries up; USD / funding-currency demand in stress.
- Menkhoff et al. (2012a): related global FX-vol / risk-off channel — this module
  uses **funding / financial-conditions** state (NFCI, TED/CPFF), *not* VIX,
  GPR, FX-RV, or EPU (those are prior waves).
- Chicago Fed NFCI: level > 0 = tighter-than-average conditions (index design).
  Trailing z ≥ 1 matches prior VIX/GPR/EPU intensity priors.

Legs
----
- ``nfci_usd``: long USD when lagged z(NFCI) ≥ z_high (primary).
- ``anfci_usd``: same with ANFCI.
- ``nfci_lvl_usd``: long USD when lagged NFCI level > 0 (absolute frozen cutoff).
- ``nfci_chg_usd``: long USD when lagged z(ΔNFCI) ≥ z_high (change vs level).
- ``cpff_usd`` / ``ted_usd`` / ``baa_usd`` / ``spread_usd``: funding/credit
  spread USD tilts.
- ``carry_nfci_cool``: scholarly carry × NFCI risk_scale ∈ [cool, 1]
  (funding-only — no VIX/GPR/FX-RV/EPU).
- ``carry_nfci_loose``: carry **only** when lagged NFCI ≤ 0 (flat in stress).
- ``carry_cpff_cool``: carry × CPFF risk_scale.
- ``funding_ew``: EW of ``nfci_usd`` + ``nfci_chg_usd`` + ``cpff_usd``.

PIT: loaders apply weekly/daily pub lags; this module adds ``signal_lag`` trading
days on aligned z / levels before forming weights. ``portfolio_returns_from_weights``
applies an extra weight lag.

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
from mt5_swing.strategies.gpr_regime import (
    GprRegimeConfig,
    USD_LONG_PAIRS,
    align_macro_to_index,
    risk_scale_from_z,
)
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class FundingLiquidityFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # trading days after pub-lagged macro is known
    z_window: int = 252
    min_periods: int = 60
    z_high: float = 1.0
    z_low: float = 0.0
    cool: float = 0.35
    usd_tilt: float = 0.5  # gross |w| when fully risk-off (match FX-RV prior)
    nfci_level_cut: float = 0.0  # Chicago Fed: >0 tighter than average
    cost_bps_side: float = 1.5
    carry_n_long: int = 2
    carry_n_short: int = 2
    binary_usd: bool = True  # USD tilt on/off at z_high (else continuous intensity)


def trailing_z(s: pd.Series, *, lookback: int, min_periods: int) -> pd.Series:
    mu = s.rolling(lookback, min_periods=min_periods).mean()
    sd = s.rolling(lookback, min_periods=min_periods).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def align_and_z(
    series: pd.Series,
    index: pd.DatetimeIndex,
    *,
    cfg: FundingLiquidityFxConfig,
) -> pd.Series:
    """As-of align → trailing z → signal_lag trading days."""
    aligned = align_macro_to_index(series, index)
    z = trailing_z(aligned, lookback=cfg.z_window, min_periods=cfg.min_periods)
    if cfg.signal_lag > 0:
        z = z.shift(int(cfg.signal_lag))
    z.name = f"{series.name or 'macro'}_z"
    return z


def align_level(
    series: pd.Series,
    index: pd.DatetimeIndex,
    *,
    cfg: FundingLiquidityFxConfig,
) -> pd.Series:
    """As-of align level → signal_lag (for absolute NFCI>0 gate)."""
    aligned = align_macro_to_index(series, index)
    if cfg.signal_lag > 0:
        aligned = aligned.shift(int(cfg.signal_lag))
    aligned.name = series.name or "level"
    return aligned


def usd_tilt_weights_from_z(
    z: pd.Series,
    pair_columns: list[str],
    *,
    cfg: FundingLiquidityFxConfig,
) -> pd.DataFrame:
    """Long-USD weights when funding stress z elevated."""
    if cfg.binary_usd:
        intensity = (z >= cfg.z_high).astype(float) * float(cfg.usd_tilt)
        intensity = intensity.where(z.notna(), 0.0)
    else:
        denom = max(float(cfg.z_high) - float(cfg.z_low), 1e-6)
        intensity = ((z - cfg.z_low) / denom).clip(0.0, 1.0) * float(cfg.usd_tilt)
        intensity = intensity.fillna(0.0)
    cols = [c for c in pair_columns if c.upper() in {p.upper() for p in USD_LONG_PAIRS}]
    if not cols:
        cols = list(pair_columns)
    n = max(len(cols), 1)
    w = pd.DataFrame(0.0, index=z.index, columns=cols)
    for sym in cols:
        sign = USD_LONG_PAIRS.get(sym.upper(), 0)
        w[sym] = (intensity / n) * sign
    return w.fillna(0.0)


def usd_tilt_weights_from_level_gate(
    level: pd.Series,
    pair_columns: list[str],
    *,
    cfg: FundingLiquidityFxConfig,
) -> pd.DataFrame:
    """Long USD when lagged NFCI level > nfci_level_cut (frozen absolute gate)."""
    on = (level > float(cfg.nfci_level_cut)).astype(float) * float(cfg.usd_tilt)
    on = on.where(level.notna(), 0.0)
    cols = [c for c in pair_columns if c.upper() in {p.upper() for p in USD_LONG_PAIRS}]
    if not cols:
        cols = list(pair_columns)
    n = max(len(cols), 1)
    w = pd.DataFrame(0.0, index=level.index, columns=cols)
    for sym in cols:
        sign = USD_LONG_PAIRS.get(sym.upper(), 0)
        w[sym] = (on / n) * sign
    return w.fillna(0.0)


def _weights_to_returns(
    w: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: FundingLiquidityFxConfig,
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


def funding_liquidity_factor_returns(
    pair_ret: pd.DataFrame,
    *,
    nfci: pd.Series | None = None,
    anfci: pd.Series | None = None,
    d_nfci: pd.Series | None = None,
    cpff: pd.Series | None = None,
    ted: pd.Series | None = None,
    baa: pd.Series | None = None,
    funding_spread: pd.Series | None = None,
    rates: pd.DataFrame | None = None,
    cfg: FundingLiquidityFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Build funding-liquidity FX factor daily returns (post optional costs)."""
    cfg = cfg or FundingLiquidityFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    cols = list(pret.columns)
    out: dict[str, pd.Series] = {}

    # --- USD tilts from NFCI / ANFCI / ΔNFCI ---
    if nfci is not None and len(nfci):
        z = align_and_z(nfci, pret.index, cfg=cfg)
        out["nfci_usd"] = _weights_to_returns(
            usd_tilt_weights_from_z(z, cols, cfg=cfg), pret, cfg=cfg, name="nfci_usd"
        )
        lvl = align_level(nfci, pret.index, cfg=cfg)
        out["nfci_lvl_usd"] = _weights_to_returns(
            usd_tilt_weights_from_level_gate(lvl, cols, cfg=cfg),
            pret,
            cfg=cfg,
            name="nfci_lvl_usd",
        )

    if anfci is not None and len(anfci):
        z = align_and_z(anfci, pret.index, cfg=cfg)
        out["anfci_usd"] = _weights_to_returns(
            usd_tilt_weights_from_z(z, cols, cfg=cfg), pret, cfg=cfg, name="anfci_usd"
        )

    chg = d_nfci
    if chg is None and nfci is not None and len(nfci):
        chg = nfci.diff()
        chg.name = "dNFCI"
    if chg is not None and len(chg):
        z = align_and_z(chg, pret.index, cfg=cfg)
        out["nfci_chg_usd"] = _weights_to_returns(
            usd_tilt_weights_from_z(z, cols, cfg=cfg), pret, cfg=cfg, name="nfci_chg_usd"
        )

    # --- Funding / credit spread USD tilts ---
    for key, series in (
        ("cpff_usd", cpff),
        ("ted_usd", ted),
        ("baa_usd", baa),
        ("spread_usd", funding_spread),
    ):
        if series is None or not len(series):
            continue
        z = align_and_z(series, pret.index, cfg=cfg)
        out[key] = _weights_to_returns(
            usd_tilt_weights_from_z(z, cols, cfg=cfg), pret, cfg=cfg, name=key
        )

    # --- Carry conditioned by funding state (scholarly sleeve only) ---
    if rates is not None and not rates.empty:
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
        if nfci is not None and len(nfci):
            z = align_and_z(nfci, pret.index, cfg=cfg)
            scale = risk_scale_from_z(z, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
            cooled = carry_daily.mul(scale, axis=0)
            r = portfolio_returns_from_weights(cooled, pret)
            r = apply_costs(r, cooled, bps_side=cfg.cost_bps_side)
            r.name = "carry_nfci_cool"
            out[r.name] = r

            # Brunnermeier-style gate: trade carry only in loose conditions (NFCI≤0)
            lvl = align_level(nfci, pret.index, cfg=cfg)
            gate = (lvl <= float(cfg.nfci_level_cut)).astype(float)
            gate = gate.where(lvl.notna(), 0.0).reindex(pret.index).fillna(0.0)
            gated = carry_daily.mul(gate, axis=0)
            rg = portfolio_returns_from_weights(gated, pret)
            rg = apply_costs(rg, gated, bps_side=cfg.cost_bps_side)
            rg.name = "carry_nfci_loose"
            out[rg.name] = rg

        if cpff is not None and len(cpff):
            z = align_and_z(cpff, pret.index, cfg=cfg)
            scale = risk_scale_from_z(z, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
            cooled = carry_daily.mul(scale, axis=0)
            r = portfolio_returns_from_weights(cooled, pret)
            r = apply_costs(r, cooled, bps_side=cfg.cost_bps_side)
            r.name = "carry_cpff_cool"
            out[r.name] = r

    # EW blend of distinct funding channels (level, change, CP spread)
    blend_keys = [k for k in ("nfci_usd", "nfci_chg_usd", "cpff_usd") if k in out]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "funding_ew"
        out["funding_ew"] = blend

    return out
