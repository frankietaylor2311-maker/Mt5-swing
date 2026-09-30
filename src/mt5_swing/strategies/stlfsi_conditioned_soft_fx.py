"""STLFSI4-conditioned Dahlquist soft-signal EW FX factors (§100).

Literature priors (fixed a priori — **not** holdout-tuned)
----------------------------------------------------------
- Dahlquist & Hasseltoft (2020), *JFE*: economic-momentum soft signals;
  equal-weight of macro-momentum XS factors (repo §66 ``soft_ew_macro5`` /
  SOFT_LEGS).
- Brunnermeier, Nagel & Pedersen (2008), "Carry Trades and Currency Crashes":
  funding-liquidity spirals / tight financial conditions coincide with FX
  risk-off. St. Louis Fed **STLFSI4** (Financial Stress Index, redesigned;
  FRED ``STLFSI4``; weekly; 0 ≈ average; >0 elevated stress; Kliesen et al.)
  via ``mt5_swing.data.fred_funding_liquidity.load_stlfsi_series`` /
  ``load_us_stlfsi_series`` (``pub_lag_days=7`` already applied in the loader).

Claim (a priori)
----------------
Soft-stack macro momentum (§66) earns more when US financial-stress is
**low** (loose STLFSI4). Trade soft EW only when lagged STLFSI z is subdued;
cool or sit out when St. Louis Fed stress is elevated. Single-stress design
(like CIP soft §71 / WUI soft §85 / NFCI soft §90 / ANFCI soft §95), not dual
EPU+TPU. STLFSI4 is a **distinct** conditioning series from NFCI/ANFCI
(§90–§99) — St. Louis Fed construction / panel, not Chicago Fed.

Legs
----
- ``soft_low_stlfsi`` (**PRIMARY**): §66 ``soft_ew_macro5`` × binary gate
  (on when lagged STLFSI z ≤ 0).
- ``soft_stlfsi_cool``: soft_ew_macro5 × continuous cool ∈ [cool, 1] from STLFSI z.
- ``soft_raw``: ungated soft_ew_macro5 honesty baseline (§66).
- ``soft_high_stlfsi``: honesty inverse — soft only when STLFSI z ≥ z_high.
- ``us_stlfsi_haven_usd``: long USD when lagged STLFSI z ≥ z_high
  (§20 / CIP-stress haven companion pattern).
- ``soft_stlfsi_stack``: soft_low_stlfsi × cool scale (sequential gate then cool).
- ``soft_stlfsi_ew``: EW of soft_low_stlfsi ⊕ soft_stlfsi_cool ⊕ us_stlfsi_haven_usd.
- ``soft_stlfsi_regime``: soft_low_stlfsi + us_stlfsi_haven_usd (exclusive regimes:
  soft when z≤0, haven when z≥z_high).

PIT
---
``load_stlfsi_series`` / ``load_us_stlfsi_series`` apply ``pub_lag_days=7``
(weekly St. Louis Fed release family). This module collapses pub-lagged weekly
STLFSI4 to **month-end** (last obs/month), then adds ``signal_lag_months``
(default 1) on **monthly** trailing z (z_window=60m, min_periods=24 — mirror
CIP §71 / EPU §76 / WUI §85 / NFCI soft §90 / ANFCI soft §95, **not** 252d
daily) + ``weight_lag_days=1`` after month-end ffill to the trading calendar.
Soft legs keep source-wave PIT from §66 (costs 1.5 bps/side already inside
source factor returns). Haven companion applies costs 1.5 bps/side on its own
weights.

Gate / cool thresholds: fixed z (z_high=1.0 / gate ≤0) — mirror §71–§99.
Literature often uses other cutoffs; documented symmetry, **not** HO-tuned.

Distinct from: funding_liquidity_fx §20 (USD tilts + carry×NFCI/ANFCI cool as
*standalone* factors — **not** a soft_ew gate), soft_ew §66, NFCI soft §90,
ANFCI soft §95, CIP soft §71, VIX/GPR soft §72, EPU/TPU soft §76, WUI soft §85,
NFCI/ANFCI×carry/mom/value/REER §91–§99, CIP/VIX/EPU/WUI × carry/mom/value/REER
§68/§73–§89, capital-sleeve §53/§70, combo §8, raw ig_oas.
Explicit: do **not** overlay coolers on locked ``fx4plus_gbpcad_d1_voltarget_0025``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.data.fred_funding_liquidity import (
    DEFAULT_WEEKLY_PUB_LAG_DAYS,
    load_stlfsi_series,
)
from mt5_swing.strategies.carry_rank import portfolio_returns_from_weights
from mt5_swing.strategies.gpr_regime import USD_LONG_PAIRS, GprRegimeConfig, risk_scale_from_z
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class StlfsiConditionedSoftFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag_months: int = 1  # months after month-end STLFSI known
    weight_lag_days: int = 1  # trading days after month-end ffill
    z_window: int = 60  # months (~5y) — mirror CIP §71 / EPU §76 / WUI §85 / NFCI §90
    min_periods: int = 24
    z_high: float = 1.0  # cool / haven thresh — §71–§99 symmetry
    z_low: float = 0.0
    cool: float = 0.35
    usd_tilt: float = 0.5  # gross |w| when haven tilt is on
    cost_bps_side: float = 1.5  # haven costs; soft legs already costed
    pub_lag_days: int = DEFAULT_WEEKLY_PUB_LAG_DAYS  # informational (loader default 7)
    # Documented: lit often uses other cutoffs; this wave keeps 1.0 for symmetry
    lit_z_high_note: float = 1.5


PRIMARY = "soft_low_stlfsi"


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _month_start(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    return pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start"), tz="UTC")


def _month_end(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    return pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="end"), tz="UTC")


def trailing_z_monthly(
    series: pd.Series, *, lookback: int, min_periods: int
) -> pd.Series:
    """Trailing z on a monthly series (mirror CIP / EPU / WUI monthly z)."""
    s = series.astype(float)
    mu = s.rolling(lookback, min_periods=min_periods).mean()
    sd = s.rolling(lookback, min_periods=min_periods).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def weekly_stlfsi_to_month_end(series: pd.Series) -> pd.Series:
    """Collapse pub-lagged weekly STLFSI to one month-end observation (last/month)."""
    s = series.astype(float).copy()
    s.index = _ensure_utc(pd.DatetimeIndex(s.index))
    s = s.sort_index()
    # Map to month-start key, keep last weekly obs in each calendar month
    keyed = s.copy()
    keyed.index = _month_start(pd.DatetimeIndex(keyed.index))
    monthly = keyed[~keyed.index.duplicated(keep="last")].sort_index()
    monthly.index = _month_end(pd.DatetimeIndex(monthly.index))
    monthly.name = series.name or "STLFSI4"
    return monthly


def _scale_series(
    soft: pd.Series,
    scale: pd.Series,
    *,
    name: str,
) -> pd.Series:
    """Element-wise soft × scale (aligned); flat when scale=0 / soft NaN preserved."""
    soft = soft.astype(float).copy()
    soft.index = _ensure_utc(pd.DatetimeIndex(soft.index))
    sc = scale.reindex(soft.index).astype(float)
    out = soft * sc
    out = out.where(soft.notna())
    out = out.where(sc.notna(), 0.0)
    out.name = name
    return out


def align_monthly_stress_z_daily(
    series: pd.Series,
    index: pd.DatetimeIndex,
    *,
    cfg: StlfsiConditionedSoftFxConfig | None = None,
) -> pd.Series:
    """Weekly (or monthly) STLFSI → month-end → trailing monthly z → lag → daily.

    Mirrors ``align_cip_stress_z_daily`` / WUI §85: z_window months on the native
    monthly frequency — **not** blindly 252d after daily ffill.
    """
    cfg = cfg or StlfsiConditionedSoftFxConfig()
    # If already month-end / monthly, weekly_stlfsi_to_month_end is idempotent
    # (one obs/month → same). Weekly series collapse to last-in-month.
    s = weekly_stlfsi_to_month_end(series)
    # Align z on month-start keys for rolling (same as WUI path)
    s_ms = s.copy()
    s_ms.index = _month_start(pd.DatetimeIndex(s_ms.index))
    s_ms = s_ms[~s_ms.index.duplicated(keep="last")].sort_index()
    z = trailing_z_monthly(s_ms, lookback=cfg.z_window, min_periods=cfg.min_periods)
    if cfg.signal_lag_months > 0:
        z = z.shift(int(cfg.signal_lag_months))
    z_me = z.copy()
    z_me.index = _month_end(pd.DatetimeIndex(z_me.index))
    pair_index = _ensure_utc(pd.DatetimeIndex(index))
    z_daily = z_me.reindex(pair_index, method="ffill")
    if cfg.weight_lag_days > 0:
        z_daily = z_daily.shift(int(cfg.weight_lag_days))
    z_daily.name = f"{series.name or 'STLFSI4'}_z"
    return z_daily


def load_us_stlfsi_series(
    *,
    series_id: str = "STLFSI4",
    pub_lag_days: int = DEFAULT_WEEKLY_PUB_LAG_DAYS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """Thin wrapper — St. Louis Fed STLFSI4 (weekly, pub_lag_days applied)."""
    return load_stlfsi_series(
        series_id, download=download, pub_lag_days=pub_lag_days, force=force
    )


def usd_tilt_from_stlfsi_z(
    z_daily: pd.Series,
    pair_columns: list[str],
    *,
    cfg: StlfsiConditionedSoftFxConfig,
) -> pd.DataFrame:
    """Long USD when lagged STLFSI z ≥ z_high (haven / risk-off tilt)."""
    on = (z_daily >= float(cfg.z_high)).astype(float) * float(cfg.usd_tilt)
    on = on.where(z_daily.notna(), 0.0)
    cols = [c for c in pair_columns if c.upper() in {p.upper() for p in USD_LONG_PAIRS}]
    if not cols:
        cols = list(pair_columns)
    n = max(len(cols), 1)
    w = pd.DataFrame(0.0, index=z_daily.index, columns=list(pair_columns))
    for sym in cols:
        sign = USD_LONG_PAIRS.get(sym.upper(), 0)
        w[sym] = (on / n) * sign
    return w.fillna(0.0)


def _haven_returns(
    z_daily: pd.Series,
    pair_ret: pd.DataFrame,
    *,
    cfg: StlfsiConditionedSoftFxConfig,
) -> pd.Series:
    cols = list(pair_ret.columns)
    w = usd_tilt_from_stlfsi_z(z_daily, cols, cfg=cfg)
    for c in cols:
        if c not in w.columns:
            w[c] = 0.0
    w = w.reindex(columns=cols).fillna(0.0)
    r = portfolio_returns_from_weights(w, pair_ret)
    r = apply_costs(r, w, bps_side=cfg.cost_bps_side)
    r.name = "us_stlfsi_haven_usd"
    return r


def stlfsi_conditioned_soft_factor_returns(
    soft_ew: pd.Series,
    *,
    us_stlfsi: pd.Series | None = None,
    pair_ret: pd.DataFrame | None = None,
    cfg: StlfsiConditionedSoftFxConfig | None = None,
    include_haven: bool = True,
) -> dict[str, pd.Series]:
    """Build STLFSI4-conditioned soft-signal EW FX factor daily returns.

    ``soft_ew`` is the §66 ``soft_ew_macro5`` / ``soft_ew5`` daily return series
    (costs already applied inside source legs). Gating/cooling multiplies that
    series by US STLFSI4 stress gate/cool — distinct from NFCI soft §90, ANFCI
    soft §95, CIP soft §71, VIX/GPR soft §72, EPU/TPU soft §76, WUI soft §85,
    and funding_liquidity §20 (standalone NFCI/ANFCI USD tilts).
    """
    cfg = cfg or StlfsiConditionedSoftFxConfig()
    soft = soft_ew.astype(float).copy()
    soft.index = _ensure_utc(pd.DatetimeIndex(soft.index))
    soft.name = soft.name or "soft_ew_macro5"

    if us_stlfsi is None:
        us_stlfsi = load_us_stlfsi_series(pub_lag_days=cfg.pub_lag_days)

    z_daily = align_monthly_stress_z_daily(us_stlfsi, soft.index, cfg=cfg)

    out: dict[str, pd.Series] = {}

    raw = soft.copy()
    raw.name = "soft_raw"
    out["soft_raw"] = raw

    gcfg = GprRegimeConfig(
        z_high=cfg.z_high,
        z_low=cfg.z_low,
        cool=cfg.cool,
        signal_lag=0,  # z already lagged
        usd_tilt=0.0,
    )
    cool_scale = (
        risk_scale_from_z(z_daily, cfg=gcfg).reindex(soft.index).ffill().fillna(1.0)
    )
    out["soft_stlfsi_cool"] = _scale_series(soft, cool_scale, name="soft_stlfsi_cool")

    # Primary: trade soft only when STLFSI z ≤ 0
    gate_lo = (z_daily <= 0.0).astype(float)
    gate_lo = gate_lo.where(z_daily.notna(), 0.0).reindex(soft.index).fillna(0.0)
    out["soft_low_stlfsi"] = _scale_series(soft, gate_lo, name="soft_low_stlfsi")

    # Honesty inverse: soft only when STLFSI z ≥ z_high
    gate_hi = (z_daily >= float(cfg.z_high)).astype(float)
    gate_hi = gate_hi.where(z_daily.notna(), 0.0).reindex(soft.index).fillna(0.0)
    out["soft_high_stlfsi"] = _scale_series(soft, gate_hi, name="soft_high_stlfsi")

    if include_haven and pair_ret is not None and not pair_ret.empty:
        pret = pair_ret.copy()
        pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
        out["us_stlfsi_haven_usd"] = _haven_returns(z_daily, pret, cfg=cfg)

    # Stack: sequential gate then cool (soft_low × cool scale)
    stack_scale = (gate_lo * cool_scale).clip(lower=0.0, upper=1.0)
    out["soft_stlfsi_stack"] = _scale_series(soft, stack_scale, name="soft_stlfsi_stack")

    # EW of primary + cool + haven (§71 / §85 mirror)
    blend_keys = [
        k
        for k in ("soft_low_stlfsi", "soft_stlfsi_cool", "us_stlfsi_haven_usd")
        if k in out
    ]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "soft_stlfsi_ew"
        out["soft_stlfsi_ew"] = blend

    # Regime switch: soft when low-STLFSI + haven when high-STLFSI (exclusive)
    if "soft_low_stlfsi" in out and "us_stlfsi_haven_usd" in out:
        regime = out["soft_low_stlfsi"].fillna(0.0) + out["us_stlfsi_haven_usd"].fillna(0.0)
        both_nan = out["soft_low_stlfsi"].isna() & out["us_stlfsi_haven_usd"].isna()
        regime = regime.where(~both_nan)
        regime.name = "soft_stlfsi_regime"
        out["soft_stlfsi_regime"] = regime

    return out


__all__ = [
    "PRIMARY",
    "StlfsiConditionedSoftFxConfig",
    "align_monthly_stress_z_daily",
    "trailing_z_monthly",
    "weekly_stlfsi_to_month_end",
    "load_us_stlfsi_series",
    "usd_tilt_from_stlfsi_z",
    "stlfsi_conditioned_soft_factor_returns",
]
