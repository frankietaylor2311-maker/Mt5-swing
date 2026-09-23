"""Baker–Bloom–Davis EPU / TPU → FX factors (frozen scholarly construction).

Literature priors (not holdout-tuned)
------------------------------------
- Baker, Bloom & Davis (2016) QJE: news-based Economic Policy Uncertainty (EPU).
- Categorical *Trade policy* EPU = US Trade Policy Uncertainty (TPU) on
  policyuncertainty.com — distinct from aggregate EPU / VIX / GPR.
- High *home-country* EPU is associated with subsequent currency depreciation
  vs USD (risk premium / capital flight), analogous to country-GPR sorts.
- High *US* EPU / TPU / GEPU episodes coincide with risk-off / safe-haven USD
  demand; carry compresses in uncertainty regimes (related Menkhoff FX-vol work).

Legs
----
- ``country_epu_xs``: long low / short high home-country EPU z (primary XS).
- ``epu_diff_xs``: sort on (home EPU − US EPU) — relative policy uncertainty.
- ``epu_us_usd``: long USD when lagged US EPU z ≥ z_high (binary tilt).
- ``tpu_us_usd``: long USD when lagged TPU z ≥ z_high.
- ``gepu_usd``: long USD when lagged GEPU z ≥ z_high.
- ``carry_epu_cool``: scholarly carry scaled by US-EPU risk_scale ∈ [cool, 1]
  (**EPU-only** — no VIX/GPR, distinct from prior combo / FX-RV waves).
- ``carry_tpu_cool``: carry scaled by TPU-only risk_scale.
- ``epu_ew``: equal-weight of ``country_epu_xs`` + ``epu_us_usd``.

PIT: loaders apply ``pub_lag_months=1``; this module adds ``signal_lag`` months
on monthly scores plus 1 trading-day lag on daily weight expansion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.data.macro_uncertainty import CURRENCY_USD_PAIR
from mt5_swing.strategies.carry_rank import (
    CarryRankConfig,
    carry_weights_from_rates,
    currency_weights_to_pair_weights,
    expand_weights_to_daily,
    portfolio_returns_from_weights,
)
from mt5_swing.strategies.country_gpr_fx import (
    currency_weights_to_pair_weights_fx,
    expand_monthly_weights_to_daily,
    portfolio_returns_from_pair_weights,
)
from mt5_swing.strategies.gpr_regime import (
    GprRegimeConfig,
    USD_LONG_PAIRS,
    align_macro_to_index,
    risk_scale_from_z,
)
from mt5_swing.strategies.yield_curve_fx import apply_costs


G10_EPU_CCY: tuple[str, ...] = ("EUR", "GBP", "AUD", "JPY", "CAD")  # CHF/NZD absent


@dataclass
class EpuTpuFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag_months: int = 1  # months after publication-lagged index is known
    signal_lag_days: int = 1  # trading days on daily expansion
    n_long: int = 2
    n_short: int = 2
    z_window: int = 60  # months for country / aggregate z
    min_periods: int = 24
    z_high: float = 1.0
    z_low: float = 0.0
    cool: float = 0.35
    usd_tilt: float = 0.15
    cost_bps_side: float = 1.5
    exclude_currencies: tuple[str, ...] = ("USD",)  # never long/short USD in XS
    carry_n_long: int = 2
    carry_n_short: int = 2
    daily_z_window: int = 252  # for US EPU/TPU/GEPU aligned to trading calendar
    daily_min_periods: int = 60


def trailing_z(panel: pd.DataFrame | pd.Series, *, lookback: int, min_periods: int):
    if isinstance(panel, pd.Series):
        mu = panel.rolling(lookback, min_periods=min_periods).mean()
        sd = panel.rolling(lookback, min_periods=min_periods).std()
        return (panel - mu) / sd.replace(0.0, np.nan)
    mu = panel.rolling(lookback, min_periods=min_periods).mean()
    sd = panel.rolling(lookback, min_periods=min_periods).std()
    return (panel - mu) / sd.replace(0.0, np.nan)


def _align_month_start(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    return pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start"), tz="UTC")


def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def prepare_country_epu_signal(
    country_epu: pd.DataFrame,
    *,
    cfg: EpuTpuFxConfig | None = None,
    us_epu: pd.Series | None = None,
    relative: bool = False,
) -> pd.DataFrame:
    """Publication-lagged country EPU → trailing z → signal_lag months.

    If ``relative``, score = z(home) − z(US) using US column or ``us_epu``.
    """
    cfg = cfg or EpuTpuFxConfig()
    g = country_epu.copy()
    g.index = _align_month_start(pd.DatetimeIndex(g.index))
    g = g[~g.index.duplicated(keep="last")].sort_index()
    z = trailing_z(g, lookback=cfg.z_window, min_periods=cfg.min_periods)
    if relative:
        if "USD" in z.columns:
            us_z = z["USD"]
        elif us_epu is not None and len(us_epu):
            u = us_epu.copy()
            u.index = _align_month_start(pd.DatetimeIndex(u.index))
            u = u[~u.index.duplicated(keep="last")].sort_index()
            us_z = trailing_z(u, lookback=cfg.z_window, min_periods=cfg.min_periods)
            us_z = us_z.reindex(z.index)
        else:
            raise ValueError("relative country EPU needs USD column or us_epu series")
        sig = z.sub(us_z, axis=0)
        # Drop USD from relative sort universe
        if "USD" in sig.columns:
            sig = sig.drop(columns=["USD"])
    else:
        sig = z
    if cfg.signal_lag_months > 0:
        sig = sig.shift(int(cfg.signal_lag_months))
    sig.attrs["signal_lag_months"] = int(cfg.signal_lag_months)
    sig.attrs["relative"] = bool(relative)
    return sig


def _rank_sort_weights(
    score: pd.DataFrame,
    *,
    n_long: int,
    n_short: int,
    exclude: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Long low score / short high score (high EPU → short FX). Each side |sum|=0.5."""
    excl = {x.upper() for x in exclude}
    cols = [c for c in score.columns if c.upper() not in excl]
    rows = []
    for dt, row in score[cols].iterrows():
        s = row.dropna()
        if len(s) < n_long + n_short:
            continue
        ranked = s.sort_values()  # low EPU first
        longs = ranked.index[:n_long]
        shorts = ranked.index[-n_short:]
        w = pd.Series(0.0, index=cols)
        w.loc[list(longs)] = 0.5 / n_long
        w.loc[list(shorts)] = -0.5 / n_short
        w.name = dt
        rows.append(w)
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_index()


def _scores_to_daily_returns(
    score: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cfg: EpuTpuFxConfig,
    *,
    name: str,
) -> pd.Series:
    ccy_w = _rank_sort_weights(
        score,
        n_long=cfg.n_long,
        n_short=cfg.n_short,
        exclude=cfg.exclude_currencies,
    )
    if ccy_w.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    keep = [c for c in ccy_w.columns if c in CURRENCY_USD_PAIR]
    ccy_w = ccy_w[keep]
    ccy_w.index = _ensure_utc(pd.DatetimeIndex(ccy_w.index))
    pair_w = currency_weights_to_pair_weights_fx(ccy_w)
    daily_w = expand_monthly_weights_to_daily(
        pair_w,
        pair_ret.index,
        signal_lag_days=cfg.signal_lag_days,
    )
    r = portfolio_returns_from_pair_weights(daily_w, pair_ret)
    r = apply_costs(r, daily_w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


def _macro_daily_z(
    series: pd.Series,
    index: pd.DatetimeIndex,
    *,
    cfg: EpuTpuFxConfig,
) -> pd.Series:
    """Align monthly macro → daily ffill, trailing daily z, then signal_lag_days."""
    aligned = align_macro_to_index(series, index)
    z = trailing_z(aligned, lookback=cfg.daily_z_window, min_periods=cfg.daily_min_periods)
    if cfg.signal_lag_days > 0:
        z = z.shift(int(cfg.signal_lag_days))
    z.name = (series.name or "macro") + "_z"
    return z


def usd_tilt_from_z(
    z: pd.Series,
    pair_columns: list[str],
    *,
    cfg: EpuTpuFxConfig,
    binary: bool = True,
) -> pd.DataFrame:
    """Long-USD weights when z elevated (binary at z_high or continuous intensity)."""
    if binary:
        intensity = (z >= cfg.z_high).astype(float) * float(cfg.usd_tilt)
        intensity = intensity.where(z.notna(), 0.0)
    else:
        intensity = ((z - 0.0) / max(cfg.z_high, 1e-6)).clip(0, 1) * float(cfg.usd_tilt)
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


def epu_tpu_factor_returns(
    pair_ret: pd.DataFrame,
    *,
    country_epu: pd.DataFrame | None = None,
    us_epu: pd.Series | None = None,
    tpu: pd.Series | None = None,
    gepu: pd.Series | None = None,
    rates: pd.DataFrame | None = None,
    cfg: EpuTpuFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Build EPU/TPU FX factor daily returns (post optional costs)."""
    cfg = cfg or EpuTpuFxConfig()
    pret = pair_ret.copy()
    pret.index = _ensure_utc(pd.DatetimeIndex(pret.index))
    cols = list(pret.columns)
    out: dict[str, pd.Series] = {}

    # --- Cross-sectional country EPU ---
    if country_epu is not None and not country_epu.empty:
        sig = prepare_country_epu_signal(country_epu, cfg=cfg, relative=False)
        out["country_epu_xs"] = _scores_to_daily_returns(sig, pret, cfg, name="country_epu_xs")
        try:
            sig_rel = prepare_country_epu_signal(
                country_epu, cfg=cfg, us_epu=us_epu, relative=True
            )
            out["epu_diff_xs"] = _scores_to_daily_returns(sig_rel, pret, cfg, name="epu_diff_xs")
        except ValueError:
            pass

    # --- Aggregate USD tilts ---
    for name, series, key in (
        ("epu_us_usd", us_epu, "epu_us_usd"),
        ("tpu_us_usd", tpu, "tpu_us_usd"),
        ("gepu_usd", gepu, "gepu_usd"),
    ):
        if series is None or not len(series):
            continue
        z = _macro_daily_z(series, pret.index, cfg=cfg)
        w = usd_tilt_from_z(z, cols, cfg=cfg, binary=True)
        for c in cols:
            if c not in w.columns:
                w[c] = 0.0
        w = w.reindex(columns=cols).fillna(0.0)
        r = portfolio_returns_from_weights(w, pret)
        # portfolio_returns_from_weights already uses w.shift(1); z already day-lagged
        r = apply_costs(r, w, bps_side=cfg.cost_bps_side)
        r.name = key
        out[key] = r

    # --- Carry cool (EPU-only / TPU-only) ---
    if rates is not None and not rates.empty:
        ccfg = CarryRankConfig(
            n_long=cfg.carry_n_long,
            n_short=cfg.carry_n_short,
            signal_lag=cfg.signal_lag_days,
        )
        ccy_w = carry_weights_from_rates(rates, cfg=ccfg)
        carry_pw = currency_weights_to_pair_weights(ccy_w)
        carry_daily = expand_weights_to_daily(
            carry_pw, pret.index, signal_lag=cfg.signal_lag_days
        )
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
            signal_lag=0,
            usd_tilt=0.0,
        )
        for tag, series in (("epu", us_epu), ("tpu", tpu)):
            if series is None or not len(series):
                continue
            z = _macro_daily_z(series, pret.index, cfg=cfg)
            scale = risk_scale_from_z(z, cfg=gcfg).reindex(pret.index).ffill().fillna(1.0)
            cooled = carry_daily.mul(scale, axis=0)
            r = portfolio_returns_from_weights(cooled, pret)
            r = apply_costs(r, cooled, bps_side=cfg.cost_bps_side)
            r.name = f"carry_{tag}_cool"
            out[r.name] = r

    # EW blend
    blend_keys = [k for k in ("country_epu_xs", "epu_us_usd") if k in out]
    if len(blend_keys) >= 2:
        blend = pd.concat([out[k] for k in blend_keys], axis=1).mean(axis=1)
        blend.name = "epu_ew"
        out["epu_ew"] = blend

    return out
