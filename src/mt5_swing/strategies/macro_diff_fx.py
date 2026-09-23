"""Dahlquist-style FX sorts on FRED macro differentials (free data only).

Currency scores use publication-lagged differentials vs USD:
- Inflation (CPI YoY): high relative inflation → short foreign (PPP / carry-risk)
- Industrial production YoY: high relative activity → long foreign
- Unemployment: high relative UR → short foreign (slack / risk)

Each factor is sorted long/short equal-weight; optional equal-weight blend of
available factors. All signals use ``signal_lag`` months on top of loader pub lags.
No holdout tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.data.macro_uncertainty import CURRENCY_USD_PAIR
from mt5_swing.strategies.country_gpr_fx import (
    currency_weights_to_pair_weights_fx,
    expand_monthly_weights_to_daily,
    portfolio_returns_from_pair_weights,
)


@dataclass
class MacroDiffFxConfig:
    signal_lag: int = 1  # months after publication-lagged macro is known
    n_long: int = 2
    n_short: int = 2
    # factor signs applied to (foreign − USD): positive score → long foreign
    # inflation: higher foreign inflation → negative score
    # ip: higher foreign IP growth → positive
    # ur: higher foreign unemployment → negative
    use_cpi: bool = True
    use_ip: bool = True
    use_ur: bool = True


def _month_start(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    return pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start"), tz="UTC")


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
        ranked = s.sort_values()  # low first
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


def prepare_macro_scores(
    cpi_diff: pd.DataFrame | None,
    ip_diff: pd.DataFrame | None,
    ur_diff: pd.DataFrame | None,
    *,
    cfg: MacroDiffFxConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build signed monthly score panels (already pub-lagged in loaders)."""
    cfg = cfg or MacroDiffFxConfig()
    out: dict[str, pd.DataFrame] = {}
    if cfg.use_cpi and cpi_diff is not None and not cpi_diff.empty:
        s = (-cpi_diff).copy()  # high inflation → low score → short
        s.index = _month_start(pd.DatetimeIndex(s.index))
        if cfg.signal_lag > 0:
            s = s.shift(int(cfg.signal_lag))
        out["cpi"] = s
    if cfg.use_ip and ip_diff is not None and not ip_diff.empty:
        s = ip_diff.copy()
        s.index = _month_start(pd.DatetimeIndex(s.index))
        if cfg.signal_lag > 0:
            s = s.shift(int(cfg.signal_lag))
        out["ip"] = s
    if cfg.use_ur and ur_diff is not None and not ur_diff.empty:
        s = (-ur_diff).copy()
        s.index = _month_start(pd.DatetimeIndex(s.index))
        if cfg.signal_lag > 0:
            s = s.shift(int(cfg.signal_lag))
        out["ur"] = s
    return out


def _scores_to_daily_returns(score: pd.DataFrame, pair_ret: pd.DataFrame, cfg: MacroDiffFxConfig) -> pd.Series:
    ccy_w = _rank_sort_weights(score, n_long=cfg.n_long, n_short=cfg.n_short)
    if ccy_w.empty:
        return pd.Series(0.0, index=pair_ret.index, name="macro_diff")
    ccy_w.index = (
        pd.DatetimeIndex(ccy_w.index)
        .tz_convert(None)
        .to_period("M")
        .to_timestamp(how="end")
        .tz_localize("UTC")
    )
    # keep only currencies we can map
    keep = [c for c in ccy_w.columns if c in CURRENCY_USD_PAIR]
    ccy_w = ccy_w[keep]
    pair_w = currency_weights_to_pair_weights_fx(ccy_w)
    daily_w = expand_monthly_weights_to_daily(pair_w, pair_ret.index, signal_lag_days=1)
    return portfolio_returns_from_pair_weights(daily_w, pair_ret)


def macro_diff_factor_returns(
    pair_ret: pd.DataFrame,
    cpi_diff: pd.DataFrame | None,
    ip_diff: pd.DataFrame | None,
    ur_diff: pd.DataFrame | None,
    *,
    cfg: MacroDiffFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Daily returns for each macro-diff sort + equal-weight blend of available."""
    cfg = cfg or MacroDiffFxConfig()
    scores = prepare_macro_scores(cpi_diff, ip_diff, ur_diff, cfg=cfg)
    factors: dict[str, pd.Series] = {}
    for name, sc in scores.items():
        r = _scores_to_daily_returns(sc, pair_ret, cfg)
        r.name = f"macro_{name}"
        factors[r.name] = r
    if factors:
        blend = sum(factors.values()) / len(factors)
        blend.name = "macro_diff_ew"
        factors["macro_diff_ew"] = blend
    return factors
