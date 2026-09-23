"""BIS private credit-to-GDP / credit-gap FX factors (Borio–Drehmann / Basel).

Literature framing (see also ``fred_bis_credit`` docstring)
---------------------------------------------------------
- Borio & Drehmann / BIS early-warning: private credit/GDP and the credit gap
  (deviation from long-run trend) predict financial-cycle stress and FX premia.
- Primary: long **low** credit gap (or low credit/GDP) / short high — lean
  balance-sheet / lower cycle-stress prior.
- Honesty alternate: long **high** credit (debtor / fragile risk-premium sort).

Fixed priors (no holdout tuning)
--------------------------------
- ``low_credit_gap_xs`` (**primary**): long negative trailing-trend gap /
  short positive gap. Gap = credit/GDP − trailing mean over ``gap_lookback``
  months (default 180 ≈ 60 quarters; HP-like one-sided). Pre-computed FRED
  gap IDs 404 — computed causally here.
- ``low_credit_xs``: long low credit/GDP level / short high.
- ``high_credit_xs``: opposite debtor/fragile premium sort.
- ``low_credit_z_xs``: long low 60m trailing z of credit/GDP (5y z board row).
- ``credit_chg_xs``: long falling credit/GDP (Δ12 < 0) / short rising.
- ``us_credit_stress_fx``: elevated US credit gap z → long foreign (stress/
  external-adjustment prior).
- ``us_credit_haven_usd``: elevated US credit gap z → long USD (haven alt).
- ``credit_ew``: EW of ``low_credit_gap_xs``, ``credit_chg_xs``,
  ``us_credit_stress_fx``.

PIT: loader ``pub_lag_months`` (default 5) + ``signal_lag`` months + 1 trading
day weight lag. Distinct from debt (§29), fiscal (§28), CA, TB (§30), BIS REER
(§31), CB-BS, funding-liq, macro-diff, equity-diff, AI-GPR, ToT, dollar-beta.

Explicit: do **not** overlay on the locked FTMO sleeve.
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
from mt5_swing.strategies.gpr_regime import USD_LONG_PAIRS
from mt5_swing.strategies.yield_curve_fx import apply_costs


@dataclass
class BisCreditFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    signal_lag: int = 1  # months after publication-lagged credit is known
    n_long: int = 2
    n_short: int = 2
    z_window: int = 60  # months for trailing z (5y level z / US tilt)
    min_periods: int = 24
    gap_lookback: int = 180  # months ≈ 60 quarters trailing mean (HP-like)
    gap_min_periods: int = 60  # require ≥5y history before gap is defined
    z_high: float = 1.0  # |z| threshold for US credit-gap tilt
    usd_tilt: float = 0.5  # gross |w| when tilt is on
    cost_bps_side: float = 1.5
    chg_periods: int = 12  # YoY change on monthly-ffilled quarterly credit


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


def trailing_z(s: pd.Series, *, lookback: int, min_periods: int) -> pd.Series:
    mu = s.rolling(lookback, min_periods=min_periods).mean()
    sd = s.rolling(lookback, min_periods=min_periods).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def credit_gap_from_level(
    panel: pd.DataFrame,
    *,
    lookback: int,
    min_periods: int,
) -> pd.DataFrame:
    """Causal HP-like credit gap: level − trailing mean (one-sided)."""
    cols = {}
    for c in panel.columns:
        s = panel[c]
        trend = s.rolling(lookback, min_periods=min_periods).mean()
        cols[c] = s - trend
    return pd.DataFrame(cols, index=panel.index)


def prepare_credit_scores(
    credit_panel: pd.DataFrame,
    *,
    cfg: BisCreditFxConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build signed monthly score panels from PIT credit/GDP (already pub-lagged).

    Scores exclude USD column when present (XS sorts are foreign vs USD).
    """
    cfg = cfg or BisCreditFxConfig()
    panel = credit_panel.copy()
    panel.index = _month_start(pd.DatetimeIndex(panel.index))
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    foreign_cols = [c for c in panel.columns if c.upper() != "USD" and c in CURRENCY_USD_PAIR]
    foreign = panel[foreign_cols]

    out: dict[str, pd.DataFrame] = {}

    gap = credit_gap_from_level(
        foreign, lookback=cfg.gap_lookback, min_periods=cfg.gap_min_periods
    )
    # Low gap (negative / below trend) → high score → long
    low_gap = (-gap).copy()
    if cfg.signal_lag > 0:
        low_gap = low_gap.shift(int(cfg.signal_lag))
    out["low_credit_gap"] = low_gap

    # Low credit level: low credit/GDP → high score → long
    low = (-foreign).copy()
    if cfg.signal_lag > 0:
        low = low.shift(int(cfg.signal_lag))
    out["low_credit"] = low

    # High credit risk premium: high credit → high score → long
    high = foreign.copy()
    if cfg.signal_lag > 0:
        high = high.shift(int(cfg.signal_lag))
    out["high_credit"] = high

    # 5y trailing z: low z → long
    z = pd.DataFrame(
        {
            c: trailing_z(foreign[c], lookback=cfg.z_window, min_periods=cfg.min_periods)
            for c in foreign.columns
        },
        index=foreign.index,
    )
    low_z = (-z).copy()
    if cfg.signal_lag > 0:
        low_z = low_z.shift(int(cfg.signal_lag))
    out["low_credit_z"] = low_z

    # Change: improving = falling credit → long (score = −Δ credit)
    chg = (-foreign.diff(int(cfg.chg_periods))).copy()
    if cfg.signal_lag > 0:
        chg = chg.shift(int(cfg.signal_lag))
    out["credit_chg"] = chg

    return out


def _scores_to_daily_returns(
    score: pd.DataFrame,
    pair_ret: pd.DataFrame,
    cfg: BisCreditFxConfig,
    *,
    name: str,
) -> pd.Series:
    ccy_w = _rank_sort_weights(score, n_long=cfg.n_long, n_short=cfg.n_short)
    if ccy_w.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)
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
    daily_w = expand_monthly_weights_to_daily(pair_w, pair_ret.index, signal_lag_days=1)
    for c in pair_ret.columns:
        if c not in daily_w.columns:
            daily_w[c] = 0.0
    daily_w = daily_w.reindex(columns=list(pair_ret.columns)).fillna(0.0)
    r = portfolio_returns_from_pair_weights(daily_w, pair_ret)
    r = apply_costs(r, daily_w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


def _us_credit_tilt_returns(
    us_credit: pd.Series,
    pair_ret: pd.DataFrame,
    *,
    cfg: BisCreditFxConfig,
    mode: str,
    name: str,
) -> pd.Series:
    """Binary USD / FX tilt from lagged US credit-gap z.

    mode='stress': z ≥ +z_high → long foreign (short USD) — cycle-stress adj.
    mode='haven': z ≥ +z_high → long USD — safe-haven alternate.
    """
    s = us_credit.copy()
    s.index = _month_start(pd.DatetimeIndex(s.index))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    gap = s - s.rolling(cfg.gap_lookback, min_periods=cfg.gap_min_periods).mean()
    z = trailing_z(gap, lookback=cfg.z_window, min_periods=cfg.min_periods)
    if cfg.signal_lag > 0:
        z = z.shift(int(cfg.signal_lag))

    z_me = z.copy()
    z_me.index = (
        pd.DatetimeIndex(z_me.index)
        .tz_convert(None)
        .to_period("M")
        .to_timestamp(how="end")
        .tz_localize("UTC")
    )
    z_daily = z_me.reindex(pair_ret.index, method="ffill").shift(1)

    on = (z_daily >= float(cfg.z_high)).astype(float)
    on = on.where(z_daily.notna(), 0.0)
    intensity = on * float(cfg.usd_tilt)

    cols = [c for c in pair_ret.columns if c.upper() in {p.upper() for p in USD_LONG_PAIRS}]
    if not cols:
        cols = list(pair_ret.columns)
    n = max(len(cols), 1)
    w = pd.DataFrame(0.0, index=pair_ret.index, columns=list(pair_ret.columns))
    for sym in cols:
        usd_sign = USD_LONG_PAIRS.get(sym.upper(), 0)
        if mode == "stress":
            w[sym] = (intensity / n) * (-usd_sign)
        else:
            w[sym] = (intensity / n) * usd_sign
    r = portfolio_returns_from_pair_weights(w, pair_ret)
    r = apply_costs(r, w, bps_side=cfg.cost_bps_side)
    r.name = name
    return r


def bis_credit_factor_returns(
    pair_ret: pd.DataFrame,
    credit_panel: pd.DataFrame,
    *,
    cfg: BisCreditFxConfig | None = None,
    us_credit_override: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Daily returns for BIS credit/GDP factors + US tilts + EW."""
    cfg = cfg or BisCreditFxConfig()
    scores = prepare_credit_scores(credit_panel, cfg=cfg)
    factors: dict[str, pd.Series] = {}

    if "low_credit_gap" in scores:
        factors["low_credit_gap_xs"] = _scores_to_daily_returns(
            scores["low_credit_gap"], pair_ret, cfg, name="low_credit_gap_xs"
        )
    if "low_credit" in scores:
        factors["low_credit_xs"] = _scores_to_daily_returns(
            scores["low_credit"], pair_ret, cfg, name="low_credit_xs"
        )
    if "high_credit" in scores:
        factors["high_credit_xs"] = _scores_to_daily_returns(
            scores["high_credit"], pair_ret, cfg, name="high_credit_xs"
        )
    if "low_credit_z" in scores:
        factors["low_credit_z_xs"] = _scores_to_daily_returns(
            scores["low_credit_z"], pair_ret, cfg, name="low_credit_z_xs"
        )
    if "credit_chg" in scores:
        factors["credit_chg_xs"] = _scores_to_daily_returns(
            scores["credit_chg"], pair_ret, cfg, name="credit_chg_xs"
        )

    us = us_credit_override
    if us is None and "USD" in credit_panel.columns:
        us = credit_panel["USD"]
    if us is not None and us.dropna().shape[0] >= cfg.gap_min_periods:
        factors["us_credit_stress_fx"] = _us_credit_tilt_returns(
            us, pair_ret, cfg=cfg, mode="stress", name="us_credit_stress_fx"
        )
        factors["us_credit_haven_usd"] = _us_credit_tilt_returns(
            us, pair_ret, cfg=cfg, mode="haven", name="us_credit_haven_usd"
        )

    blend_keys = [
        k
        for k in ("low_credit_gap_xs", "credit_chg_xs", "us_credit_stress_fx")
        if k in factors
    ]
    if blend_keys:
        ew = sum(factors[k] for k in blend_keys) / len(blend_keys)
        ew.name = "credit_ew"
        factors["credit_ew"] = ew

    return factors
