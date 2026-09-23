"""Swap- / forward-aware FX carry from free money-market rate proxies.

Literature (Lustig–Roussanov–Verdelhan 2011; Lustig–Verdelhan 2007; Menkhoff
et al. 2012a; Fama 1984): currencies are sorted on *forward discounts*. Under
CIP those discounts equal interest differentials. We implement:

- ``carry_ir3m_xs`` — HML-FX style long-high / short-low on OECD 3M differentials
  (primary free *forward-proxy*; closer tenor than immediate cash rates).
- ``carry_irstci_xs`` — same sort on OECD immediate rates (cash-rate baseline).
- ``carry_cip_fd_xs`` — sort on exact discrete CIP-implied FD from IR3M.
- ``carry_ir3m_ew`` — equal-weight signed carry (all +diffs long / −diffs short).
- ``carry_blend_xs`` — EW of IR3M + IRSTCI XS returns.
- ``carry_ir3m_tc5`` — same IR3M XS with **5 bps/side** TC haircut (vs 1.5).

Frozen priors: ``pub_lag`` in loaders + ``signal_lag`` months + 1 trading-day
weight lag. No holdout tuning. Explicit: still cash / MM approximation — **not**
Bloomberg FX swap or outright forward points.
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
from mt5_swing.strategies.yield_curve_fx import apply_costs, apply_signal_lag


@dataclass
class ForwardCarryFxConfig:
    signal_lag: int = 1  # months after publication-lagged rates are known
    n_long: int = 2
    n_short: int = 2
    cost_bps_side: float = 1.5
    tc_haircut_bps: float = 5.0  # honest higher-cost variant
    build_ir3m_xs: bool = True
    build_irstci_xs: bool = True
    build_cip_fd_xs: bool = True
    build_ir3m_ew: bool = True
    build_blend: bool = True
    build_tc_haircut: bool = True


def _month_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    naive = idx.tz_convert(None) if getattr(idx, "tz", None) is not None else idx
    return (
        pd.DatetimeIndex(naive)
        .to_period("M")
        .to_timestamp(how="end")
        .tz_localize("UTC")
    )


def rank_sort_weights(
    score: pd.DataFrame,
    *,
    n_long: int,
    n_short: int,
) -> pd.DataFrame:
    """Long high score / short low; each side |sum| = 0.5."""
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


def ew_signed_weights(score: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight signed carry: +diffs long / −diffs short; sides sum ±0.5.

    Currencies with zero / NaN score are flat. If one side is empty, that month
    is skipped (no invented USD residual trade).
    """
    rows = []
    cols = list(score.columns)
    for dt, row in score.iterrows():
        s = row.dropna()
        pos = s[s > 0]
        neg = s[s < 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        w = pd.Series(0.0, index=cols)
        w.loc[pos.index] = 0.5 / len(pos)
        w.loc[neg.index] = -0.5 / len(neg)
        w.name = dt
        rows.append(w)
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_index()


def _scores_to_daily_returns(
    score: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    name: str,
    n_long: int,
    n_short: int,
    cost_bps: float,
    mode: str = "xs",
) -> tuple[pd.Series, pd.DataFrame]:
    if mode == "ew":
        ccy_w = ew_signed_weights(score)
    else:
        ccy_w = rank_sort_weights(score, n_long=n_long, n_short=n_short)
    if ccy_w.empty:
        empty = pd.Series(0.0, index=pair_ret.index, name=name)
        return empty, pd.DataFrame(0.0, index=pair_ret.index, columns=[])
    ccy_w.index = _month_end_index(pd.DatetimeIndex(ccy_w.index))
    keep = [c for c in ccy_w.columns if c in CURRENCY_USD_PAIR]
    ccy_w = ccy_w[keep]
    pair_w = currency_weights_to_pair_weights_fx(ccy_w)
    daily_w = expand_monthly_weights_to_daily(pair_w, pair_ret.index, signal_lag_days=1)
    r = portfolio_returns_from_pair_weights(daily_w, pair_ret)
    r.name = name
    r = apply_costs(r, daily_w, bps_side=cost_bps)
    r.name = name
    return r, daily_w


def prepare_forward_carry_scores(
    ir3m_diff: pd.DataFrame | None,
    irstci_diff: pd.DataFrame | None,
    cip_fd: pd.DataFrame | None,
    *,
    cfg: ForwardCarryFxConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Monthly score panels after ``signal_lag`` (pub_lag already in loaders)."""
    cfg = cfg or ForwardCarryFxConfig()
    out: dict[str, pd.DataFrame] = {}
    if cfg.build_ir3m_xs and ir3m_diff is not None and not ir3m_diff.empty:
        out["carry_ir3m_xs"] = apply_signal_lag(ir3m_diff, cfg.signal_lag)
    if cfg.build_irstci_xs and irstci_diff is not None and not irstci_diff.empty:
        out["carry_irstci_xs"] = apply_signal_lag(irstci_diff, cfg.signal_lag)
    if cfg.build_cip_fd_xs and cip_fd is not None and not cip_fd.empty:
        out["carry_cip_fd_xs"] = apply_signal_lag(cip_fd, cfg.signal_lag)
    if cfg.build_ir3m_ew and ir3m_diff is not None and not ir3m_diff.empty:
        out["carry_ir3m_ew"] = apply_signal_lag(ir3m_diff, cfg.signal_lag)
    return out


def forward_carry_factor_returns(
    pair_ret: pd.DataFrame,
    *,
    ir3m_diff: pd.DataFrame | None = None,
    irstci_diff: pd.DataFrame | None = None,
    cip_fd: pd.DataFrame | None = None,
    cfg: ForwardCarryFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Daily net-of-cost returns for forward-proxy carry legs."""
    cfg = cfg or ForwardCarryFxConfig()
    scores = prepare_forward_carry_scores(
        ir3m_diff, irstci_diff, cip_fd, cfg=cfg
    )
    factors: dict[str, pd.Series] = {}

    for name, sc in scores.items():
        mode = "ew" if name.endswith("_ew") else "xs"
        r, _ = _scores_to_daily_returns(
            sc,
            pair_ret,
            name=name,
            n_long=cfg.n_long,
            n_short=cfg.n_short,
            cost_bps=cfg.cost_bps_side,
            mode=mode,
        )
        factors[name] = r

    # Higher TC haircut on primary IR3M XS (same weights / signal, more cost)
    if cfg.build_tc_haircut and "carry_ir3m_xs" in scores:
        r_tc, _ = _scores_to_daily_returns(
            scores["carry_ir3m_xs"],
            pair_ret,
            name="carry_ir3m_tc5",
            n_long=cfg.n_long,
            n_short=cfg.n_short,
            cost_bps=cfg.tc_haircut_bps,
            mode="xs",
        )
        factors["carry_ir3m_tc5"] = r_tc

    if cfg.build_blend:
        legs = [factors[k] for k in ("carry_ir3m_xs", "carry_irstci_xs") if k in factors]
        if legs:
            blend = sum(legs) / len(legs)
            blend.name = "carry_blend_xs"
            factors["carry_blend_xs"] = blend

    return factors


def _spearman_corr(x: pd.Series, y: pd.Series) -> float:
    """Spearman via rank+Pearson (no scipy dependency)."""
    if len(x) < 4:
        return float("nan")
    rx = x.rank()
    ry = y.rank()
    if float(rx.std(ddof=0)) == 0.0 or float(ry.std(ddof=0)) == 0.0:
        return float("nan")
    return float(rx.corr(ry, method="pearson"))


def score_rank_agreement(
    a: pd.DataFrame,
    b: pd.DataFrame,
) -> float:
    """Mean monthly Spearman correlation between two score panels (audit)."""
    common_cols = sorted(set(a.columns) & set(b.columns))
    common_idx = a.dropna(how="all").index.intersection(b.dropna(how="all").index)
    if not common_cols or len(common_idx) < 6:
        return float("nan")
    cors = []
    for dt in common_idx:
        x = a.loc[dt, common_cols]
        y = b.loc[dt, common_cols]
        m = x.notna() & y.notna()
        if int(m.sum()) < 3:
            continue
        c = _spearman_corr(x[m], y[m])
        if np.isfinite(c):
            cors.append(c)
    return float(np.nanmean(cors)) if cors else float("nan")
