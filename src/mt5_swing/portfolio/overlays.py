"""Causal portfolio overlays: clock budgets, corr, max-concurrent + idle recycle.

All time-varying scales use information from t-1 (or earlier) to size t.
Exposure after recycle is clipped so sum(weights) <= 1 (no extra leverage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INITIAL = 100_000.0


def apply_vol_target(
    port: pd.Series,
    target: float,
    look: int = 60,
    lo: float = 0.25,
    hi: float = 3.0,
) -> pd.Series:
    """Scale next-bar returns by target / trailing vol (lagged)."""
    if port is None or len(port) < 5:
        return port if port is not None else pd.Series(dtype=float)
    r = port.pct_change()
    trail = r.shift(1).rolling(look, min_periods=max(20, look // 3)).std()
    scale = (target / trail.replace(0, np.nan)).clip(lo, hi).fillna(1.0)
    return (1.0 + r.fillna(0) * scale).cumprod() * float(port.iloc[0])


def daily_return_corr(a: pd.Series, b: pd.Series, min_days: int = 20) -> float:
    """Pearson corr of daily pct-changes. Fail-closed to 1.0 if too short."""
    if a is None or b is None or a.empty or b.empty:
        return 1.0
    da = a.resample("1D").last().pct_change()
    db = b.resample("1D").last().pct_change()
    both = pd.concat([da.rename("a"), db.rename("b")], axis=1).dropna()
    if len(both) < min_days:
        return 1.0
    c = float(both["a"].corr(both["b"]))
    if c != c:
        return 1.0
    return c


def clock_budget_weights(
    timeframes: list[str],
    base_weights: np.ndarray,
    h4_share: float,
) -> np.ndarray:
    """Split a unit budget between H4 and D1; keep relative weights within clock.

    If only one clock is present, returns renormalized base_weights (share ignored).
    """
    n = len(timeframes)
    w = np.asarray(base_weights, dtype=float).reshape(-1)
    if w.size != n:
        raise ValueError("base_weights length must match timeframes")
    w = np.maximum(w, 0.0)
    tfs = [str(t).upper() for t in timeframes]
    h4 = np.array([tf == "H4" for tf in tfs])
    d1 = np.array([tf == "D1" for tf in tfs])
    out = np.zeros(n, dtype=float)
    hs = float(np.clip(h4_share, 0.0, 1.0))
    if h4.any() and d1.any():
        hw, dw = w[h4], w[d1]
        hs_sum = hw.sum()
        ds_sum = dw.sum()
        if hs_sum <= 0:
            hw = np.ones_like(hw)
            hs_sum = hw.sum()
        if ds_sum <= 0:
            dw = np.ones_like(dw)
            ds_sum = dw.sum()
        out[h4] = (hw / hs_sum) * hs
        out[d1] = (dw / ds_sum) * (1.0 - hs)
    else:
        s = w.sum()
        out = (w / s) if s > 0 else np.ones(n) / n
    tot = out.sum()
    if tot <= 0:
        return np.ones(n) / n
    return out / tot


def occupancy_frame(positions: list[pd.Series], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex positions onto a common clock and ffill occupancy (0/1)."""
    cols = []
    for i, p in enumerate(positions):
        s = p.reindex(index).ffill().fillna(0.0)
        cols.append((s.abs() > 0).astype(float).rename(f"p{i}"))
    if not cols:
        return pd.DataFrame(index=index)
    return pd.concat(cols, axis=1)


def causal_max_concurrent_recycle(
    returns: pd.DataFrame,
    occupancy: pd.DataFrame,
    weights: np.ndarray,
    *,
    max_k: int | None = None,
    recycle_cap: float = 1.0,
    priority: np.ndarray | None = None,
    initial: float = INITIAL,
) -> pd.Series:
    """Mix lagged-occupancy legs, cap concurrent, recycle idle weight.

    - Occupancy is shifted by 1 bar so today's mix cannot see today's fills.
    - Among occupied legs, keep at most ``max_k`` by a-priori ``priority``.
    - Idle / dropped weight is recycled into keepers: target exposure =
      min(1, recycle_cap * keeper_weight_sum). ``recycle_cap=1`` is no recycle.
    - Combined exposure never exceeds 1 (no leverage vs fully-invested basket).
    """
    if returns.empty:
        return pd.Series(dtype=float)
    rets = returns.sort_index()
    occ = occupancy.reindex(rets.index).fillna(0.0)
    if occ.shape[1] != rets.shape[1]:
        raise ValueError("occupancy columns must match returns")
    n = rets.shape[1]
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != n:
        raise ValueError("weights length mismatch")
    w = np.maximum(w, 0.0)
    ws = w.sum()
    w = (w / ws) if ws > 0 else np.ones(n) / n
    pr = np.asarray(priority if priority is not None else w, dtype=float).reshape(-1)
    if pr.size != n:
        pr = w
    k = int(max_k) if max_k is not None else n
    k = max(1, min(k, n))
    cap = float(max(recycle_cap, 0.0))

    occ_lag = occ.shift(1).fillna(0.0).clip(lower=0.0, upper=1.0)
    occ_v = occ_lag.to_numpy(dtype=float)
    mask = np.zeros_like(occ_v)
    for t in range(occ_v.shape[0]):
        idx = np.flatnonzero(occ_v[t] > 0.5)
        if idx.size == 0:
            continue
        if idx.size <= k:
            mask[t, idx] = 1.0
        else:
            order = idx[np.argsort(-pr[idx], kind="stable")]
            mask[t, order[:k]] = 1.0

    raw = w.reshape(1, -1) * mask
    s = raw.sum(axis=1, keepdims=True)
    target = np.minimum(1.0, cap * s)
    scale = np.divide(target, s, out=np.ones_like(s), where=s > 1e-12)
    # idle rows keep zeros
    scale = np.where(s > 1e-12, scale, 0.0)
    w_t = raw * scale
    r = (rets.fillna(0.0).to_numpy(dtype=float) * w_t).sum(axis=1)
    eq = (1.0 + r).cumprod() * float(initial)
    return pd.Series(eq, index=rets.index, name="equity")


def combine_weighted(curves: list[pd.Series], weights: np.ndarray, initial: float = INITIAL) -> pd.Series:
    eq = pd.concat(curves, axis=1, sort=True).sort_index().ffill().dropna(how="any")
    if eq.empty:
        return eq
    w = np.asarray(weights, dtype=float)
    w = w / w.sum() if w.sum() else np.ones(len(curves)) / len(curves)
    norms = eq / eq.iloc[0]
    return (norms * w).sum(axis=1) * initial


def apply_month_aware_scale(
    port: pd.Series,
    *,
    strong_mo: float = 0.025,
    after_strong: float = 0.55,
    dd_trigger: float = 0.035,
    after_dd: float = 0.55,
    lo: float = 0.25,
) -> pd.Series:
    """Causal month / drawdown throttle — scales *down* only (never > 1).

    For bars in calendar month M, month_scale uses the fully completed return of
    month M-1 (first/last within M-1). Drawdown uses equity peak-to-trough lagged
    one bar. Combined scale = min(month_scale, dd_scale) clipped to [lo, 1], then
    lagged one more bar before multiplying returns.
    """
    if port is None or len(port) < 10:
        return port if port is not None else pd.Series(dtype=float)
    eq = port.astype(float)
    r = eq.pct_change().fillna(0.0)
    idx = eq.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    per = idx.to_period("M")
    periods = list(dict.fromkeys(per))  # ordered unique
    mo_ret: dict = {}
    for p in periods:
        chunk = eq.loc[per == p]
        if len(chunk) < 2:
            mo_ret[p] = 0.0
        else:
            mo_ret[p] = float(chunk.iloc[-1] / chunk.iloc[0] - 1.0)
    p_index = {p: i for i, p in enumerate(periods)}
    prior_vals = []
    for p in per:
        i = p_index[p]
        prior_vals.append(mo_ret[periods[i - 1]] if i > 0 else 0.0)
    prior_on_bars = pd.Series(prior_vals, index=eq.index, dtype=float)
    mo_scale = pd.Series(1.0, index=eq.index)
    mo_scale = mo_scale.where(prior_on_bars <= float(strong_mo), float(after_strong))

    peak = eq.cummax()
    dd = (peak - eq) / peak.replace(0, np.nan)
    dd_lag = dd.shift(1).fillna(0.0)
    dd_scale = pd.Series(1.0, index=eq.index)
    dd_scale = dd_scale.where(dd_lag <= float(dd_trigger), float(after_dd))

    scale = np.minimum(mo_scale.to_numpy(), dd_scale.to_numpy())
    scale = np.clip(scale, float(lo), 1.0)
    scale_s = pd.Series(scale, index=eq.index).shift(1).fillna(1.0)
    return (1.0 + r * scale_s).cumprod() * float(eq.iloc[0])


def apply_equity_curve_target(
    port: pd.Series,
    *,
    target_mo_vol: float = 0.012,
    lookback_months: int = 6,
    lo: float = 0.25,
    hi: float = 1.0,
) -> pd.Series:
    """Scale bar returns so trailing monthly vol approaches ``target_mo_vol``.

    Uses only completed months (shifted) — causal. ``hi`` defaults to 1.0 so this
    is a flattener / de-leverager vs the input curve, not a leverage booster.
    """
    if port is None or len(port) < 40:
        return port if port is not None else pd.Series(dtype=float)
    eq = port.astype(float)
    r = eq.pct_change().fillna(0.0)
    m_last = eq.resample("ME").last()
    m_ret = m_last.pct_change().dropna()
    if len(m_ret) < 3:
        return eq
    lb = max(3, int(lookback_months))
    trail = m_ret.rolling(lb, min_periods=3).std()
    # shift so month-end vol estimate is usable only after that month closes
    trail_s = trail.shift(1)
    scale_m = (float(target_mo_vol) / trail_s.replace(0, np.nan)).clip(lo, hi)
    scale_m = scale_m.fillna(1.0)
    scale_bars = scale_m.reindex(eq.index, method="ffill").fillna(1.0)
    scale_bars = scale_bars.shift(1).fillna(1.0).clip(lo, hi)
    return (1.0 + r * scale_bars).cumprod() * float(eq.iloc[0])


def apply_runup_throttle(
    port: pd.Series,
    *,
    trail_bars: int = 42,
    runup_thresh: float = 0.04,
    cool_scale: float = 0.5,
    lo: float = 0.25,
) -> pd.Series:
    """If lagged trailing return exceeds ``runup_thresh``, cut size (causal)."""
    if port is None or len(port) < max(10, trail_bars + 2):
        return port if port is not None else pd.Series(dtype=float)
    eq = port.astype(float)
    r = eq.pct_change().fillna(0.0)
    trail = eq / eq.shift(int(trail_bars)) - 1.0
    trail_lag = trail.shift(1)
    scale = pd.Series(1.0, index=eq.index)
    scale = scale.where(~(trail_lag > float(runup_thresh)), float(cool_scale))
    scale = scale.clip(lower=float(lo), upper=1.0).fillna(1.0)
    return (1.0 + r * scale).cumprod() * float(eq.iloc[0])


def apply_mtd_gain_clip(
    port: pd.Series,
    *,
    tau: float = 0.03,
    after_clip: float = 0.0,
    lo: float = 0.0,
) -> pd.Series:
    """Causal intra-month MTD gain clip — cut exposure once month-to-date exceeds tau.

    At bar t, MTD is computed from month-start equity through bar t-1 only.
    If that lagged MTD > ``tau``, next-bar scale becomes ``after_clip`` (clipped
    to [lo, 1]). Never increases leverage (hi=1). Designed to soft-cap top-month
    concentration without peeking at unfinished-bar or future months.
    """
    if port is None or len(port) < 10:
        return port if port is not None else pd.Series(dtype=float)
    eq = port.astype(float)
    r = eq.pct_change().fillna(0.0)
    idx = eq.index
    # Month keys in the series timezone
    if getattr(idx, "tz", None) is not None:
        months = idx.tz_convert("UTC").to_period("M")
    else:
        months = idx.to_period("M")
    # Month-start equity (first bar of each calendar month)
    mo_start = eq.groupby(months).transform("first")
    # MTD through prior bar: eq[t-1] / mo_start[t] - 1  (mo_start known at month open)
    eq_lag = eq.shift(1)
    mtd_lag = (eq_lag / mo_start.replace(0, np.nan) - 1.0).fillna(0.0)
    # First bar of month: no prior bar in-month → mtd_lag uses prior month's last /
    # same mo_start after shift can be stale; force 0 when month changes
    mo_change = pd.Series(months, index=eq.index) != pd.Series(months, index=eq.index).shift(1)
    mtd_lag = mtd_lag.where(~mo_change.fillna(True), 0.0)
    scale = pd.Series(1.0, index=eq.index)
    scale = scale.where(~(mtd_lag > float(tau)), float(after_clip))
    scale = scale.clip(lower=float(lo), upper=1.0).fillna(1.0)
    return (1.0 + r * scale).cumprod() * float(eq.iloc[0])


def apply_mtd_loss_halt(
    port: pd.Series,
    *,
    tau: float = 0.02,
    after_halt: float = 0.0,
    lo: float = 0.0,
) -> pd.Series:
    """Causal intra-month MTD loss halt — flatten once month-to-date drops below -tau.

    At bar t, MTD is computed from month-start equity through bar t-1 only
    (same month-start / month-change rules as ``apply_mtd_gain_clip``).
    If that lagged MTD < ``-tau``, next-bar scale becomes ``after_halt``
    (typically 0 or 0.25), clipped to [lo, 1]. Never increases leverage (hi=1).
    """
    if port is None or len(port) < 10:
        return port if port is not None else pd.Series(dtype=float)
    eq = port.astype(float)
    r = eq.pct_change().fillna(0.0)
    idx = eq.index
    if getattr(idx, "tz", None) is not None:
        months = idx.tz_convert("UTC").to_period("M")
    else:
        months = idx.to_period("M")
    mo_start = eq.groupby(months).transform("first")
    eq_lag = eq.shift(1)
    mtd_lag = (eq_lag / mo_start.replace(0, np.nan) - 1.0).fillna(0.0)
    mo_change = pd.Series(months, index=eq.index) != pd.Series(months, index=eq.index).shift(1)
    mtd_lag = mtd_lag.where(~mo_change.fillna(True), 0.0)
    scale = pd.Series(1.0, index=eq.index)
    scale = scale.where(~(mtd_lag < -float(tau)), float(after_halt))
    scale = scale.clip(lower=float(lo), upper=1.0).fillna(1.0)
    return (1.0 + r * scale).cumprod() * float(eq.iloc[0])


def apply_after_loss_throttle(
    port: pd.Series,
    *,
    after_loss: float = 0.5,
    lo: float = 0.25,
) -> pd.Series:
    """Causal prior-month loss throttle — scale current month if M-1 finished negative.

    Uses only the *fully completed* calendar month M-1 return (first/last within
    M-1), identical causal contract to ``apply_month_aware_scale``'s month leg.
    Bars in month M see prior_month_return = return(M-1); if that return < 0,
    scale = ``after_loss`` (in {0.25, 0.5, 0.75} typically), else 1.0.
    Scale is clipped to [lo, 1] (never leverage) and lagged one bar before
    multiplying returns so the decision at t uses information available at t-1.
    Losing January does not change January scales; February is scaled down.
    """
    if port is None or len(port) < 10:
        return port if port is not None else pd.Series(dtype=float)
    eq = port.astype(float)
    r = eq.pct_change().fillna(0.0)
    idx = eq.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    per = idx.to_period("M")
    periods = list(dict.fromkeys(per))  # ordered unique
    mo_ret: dict = {}
    for p in periods:
        chunk = eq.loc[per == p]
        if len(chunk) < 2:
            mo_ret[p] = 0.0
        else:
            mo_ret[p] = float(chunk.iloc[-1] / chunk.iloc[0] - 1.0)
    p_index = {p: i for i, p in enumerate(periods)}
    prior_vals = []
    for p in per:
        i = p_index[p]
        prior_vals.append(mo_ret[periods[i - 1]] if i > 0 else 0.0)
    prior_on_bars = pd.Series(prior_vals, index=eq.index, dtype=float)
    scale = pd.Series(1.0, index=eq.index)
    scale = scale.where(~(prior_on_bars < 0.0), float(after_loss))
    scale = scale.clip(lower=float(lo), upper=1.0).fillna(1.0)
    scale_s = scale.shift(1).fillna(1.0)
    return (1.0 + r * scale_s).cumprod() * float(eq.iloc[0])


def apply_trailing_gain_concentration_dampen(
    port: pd.Series,
    *,
    lookback_months: int = 6,
    thresh: float = 0.60,
    cool_scale: float = 0.5,
    k: int = 3,
    lo: float = 0.25,
) -> pd.Series:
    """Causal trailing gain-concentration dampener — scale down only (never > 1).

    For bars in calendar month M, inspect the last ``lookback_months`` *fully
    completed* months before M (or fewer early in the series). Among positive
    completed-month returns in that window, concentration = sum(top-K positives)
    / sum(all positives). When fewer than 2 positive months exist in the window,
    concentration is treated as 1.0 (max dampen trigger if above thresh) — this
    is the fail-closed choice so sparse early history does not silently skip.

    If concentration > ``thresh``, scale = ``cool_scale``, else 1.0. Scale is
    clipped to [lo, 1] and lagged one bar before multiplying returns (same
    causal contract as ``apply_month_aware_scale`` / ``apply_after_loss_throttle``).
    """
    if port is None or len(port) < 10:
        return port if port is not None else pd.Series(dtype=float)
    eq = port.astype(float)
    r = eq.pct_change().fillna(0.0)
    idx = eq.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    per = idx.to_period("M")
    periods = list(dict.fromkeys(per))  # ordered unique
    mo_ret: dict = {}
    for p in periods:
        chunk = eq.loc[per == p]
        if len(chunk) < 2:
            mo_ret[p] = 0.0
        else:
            mo_ret[p] = float(chunk.iloc[-1] / chunk.iloc[0] - 1.0)
    p_index = {p: i for i, p in enumerate(periods)}
    lb = max(1, int(lookback_months))
    kk = max(1, int(k))
    conc_vals = []
    for p in per:
        i = p_index[p]
        # Fully completed months strictly before current month p
        start = max(0, i - lb)
        window = periods[start:i]
        pos = [mo_ret[q] for q in window if mo_ret[q] > 0.0]
        if len(pos) < 2:
            # Fail-closed: treat as fully concentrated when <2 positives
            conc = 1.0
        else:
            pos_arr = np.asarray(pos, dtype=float)
            total = float(pos_arr.sum())
            if total <= 0:
                conc = 1.0
            else:
                top = np.sort(pos_arr)[::-1][: min(kk, len(pos_arr))]
                conc = float(top.sum() / total)
        conc_vals.append(conc)
    conc_on_bars = pd.Series(conc_vals, index=eq.index, dtype=float)
    scale = pd.Series(1.0, index=eq.index)
    scale = scale.where(~(conc_on_bars > float(thresh)), float(cool_scale))
    scale = scale.clip(lower=float(lo), upper=1.0).fillna(1.0)
    scale_s = scale.shift(1).fillna(1.0)
    return (1.0 + r * scale_s).cumprod() * float(eq.iloc[0])

def apply_prior_month_win_throttle(
    port: pd.Series,
    *,
    win_tau: float = 0.02,
    cool_scale: float = 0.5,
    lo: float = 0.25,
) -> pd.Series:
    """Causal prior-month win throttle — scale current month if M-1 finished strong.

    Uses only the *fully completed* calendar month M-1 return (first/last within
    M-1), identical causal contract to ``apply_after_loss_throttle``.
    Bars in month M see prior_month_return = return(M-1); if that return >
    ``win_tau``, scale = ``cool_scale``, else 1.0.
    Scale is clipped to [lo, 1] (never leverage) and lagged one bar before
    multiplying returns so the decision at t uses information available at t-1.
    A strong January does not change January scales; February is scaled down.
    """
    if port is None or len(port) < 10:
        return port if port is not None else pd.Series(dtype=float)
    eq = port.astype(float)
    r = eq.pct_change().fillna(0.0)
    idx = eq.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    per = idx.to_period("M")
    periods = list(dict.fromkeys(per))  # ordered unique
    mo_ret: dict = {}
    for p in periods:
        chunk = eq.loc[per == p]
        if len(chunk) < 2:
            mo_ret[p] = 0.0
        else:
            mo_ret[p] = float(chunk.iloc[-1] / chunk.iloc[0] - 1.0)
    p_index = {p: i for i, p in enumerate(periods)}
    prior_vals = []
    for p in per:
        i = p_index[p]
        prior_vals.append(mo_ret[periods[i - 1]] if i > 0 else 0.0)
    prior_on_bars = pd.Series(prior_vals, index=eq.index, dtype=float)
    scale = pd.Series(1.0, index=eq.index)
    scale = scale.where(~(prior_on_bars > float(win_tau)), float(cool_scale))
    scale = scale.clip(lower=float(lo), upper=1.0).fillna(1.0)
    scale_s = scale.shift(1).fillna(1.0)
    return (1.0 + r * scale_s).cumprod() * float(eq.iloc[0])


def apply_trailing_downside_vol_scale(
    port: pd.Series,
    *,
    lookback_days: int = 63,
    target_ddown: float = 0.006,
    cool_floor: float = 0.35,
    lo: float = 0.25,
) -> pd.Series:
    """Causal trailing downside-vol scale — cool only, never leverage (hi=1).

    Converts equity to daily returns, estimates trailing downside volatility as
    the rolling std of ``min(r, 0)`` over ``lookback_days`` (semideviation-style),
    then sets ``scale = clip(target_ddown / max(trail, eps), floor, 1.0)``.

    Scale is lagged one calendar day before multiplying returns and equity is
    rebuilt from the portfolio start. Uses only information available at t-1 —
    no look-ahead. Never increases leverage (upper clip is always 1.0).
    ``cool_floor`` raises the effective lower clip above ``lo`` when larger
    (default floor 0.35). Timezone-safe: daily resampling uses UTC-normalized
    timestamps like sibling month overlays.
    """
    if port is None or len(port) < 10:
        return port if port is not None else pd.Series(dtype=float)
    eq = port.astype(float)
    r_native = eq.pct_change().fillna(0.0)
    # Normalize to naive UTC for daily resample / date mapping
    eq_work = eq.copy()
    if getattr(eq_work.index, "tz", None) is not None:
        eq_work.index = eq_work.index.tz_convert("UTC").tz_localize(None)
    eq_d = eq_work.resample("1D").last().dropna()
    if len(eq_d) < 5:
        return eq
    r_d = eq_d.pct_change().fillna(0.0)
    down = r_d.clip(upper=0.0)
    lb = max(5, int(lookback_days))
    min_p = max(10, lb // 3)
    trail = down.rolling(lb, min_periods=min_p).std()
    eps = 1e-12
    eff_lo = max(float(lo), float(cool_floor))
    # trail at day t includes r_t; lag so day t uses trail through t-1 only
    scale_d = (
        (float(target_ddown) / trail.replace(0, np.nan).clip(lower=eps))
        .clip(lower=eff_lo, upper=1.0)
        .fillna(1.0)
    )
    scale_d = scale_d.shift(1).fillna(1.0)
    scale_d.index = pd.DatetimeIndex(scale_d.index).normalize()
    if getattr(eq.index, "tz", None) is not None:
        day_keys = eq.index.tz_convert("UTC").tz_localize(None).normalize()
    else:
        day_keys = pd.DatetimeIndex(eq.index).normalize()
    scale_bars = (
        pd.Series(day_keys, index=eq.index, dtype="datetime64[ns]")
        .map(scale_d)
        .ffill()
        .fillna(1.0)
    )
    scale_bars = scale_bars.clip(lower=eff_lo, upper=1.0)
    return (1.0 + r_native * scale_bars).cumprod() * float(eq.iloc[0])
