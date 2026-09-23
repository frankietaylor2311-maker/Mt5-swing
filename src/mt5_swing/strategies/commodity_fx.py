"""Commodity-currency FX strategies (Chen–Rogoff–Rossi spirit).

Literature prior (fixed — not holdout-tuned)
--------------------------------------------
Commodity price moves co-move with / can forecast commodity-currency FX
(AUD, CAD, NZD). Classic refs: Chen, Rogoff & Rossi (2010); Cashin, Céspedes
& Sahay; Amano & van Norden (oil–CAD).

Implementation
--------------
1. **Country-linked TS:** for AUD/CAD/NZD, formation = trailing cumulative
   commodity return over ``formation_days`` skipping most recent ``skip_days``,
   then ``signal_lag`` trading days. Long commodity currency when lagged
   commodity momentum > 0 (equal-weight across the three).
2. **Cross-sectional basket:** long AUD/CAD/NZD EW vs short EUR/GBP/JPY EW
   when lagged *broad* commodity momentum > 0; flat otherwise.
3. **Local projection (diagnostic):** monthly cum FX return of AUD/CAD/NZD on
   lagged commodity z/momentum at h=1,3 months.

All signals use ``signal_lag≥1`` on top of commodity ``pub_lag_days``.
Costs: ~1.5 bps/side on turnover when reporting strategy returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.data.commodity_prices import COUNTRY_COMMODITY_MAP, commodity_for_currency
from mt5_swing.data.macro_uncertainty import CURRENCY_USD_PAIR
from mt5_swing.strategies.carry_rank import (
    USD_PAIRS,
    currency_weights_to_pair_weights,
    expand_weights_to_daily,
    portfolio_returns_from_weights,
)
from mt5_swing.strategies.country_gpr_fx import (
    foreign_vs_usd_returns,
    local_projection_beta,
    monthly_fx_returns,
)


COMMODITY_CCYS: tuple[str, ...] = ("AUD", "CAD", "NZD")
NON_COMMODITY_CCYS: tuple[str, ...] = ("EUR", "GBP", "JPY", "CHF")


@dataclass
class CommodityFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    formation_days: int = 63
    skip_days: int = 21
    signal_lag: int = 1  # trading days after lagged commodity signal known
    cost_bps_per_side: float = 1.5
    lp_horizons: tuple[int, ...] = (1, 3)  # months
    z_window: int = 252  # trading days for commodity z (LP diagnostic)
    min_periods: int = 63
    # Cross-sectional: short side
    short_non_commodity: tuple[str, ...] = ("EUR", "GBP", "JPY")



def _ensure_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def align_daily_to_index(obj: pd.Series | pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series | pd.DataFrame:
    """As-of ffill align commodity/daily signals onto an FX calendar (handles session-time mismatch)."""
    out = obj.copy()
    out.index = _ensure_utc_index(pd.DatetimeIndex(out.index))
    idx = _ensure_utc_index(pd.DatetimeIndex(index))
    # Prefer calendar-day keys so Yahoo 04:00 and FX 23:00 on the same UTC date join cleanly
    day = out.copy()
    day.index = out.index.normalize()
    day = day[~day.index.duplicated(keep="last")].sort_index()
    target_days = pd.DatetimeIndex(idx.normalize())
    aligned_days = day.reindex(target_days, method="ffill")
    aligned_days.index = idx
    return aligned_days

def trailing_commodity_momentum(
    prices: pd.Series,
    *,
    formation_days: int = 63,
    skip_days: int = 21,
) -> pd.Series:
    """Cumulative return over [t-formation-skip, t-skip] using log-sum of daily rets.

    ``prices`` should already include publication lag (close known next session).
    """
    r = prices.pct_change()
    # Sum of returns from lag=(formation+skip) to lag=skip (inclusive of start)
    # Equivalent: cumret = px_{t-skip} / px_{t-formation-skip} - 1
    lag_near = int(skip_days)
    lag_far = int(formation_days + skip_days)
    near = prices.shift(lag_near)
    far = prices.shift(lag_far)
    mom = near / far - 1.0
    mom.name = prices.name or "commodity_mom"
    return mom


def prepare_country_commodity_signals(
    commodity_panel: pd.DataFrame,
    *,
    cfg: CommodityFxConfig | None = None,
    currencies: tuple[str, ...] = COMMODITY_CCYS,
) -> pd.DataFrame:
    """Daily commodity-momentum signals per commodity currency (pre signal_lag)."""
    cfg = cfg or CommodityFxConfig()
    cols = {}
    for ccy in currencies:
        try:
            px = commodity_for_currency(commodity_panel, ccy)
        except KeyError:
            continue
        cols[ccy] = trailing_commodity_momentum(
            px, formation_days=cfg.formation_days, skip_days=cfg.skip_days
        )
    if not cols:
        return pd.DataFrame()
    sig = pd.DataFrame(cols).sort_index()
    if cfg.signal_lag > 0:
        sig = sig.shift(int(cfg.signal_lag))
    sig.attrs["signal_lag"] = int(cfg.signal_lag)
    sig.attrs["formation_days"] = int(cfg.formation_days)
    sig.attrs["skip_days"] = int(cfg.skip_days)
    return sig


def prepare_broad_commodity_signal(
    commodity_panel: pd.DataFrame,
    *,
    cfg: CommodityFxConfig | None = None,
) -> pd.Series:
    """Lagged broad basket commodity momentum (for cross-sectional basket rule)."""
    cfg = cfg or CommodityFxConfig()
    col = "basket" if "basket" in commodity_panel.columns else (
        "oil" if "oil" in commodity_panel.columns else commodity_panel.columns[0]
    )
    mom = trailing_commodity_momentum(
        commodity_panel[col],
        formation_days=cfg.formation_days,
        skip_days=cfg.skip_days,
    )
    if cfg.signal_lag > 0:
        mom = mom.shift(int(cfg.signal_lag))
    mom.name = "broad_commodity_mom"
    mom.attrs["signal_lag"] = int(cfg.signal_lag)
    return mom


def country_ts_currency_weights(
    signals: pd.DataFrame,
    *,
    cfg: CommodityFxConfig | None = None,
) -> pd.DataFrame:
    """Equal-weight long commodity currencies with positive lagged commodity mom.

    Flat currencies with non-positive signal. Gross long-only in FX currency space
    (no short non-commodity here). When k>0 actives, each gets 1/k.
    """
    cfg = cfg or CommodityFxConfig()
    cols = [c for c in signals.columns if c in COMMODITY_CCYS]
    if not cols:
        return pd.DataFrame()
    rows = []
    for dt, row in signals[cols].iterrows():
        s = row.dropna()
        active = s[s > 0]
        w = pd.Series(0.0, index=cols)
        if len(active) > 0:
            w.loc[list(active.index)] = 1.0 / len(active)
        w.name = dt
        rows.append(w)
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows).sort_index()
    out.attrs["strategy"] = "commodity_country_ts"
    return out


def cross_sectional_basket_weights(
    broad_signal: pd.Series,
    *,
    cfg: CommodityFxConfig | None = None,
    daily_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Long AUD/CAD/NZD EW vs short EUR/GBP/JPY EW when broad mom > 0; else flat.

    Weights in currency space; each side |sum|=0.5 when on.
    """
    cfg = cfg or CommodityFxConfig()
    longs = list(COMMODITY_CCYS)
    shorts = [c for c in cfg.short_non_commodity if c in NON_COMMODITY_CCYS or c in CURRENCY_USD_PAIR]
    idx = daily_index if daily_index is not None else broad_signal.index
    # Align signal to index via ffill after reindex
    sig = broad_signal.reindex(idx).ffill()
    cols = longs + [c for c in shorts if c not in longs]
    out = pd.DataFrame(0.0, index=idx, columns=cols)
    on = sig > 0
    if on.any():
        lw = 0.5 / max(len(longs), 1)
        sw = 0.5 / max(len(shorts), 1)
        for c in longs:
            out.loc[on, c] = lw
        for c in shorts:
            if c not in out.columns:
                out[c] = 0.0
            out.loc[on, c] = -sw
    out.attrs["strategy"] = "commodity_xs_basket"
    return out


def apply_turnover_costs(
    gross: pd.Series,
    pair_weights_daily: pd.DataFrame,
    *,
    cost_bps_per_side: float = 1.5,
) -> pd.Series:
    """Subtract ``cost_bps_per_side`` on |Δw| turnover (weights already for today's return)."""
    # portfolio_returns_from_weights uses w.shift(1); costs should match that lag
    w = pair_weights_daily.fillna(0.0)
    w_lag = w.shift(1).fillna(0.0)
    turnover = w_lag.diff().abs().sum(axis=1).fillna(0.0)
    # First day: treat initial gross deployment as turnover
    if len(turnover) and np.isfinite(turnover.iloc[0]):
        turnover.iloc[0] = float(w_lag.iloc[0].abs().sum()) if len(w_lag) else 0.0
    cost = turnover * (float(cost_bps_per_side) / 10_000.0)
    net = gross.fillna(0.0) - cost.reindex(gross.index).fillna(0.0)
    net.name = gross.name
    net.attrs["cost_bps_per_side"] = float(cost_bps_per_side)
    return net


def _ccy_weights_to_daily_pair_port(
    ccy_w: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    signal_lag: int = 1,
    cost_bps_per_side: float = 1.5,
    name: str = "commodity_fx",
) -> pd.Series:
    if ccy_w.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    # Ensure currency columns known to USD_PAIRS
    keep = [c for c in ccy_w.columns if c in USD_PAIRS]
    if not keep:
        return pd.Series(0.0, index=pair_ret.index, name=name)
    cw = ccy_w[keep].copy()
    cw.index = pd.DatetimeIndex(cw.index)
    if cw.index.tz is None:
        cw.index = cw.index.tz_localize("UTC")
    else:
        cw.index = cw.index.tz_convert("UTC")
    pair_w = currency_weights_to_pair_weights(cw)
    daily_w = expand_weights_to_daily(pair_w, pair_ret.index, signal_lag=0)
    # signal_lag already applied on commodity signal; expand without extra day,
    # but portfolio_returns_from_weights still shifts weights by 1 for causality.
    # If signal_lag was on signal, weights known at t should trade t+1 — the
    # internal shift(1) in portfolio_returns_from_weights handles that.
    # Additional signal_lag>1 already baked into ccy_w via prepare_* shift.
    _ = signal_lag  # documented; already in signal
    gross = portfolio_returns_from_weights(daily_w, pair_ret)
    gross.name = name
    net = apply_turnover_costs(gross, daily_w, cost_bps_per_side=cost_bps_per_side)
    net.name = name
    return net


def commodity_country_ts_returns(
    commodity_panel: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: CommodityFxConfig | None = None,
) -> pd.Series:
    """Country-linked TS: long commodity-ccy when own lagged commodity mom > 0."""
    cfg = cfg or CommodityFxConfig()
    sig = prepare_country_commodity_signals(commodity_panel, cfg=cfg)
    # Align signals onto pair calendar (Yahoo session times ≠ FX D1 bar times)
    sig = align_daily_to_index(sig, pair_ret.index)
    ccy_w = country_ts_currency_weights(sig, cfg=cfg)
    return _ccy_weights_to_daily_pair_port(
        ccy_w,
        pair_ret,
        signal_lag=cfg.signal_lag,
        cost_bps_per_side=cfg.cost_bps_per_side,
        name="commodity_country_ts",
    )


def commodity_xs_basket_returns(
    commodity_panel: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: CommodityFxConfig | None = None,
) -> pd.Series:
    """Long commodity-ccy vs short non-commodity when broad commodity mom > 0."""
    cfg = cfg or CommodityFxConfig()
    broad = prepare_broad_commodity_signal(commodity_panel, cfg=cfg)
    broad = align_daily_to_index(broad, pair_ret.index)
    ccy_w = cross_sectional_basket_weights(broad, cfg=cfg, daily_index=pair_ret.index)
    return _ccy_weights_to_daily_pair_port(
        ccy_w,
        pair_ret,
        signal_lag=cfg.signal_lag,
        cost_bps_per_side=cfg.cost_bps_per_side,
        name="commodity_xs_basket",
    )


def oil_impulse_cad_pair_weight_sign(oil_mom: float) -> float:
    """Diagnostic helper: positive oil mom → long CAD → short USDCAD (weight < 0).

    Returns the expected USDCAD pair weight sign contribution from +CAD.
    """
    # USD_PAIRS CAD = (USDCAD, -1); long CAD → weight * sign = +1 * -1 = -1 on USDCAD
    _sym, sign = USD_PAIRS["CAD"]
    w_ccy = 1.0 if oil_mom > 0 else (-1.0 if oil_mom < 0 else 0.0)
    return w_ccy * sign


def commodity_local_projection_panel(
    commodity_panel: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: CommodityFxConfig | None = None,
) -> pd.DataFrame:
    """Monthly LP: cum FX return of AUD/CAD/NZD on lagged commodity momentum/z.

    Sign prior: beta > 0 (positive commodity impulse → commodity FX appreciates vs USD).
    """
    cfg = cfg or CommodityFxConfig()
    # Daily mom → month-end
    sig_d = prepare_country_commodity_signals(commodity_panel, cfg=cfg)
    if sig_d.empty:
        return pd.DataFrame()
    sig_d = align_daily_to_index(sig_d, pair_ret.index)
    # Also build z-score of commodity levels for robustness diagnostic
    z_cols = {}
    for ccy in COMMODITY_CCYS:
        try:
            px = commodity_for_currency(commodity_panel, ccy)
        except KeyError:
            continue
        mu = px.rolling(cfg.z_window, min_periods=cfg.min_periods).mean()
        sd = px.rolling(cfg.z_window, min_periods=cfg.min_periods).std()
        z = (px - mu) / sd.replace(0.0, np.nan)
        if cfg.signal_lag > 0:
            z = z.shift(int(cfg.signal_lag))
        z_cols[ccy] = z
    z_d = pd.DataFrame(z_cols).sort_index() if z_cols else pd.DataFrame()
    if not z_d.empty:
        z_d = align_daily_to_index(z_d, pair_ret.index)

    fx_m = monthly_fx_returns(pair_ret)
    # Month-end signals
    sig_m = sig_d.resample("ME").last()
    z_m = z_d.resample("ME").last() if not z_d.empty else pd.DataFrame()

    rows = []
    for h in cfg.lp_horizons:
        for ccy in COMMODITY_CCYS:
            if ccy not in fx_m.columns:
                continue
            if ccy in sig_m.columns:
                res = local_projection_beta(sig_m[ccy], fx_m[ccy], horizon=h)
                res["currency"] = ccy
                res["regressor"] = "commodity_mom"
                res["scope"] = "currency"
                rows.append(res)
            if ccy in z_m.columns:
                resz = local_projection_beta(z_m[ccy], fx_m[ccy], horizon=h)
                resz["currency"] = ccy
                resz["regressor"] = "commodity_z"
                resz["scope"] = "currency"
                rows.append(resz)
        # Pooled mom
        xs, ys = [], []
        for ccy in COMMODITY_CCYS:
            if ccy not in sig_m.columns or ccy not in fx_m.columns:
                continue
            g = sig_m[ccy].copy()
            r = fx_m[ccy].copy()
            g.index = pd.DatetimeIndex(g.index).tz_convert(None).to_period("M").to_timestamp(how="start")
            r.index = pd.DatetimeIndex(r.index).tz_convert(None).to_period("M").to_timestamp(how="start")
            if g.index.tz is None:
                g.index = g.index.tz_localize("UTC")
            if r.index.tz is None:
                r.index = r.index.tz_localize("UTC")
            fwd = pd.Series(index=r.index, dtype=float)
            for i in range(len(r) - h):
                fwd.iloc[i] = float(r.iloc[i + 1 : i + 1 + h].sum())
            df = pd.concat([g.rename("x"), fwd.rename("y")], axis=1).dropna()
            if len(df):
                xs.append(df["x"].to_numpy())
                ys.append(df["y"].to_numpy())
        if xs:
            x = np.concatenate(xs)
            y = np.concatenate(ys)
            if len(x) >= 24:
                x1 = np.column_stack([np.ones(len(x)), x])
                coef, _, _, _ = np.linalg.lstsq(x1, y, rcond=None)
                resid = y - x1 @ coef
                n, k = len(y), 2
                sigma2 = float(np.dot(resid, resid) / max(n - k, 1))
                se_b = float(np.sqrt(sigma2 * np.linalg.inv(x1.T @ x1)[1, 1]))
                rows.append(
                    {
                        "horizon": h,
                        "currency": "POOLED",
                        "regressor": "commodity_mom",
                        "scope": "pooled",
                        "beta": float(coef[1]),
                        "alpha": float(coef[0]),
                        "tstat": float(coef[1] / se_b) if se_b > 0 else float("nan"),
                        "n": int(n),
                        "r2": float(
                            1.0
                            - np.dot(resid, resid)
                            / max(np.dot(y - y.mean(), y - y.mean()), 1e-18)
                        ),
                        "mean_y": float(y.mean()),
                    }
                )
    return pd.DataFrame(rows)


# Re-export helpers used by tests / wave
__all__ = [
    "COMMODITY_CCYS",
    "NON_COMMODITY_CCYS",
    "CommodityFxConfig",
    "align_daily_to_index",
    "trailing_commodity_momentum",
    "prepare_country_commodity_signals",
    "prepare_broad_commodity_signal",
    "country_ts_currency_weights",
    "cross_sectional_basket_weights",
    "commodity_country_ts_returns",
    "commodity_xs_basket_returns",
    "commodity_local_projection_panel",
    "oil_impulse_cad_pair_weight_sign",
    "foreign_vs_usd_returns",
    "monthly_fx_returns",
    "CURRENCY_USD_PAIR",
    "COUNTRY_COMMODITY_MAP",
]
