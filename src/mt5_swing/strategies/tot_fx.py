"""Terms-of-trade / commodity-currency FX (Cashin–Céspedes–Sahay refinement).

Distinct from the prior Chen–Rogoff–Rossi commodity-*momentum* wave
(``commodity_fx.py``):
- Prior CRR: single mapped commodity momentum (AUD→copper, CAD→oil, NZD→basket)
  → long if mom > 0.
- This ToT wave: country **export basket − import proxy** momentum differential
  (frozen ``COUNTRY_TOT_MAP``), then country-TS and cross-sectional sorts.

Literature priors (fixed — not holdout-tuned)
---------------------------------------------
Cashin, Céspedes & Sahay (2004) — commodity currencies and the terms of trade;
Chen, Rogoff & Rossi (2010) — commodity prices and exchange rates (export side);
Amano & van Norden — oil–CAD.

PIT: commodity ``pub_lag_days`` (loader) + ``signal_lag`` trading days.
Calendar-date align via ``align_daily_to_index`` (Yahoo commodity clocks ≠ FX D1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.data.commodity_prices import (
    COUNTRY_TOT_MAP,
    TOT_TRADEABLE_CCYS,
    tot_export_import_columns,
)
from mt5_swing.strategies.carry_rank import (
    USD_PAIRS,
    currency_weights_to_pair_weights,
    expand_weights_to_daily,
    portfolio_returns_from_weights,
)
from mt5_swing.strategies.commodity_fx import (
    NON_COMMODITY_CCYS,
    align_daily_to_index,
    apply_turnover_costs,
    trailing_commodity_momentum,
)


@dataclass
class TotFxConfig:
    """Fixed research priors — do not grid / holdout-tune."""

    formation_days: int = 63
    skip_days: int = 21
    signal_lag: int = 1  # trading days after lagged ToT signal known
    cost_bps_per_side: float = 1.5
    n_long: int = 1  # XS among 3 tradeable commodity ccys
    n_short: int = 1
    short_non_commodity: tuple[str, ...] = ("EUR", "GBP", "JPY")
    tradeable: tuple[str, ...] = TOT_TRADEABLE_CCYS


def _resolve_col(panel: pd.DataFrame, col: str) -> str:
    if col in panel.columns:
        return col
    if col == "basket" and "etf" in panel.columns:
        return "etf"
    # Fallback: prefer oil then first available
    for alt in ("oil", "copper", "gold", "basket"):
        if alt in panel.columns:
            return alt
    raise KeyError(f"Commodity column {col} missing from panel; have {list(panel.columns)}")


def country_tot_momentum(
    commodity_panel: pd.DataFrame,
    *,
    cfg: TotFxConfig | None = None,
    currencies: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Daily ToT change proxy = export_mom − import_mom per currency (pre signal_lag).

    Both legs use the same formation/skip window so the differential is an honest
    relative ToT impulse, not a mixed-horizon construct.
    """
    cfg = cfg or TotFxConfig()
    ccys = currencies or cfg.tradeable
    cols: dict[str, pd.Series] = {}
    for ccy in ccys:
        if ccy.upper() not in COUNTRY_TOT_MAP:
            continue
        exp_name, imp_name = tot_export_import_columns(ccy)
        try:
            exp_col = _resolve_col(commodity_panel, exp_name)
            imp_col = _resolve_col(commodity_panel, imp_name)
        except KeyError:
            continue
        if exp_col == imp_col:
            # Degenerate mapping — skip rather than invent a zero ToT series
            continue
        exp_mom = trailing_commodity_momentum(
            commodity_panel[exp_col],
            formation_days=cfg.formation_days,
            skip_days=cfg.skip_days,
        )
        imp_mom = trailing_commodity_momentum(
            commodity_panel[imp_col],
            formation_days=cfg.formation_days,
            skip_days=cfg.skip_days,
        )
        tot = exp_mom - imp_mom
        tot.name = ccy.upper()
        cols[ccy.upper()] = tot
    if not cols:
        return pd.DataFrame()
    sig = pd.DataFrame(cols).sort_index()
    if cfg.signal_lag > 0:
        sig = sig.shift(int(cfg.signal_lag))
    sig.attrs["signal_lag"] = int(cfg.signal_lag)
    sig.attrs["formation_days"] = int(cfg.formation_days)
    sig.attrs["skip_days"] = int(cfg.skip_days)
    sig.attrs["factor"] = "tot_chg"
    return sig


def tot_country_ts_weights(
    tot_signals: pd.DataFrame,
    *,
    cfg: TotFxConfig | None = None,
) -> pd.DataFrame:
    """Equal-weight long commodity currencies with positive lagged ToT change."""
    cfg = cfg or TotFxConfig()
    cols = [c for c in tot_signals.columns if c in cfg.tradeable]
    if not cols:
        return pd.DataFrame()
    rows = []
    for dt, row in tot_signals[cols].iterrows():
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
    out.attrs["strategy"] = "tot_country_ts"
    return out


def tot_xs_weights(
    tot_signals: pd.DataFrame,
    *,
    cfg: TotFxConfig | None = None,
) -> pd.DataFrame:
    """Cross-sectional: long n_long highest ToT / short n_short lowest among commodity ccys."""
    cfg = cfg or TotFxConfig()
    cols = [c for c in tot_signals.columns if c in cfg.tradeable]
    if len(cols) < 2:
        return pd.DataFrame()
    n_long = min(int(cfg.n_long), len(cols))
    n_short = min(int(cfg.n_short), len(cols))
    rows = []
    for dt, row in tot_signals[cols].iterrows():
        s = row.dropna()
        w = pd.Series(0.0, index=cols)
        if len(s) < 2:
            w.name = dt
            rows.append(w)
            continue
        ranked = s.sort_values(ascending=False)
        longs = list(ranked.index[:n_long])
        shorts = list(ranked.index[-n_short:])
        # Avoid overlapping long/short when n is large relative to panel
        shorts = [c for c in shorts if c not in longs]
        if longs:
            w.loc[longs] = 0.5 / len(longs)
        if shorts:
            w.loc[shorts] = -0.5 / len(shorts)
        w.name = dt
        rows.append(w)
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows).sort_index()
    out.attrs["strategy"] = "tot_xs"
    return out


def tot_vs_g10_weights(
    tot_signals: pd.DataFrame,
    *,
    cfg: TotFxConfig | None = None,
    daily_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Long positive-ToT commodity ccys EW vs short EUR/GBP/JPY when mean ToT > 0."""
    cfg = cfg or TotFxConfig()
    longs = [c for c in cfg.tradeable if c in tot_signals.columns]
    shorts = list(cfg.short_non_commodity)
    idx = daily_index if daily_index is not None else tot_signals.index
    sig = tot_signals.reindex(idx).ffill()
    cols = list(dict.fromkeys(longs + shorts))
    out = pd.DataFrame(0.0, index=idx, columns=cols)
    if not longs:
        out.attrs["strategy"] = "tot_vs_g10"
        return out
    mean_tot = sig[longs].mean(axis=1)
    on = mean_tot > 0
    # Within-on days: only long those with positive own ToT (refinement vs CRR broad basket)
    for dt in out.index[on.fillna(False)]:
        row = sig.loc[dt, longs] if dt in sig.index else pd.Series(dtype=float)
        active = [c for c in longs if pd.notna(row.get(c, np.nan)) and float(row[c]) > 0]
        if not active:
            continue
        lw = 0.5 / len(active)
        sw = 0.5 / max(len(shorts), 1)
        for c in active:
            out.loc[dt, c] = lw
        for c in shorts:
            out.loc[dt, c] = -sw
    out.attrs["strategy"] = "tot_vs_g10"
    return out


def _ccy_weights_to_daily_pair_port(
    ccy_w: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cost_bps_per_side: float = 1.5,
    name: str = "tot_fx",
) -> pd.Series:
    if ccy_w.empty:
        return pd.Series(0.0, index=pair_ret.index, name=name)
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
    gross = portfolio_returns_from_weights(daily_w, pair_ret)
    gross.name = name
    net = apply_turnover_costs(gross, daily_w, cost_bps_per_side=cost_bps_per_side)
    net.name = name
    return net


def tot_factor_returns(
    commodity_panel: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: TotFxConfig | None = None,
) -> dict[str, pd.Series]:
    """Build ToT strategy return series (net of costs)."""
    cfg = cfg or TotFxConfig()
    tot = country_tot_momentum(commodity_panel, cfg=cfg)
    if tot.empty:
        return {}
    tot = align_daily_to_index(tot, pair_ret.index)

    out: dict[str, pd.Series] = {}
    w_ts = tot_country_ts_weights(tot, cfg=cfg)
    out["tot_country_ts"] = _ccy_weights_to_daily_pair_port(
        w_ts, pair_ret, cost_bps_per_side=cfg.cost_bps_per_side, name="tot_country_ts"
    )

    w_xs = tot_xs_weights(tot, cfg=cfg)
    out["tot_xs"] = _ccy_weights_to_daily_pair_port(
        w_xs, pair_ret, cost_bps_per_side=cfg.cost_bps_per_side, name="tot_xs"
    )

    w_g10 = tot_vs_g10_weights(tot, cfg=cfg, daily_index=pair_ret.index)
    out["tot_vs_g10"] = _ccy_weights_to_daily_pair_port(
        w_g10, pair_ret, cost_bps_per_side=cfg.cost_bps_per_side, name="tot_vs_g10"
    )

    # Equal-weight blend of the three legs (soft diagnostic)
    stack = pd.DataFrame({k: v for k, v in out.items()})
    ew = stack.mean(axis=1)
    ew.name = "tot_ew"
    out["tot_ew"] = ew
    return out


def tot_vs_crr_correlation(
    tot_ret: pd.Series,
    crr_ret: pd.Series,
) -> float:
    """Pearson corr of overlapping daily returns (document distinctness vs CRR)."""
    df = pd.concat([tot_ret.rename("tot"), crr_ret.rename("crr")], axis=1).dropna()
    if len(df) < 60:
        return float("nan")
    return float(df["tot"].corr(df["crr"]))


__all__ = [
    "TotFxConfig",
    "country_tot_momentum",
    "tot_country_ts_weights",
    "tot_xs_weights",
    "tot_vs_g10_weights",
    "tot_factor_returns",
    "tot_vs_crr_correlation",
    "COUNTRY_TOT_MAP",
    "TOT_TRADEABLE_CCYS",
]
