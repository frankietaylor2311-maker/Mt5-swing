"""FX event study around high geopolitics-news intensity days.

Prior (literature-aligned, fixed — no holdout tuning)
------------------------------------------------------
Elevated conflict/geopolitics news intensity is associated with safe-haven USD
demand vs cyclical FX (EUR, GBP, AUD) and sometimes vs JPY (mixed / haven).

We study cumulative USD-vs-ccy returns around high-intensity event days versus
matched control days. Intensity comes from ``mt5_swing.data.news_events``
(GDELT if usable; else GPR daily spike proxy).

Trading rule (only if event-vs-control post-window shows USD edge):
  After a high-intensity day is known (signal_lag≥1), hold long-USD basket for
  ``hold_days`` sessions. Costs applied on position changes. No HO tuning of
  thresholds — fixed top-decile + hold priors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.data.news_events import control_dates, high_intensity_event_dates
from mt5_swing.strategies.gpr_regime import align_macro_to_index

# USD vs foreign: +1 = long USD vs that currency
USD_VS_CCY: dict[str, tuple[str, int]] = {
    "EUR": ("EURUSD", -1),  # long USD = short EURUSD
    "GBP": ("GBPUSD", -1),
    "JPY": ("USDJPY", +1),  # long USD = long USDJPY
    "AUD": ("AUDUSD", -1),
}


@dataclass
class NewsEventStudyConfig:
    pre: int = 5
    post: int = 10
    decile: float = 0.90
    min_gap_days: int = 5
    extra_lag: int = 1  # bar lag on intensity for event definition
    control_seed: int = 42
    # Signal gate for enabling trading rule (fixed prior, not HO-tuned)
    signal_horizon: int = 5  # evaluate USD edge at h=+signal_horizon
    min_edge: float = 0.0  # require event - control mean > 0 at signal_horizon
    min_t: float = 1.0  # weak bar; literature not prop-firm
    hold_days: int = 5
    signal_lag: int = 1  # days after event before entering
    cost_bps_per_side: float = 1.5  # Yahoo approx; not FTMO spread
    currencies: tuple[str, ...] = ("EUR", "GBP", "JPY", "AUD")


def usd_vs_ccy_returns(pair_ret: pd.DataFrame, currencies: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Daily returns of long-USD vs each currency (positive = USD strengthens)."""
    ccys = currencies or tuple(USD_VS_CCY.keys())
    out = {}
    for ccy in ccys:
        if ccy not in USD_VS_CCY:
            continue
        pair, sign = USD_VS_CCY[ccy]
        if pair not in pair_ret.columns:
            continue
        out[ccy] = sign * pair_ret[pair]
    return pd.DataFrame(out, index=pair_ret.index).sort_index()


def usd_basket_returns(pair_ret: pd.DataFrame, currencies: tuple[str, ...] | None = None) -> pd.Series:
    """Equal-weight long-USD vs EUR/GBP/JPY/AUD."""
    panel = usd_vs_ccy_returns(pair_ret, currencies=currencies)
    if panel.empty:
        return pd.Series(dtype=float, name="usd_basket")
    s = panel.mean(axis=1)
    s.name = "usd_basket"
    return s


def _cum_from_event(r: pd.Series, loc: int, h: int) -> float:
    if h >= 0:
        return float(r.iloc[loc : loc + h + 1].sum())
    return float(r.iloc[loc + h : loc + 1].sum())


def event_study_paths(
    returns: pd.Series,
    events: pd.DatetimeIndex,
    *,
    pre: int = 5,
    post: int = 10,
) -> pd.DataFrame:
    """Per-event cumulative return paths indexed by horizon."""
    rows = []
    idx = returns.index
    for e in events:
        loc = idx.get_indexer([e], method="pad")[0]
        if loc < 0:
            continue
        for h in range(-pre, post + 1):
            j = loc + h
            if j < 0 or j >= len(returns):
                continue
            rows.append({"event": e, "horizon": h, "cum": _cum_from_event(returns, loc, h)})
    return pd.DataFrame(rows)


def summarize_event_vs_control(
    returns: pd.Series,
    events: pd.DatetimeIndex,
    controls: pd.DatetimeIndex,
    *,
    pre: int = 5,
    post: int = 10,
) -> pd.DataFrame:
    """Mean cum path for events vs controls + difference and simple t on cross-section at each h."""
    rows = []
    for h in range(-pre, post + 1):
        ev = []
        ct = []
        idx = returns.index
        for e in events:
            loc = idx.get_indexer([e], method="pad")[0]
            if loc < 0:
                continue
            j = loc + h
            if 0 <= j < len(returns):
                ev.append(_cum_from_event(returns, loc, h))
        for c in controls:
            loc = idx.get_indexer([c], method="pad")[0]
            if loc < 0:
                continue
            j = loc + h
            if 0 <= j < len(returns):
                ct.append(_cum_from_event(returns, loc, h))
        ev_a = np.asarray(ev, dtype=float)
        ct_a = np.asarray(ct, dtype=float)
        diff = float(ev_a.mean() - ct_a.mean()) if len(ev_a) and len(ct_a) else float("nan")
        # Two-sample t (unequal var approx) on cum returns at horizon h
        tstat = float("nan")
        if len(ev_a) >= 3 and len(ct_a) >= 3:
            se = np.sqrt(ev_a.var(ddof=1) / len(ev_a) + ct_a.var(ddof=1) / len(ct_a))
            if se > 0:
                tstat = float((ev_a.mean() - ct_a.mean()) / se)
        rows.append(
            {
                "horizon": h,
                "n_events": int(len(ev_a)),
                "n_controls": int(len(ct_a)),
                "event_cum_mean": float(ev_a.mean()) if len(ev_a) else float("nan"),
                "control_cum_mean": float(ct_a.mean()) if len(ct_a) else float("nan"),
                "diff_event_minus_control": diff,
                "tstat_diff": tstat,
            }
        )
    return pd.DataFrame(rows)


def run_news_event_study(
    pair_ret: pd.DataFrame,
    intensity: pd.Series,
    *,
    cfg: NewsEventStudyConfig | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """Full board: per-ccy + USD basket event vs control summaries."""
    cfg = cfg or NewsEventStudyConfig()
    usd_panel = usd_vs_ccy_returns(pair_ret, currencies=cfg.currencies)
    basket = usd_basket_returns(pair_ret, currencies=cfg.currencies)
    events, ev_meta = high_intensity_event_dates(
        intensity,
        pair_ret.index,
        decile=cfg.decile,
        min_gap_days=cfg.min_gap_days,
        extra_lag=cfg.extra_lag,
    )
    controls = control_dates(
        pair_ret.index,
        events,
        n=len(events),
        seed=cfg.control_seed,
        exclude_window=cfg.min_gap_days,
    )
    summaries = {}
    for ccy in usd_panel.columns:
        summaries[ccy] = summarize_event_vs_control(
            usd_panel[ccy], events, controls, pre=cfg.pre, post=cfg.post
        )
    summaries["USD_BASKET"] = summarize_event_vs_control(
        basket, events, controls, pre=cfg.pre, post=cfg.post
    )
    signal = evaluate_signal_gate(summaries["USD_BASKET"], cfg=cfg)
    return {
        "events": pd.DataFrame({"event_date": events}),
        "controls": pd.DataFrame({"control_date": controls}),
        "summaries": summaries,
        "signal": signal,
        "meta": {
            "n_events": int(len(events)),
            "n_controls": int(len(controls)),
            "threshold": float(ev_meta.get("threshold", float("nan"))),
            "decile": cfg.decile,
            "currencies": list(cfg.currencies),
        },
    }


def evaluate_signal_gate(
    basket_summary: pd.DataFrame,
    *,
    cfg: NewsEventStudyConfig | None = None,
) -> dict:
    """Whether post-event USD basket edge vs control clears fixed prior gates."""
    cfg = cfg or NewsEventStudyConfig()
    row = basket_summary.loc[basket_summary["horizon"] == cfg.signal_horizon]
    if row.empty:
        return {
            "trade": False,
            "reason": f"missing horizon h={cfg.signal_horizon}",
            "diff": float("nan"),
            "tstat": float("nan"),
            "horizon": cfg.signal_horizon,
        }
    r = row.iloc[0]
    diff = float(r["diff_event_minus_control"])
    tstat = float(r["tstat_diff"])
    ok = (
        np.isfinite(diff)
        and np.isfinite(tstat)
        and diff > cfg.min_edge
        and tstat >= cfg.min_t
    )
    return {
        "trade": bool(ok),
        "reason": (
            "USD basket event−control edge clears min_edge/min_t"
            if ok
            else "no reliable USD edge vs control at fixed prior gate"
        ),
        "diff": diff,
        "tstat": tstat,
        "horizon": int(cfg.signal_horizon),
        "event_cum_mean": float(r["event_cum_mean"]),
        "control_cum_mean": float(r["control_cum_mean"]),
    }


def lagged_usd_event_strategy_returns(
    pair_ret: pd.DataFrame,
    intensity: pd.Series,
    *,
    cfg: NewsEventStudyConfig | None = None,
    force_trade: bool | None = None,
) -> pd.Series:
    """Long USD basket for ``hold_days`` after lagged high-intensity events.

    If ``force_trade`` is None, uses ``evaluate_signal_gate`` on the full-sample
    event study (research board only — not walk-forward gated). Costs haircut
    on position changes. Returns zeros if gate fails (unless force_trade=True).
    """
    cfg = cfg or NewsEventStudyConfig()
    study = run_news_event_study(pair_ret, intensity, cfg=cfg)
    trade = bool(study["signal"]["trade"]) if force_trade is None else bool(force_trade)
    basket = usd_basket_returns(pair_ret, currencies=cfg.currencies)
    out = pd.Series(0.0, index=pair_ret.index, name="news_usd_event")
    if not trade:
        out.attrs["trade_enabled"] = False
        out.attrs["signal"] = study["signal"]
        return out

    events, _ev_meta = high_intensity_event_dates(
        intensity,
        pair_ret.index,
        decile=cfg.decile,
        min_gap_days=cfg.min_gap_days,
        extra_lag=cfg.extra_lag,
    )
    pos = pd.Series(0.0, index=pair_ret.index)
    idx = pair_ret.index
    for e in events:
        loc = idx.get_indexer([e], method="pad")[0]
        if loc < 0:
            continue
        # Enter after signal_lag bars; hold hold_days (inclusive of entry bar count)
        start = loc + int(cfg.signal_lag)
        end = start + int(cfg.hold_days)
        if start >= len(idx):
            continue
        pos.iloc[start : min(end, len(idx))] = 1.0

    # Apply costs on |Δposition|
    turnover = pos.diff().abs().fillna(pos.iloc[0])
    cost = turnover * (2.0 * cfg.cost_bps_per_side / 10_000.0)  # round-trip style per flip unit
    # Actually per-side on each unit change: cost_bps_per_side / 1e4 * |Δw|
    cost = turnover * (cfg.cost_bps_per_side / 10_000.0)
    raw = pos.shift(0) * basket  # position known at open of bar after lag already in start
    # Causality: position for day t uses info through t-1 (event lagged). Returns at t realized with pos_t.
    net = raw - cost
    net.name = "news_usd_event"
    net.attrs["trade_enabled"] = True
    net.attrs["signal"] = study["signal"]
    net.attrs["cost_bps_per_side"] = float(cfg.cost_bps_per_side)
    net.attrs["hold_days"] = int(cfg.hold_days)
    net.attrs["n_events"] = int(len(events))
    return net.fillna(0.0)
