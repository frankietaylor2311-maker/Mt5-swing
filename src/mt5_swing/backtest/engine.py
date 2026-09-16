"""
Simple bar backtester.

Execution model (anti look-ahead):
- Signals are computed from lagged features (see features.apply_feature_pipeline).
- Target position at bar i is taken from signal[i]; fills occur at open[i] when
  signal changes vs prior bar (signal already lagged by 1 by default, so this is
  effectively open of the bar after signal formation).
- Costs: spread (from bar or config) + commission per lot + slippage in price units.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mt5_swing.backtest.metrics import Metrics, compute_metrics
from mt5_swing.data.symbols import get_symbol_meta, pip_value_per_lot
from mt5_swing.features.indicators import apply_feature_pipeline
from mt5_swing.risk.monitors import KillSwitch, RiskLimits
from mt5_swing.risk.sizing import atr_position_size, fixed_fractional_size


@dataclass
class BacktestConfig:
    symbol: str = "EURUSD"
    initial_equity: float = 10_000.0
    commission_per_lot: float = 7.0  # round-turn USD approx
    slippage_pips: float = 0.5
    default_spread_pips: float = 1.2
    risk_fraction: float = 0.01
    atr_stop_mult: float = 2.0
    sizing: str = "atr"  # atr | fixed
    fixed_lot: float = 0.1
    signal_lag: int = 1
    max_dd: float = 0.10
    daily_dd: float = 0.05
    flatten_on_breach: bool = True
    periods_per_year: float = 252 * 6


@dataclass
class BacktestResult:
    equity: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: Metrics
    signals: pd.Series
    kill_events: list[dict] = field(default_factory=list)
    config: BacktestConfig | None = None


def run_backtest(
    ohlc: pd.DataFrame,
    strategy,
    config: BacktestConfig | None = None,
    *,
    features: pd.DataFrame | None = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    meta = get_symbol_meta(cfg.symbol)
    pip = meta.pip_size

    data = (
        features
        if features is not None
        else apply_feature_pipeline(ohlc, signal_lag=cfg.signal_lag)
    )
    for c in ("open", "high", "low", "close"):
        if c not in data.columns:
            data[c] = ohlc[c]

    signals = strategy.generate_signals(data).astype(int)
    signals = signals.reindex(data.index).fillna(0).astype(int)

    equity = float(cfg.initial_equity)
    position = 0
    lots = 0.0
    entry_price = 0.0
    entry_time = None
    eq_curve: list[float] = []
    pos_series: list[int] = []
    trades: list[dict] = []
    kill_events: list[dict] = []

    ks = KillSwitch(
        RiskLimits(
            max_peak_to_trough_dd=cfg.max_dd,
            max_daily_dd=cfg.daily_dd,
            flatten_on_breach=cfg.flatten_on_breach,
        )
    )
    ks.reset(cfg.initial_equity)

    idx = data.index
    opens = data["open"].to_numpy(dtype=float)
    closes = data["close"].to_numpy(dtype=float)
    atrs = (
        data["atr"].to_numpy(dtype=float)
        if "atr" in data.columns
        else np.full(len(data), np.nan)
    )
    spreads_col = (
        data["spread"].to_numpy(dtype=float) if "spread" in data.columns else None
    )
    sigs = signals.to_numpy(dtype=int)

    def _cost_offset(side_sign: int, bar_spread: float) -> float:
        slip = cfg.slippage_pips * pip
        half_spread = (
            bar_spread / 2.0
            if bar_spread > 0
            else (cfg.default_spread_pips * pip) / 2.0
        )
        return side_sign * (half_spread + slip)

    def _pnl_usd(direction: int, trade_lots: float, exit_px: float, entry_px: float) -> float:
        raw = direction * trade_lots * meta.contract_size * (exit_px - entry_px)
        if meta.quote_currency != "USD" and exit_px > 0:
            raw = raw / exit_px
        return raw - abs(trade_lots) * cfg.commission_per_lot

    def _close_position(i: int, bar_spread: float, reason: str) -> None:
        nonlocal equity, position, lots, entry_price, entry_time
        if position == 0 or lots <= 0:
            return
        fill = opens[i] + _cost_offset(-position, bar_spread)
        pnl = _pnl_usd(position, lots, fill, entry_price)
        equity += pnl
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": idx[i],
                "direction": position,
                "lots": lots,
                "entry_price": entry_price,
                "exit_price": fill,
                "pnl": pnl,
                "reason": reason,
            }
        )
        position = 0
        lots = 0.0
        entry_price = 0.0
        entry_time = None

    for i in range(len(data)):
        ts = idx[i]
        target = int(sigs[i])
        if spreads_col is not None:
            bar_spread = float(spreads_col[i])
            if np.isnan(bar_spread) or bar_spread <= 0:
                bar_spread = cfg.default_spread_pips * pip
        else:
            bar_spread = cfg.default_spread_pips * pip

        # Mark-to-market before risk check
        mtm = equity
        if position != 0 and lots > 0:
            unreal = position * lots * meta.contract_size * (closes[i] - entry_price)
            if meta.quote_currency != "USD" and closes[i] > 0:
                unreal = unreal / closes[i]
            mtm = equity + unreal

        state = ks.update(ts, mtm)

        if state.flatten_requested and position != 0:
            _close_position(i, bar_spread, "kill_switch_flatten")
            kill_events.append(
                {"time": ts, "reason": state.breach_reason, "equity": equity}
            )
            target = 0

        if not ks.allow_new_entry():
            if position == 0:
                target = 0
            elif int(np.sign(target)) != position:
                target = 0  # only allow flatten, not reverse

        desired = int(np.sign(target))

        if desired != position:
            if position != 0:
                _close_position(i, bar_spread, "signal")
            if desired != 0 and ks.allow_new_entry():
                atr_v = atrs[i]
                if cfg.sizing == "atr" and atr_v == atr_v and atr_v > 0:
                    new_lots = atr_position_size(
                        equity,
                        float(atr_v),
                        risk_fraction=cfg.risk_fraction,
                        atr_stop_mult=cfg.atr_stop_mult,
                        pip_size=pip,
                        pip_value=pip_value_per_lot(cfg.symbol, float(opens[i])),
                    )
                elif cfg.sizing == "fixed":
                    new_lots = cfg.fixed_lot
                else:
                    new_lots = fixed_fractional_size(equity)
                if new_lots > 0:
                    fill = opens[i] + _cost_offset(desired, bar_spread)
                    entry_price = fill
                    entry_time = ts
                    position = desired
                    lots = new_lots

        mtm = equity
        if position != 0 and lots > 0:
            unreal = position * lots * meta.contract_size * (closes[i] - entry_price)
            if meta.quote_currency != "USD" and closes[i] > 0:
                unreal = unreal / closes[i]
            mtm = equity + unreal

        eq_curve.append(mtm)
        pos_series.append(position)

    equity_s = pd.Series(eq_curve, index=idx, name="equity")
    pos_s = pd.Series(pos_series, index=idx, name="position")
    trades_df = pd.DataFrame(trades)
    metrics = compute_metrics(
        equity_s,
        trades_df if len(trades_df) else None,
        max_dd_gate=cfg.max_dd,
        daily_dd_gate=cfg.daily_dd,
        periods_per_year=cfg.periods_per_year,
    )
    return BacktestResult(
        equity=equity_s,
        positions=pos_s,
        trades=trades_df,
        metrics=metrics,
        signals=signals,
        kill_events=kill_events,
        config=cfg,
    )
