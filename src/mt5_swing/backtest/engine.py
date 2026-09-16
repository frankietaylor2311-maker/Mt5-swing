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
from mt5_swing.risk.sizing import atr_position_size, fixed_fractional_size, volatility_scale


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
    vol_target: bool = False
    vol_lookback: int = 100
    max_lot: float = 2.0
    min_lot: float = 0.01
    max_loss_mode: str = "static_initial"  # static_initial | peak_to_trough
    daily_loss_mode: str = "ftmo_initial"  # ftmo_initial | pct_of_day_open
    daily_tz: str = "Europe/Prague"
    use_atr_exits: bool = True  # ATR stop (+ optional target) after entry bar
    atr_target_mult: float = 3.0  # 0 disables take-profit; stop uses atr_stop_mult
    no_same_bar_exit: bool = True  # conservative: first check stops on bar after entry
    atr_trail_mult: float = 0.0  # >0 enables ATR trailing stop from favorable extreme
    max_hold_bars: int = 0  # >0 flat after N bars in trade (time stop)


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
    entry_bar = -1
    stop_price = float("nan")
    target_price = float("nan")
    extreme_px = float("nan")  # favorable extreme for trailing stop
    eq_curve: list[float] = []
    pos_series: list[int] = []
    trades: list[dict] = []
    kill_events: list[dict] = []

    ks = KillSwitch(
        RiskLimits(
            max_peak_to_trough_dd=cfg.max_dd,
            max_daily_dd=cfg.daily_dd,
            flatten_on_breach=cfg.flatten_on_breach,
            max_loss_mode=cfg.max_loss_mode,  # type: ignore[arg-type]
            daily_loss_mode=cfg.daily_loss_mode,  # type: ignore[arg-type]
            daily_tz=cfg.daily_tz,
            initial_capital=cfg.initial_equity,
        )
    )
    ks.reset(cfg.initial_equity)

    idx = data.index
    opens = data["open"].to_numpy(dtype=float)
    highs = data["high"].to_numpy(dtype=float)
    lows = data["low"].to_numpy(dtype=float)
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

    def _close_at(i: int, fill: float, reason: str) -> None:
        nonlocal equity, position, lots, entry_price, entry_time, entry_bar, stop_price, target_price
        if position == 0 or lots <= 0:
            return
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
        entry_bar = -1
        stop_price = float("nan")
        target_price = float("nan")

    def _close_position(i: int, bar_spread: float, reason: str) -> None:
        if position == 0 or lots <= 0:
            return
        fill = opens[i] + _cost_offset(-position, bar_spread)
        _close_at(i, fill, reason)

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

        # ATR stop / take-profit / trail / time-stop (no same-bar exit by default)
        if position != 0 and lots > 0 and (not cfg.no_same_bar_exit or i > entry_bar):
            atr_v_exit = atrs[i]
            # Update trailing stop from favorable extreme (causal: uses this bar's high/low)
            if (
                cfg.use_atr_exits
                and cfg.atr_trail_mult
                and cfg.atr_trail_mult > 0
                and atr_v_exit == atr_v_exit
                and atr_v_exit > 0
            ):
                if position > 0:
                    extreme_px = highs[i] if extreme_px != extreme_px else max(extreme_px, highs[i])
                    trail = extreme_px - cfg.atr_trail_mult * float(atr_v_exit)
                    if stop_price != stop_price or trail > stop_price:
                        stop_price = trail
                else:
                    extreme_px = lows[i] if extreme_px != extreme_px else min(extreme_px, lows[i])
                    trail = extreme_px + cfg.atr_trail_mult * float(atr_v_exit)
                    if stop_price != stop_price or trail < stop_price:
                        stop_price = trail
            if cfg.use_atr_exits and stop_price == stop_price:
                hit_stop = False
                hit_tp = False
                if position > 0:
                    hit_stop = lows[i] <= stop_price
                    hit_tp = (target_price == target_price) and highs[i] >= target_price
                else:
                    hit_stop = highs[i] >= stop_price
                    hit_tp = (target_price == target_price) and lows[i] <= target_price
                if hit_stop and hit_tp:
                    # Conservative: assume stop hit first when both in same bar
                    _close_at(i, float(stop_price), "atr_stop")
                    target = 0
                elif hit_stop:
                    reason = "atr_trail" if (cfg.atr_trail_mult and cfg.atr_trail_mult > 0) else "atr_stop"
                    _close_at(i, float(stop_price), reason)
                    target = 0
                elif hit_tp:
                    _close_at(i, float(target_price), "atr_target")
                    target = 0
            # Time stop after ATR checks (still no same-bar by gate above)
            if position != 0 and cfg.max_hold_bars and cfg.max_hold_bars > 0:
                if (i - entry_bar) >= cfg.max_hold_bars:
                    _close_position(i, bar_spread, "time_stop")
                    target = 0

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
                        min_lot=cfg.min_lot,
                        max_lot=cfg.max_lot,
                    )
                elif cfg.sizing == "fixed":
                    new_lots = cfg.fixed_lot
                else:
                    new_lots = fixed_fractional_size(equity, min_lot=cfg.min_lot, max_lot=cfg.max_lot)
                if cfg.vol_target and atr_v == atr_v and atr_v > 0 and i >= 1:
                    # Causal median ATR over lookback ending at i-1 / current lagged atr
                    lo = max(0, i - cfg.vol_lookback)
                    window = atrs[lo : i + 1]
                    window = window[~np.isnan(window)]
                    if len(window) >= 10:
                        med = float(np.median(window))
                        new_lots *= volatility_scale(float(atr_v), med)
                        new_lots = max(cfg.min_lot, min(cfg.max_lot, round(int(new_lots / 0.01) * 0.01, 2)))
                if new_lots > 0:
                    fill = opens[i] + _cost_offset(desired, bar_spread)
                    entry_price = fill
                    entry_time = ts
                    entry_bar = i
                    position = desired
                    lots = new_lots
                    atr_v2 = atrs[i]
                    extreme_px = fill
                    if cfg.use_atr_exits and atr_v2 == atr_v2 and atr_v2 > 0:
                        stop_price = fill - desired * cfg.atr_stop_mult * float(atr_v2)
                        if cfg.atr_target_mult and cfg.atr_target_mult > 0:
                            target_price = fill + desired * cfg.atr_target_mult * float(atr_v2)
                        else:
                            target_price = float("nan")
                    else:
                        stop_price = float("nan")
                        target_price = float("nan")

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
        max_loss_mode=cfg.max_loss_mode,  # type: ignore[arg-type]
        daily_loss_mode=cfg.daily_loss_mode,  # type: ignore[arg-type]
        daily_tz=cfg.daily_tz,
        initial_equity=cfg.initial_equity,
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
