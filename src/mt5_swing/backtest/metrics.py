"""Performance and risk metrics from an equity curve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from mt5_swing.risk.monitors import FTMO_DAILY_TZ


@dataclass
class Metrics:
    total_return: float
    max_drawdown: float  # peak-to-trough (informational)
    max_daily_dd: float  # per configured daily_loss_mode
    static_loss_from_initial: float  # (initial − min_equity) / initial
    sharpe: float
    n_trades: int
    win_rate: float
    profit_factor: float
    final_equity: float
    start_equity: float
    passed_max_dd_gate: bool
    passed_daily_dd_gate: bool
    gates_pass: bool
    max_loss_mode: str = "static_initial"
    daily_loss_mode: str = "ftmo_initial"
    daily_tz: str = FTMO_DAILY_TZ

    def as_dict(self) -> dict:
        return {
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "max_daily_dd": self.max_daily_dd,
            "static_loss_from_initial": self.static_loss_from_initial,
            "sharpe": self.sharpe,
            "n_trades": self.n_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "final_equity": self.final_equity,
            "start_equity": self.start_equity,
            "passed_max_dd_gate": self.passed_max_dd_gate,
            "passed_daily_dd_gate": self.passed_daily_dd_gate,
            "gates_pass": self.gates_pass,
            "max_loss_mode": self.max_loss_mode,
            "daily_loss_mode": self.daily_loss_mode,
            "daily_tz": self.daily_tz,
        }


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0, np.nan)
    return float(dd.max()) if len(dd) else 0.0


def _static_loss_from_initial(equity: pd.Series, initial: float) -> float:
    if equity.empty or initial <= 0:
        return 0.0
    return float(max(0.0, (initial - float(equity.min())) / initial))


def _max_daily_loss(
    equity: pd.Series,
    *,
    initial: float,
    mode: Literal["ftmo_initial", "pct_of_day_open"] = "ftmo_initial",
    tz: str = FTMO_DAILY_TZ,
) -> float:
    """
    Max daily loss over the series.

    FTMO 2-Step: for each Prague calendar day, (day_open_equity − intraday_equity) / initial.
    Legacy: (day_open − equity) / day_open.
    """
    if equity.empty:
        return 0.0
    eq = equity.copy()
    eq.index = pd.to_datetime(eq.index, utc=True)
    local_dates = eq.index.tz_convert(tz).date
    max_ddd = 0.0
    current_day = local_dates[0]
    day_open = float(eq.iloc[0])
    prev_close = day_open
    for val, d in zip(eq.values, local_dates):
        if d != current_day:
            day_open = prev_close
            current_day = d
        if mode == "ftmo_initial":
            denom = initial if initial > 0 else day_open
            ddd = max(0.0, (day_open - float(val)) / denom) if denom > 0 else 0.0
        else:
            ddd = max(0.0, (day_open - float(val)) / day_open) if day_open > 0 else 0.0
        max_ddd = max(max_ddd, ddd)
        prev_close = float(val)
    return max_ddd


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame | None = None,
    *,
    max_dd_gate: float = 0.10,
    daily_dd_gate: float = 0.05,
    periods_per_year: float = 252 * 6,
    max_loss_mode: Literal["static_initial", "peak_to_trough"] = "static_initial",
    daily_loss_mode: Literal["ftmo_initial", "pct_of_day_open"] = "ftmo_initial",
    daily_tz: str = FTMO_DAILY_TZ,
    initial_equity: float | None = None,
) -> Metrics:
    start = float(equity.iloc[0]) if len(equity) else 0.0
    initial = float(initial_equity) if initial_equity is not None else start
    end = float(equity.iloc[-1]) if len(equity) else 0.0
    total_return = (end / start - 1.0) if start > 0 else 0.0
    max_dd = _max_drawdown(equity)
    static_loss = _static_loss_from_initial(equity, initial)
    max_ddd = _max_daily_loss(equity, initial=initial, mode=daily_loss_mode, tz=daily_tz)
    rets = equity.pct_change().dropna()
    if len(rets) > 1 and rets.std() > 0:
        sharpe = float(np.sqrt(periods_per_year) * rets.mean() / rets.std())
    else:
        sharpe = 0.0

    n_trades = 0
    win_rate = 0.0
    profit_factor = 0.0
    if trades is not None and len(trades):
        n_trades = len(trades)
        wins = trades.loc[trades["pnl"] > 0, "pnl"]
        losses = trades.loc[trades["pnl"] < 0, "pnl"]
        win_rate = float((trades["pnl"] > 0).mean()) if n_trades else 0.0
        gp = float(wins.sum()) if len(wins) else 0.0
        gl = float(-losses.sum()) if len(losses) else 0.0
        profit_factor = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)

    if max_loss_mode == "static_initial":
        pass_mdd = static_loss < max_dd_gate
    else:
        pass_mdd = max_dd < max_dd_gate
    pass_ddd = max_ddd < daily_dd_gate
    return Metrics(
        total_return=total_return,
        max_drawdown=max_dd,
        max_daily_dd=max_ddd,
        static_loss_from_initial=static_loss,
        sharpe=sharpe,
        n_trades=n_trades,
        win_rate=win_rate,
        profit_factor=profit_factor if np.isfinite(profit_factor) else 999.0,
        final_equity=end,
        start_equity=start,
        passed_max_dd_gate=pass_mdd,
        passed_daily_dd_gate=pass_ddd,
        gates_pass=pass_mdd and pass_ddd,
        max_loss_mode=max_loss_mode,
        daily_loss_mode=daily_loss_mode,
        daily_tz=daily_tz,
    )
