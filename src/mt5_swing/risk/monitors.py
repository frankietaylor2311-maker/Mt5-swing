"""
Drawdown monitors and kill-switch.

FTMO Challenge: 2-Step (default for prop research)
-------------------------------------------------
- **Max Daily Loss (5%)**: equity may not fall below
  ``balance_at_00:00_Europe/Prague − 0.05 × initial_capital``.
  Equivalently, loss from the Prague midnight balance scaled by **initial
  capital** (not by day-open equity %) must stay &lt; 5%.
  Day boundary = calendar date in ``Europe/Prague`` (CET/CEST), matching FTMO
  CE(S)T reset — **not** UTC midnight.

- **Max Loss (10%)**: **static** from initial simulated capital.
  Equity must stay above ``0.90 × initial_capital``.
  This is **not** peak-to-trough trailing DD. Peak-to-trough remains available
  as an informational / alternate mode for non-FTMO research.

Legacy / research modes
-----------------------
- ``max_loss_mode="peak_to_trough"``: classic (peak − equity) / peak
- ``daily_loss_mode="pct_of_day_open"``: (day_open − equity) / day_open

On breach: halt new entries; optionally flatten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd

FTMO_DAILY_TZ = "Europe/Prague"


@dataclass
class RiskLimits:
    max_peak_to_trough_dd: float = 0.10
    max_daily_dd: float = 0.05
    flatten_on_breach: bool = True
    flatten_threshold_max_dd: float | None = None
    flatten_threshold_daily_dd: float | None = None
    # FTMO 2-Step defaults
    max_loss_mode: Literal["static_initial", "peak_to_trough"] = "static_initial"
    daily_loss_mode: Literal["ftmo_initial", "pct_of_day_open"] = "ftmo_initial"
    daily_tz: str = FTMO_DAILY_TZ
    initial_capital: float | None = None  # required for FTMO modes; set on reset


@dataclass
class DrawdownState:
    equity: float
    peak_equity: float
    day_open_equity: float
    current_date: date | None
    peak_to_trough_dd: float = 0.0
    daily_dd: float = 0.0
    static_loss_from_initial: float = 0.0
    halted: bool = False
    flatten_requested: bool = False
    breach_reason: str | None = None


@dataclass
class MaxDDMonitor:
    """Peak-to-trough drawdown (research / non-FTMO alternate)."""

    limit: float = 0.10
    flatten_threshold: float | None = None
    peak: float = 0.0
    current_dd: float = 0.0

    def update(self, equity: float) -> tuple[float, bool, bool]:
        if equity > self.peak:
            self.peak = equity
        dd = 0.0 if self.peak <= 0 else (self.peak - equity) / self.peak
        self.current_dd = max(0.0, dd)
        halt = self.current_dd >= self.limit
        thresh = self.flatten_threshold if self.flatten_threshold is not None else self.limit
        flatten = self.current_dd >= thresh
        return self.current_dd, halt, flatten


@dataclass
class StaticMaxLossMonitor:
    """
    FTMO 2-Step Max Loss: static floor at (1 − limit) × initial_capital.

    ``current_loss = (initial − equity) / initial`` (0 if equity ≥ initial).
    """

    limit: float = 0.10
    initial_capital: float = 0.0
    flatten_threshold: float | None = None
    current_loss: float = 0.0
    min_equity: float = 0.0

    def update(self, equity: float) -> tuple[float, bool, bool]:
        if self.initial_capital <= 0:
            return 0.0, False, False
        self.min_equity = equity if self.min_equity <= 0 else min(self.min_equity, equity)
        loss = max(0.0, (self.initial_capital - equity) / self.initial_capital)
        self.current_loss = loss
        halt = loss >= self.limit
        thresh = self.flatten_threshold if self.flatten_threshold is not None else self.limit
        flatten = loss >= thresh
        return loss, halt, flatten


@dataclass
class DailyDDMonitor:
    """
    Daily loss monitor.

    ``daily_loss_mode``:
    - ``ftmo_initial`` (FTMO 2-Step): loss from Prague midnight balance /
      **initial_capital** ≥ limit → breach.
    - ``pct_of_day_open``: (day_open − equity) / day_open ≥ limit.

    Day boundary uses ``tz`` (default Europe/Prague).
    """

    limit: float = 0.05
    flatten_threshold: float | None = None
    daily_loss_mode: Literal["ftmo_initial", "pct_of_day_open"] = "ftmo_initial"
    tz: str = FTMO_DAILY_TZ
    initial_capital: float = 0.0
    day_open_equity: float | None = None
    current_date: date | None = None
    prior_day_close_equity: float | None = None
    last_equity: float | None = None
    current_dd: float = 0.0

    def _local_date(self, ts: pd.Timestamp) -> date:
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(self.tz).date()

    def update(self, ts: pd.Timestamp, equity: float) -> tuple[float, bool, bool]:
        d = self._local_date(ts)
        if self.current_date is None:
            self.current_date = d
            self.day_open_equity = equity
            self.prior_day_close_equity = equity
        elif d != self.current_date:
            self.prior_day_close_equity = self.last_equity
            self.day_open_equity = self.prior_day_close_equity
            self.current_date = d

        base = self.day_open_equity if self.day_open_equity and self.day_open_equity > 0 else equity
        if self.daily_loss_mode == "ftmo_initial":
            denom = self.initial_capital if self.initial_capital > 0 else base
            dd = max(0.0, (base - equity) / denom)
        else:
            dd = max(0.0, (base - equity) / base) if base > 0 else 0.0

        self.current_dd = dd
        self.last_equity = equity
        halt = dd >= self.limit
        thresh = self.flatten_threshold if self.flatten_threshold is not None else self.limit
        flatten = dd >= thresh
        return dd, halt, flatten


@dataclass
class KillSwitch:
    """Combines max-loss + daily-loss monitors; manages halt/flatten flags."""

    limits: RiskLimits = field(default_factory=RiskLimits)
    max_dd: MaxDDMonitor = field(init=False)
    static_loss: StaticMaxLossMonitor = field(init=False)
    daily_dd: DailyDDMonitor = field(init=False)
    halted: bool = False
    flatten_requested: bool = False
    breach_reason: str | None = None
    sticky_halt: bool = True

    def __post_init__(self) -> None:
        lim = self.limits
        init = float(lim.initial_capital or 0.0)
        self.max_dd = MaxDDMonitor(
            limit=lim.max_peak_to_trough_dd,
            flatten_threshold=lim.flatten_threshold_max_dd,
        )
        self.static_loss = StaticMaxLossMonitor(
            limit=lim.max_peak_to_trough_dd,
            initial_capital=init,
            flatten_threshold=lim.flatten_threshold_max_dd,
        )
        self.daily_dd = DailyDDMonitor(
            limit=lim.max_daily_dd,
            flatten_threshold=lim.flatten_threshold_daily_dd,
            daily_loss_mode=lim.daily_loss_mode,
            tz=lim.daily_tz,
            initial_capital=init,
        )

    def reset(self, starting_equity: float) -> None:
        lim = self.limits
        lim.initial_capital = starting_equity
        self.max_dd = MaxDDMonitor(
            limit=lim.max_peak_to_trough_dd,
            flatten_threshold=lim.flatten_threshold_max_dd,
            peak=starting_equity,
        )
        self.static_loss = StaticMaxLossMonitor(
            limit=lim.max_peak_to_trough_dd,
            initial_capital=starting_equity,
            flatten_threshold=lim.flatten_threshold_max_dd,
            min_equity=starting_equity,
        )
        self.daily_dd = DailyDDMonitor(
            limit=lim.max_daily_dd,
            flatten_threshold=lim.flatten_threshold_daily_dd,
            daily_loss_mode=lim.daily_loss_mode,
            tz=lim.daily_tz,
            initial_capital=starting_equity,
        )
        self.halted = False
        self.flatten_requested = False
        self.breach_reason = None

    def update(self, ts: pd.Timestamp, equity: float) -> DrawdownState:
        mdd, halt_m_pt, flat_m_pt = self.max_dd.update(equity)
        static_loss, halt_m_st, flat_m_st = self.static_loss.update(equity)
        ddd, halt_d, flat_d = self.daily_dd.update(ts, equity)

        if self.limits.max_loss_mode == "static_initial":
            halt_m, flat_m, m_label, m_val = halt_m_st, flat_m_st, "static_max_loss", static_loss
        else:
            halt_m, flat_m, m_label, m_val = halt_m_pt, flat_m_pt, "max_dd", mdd

        reasons = []
        if halt_m:
            reasons.append(f"{m_label}={m_val:.2%}≥{self.limits.max_peak_to_trough_dd:.2%}")
        if halt_d:
            reasons.append(f"daily_loss={ddd:.2%}≥{self.limits.max_daily_dd:.2%}")
        if reasons:
            if not self.halted or not self.sticky_halt:
                self.breach_reason = "; ".join(reasons)
            self.halted = True
        flatten = False
        if self.limits.flatten_on_breach and (flat_m or flat_d):
            flatten = True
            self.flatten_requested = True
        return DrawdownState(
            equity=equity,
            peak_equity=self.max_dd.peak,
            day_open_equity=self.daily_dd.day_open_equity or equity,
            current_date=self.daily_dd.current_date,
            peak_to_trough_dd=mdd,
            daily_dd=ddd,
            static_loss_from_initial=static_loss,
            halted=self.halted,
            flatten_requested=flatten,
            breach_reason=self.breach_reason,
        )

    def allow_new_entry(self) -> bool:
        return not self.halted
