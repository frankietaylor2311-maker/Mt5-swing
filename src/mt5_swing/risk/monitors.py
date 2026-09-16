"""
Drawdown monitors and kill-switch.

Daily DD definition
-------------------
Daily drawdown at timestamp t is:

    daily_dd(t) = (E_day_open - E_t) / E_day_open

where ``E_day_open`` is equity at the last bar of the previous calendar day
(UTC date of the bar index), i.e. equity vs prior calendar-day close.
If the current day is the first day in the series, E_day_open = starting equity.

Peak-to-trough (max) DD:

    max_dd(t) = (peak_equity - E_t) / peak_equity

On breach of configured limits: halt new entries; optionally flatten if past
``flatten_threshold`` (defaults to the same limit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd


@dataclass
class RiskLimits:
    max_peak_to_trough_dd: float = 0.10  # 10%
    max_daily_dd: float = 0.05  # 5%
    flatten_on_breach: bool = True
    # Flatten if DD exceeds this; None → same as the breached limit
    flatten_threshold_max_dd: float | None = None
    flatten_threshold_daily_dd: float | None = None


@dataclass
class DrawdownState:
    equity: float
    peak_equity: float
    day_open_equity: float
    current_date: date | None
    peak_to_trough_dd: float = 0.0
    daily_dd: float = 0.0
    halted: bool = False
    flatten_requested: bool = False
    breach_reason: str | None = None


@dataclass
class MaxDDMonitor:
    limit: float = 0.10
    flatten_threshold: float | None = None
    peak: float = 0.0
    current_dd: float = 0.0

    def update(self, equity: float) -> tuple[float, bool, bool]:
        """Return (dd, halt_new_entries, flatten)."""
        if equity > self.peak:
            self.peak = equity
        dd = 0.0 if self.peak <= 0 else (self.peak - equity) / self.peak
        self.current_dd = max(0.0, dd)
        halt = self.current_dd >= self.limit
        thresh = self.flatten_threshold if self.flatten_threshold is not None else self.limit
        flatten = self.current_dd >= thresh
        return self.current_dd, halt, flatten


@dataclass
class DailyDDMonitor:
    """
    Tracks equity vs prior calendar-day close.

    Call ``update(timestamp, equity)`` on each bar in chronological order.
    """

    limit: float = 0.05
    flatten_threshold: float | None = None
    day_open_equity: float | None = None
    current_date: date | None = None
    prior_day_close_equity: float | None = None
    last_equity: float | None = None
    current_dd: float = 0.0

    def update(self, ts: pd.Timestamp, equity: float) -> tuple[float, bool, bool]:
        d = ts.tz_convert("UTC").date() if ts.tzinfo else ts.date()
        if self.current_date is None:
            self.current_date = d
            self.day_open_equity = equity
            self.prior_day_close_equity = equity
        elif d != self.current_date:
            # New calendar day: prior close is last equity of previous day
            self.prior_day_close_equity = self.last_equity
            self.day_open_equity = self.prior_day_close_equity
            self.current_date = d

        base = self.day_open_equity if self.day_open_equity and self.day_open_equity > 0 else equity
        dd = max(0.0, (base - equity) / base)
        self.current_dd = dd
        self.last_equity = equity
        halt = dd >= self.limit
        thresh = self.flatten_threshold if self.flatten_threshold is not None else self.limit
        flatten = dd >= thresh
        return dd, halt, flatten


@dataclass
class KillSwitch:
    """Combines max DD + daily DD monitors; manages halt/flatten flags."""

    limits: RiskLimits = field(default_factory=RiskLimits)
    max_dd: MaxDDMonitor = field(init=False)
    daily_dd: DailyDDMonitor = field(init=False)
    halted: bool = False
    flatten_requested: bool = False
    breach_reason: str | None = None
    # Once halted, stay halted for the session unless reset
    sticky_halt: bool = True

    def __post_init__(self) -> None:
        lim = self.limits
        self.max_dd = MaxDDMonitor(
            limit=lim.max_peak_to_trough_dd,
            flatten_threshold=lim.flatten_threshold_max_dd,
        )
        self.daily_dd = DailyDDMonitor(
            limit=lim.max_daily_dd,
            flatten_threshold=lim.flatten_threshold_daily_dd,
        )

    def reset(self, starting_equity: float) -> None:
        self.max_dd = MaxDDMonitor(
            limit=self.limits.max_peak_to_trough_dd,
            flatten_threshold=self.limits.flatten_threshold_max_dd,
            peak=starting_equity,
        )
        self.daily_dd = DailyDDMonitor(
            limit=self.limits.max_daily_dd,
            flatten_threshold=self.limits.flatten_threshold_daily_dd,
        )
        self.halted = False
        self.flatten_requested = False
        self.breach_reason = None

    def update(self, ts: pd.Timestamp, equity: float) -> DrawdownState:
        mdd, halt_m, flat_m = self.max_dd.update(equity)
        ddd, halt_d, flat_d = self.daily_dd.update(ts, equity)
        reasons = []
        if halt_m:
            reasons.append(f"max_dd={mdd:.2%}≥{self.limits.max_peak_to_trough_dd:.2%}")
        if halt_d:
            reasons.append(f"daily_dd={ddd:.2%}≥{self.limits.max_daily_dd:.2%}")
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
            halted=self.halted,
            flatten_requested=flatten,
            breach_reason=self.breach_reason,
        )

    def allow_new_entry(self) -> bool:
        return not self.halted
