"""Risk monitor unit tests — daily DD vs prior calendar-day close."""

from __future__ import annotations

import pandas as pd

from mt5_swing.risk.monitors import DailyDDMonitor, KillSwitch, MaxDDMonitor, RiskLimits


def test_max_dd_monitor():
    m = MaxDDMonitor(limit=0.10)
    m.update(10_000)
    dd, halt, flat = m.update(9_500)
    assert abs(dd - 0.05) < 1e-9
    assert not halt
    dd, halt, flat = m.update(8_900)
    assert dd >= 0.10
    assert halt and flat


def test_daily_dd_prior_calendar_close():
    mon = DailyDDMonitor(limit=0.05)
    # Day 1
    t0 = pd.Timestamp("2024-01-01 08:00", tz="UTC")
    t1 = pd.Timestamp("2024-01-01 16:00", tz="UTC")
    mon.update(t0, 10_000)
    mon.update(t1, 10_200)  # day close equity path ends at 10200
    # Day 2: prior close = 10200
    t2 = pd.Timestamp("2024-01-02 08:00", tz="UTC")
    dd, halt, _ = mon.update(t2, 9_690)  # ~5.0% from 10200
    assert dd >= 0.05
    assert halt


def test_kill_switch_halts_entries():
    ks = KillSwitch(RiskLimits(max_peak_to_trough_dd=0.10, max_daily_dd=0.05))
    ks.reset(10_000)
    ts = pd.Timestamp("2024-06-01", tz="UTC")
    ks.update(ts, 10_000)
    ks.update(ts + pd.Timedelta(hours=4), 8_500)
    assert ks.halted
    assert not ks.allow_new_entry()
