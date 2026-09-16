"""Risk monitor unit tests — FTMO 2-Step Prague daily + static max loss."""

from __future__ import annotations

import pandas as pd

from mt5_swing.risk.monitors import (
    DailyDDMonitor,
    KillSwitch,
    MaxDDMonitor,
    RiskLimits,
    StaticMaxLossMonitor,
)


def test_max_dd_monitor():
    m = MaxDDMonitor(limit=0.10)
    m.update(10_000)
    dd, halt, flat = m.update(9_500)
    assert abs(dd - 0.05) < 1e-9
    assert not halt
    dd, halt, flat = m.update(8_900)
    assert dd >= 0.10
    assert halt and flat


def test_static_max_loss_from_initial():
    m = StaticMaxLossMonitor(limit=0.10, initial_capital=100_000)
    loss, halt, _ = m.update(95_000)
    assert abs(loss - 0.05) < 1e-9
    assert not halt
    loss, halt, flat = m.update(89_000)
    assert loss >= 0.10
    assert halt and flat


def test_daily_dd_ftmo_prague_midnight():
    """Day boundary is Europe/Prague; loss scaled by initial capital."""
    mon = DailyDDMonitor(
        limit=0.05,
        daily_loss_mode="ftmo_initial",
        tz="Europe/Prague",
        initial_capital=100_000,
    )
    # 23:00 UTC = 00:00 Prague in winter (CET=UTC+1) on 2024-01-02
    # Use summer: CEST UTC+2 — 22:00 UTC = midnight Prague
    t0 = pd.Timestamp("2024-07-01 10:00", tz="UTC")
    mon.update(t0, 100_000)
    t1 = pd.Timestamp("2024-07-01 20:00", tz="UTC")
    mon.update(t1, 102_000)  # day ends ~102k
    # Next Prague day: 2024-07-02 00:30 Prague = 2024-07-01 22:30 UTC
    t2 = pd.Timestamp("2024-07-01 22:30", tz="UTC")
    dd, halt, _ = mon.update(t2, 96_500)  # loss 5500 from 102k vs initial 100k = 5.5%
    assert dd >= 0.05
    assert halt


def test_daily_dd_pct_of_day_open_legacy():
    mon = DailyDDMonitor(limit=0.05, daily_loss_mode="pct_of_day_open", tz="UTC")
    t0 = pd.Timestamp("2024-01-01 08:00", tz="UTC")
    t1 = pd.Timestamp("2024-01-01 16:00", tz="UTC")
    mon.update(t0, 10_000)
    mon.update(t1, 10_200)
    t2 = pd.Timestamp("2024-01-02 08:00", tz="UTC")
    dd, halt, _ = mon.update(t2, 9_690)
    assert dd >= 0.05
    assert halt


def test_kill_switch_static_ftmo_halts():
    ks = KillSwitch(
        RiskLimits(
            max_peak_to_trough_dd=0.10,
            max_daily_dd=0.05,
            max_loss_mode="static_initial",
            daily_loss_mode="ftmo_initial",
            daily_tz="Europe/Prague",
        )
    )
    ks.reset(100_000)
    ts = pd.Timestamp("2024-06-01 12:00", tz="UTC")
    ks.update(ts, 100_000)
    ks.update(ts + pd.Timedelta(hours=4), 89_000)
    assert ks.halted
    assert not ks.allow_new_entry()
