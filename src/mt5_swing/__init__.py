"""mt5_swing — swing-trading research + execution toolkit for MetaTrader 5."""

__version__ = "0.1.0"

# Hard risk gates (documented in README): peak-to-trough max DD < 10%, daily DD < 5%.
MAX_PEAK_TO_TROUGH_DD = 0.10
MAX_DAILY_DD = 0.05
