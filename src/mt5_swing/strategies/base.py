"""Strategy protocol — strategies emit target position direction per bar."""

from __future__ import annotations

from enum import IntEnum
from typing import Protocol, runtime_checkable

import pandas as pd


class Signal(IntEnum):
    FLAT = 0
    LONG = 1
    SHORT = -1


@runtime_checkable
class Strategy(Protocol):
    """Point-in-time strategy: generate signals from a features DataFrame."""

    name: str

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Return a Series of Signal values aligned to ``data.index``.

        Must use only columns available at each timestamp (lagged features).
        """
        ...
