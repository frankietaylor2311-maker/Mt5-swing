"""Shared fixtures."""

from __future__ import annotations

import pytest

from mt5_swing.data.loader import generate_sample_ohlc


@pytest.fixture
def sample_ohlc():
    return generate_sample_ohlc(symbol="EURUSD", timeframe="H4", n_bars=800, seed=7)
