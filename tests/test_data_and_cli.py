"""Data loader / sample / package smoke."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from mt5_swing.cli import main
from mt5_swing.data.loader import ensure_sample_data, load_ohlc_csv
from mt5_swing.data.symbols import pip_size, pip_value_per_lot


def test_pip_sizing():
    assert abs(pip_size("EURUSD") - 0.0001) < 1e-12
    assert abs(pip_size("USDJPY") - 0.01) < 1e-12
    assert pip_value_per_lot("EURUSD") > 0


def test_ensure_sample_and_load(tmp_path: Path):
    paths = ensure_sample_data(tmp_path, symbols=["EURUSD"], timeframes=["H4"])
    df = load_ohlc_csv(paths["EURUSD_H4"], symbol="EURUSD")
    assert {"open", "high", "low", "close"}.issubset(df.columns)
    assert df.index.is_monotonic_increasing


def test_cli_walk_forward_smoke():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "walk-forward",
            "--symbol",
            "EURUSD",
            "--timeframe",
            "H4",
            "--train-bars",
            "300",
            "--test-bars",
            "50",
            "--step-bars",
            "50",
        ],
    )
    # Exit 0 or 1 depending on gates — both OK; must print report
    assert result.exit_code in (0, 1)
    assert "Walk-forward" in result.output
    assert "Out-of-sample" in result.output
    assert "Overall risk gates" in result.output
