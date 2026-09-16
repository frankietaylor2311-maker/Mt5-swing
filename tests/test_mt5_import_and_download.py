"""MT5 FTMO CSV import + interim downloader (mocked network)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mt5_swing.data.mt5_import import DATA_SOURCE_FTMO_MT5, import_mt5_exports, load_mt5_export_csv
from mt5_swing.features.indicators import apply_feature_pipeline
from mt5_swing.data.loader import generate_sample_ohlc


def test_load_mt5_export_headerless(tmp_path: Path):
    # Classic MT5 History Center tab export (headerless)
    lines = [
        "2020.01.02\t00:00\t1.12000\t1.12100\t1.11900\t1.12050\t100\t0\t12",
        "2020.01.02\t04:00\t1.12050\t1.12200\t1.12000\t1.12150\t110\t0\t12",
        "2020.01.02\t08:00\t1.12150\t1.12300\t1.12100\t1.12250\t120\t0\t10",
    ]
    p = tmp_path / "EURUSD_H4.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    df = load_mt5_export_csv(p, symbol="EURUSD", timeframe="H4")
    assert len(df) == 3
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "spread"]
    assert df.index.tz is not None
    assert df.attrs["data_source"] == DATA_SOURCE_FTMO_MT5
    assert df["close"].iloc[-1] == pytest.approx(1.12250)


def test_load_mt5_export_with_header(tmp_path: Path):
    p = tmp_path / "GBPUSD_D1.csv"
    p.write_text(
        "Date,Time,Open,High,Low,Close,TickVol,Volume,Spread\n"
        "2021.03.01,00:00,1.3900,1.3950,1.3880,1.3920,1000,0,14\n"
        "2021.03.02,00:00,1.3920,1.3980,1.3910,1.3960,1100,0,14\n",
        encoding="utf-8",
    )
    df = load_mt5_export_csv(p, symbol="GBPUSD", timeframe="D1")
    assert len(df) == 2
    assert df.attrs["data_source"] == DATA_SOURCE_FTMO_MT5


def test_import_mt5_exports_writes_meta(tmp_path: Path):
    src = tmp_path / "raw.csv"
    src.write_text(
        "2022.01.03\t00:00\t110.00\t110.50\t109.80\t110.20\t50\t0\t20\n"
        "2022.01.03\t04:00\t110.20\t110.80\t110.00\t110.60\t55\t0\t20\n",
        encoding="utf-8",
    )
    out = tmp_path / "ftmo"
    paths = import_mt5_exports({"USDJPY_H4": src}, out)
    assert paths["USDJPY_H4"].exists()
    meta = paths["USDJPY_H4"].with_suffix(".meta.json")
    assert meta.exists()
    assert "ftmo_mt5_export" in meta.read_text()


def test_download_normalize_and_tag_approximate(tmp_path: Path):
    """Mock yfinance history → must tag approximate_non_ftmo."""
    idx = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC")
    fake = pd.DataFrame(
        {
            "Open": 1.1,
            "High": 1.11,
            "Low": 1.09,
            "Close": 1.105,
            "Volume": 0,
        },
        index=idx,
    )
    with patch("yfinance.Ticker") as T:
        inst = MagicMock()
        inst.history.return_value = fake
        T.return_value = inst
        from mt5_swing.data.download import download_symbol_timeframe, download_history

        h4 = download_symbol_timeframe("EURUSD", "H4")
        assert h4.attrs.get("data_source") == "approximate_non_ftmo"
        assert len(h4) >= 1
        paths = download_history(["EURUSD"], ["H4"], out_dir=tmp_path)
        assert paths["EURUSD_H4"].exists()
        meta = paths["EURUSD_H4"].with_suffix(".meta.json")
        assert "approximate_non_ftmo" in meta.read_text()


def test_signal_lag_still_holds_after_feature_update():
    df = generate_sample_ohlc(n_bars=150, seed=11)
    feat = apply_feature_pipeline(df, signal_lag=1)
    assert "signal_close" in feat.columns
    # signal_close must equal close.shift(1)
    expected = df["close"].shift(1)
    mask = expected.notna() & feat["signal_close"].notna()
    assert (feat.loc[mask, "signal_close"] == expected.loc[mask]).all()
