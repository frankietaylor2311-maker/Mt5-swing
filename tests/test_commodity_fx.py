"""Tests for commodity-currency scholarly FX (Chen–Rogoff–Rossi style)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.commodity_prices import (
    COUNTRY_COMMODITY_MAP,
    commodity_returns,
    load_commodity_panel,
)
from mt5_swing.strategies.carry_rank import USD_PAIRS, currency_weights_to_pair_weights
from mt5_swing.strategies.commodity_fx import (
    CommodityFxConfig,
    commodity_country_ts_returns,
    commodity_xs_basket_returns,
    country_ts_currency_weights,
    oil_impulse_cad_pair_weight_sign,
    prepare_country_commodity_signals,
    trailing_commodity_momentum,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_commodity_signal_no_lookahead():
    """Future commodity returns must not affect earlier signals."""
    idx = pd.date_range("2020-01-01", periods=300, freq="B", tz="UTC")
    # Synthetic price path
    rng = np.random.default_rng(42)
    px = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)), index=idx, name="oil")
    cfg = CommodityFxConfig(formation_days=63, skip_days=21, signal_lag=1)
    mom = trailing_commodity_momentum(px, formation_days=cfg.formation_days, skip_days=cfg.skip_days)
    mom_lag = mom.shift(cfg.signal_lag)

    # Mutate last 5 commodity prices — early signal must be unchanged
    px2 = px.copy()
    px2.iloc[-5:] *= 1.5
    mom2 = trailing_commodity_momentum(px2, formation_days=cfg.formation_days, skip_days=cfg.skip_days)
    mom2_lag = mom2.shift(cfg.signal_lag)

    # Signal at t depends on prices up to t-skip (and shift); last formation window
    # that doesn't touch the last 5 bars should match. Conservatively: all but last
    # formation+skip+lag observations.
    cut = -(cfg.formation_days + cfg.skip_days + cfg.signal_lag + 2)
    a = mom_lag.iloc[:cut].dropna()
    b = mom2_lag.iloc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 50
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True)


def test_oil_impulse_long_cad_short_usdcad():
    """Positive oil impulse → long CAD exposure → negative USDCAD weight."""
    assert oil_impulse_cad_pair_weight_sign(0.05) < 0
    assert oil_impulse_cad_pair_weight_sign(-0.05) > 0
    assert oil_impulse_cad_pair_weight_sign(0.0) == 0.0

    # Via weight mapper: +CAD → USDCAD weight negative
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-31", tz="UTC")])
    ccy_w = pd.DataFrame({"CAD": [1.0], "AUD": [0.0], "NZD": [0.0]}, index=idx)
    pw = currency_weights_to_pair_weights(ccy_w)
    assert "USDCAD" in pw.columns
    assert float(pw.iloc[0]["USDCAD"]) < 0
    sym, sign = USD_PAIRS["CAD"]
    assert sym == "USDCAD" and sign == -1


def test_country_ts_weights_positive_mom_long():
    idx = pd.date_range("2021-01-01", periods=10, freq="B", tz="UTC")
    sig = pd.DataFrame(
        {
            "AUD": [0.1, -0.1, 0.05, 0.0, 0.2, -0.05, 0.1, 0.1, -0.2, 0.3],
            "CAD": [0.1, 0.1, -0.05, 0.0, 0.2, 0.05, -0.1, 0.1, 0.2, 0.3],
            "NZD": [-0.1, -0.1, 0.05, 0.0, -0.2, 0.05, 0.1, -0.1, 0.2, 0.3],
        },
        index=idx,
    )
    w = country_ts_currency_weights(sig)
    # Day 0: AUD+CAD positive → equal 0.5 each, NZD 0
    assert w.iloc[0]["AUD"] == pytest.approx(0.5)
    assert w.iloc[0]["CAD"] == pytest.approx(0.5)
    assert w.iloc[0]["NZD"] == pytest.approx(0.0)
    # Day 3: all non-positive → flat
    assert float(w.iloc[3].abs().sum()) == pytest.approx(0.0)


def test_commodity_strategy_smoke_empty_safe_and_causal():
    idx = pd.date_range("2019-01-01", periods=500, freq="B", tz="UTC")
    rng = np.random.default_rng(7)
    # Commodity panel
    oil = pd.Series(50 * np.cumprod(1 + rng.normal(0.0005, 0.02, 500)), index=idx)
    copper = pd.Series(3 * np.cumprod(1 + rng.normal(0.0002, 0.015, 500)), index=idx)
    gold = pd.Series(1500 * np.cumprod(1 + rng.normal(0.0001, 0.01, 500)), index=idx)
    basket = (oil / oil.iloc[0] + copper / copper.iloc[0] + gold / gold.iloc[0]) / 3 * 100
    panel = pd.DataFrame({"oil": oil, "copper": copper, "gold": gold, "basket": basket})

    pret = pd.DataFrame(
        {
            "EURUSD": rng.normal(0, 0.004, 500),
            "GBPUSD": rng.normal(0, 0.004, 500),
            "AUDUSD": rng.normal(0, 0.005, 500),
            "NZDUSD": rng.normal(0, 0.005, 500),
            "USDJPY": rng.normal(0, 0.004, 500),
            "USDCAD": rng.normal(0, 0.004, 500),
            "USDCHF": rng.normal(0, 0.004, 500),
        },
        index=idx,
    )
    cfg = CommodityFxConfig(signal_lag=1, cost_bps_per_side=1.5)
    port_ts = commodity_country_ts_returns(panel, pret, cfg=cfg)
    port_xs = commodity_xs_basket_returns(panel, pret, cfg=cfg)
    assert len(port_ts) == len(pret)
    assert len(port_xs) == len(pret)
    assert port_ts.name == "commodity_country_ts"
    assert np.isfinite(port_ts.fillna(0).sum())

    # Causality: zeroing last FX return shouldn't change earlier portfolio
    pret2 = pret.copy()
    pret2.iloc[-1] = 0.0
    port_ts2 = commodity_country_ts_returns(panel, pret2, cfg=cfg)
    assert np.allclose(port_ts.iloc[:-1].values, port_ts2.iloc[:-1].values, equal_nan=True)

    # Empty-ish panel still returns a series
    empty_panel = pd.DataFrame({"oil": oil.iloc[:5], "basket": basket.iloc[:5]})
    port_e = commodity_xs_basket_returns(empty_panel, pret, cfg=cfg)
    assert isinstance(port_e, pd.Series)
    assert len(port_e) == len(pret)


def test_prepare_signals_respect_country_map():
    assert COUNTRY_COMMODITY_MAP["CAD"] == "oil"
    assert COUNTRY_COMMODITY_MAP["AUD"] == "copper"
    assert COUNTRY_COMMODITY_MAP["NZD"] == "basket"
    idx = pd.date_range("2020-01-01", periods=200, freq="B", tz="UTC")
    panel = pd.DataFrame(
        {
            "oil": np.linspace(40, 80, 200),
            "copper": np.linspace(2, 4, 200),
            "gold": np.linspace(1200, 1800, 200),
            "basket": np.linspace(100, 120, 200),
        },
        index=idx,
    )
    cfg = CommodityFxConfig(signal_lag=1)
    sig = prepare_country_commodity_signals(panel, cfg=cfg)
    assert set(sig.columns) == {"AUD", "CAD", "NZD"}
    # After lag, first rows NaN
    assert sig.iloc[0].isna().all() or sig.iloc[: cfg.signal_lag].isna().any().any()


def test_load_commodity_panel_if_cached():
    path = MACRO / "commodity_yahoo_panel.csv"
    if not path.exists():
        pytest.skip("commodity cache not yet downloaded")
    panel = load_commodity_panel(download=False, pub_lag_days=1)
    assert panel.attrs.get("pub_lag_days") == 1
    assert "oil" in panel.columns or "basket" in panel.columns
    ret = commodity_returns(panel)
    assert len(ret) == len(panel)


def test_align_yahoo_session_vs_fx_bar_times():
    """Commodity Yahoo 04:00 UTC bars must still drive FX D1 23:00 calendars."""
    from mt5_swing.strategies.commodity_fx import align_daily_to_index

    cidx = pd.date_range("2020-01-01 04:00", periods=120, freq="B", tz="UTC")
    fidx = pd.date_range("2020-01-01 23:00", periods=100, freq="B", tz="UTC")
    sig = pd.DataFrame(
        {"AUD": np.linspace(-0.1, 0.2, 120), "CAD": np.linspace(0.05, -0.05, 120), "NZD": 0.1},
        index=cidx,
    )
    aligned = align_daily_to_index(sig, fidx)
    assert aligned.index.equals(fidx)
    assert aligned.notna().any().any()
    # Positive NZD every day → country_ts should put weight on NZD
    w = country_ts_currency_weights(aligned)
    assert float(w["NZD"].mean()) > 0

    # Full pipeline with mismatched times must produce non-flat returns
    panel = pd.DataFrame(
        {
            "oil": np.linspace(40, 80, 120),
            "copper": np.linspace(2, 4, 120),
            "gold": np.linspace(1200, 1800, 120),
            "basket": np.linspace(100, 140, 120),
        },
        index=cidx,
    )
    rng = np.random.default_rng(0)
    pret = pd.DataFrame(
        {s: rng.normal(0.0005, 0.005, 100) for s in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]},
        index=fidx,
    )
    port = commodity_country_ts_returns(panel, pret, cfg=CommodityFxConfig(signal_lag=1))
    assert (port.fillna(0).abs() > 0).sum() > 10
