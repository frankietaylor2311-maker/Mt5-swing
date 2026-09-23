"""Tests for terms-of-trade / commodity-currency scholarly FX."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.commodity_prices import (
    COUNTRY_COMMODITY_MAP,
    COUNTRY_TOT_MAP,
    TOT_TRADEABLE_CCYS,
    load_commodity_panel,
    tot_export_import_columns,
)
from mt5_swing.strategies.commodity_fx import (
    CommodityFxConfig,
    commodity_country_ts_returns,
    trailing_commodity_momentum,
)
from mt5_swing.strategies.tot_fx import (
    TotFxConfig,
    country_tot_momentum,
    tot_country_ts_weights,
    tot_factor_returns,
    tot_vs_crr_correlation,
    tot_xs_weights,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_tot_map_distinct_from_crr_single_map():
    """ToT uses export AND import; CRR map is single commodity per ccy."""
    for ccy in TOT_TRADEABLE_CCYS:
        exp, imp = tot_export_import_columns(ccy)
        assert exp != imp
        assert ccy in COUNTRY_COMMODITY_MAP
        # CAD: CRR maps to oil alone; ToT export=oil, import=copper
        if ccy == "CAD":
            assert COUNTRY_COMMODITY_MAP[ccy] == "oil"
            assert exp == "oil" and imp == "copper"
        if ccy == "AUD":
            assert COUNTRY_COMMODITY_MAP[ccy] == "copper"
            assert exp == "copper" and imp == "oil"
    assert "NOK" in COUNTRY_TOT_MAP and "ZAR" in COUNTRY_TOT_MAP


def test_tot_signal_no_lookahead():
    idx = pd.date_range("2020-01-01", periods=300, freq="B", tz="UTC")
    rng = np.random.default_rng(42)
    oil = pd.Series(50 * np.cumprod(1 + rng.normal(0, 0.02, 300)), index=idx)
    copper = pd.Series(3 * np.cumprod(1 + rng.normal(0, 0.015, 300)), index=idx)
    gold = pd.Series(1500 * np.cumprod(1 + rng.normal(0, 0.01, 300)), index=idx)
    basket = (oil / oil.iloc[0] + copper / copper.iloc[0] + gold / gold.iloc[0]) / 3 * 100
    panel = pd.DataFrame({"oil": oil, "copper": copper, "gold": gold, "basket": basket})
    cfg = TotFxConfig(formation_days=63, skip_days=21, signal_lag=1)
    tot = country_tot_momentum(panel, cfg=cfg)

    panel2 = panel.copy()
    panel2.iloc[-5:] *= 1.5
    tot2 = country_tot_momentum(panel2, cfg=cfg)

    cut = -(cfg.formation_days + cfg.skip_days + cfg.signal_lag + 2)
    for ccy in ("AUD", "CAD", "NZD"):
        a = tot[ccy].iloc[:cut].dropna()
        b = tot2[ccy].iloc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 50
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True)


def test_tot_equals_export_minus_import_mom():
    """Identity: ToT_AUD = copper_mom − oil_mom (after same lag)."""
    idx = pd.date_range("2019-01-01", periods=400, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    oil = pd.Series(60 * np.cumprod(1 + rng.normal(0.0001, 0.02, 400)), index=idx)
    copper = pd.Series(2.5 * np.cumprod(1 + rng.normal(0.0002, 0.015, 400)), index=idx)
    gold = pd.Series(1400 * np.cumprod(1 + rng.normal(0, 0.01, 400)), index=idx)
    basket = (oil / oil.iloc[0] + copper / copper.iloc[0]) / 2 * 100
    panel = pd.DataFrame({"oil": oil, "copper": copper, "gold": gold, "basket": basket})
    cfg = TotFxConfig(formation_days=63, skip_days=21, signal_lag=1)
    tot = country_tot_momentum(panel, cfg=cfg)
    exp = trailing_commodity_momentum(copper, formation_days=63, skip_days=21).shift(1)
    imp = trailing_commodity_momentum(oil, formation_days=63, skip_days=21).shift(1)
    expected = exp - imp
    common = tot["AUD"].dropna().index.intersection(expected.dropna().index)
    assert len(common) > 100
    assert np.allclose(tot["AUD"].loc[common], expected.loc[common], atol=1e-12)


def test_tot_weights_positive_long():
    idx = pd.date_range("2021-01-01", periods=5, freq="B", tz="UTC")
    sig = pd.DataFrame(
        {
            "AUD": [0.1, -0.1, 0.05, 0.0, 0.2],
            "CAD": [0.1, 0.1, -0.05, 0.0, -0.2],
            "NZD": [-0.1, -0.1, 0.05, 0.0, 0.1],
        },
        index=idx,
    )
    w = tot_country_ts_weights(sig)
    assert w.iloc[0]["AUD"] == pytest.approx(0.5)
    assert w.iloc[0]["CAD"] == pytest.approx(0.5)
    assert w.iloc[0]["NZD"] == pytest.approx(0.0)
    assert float(w.iloc[3].abs().sum()) == pytest.approx(0.0)

    wxs = tot_xs_weights(sig, cfg=TotFxConfig(n_long=1, n_short=1))
    # Day 0: AUD=CAD=0.1, NZD=-0.1 → long AUD (or CAD; stable sort by value then index)
    assert float(wxs.iloc[0].sum()) == pytest.approx(0.0)
    assert float(wxs.iloc[0].abs().sum()) == pytest.approx(1.0)


def test_tot_strategy_smoke_and_distinct_from_crr():
    idx = pd.date_range("2018-01-01", periods=800, freq="B", tz="UTC")
    rng = np.random.default_rng(11)
    oil = pd.Series(55 * np.cumprod(1 + rng.normal(0.0003, 0.02, 800)), index=idx)
    copper = pd.Series(3 * np.cumprod(1 + rng.normal(0.0001, 0.015, 800)), index=idx)
    gold = pd.Series(1600 * np.cumprod(1 + rng.normal(0.00005, 0.01, 800)), index=idx)
    basket = (oil / oil.iloc[0] + copper / copper.iloc[0] + gold / gold.iloc[0]) / 3 * 100
    panel = pd.DataFrame({"oil": oil, "copper": copper, "gold": gold, "basket": basket})

    # Synthetic FX returns
    pair_ret = pd.DataFrame(
        {
            "EURUSD": rng.normal(0, 0.005, 800),
            "GBPUSD": rng.normal(0, 0.005, 800),
            "AUDUSD": rng.normal(0, 0.006, 800),
            "NZDUSD": rng.normal(0, 0.006, 800),
            "USDJPY": rng.normal(0, 0.005, 800),
            "USDCAD": rng.normal(0, 0.005, 800),
            "USDCHF": rng.normal(0, 0.004, 800),
        },
        index=idx,
    )

    factors = tot_factor_returns(panel, pair_ret, cfg=TotFxConfig())
    assert set(factors) >= {"tot_country_ts", "tot_xs", "tot_vs_g10", "tot_ew"}
    for name, r in factors.items():
        assert len(r) == len(pair_ret)
        assert r.notna().sum() > 100

    crr = commodity_country_ts_returns(panel, pair_ret, cfg=CommodityFxConfig())
    corr = tot_vs_crr_correlation(factors["tot_country_ts"], crr)
    # Must be defined and not identically 1.0 (ToT ≠ CRR)
    assert np.isfinite(corr)
    assert corr < 0.999


@pytest.mark.skipif(
    not (MACRO / "commodity_yahoo_panel.csv").exists(),
    reason="commodity panel cache missing",
)
def test_load_panel_tot_map_attrs():
    panel = load_commodity_panel(download=False, pub_lag_days=1)
    assert "oil" in panel.columns or "copper" in panel.columns
    tot = country_tot_momentum(panel, cfg=TotFxConfig())
    assert set(tot.columns) >= {"AUD", "CAD", "NZD"}
    # After pub lag + formation, early NaNs then finite values
    assert tot.dropna(how="all").shape[0] > 200
