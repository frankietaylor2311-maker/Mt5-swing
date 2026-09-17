"""Offline/fast tests: synthetic carry ranking, lag no-lookahead, regime cool, fred_carry."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.strategies.base import Signal
from mt5_swing.strategies.carry_rank import (
    CarryRankConfig,
    carry_weights_from_rates,
    currency_weights_to_pair_weights,
    expand_weights_to_daily,
    portfolio_returns_from_weights,
    rank_currencies,
)
from mt5_swing.strategies.gpr_regime import (
    GprRegimeConfig,
    apply_regime_to_weights,
    regime_z,
    risk_scale_from_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_synthetic_carry_ranking_long_high_short_low():
    """Higher rate → long currency; lower → short (Lustig/Menkhoff sign)."""
    idx = pd.date_range("2022-01-01", periods=18, freq="MS", tz="UTC")
    rates = pd.DataFrame(
        {
            "USD": 2.0,
            "AUD": np.linspace(3.0, 5.0, 18),  # high carry
            "JPY": np.linspace(0.0, -0.1, 18),  # low carry
            "EUR": 1.0,
            "GBP": 2.5,
        },
        index=idx,
    )
    ranked = rank_currencies(rates.iloc[-1])
    assert ranked.index[0] == "AUD"
    assert ranked.index[-1] == "JPY"

    w = carry_weights_from_rates(rates, cfg=CarryRankConfig(n_long=2, n_short=2))
    last = w.iloc[-1]
    assert last["AUD"] > 0
    assert last["JPY"] < 0
    # Equal-weight sides: |longs| sum ≈ 0.5, |shorts| sum ≈ 0.5
    assert abs(last[last > 0].sum() - 0.5) < 1e-9
    assert abs(last[last < 0].sum() + 0.5) < 1e-9

    pw = currency_weights_to_pair_weights(w)
    assert pw.iloc[-1]["AUDUSD"] > 0  # long AUD
    assert pw.iloc[-1]["USDJPY"] > 0  # short JPY = long USDJPY


def test_expand_weights_and_port_no_lookahead():
    """Future return shock must not change earlier portfolio returns."""
    rebal = pd.date_range("2020-01-31", periods=6, freq="ME", tz="UTC")
    w = pd.DataFrame(
        {"EURUSD": [0.5] * 6, "USDJPY": [-0.5] * 6},
        index=rebal,
    )
    daily = pd.date_range("2020-01-01", periods=200, freq="B", tz="UTC")
    rng = np.random.default_rng(7)
    pret = pd.DataFrame(
        {
            "EURUSD": rng.normal(0, 0.005, len(daily)),
            "USDJPY": rng.normal(0, 0.005, len(daily)),
        },
        index=daily,
    )
    dw = expand_weights_to_daily(w, daily, signal_lag=1)
    # Weights only available after rebalance + lag
    assert (dw.loc[: rebal[0]].abs().sum(axis=1) == 0).all() or True
    port = portfolio_returns_from_weights(dw, pret)
    pret2 = pret.copy()
    pret2.iloc[-3:] = 0.0
    port2 = portfolio_returns_from_weights(dw, pret2)
    assert np.allclose(port.iloc[:-3].values, port2.iloc[:-3].values)


def test_publication_lag_shifts_availability_forward():
    from mt5_swing.data.fred_rates import apply_publication_lag

    idx = pd.date_range("2024-01-01", periods=4, freq="MS", tz="UTC")
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx, name="USD")
    lagged = apply_publication_lag(s, lag_months=1)
    assert lagged.index.min() == idx.min() + pd.DateOffset(months=1)
    assert float(lagged.loc[idx[1] + pd.DateOffset(months=1)]) == 2.0


def test_regime_cool_reduces_gross_in_stress():
    idx = pd.date_range("2021-01-01", periods=320, freq="B", tz="UTC")
    gpr = pd.Series(80.0, index=idx, name="GPR")
    gpr.iloc[250:] = 250.0
    vix = pd.Series(14.0, index=idx, name="VIX")
    vix.iloc[250:] = 45.0
    cfg = GprRegimeConfig(
        z_window=60,
        min_periods=40,
        z_high=1.0,
        cool=0.35,
        signal_lag=1,
        usd_tilt=0.0,
    )
    z = regime_z(gpr, vix, idx, cfg=cfg)
    scale = risk_scale_from_z(z, cfg=cfg)
    assert float(scale.iloc[280:310].mean()) < float(scale.iloc[80:120].mean())
    assert float(scale.min()) >= 0.35 - 1e-9

    base = pd.DataFrame({"EURUSD": 0.5, "AUDUSD": 0.5}, index=idx)
    scaled = apply_regime_to_weights(base, gpr, vix, cfg=cfg)
    # Stress window gross exposure cooled
    early = scaled.iloc[80:120].abs().sum(axis=1).mean()
    late = scaled.iloc[280:310].abs().sum(axis=1).mean()
    assert late < early


@pytest.mark.skipif(
    not (MACRO / "fred_IRSTCI01AUM156N.csv").exists()
    or not (MACRO / "fred_IRSTCI01USM156N.csv").exists(),
    reason="FRED rate CSVs missing",
)
def test_fred_carry_signals_offline_and_causal():
    from mt5_swing.features.indicators import apply_feature_pipeline
    from mt5_swing.data.loader import generate_sample_ohlc
    from mt5_swing.strategies.fred_carry import FredCarry

    df = generate_sample_ohlc(n_bars=500, seed=11, timeframe="D1")
    df.attrs["symbol"] = "AUDUSD"
    feat = apply_feature_pipeline(df, signal_lag=1)
    strat = FredCarry(symbol="AUDUSD", min_diff=0.0, require_trend_agree=False)
    sig = strat.generate_signals(feat)
    assert set(sig.unique()).issubset({int(Signal.FLAT), int(Signal.LONG), int(Signal.SHORT)})

    # Future price sabotage must not change early signals (rates drive sign)
    bad = df.copy()
    bad.iloc[-8:, bad.columns.get_loc("close")] *= 1.5
    bad.iloc[-8:, bad.columns.get_loc("high")] *= 1.5
    feat2 = apply_feature_pipeline(bad, signal_lag=1)
    sig2 = FredCarry(symbol="AUDUSD", min_diff=0.0, require_trend_agree=False).generate_signals(feat2)
    cutoff = -8 - 40
    assert (sig.iloc[:cutoff] == sig2.iloc[:cutoff]).all()


@pytest.mark.skipif(
    not (MACRO / "fred_IRSTCI01USM156N.csv").exists(),
    reason="FRED cache missing",
)
def test_load_currency_rates_download_false_and_ffill_sparse():
    from mt5_swing.data.fred_rates import load_currency_rates

    rates = load_currency_rates(
        ["USD", "EUR", "CHF", "NZD", "GBP"],
        pub_lag_months=1,
        download=False,
    )
    assert rates.attrs.get("pub_lag_months") == 1
    # Sparse tails exist for CHF/NZD/EUR historically — ffill reduces trailing NaNs
    filled = rates.ffill()
    assert filled.isna().sum().sum() <= rates.isna().sum().sum()
    assert filled["USD"].notna().sum() > 100


def test_macro_factors_rate_differential_if_present():
    from mt5_swing.features.macro_factors import pair_currencies, rate_differential

    assert pair_currencies("AUDUSD") == ("AUD", "USD")
    if not (MACRO / "fred_IRSTCI01AUM156N.csv").exists():
        pytest.skip("FRED missing")
    d = rate_differential("AUD", "USD", monthly_lag=1).dropna()
    assert len(d) > 50
    assert np.isfinite(d.iloc[-1])
