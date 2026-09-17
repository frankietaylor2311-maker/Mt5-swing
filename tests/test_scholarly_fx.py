"""Tests for scholarly FX loaders + carry / momentum / GPR regime modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_rates import (
    apply_publication_lag,
    load_currency_rates,
    load_fred_series,
    rate_differentials_vs_usd,
)
from mt5_swing.data.macro_uncertainty import GprIndex, load_gpr, load_vix
from mt5_swing.strategies.base import Signal
from mt5_swing.strategies.carry_rank import (
    CarryRankBasket,
    CarryRankConfig,
    CarryRankPairSignal,
    carry_weights_from_rates,
    currency_weights_to_pair_weights,
    expand_weights_to_daily,
    portfolio_returns_from_weights,
)
from mt5_swing.strategies.fx_momentum import (
    FxMomentumBasket,
    FxMomentumPairSignal,
    dollar_factor_returns,
    momentum_weights_from_returns,
    pair_returns_to_currency_returns,
)
from mt5_swing.strategies.gpr_regime import (
    GprRegimeConfig,
    GprRegimeFilter,
    GprRegimePairSignal,
    apply_regime_to_returns,
    regime_z,
    risk_scale_from_z,
)
from mt5_swing.strategies.registry import get_strategy, list_strategies

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_fred_series_loads_and_lags():
    if not (MACRO / "fred_IRSTCI01USM156N.csv").exists():
        pytest.skip("FRED cache missing")
    s = load_fred_series("IRSTCI01USM156N", download=False)
    assert len(s) > 100
    lagged = apply_publication_lag(s, lag_months=1)
    # First usable date moves forward
    assert lagged.index.min() > s.index.min()


def test_currency_rate_panel_point_in_time():
    if not (MACRO / "fred_IRSTCI01USM156N.csv").exists():
        pytest.skip("FRED cache missing")
    rates = load_currency_rates(
        ["USD", "EUR", "JPY", "AUD"],
        pub_lag_months=1,
        download=False,
    )
    assert "USD" in rates.columns and "EUR" in rates.columns
    assert rates.attrs.get("pub_lag_months") == 1
    diff = rate_differentials_vs_usd(rates)
    assert "USD" not in diff.columns
    assert "EUR" in diff.columns


def test_vix_and_gpr_loaders():
    if not (MACRO / "vix_yahoo.csv").exists():
        pytest.skip("VIX cache missing")
    vix = load_vix(download=False, pub_lag_days=1)
    assert len(vix) > 1000
    assert vix.attrs.get("pub_lag_days") == 1
    if not (MACRO / "gpr_monthly.csv").exists():
        stub = GprIndex.stub()
        assert not stub.available
        pytest.skip("GPR cache missing")
    gpr = load_gpr(freq="M", download=False, pub_lag_months=1)
    assert gpr.notna().sum() > 100


def test_carry_rank_long_high_short_low_and_causal_port():
    idx = pd.date_range("2020-01-01", periods=36, freq="MS", tz="UTC")
    rates = pd.DataFrame(
        {
            "USD": 1.0,
            "AUD": np.linspace(2.0, 4.0, 36),
            "JPY": np.linspace(0.1, 0.0, 36),
            "EUR": np.linspace(0.5, 1.5, 36),
            "GBP": np.linspace(1.0, 2.0, 36),
        },
        index=idx,
    )
    w = carry_weights_from_rates(rates, cfg=CarryRankConfig(n_long=1, n_short=1))
    assert not w.empty
    # Last row: AUD highest, JPY lowest
    last = w.iloc[-1]
    assert last["AUD"] > 0
    assert last["JPY"] < 0
    pw = currency_weights_to_pair_weights(w)
    assert "AUDUSD" in pw.columns and "USDJPY" in pw.columns
    # Long AUD → +AUDUSD; short JPY → +USDJPY (short foreign = long USDJPY)
    assert pw.iloc[-1]["AUDUSD"] > 0
    assert pw.iloc[-1]["USDJPY"] > 0

    daily = pd.date_range("2020-01-01", periods=400, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    pret = pd.DataFrame(
        rng.normal(0, 0.005, size=(len(daily), len(pw.columns))),
        index=daily,
        columns=pw.columns,
    )
    dw = expand_weights_to_daily(pw, daily, signal_lag=1)
    port = portfolio_returns_from_weights(dw, pret)
    # Causality: changing last return shouldn't change early port
    pret2 = pret.copy()
    pret2.iloc[-1] *= 0
    port2 = portfolio_returns_from_weights(dw, pret2)
    assert np.allclose(port.iloc[:-1].values, port2.iloc[:-1].values)


def test_carry_rank_basket_and_pair_signal():
    basket = CarryRankBasket(n_long=2, n_short=2)
    idx = pd.date_range("2021-01-01", periods=24, freq="MS", tz="UTC")
    rates = pd.DataFrame(
        {c: np.random.default_rng(1).normal(1, 0.2, 24) for c in ["USD", "EUR", "GBP", "AUD", "JPY"]},
        index=idx,
    )
    rates["USD"] = 1.0
    pw = basket.weights(rates)
    assert isinstance(pw, pd.DataFrame)
    data = pd.DataFrame({"close": np.linspace(1, 1.1, 50), "carry_w": [0.1] * 25 + [-0.1] * 25})
    sig = CarryRankPairSignal().generate_signals(data)
    assert sig.iloc[0] == Signal.LONG
    assert sig.iloc[-1] == Signal.SHORT


def test_fx_momentum_and_dollar_factor():
    idx = pd.date_range("2019-01-01", periods=500, freq="B", tz="UTC")
    rng = np.random.default_rng(2)
    # AUD drifts up, JPY drifts down via USDJPY up
    pret = pd.DataFrame(
        {
            "EURUSD": rng.normal(0, 0.004, 500),
            "GBPUSD": rng.normal(0, 0.004, 500),
            "AUDUSD": rng.normal(0.0008, 0.004, 500),
            "NZDUSD": rng.normal(0, 0.004, 500),
            "USDJPY": rng.normal(0.0005, 0.004, 500),
            "USDCAD": rng.normal(0, 0.004, 500),
            "USDCHF": rng.normal(0, 0.004, 500),
        },
        index=idx,
    )
    ccy = pair_returns_to_currency_returns(pret)
    assert "AUD" in ccy.columns and np.allclose(ccy["USD"], 0.0)
    w = momentum_weights_from_returns(pret)
    assert not w.empty
    basket = FxMomentumBasket(formation_days=63, skip_days=21, signal_lag=1)
    pr = basket.portfolio_returns(pret)
    assert len(pr) == len(pret)
    dol = dollar_factor_returns(pret)
    assert dol.name == "dollar_factor"
    # Pair signal causal
    px = (1 + pret["EURUSD"]).cumprod()
    data = pd.DataFrame({"close": px})
    sig = FxMomentumPairSignal(lookback=20, skip=1).generate_signals(data)
    assert set(sig.unique()).issubset({-1, 0, 1})


def test_gpr_regime_scales_down_in_stress_and_is_lagged():
    idx = pd.date_range("2020-01-01", periods=400, freq="B", tz="UTC")
    gpr = pd.Series(100.0, index=idx, name="GPR")
    gpr.iloc[300:] = 300.0  # stress spike
    vix = pd.Series(15.0, index=idx, name="VIX")
    vix.iloc[300:] = 40.0
    cfg = GprRegimeConfig(z_window=60, min_periods=30, z_high=1.0, cool=0.35, signal_lag=1, usd_tilt=0.0)
    z = regime_z(gpr, vix, idx, cfg=cfg)
    scale = risk_scale_from_z(z, cfg=cfg)
    # Late window should be cooler than early (after warmup)
    assert float(scale.iloc[350:390].mean()) < float(scale.iloc[80:120].mean())
    base = pd.DataFrame({"EURUSD": 0.5, "USDJPY": 0.5}, index=idx)
    filt = GprRegimeFilter(z_window=60, min_periods=30, z_high=1.0, cool=0.35, signal_lag=1, usd_tilt=0.0)
    scaled = filt.scale_weights(base, gpr, vix)
    assert scaled["EURUSD"].iloc[350] <= base["EURUSD"].iloc[350] + 1e-9
    r = pd.Series(rng_ret(idx), index=idx)
    out = apply_regime_to_returns(r, gpr, vix, cfg=cfg)
    assert len(out) == len(r)
    # Pair flatten
    data = pd.DataFrame({"base_signal": 1, "stress_z": [0.0] * 10 + [2.0] * 10})
    sig = GprRegimePairSignal(z_high=1.0).generate_signals(data)
    assert sig.iloc[0] == 1 and sig.iloc[-1] == 0


def rng_ret(idx):
    return np.random.default_rng(3).normal(0, 0.01, len(idx))


def test_registry_has_scholarly_pair_adapters():
    names = list_strategies()
    assert "carry_rank_pair" in names
    assert "fx_momentum_pair" in names
    assert "gpr_regime_pair" in names
    assert get_strategy("fx_momentum_pair", lookback=10).name == "fx_momentum_pair"


def test_gpr_stub_interface():
    stub = GprIndex.stub()
    assert not stub.available
    with pytest.raises(RuntimeError):
        stub.series()


def test_scholarly_combo_weights_cool_carry_and_causal():
    from mt5_swing.strategies.scholarly_combo import (
        ScholarlyComboConfig,
        build_combo_weights,
        combo_portfolio_returns,
        sleeve_returns_bundle,
    )

    idx = pd.date_range("2019-01-01", periods=520, freq="B", tz="UTC")
    rng = np.random.default_rng(7)
    rates_idx = pd.date_range("2018-01-01", periods=40, freq="MS", tz="UTC")
    rates = pd.DataFrame(
        {
            "USD": 1.0,
            "AUD": np.linspace(2.0, 4.0, 40),
            "JPY": np.linspace(0.0, 0.2, 40),
            "EUR": np.linspace(0.5, 1.0, 40),
            "GBP": np.linspace(1.0, 1.5, 40),
            "CAD": np.linspace(0.8, 1.2, 40),
            "CHF": np.linspace(0.0, 0.3, 40),
            "NZD": np.linspace(1.5, 2.5, 40),
        },
        index=rates_idx,
    )
    pret = pd.DataFrame(
        rng.normal(0, 0.004, size=(len(idx), 7)),
        index=idx,
        columns=["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"],
    )
    gpr = pd.Series(100.0, index=idx, name="GPR")
    gpr.iloc[400:] = 400.0
    vix = pd.Series(15.0, index=idx, name="VIX")
    vix.iloc[400:] = 45.0
    cfg = ScholarlyComboConfig(
        signal_lag=1,
        z_window=60,
        min_periods=30,
        z_high=1.0,
        carry_cool=0.35,
        usd_tilt=0.15,
    )
    w = build_combo_weights(rates, pret, gpr, vix, cfg=cfg)
    assert not w.empty
    assert set(["EURUSD", "USDJPY"]).issubset(set(w.columns))
    # Stress window: gross exposure should not explode above ~1
    late_gross = float(w.iloc[450:500].abs().sum(axis=1).mean())
    assert late_gross <= 1.05
    r = combo_portfolio_returns(rates, pret, gpr, vix, cfg=cfg)
    assert len(r) == len(pret)
    # Causality: last return change does not affect earlier combo returns
    pret2 = pret.copy()
    pret2.iloc[-1] = 0.0
    r2 = combo_portfolio_returns(rates, pret2, gpr, vix, cfg=cfg)
    assert np.allclose(r.iloc[:-1].values, r2.iloc[:-1].values, equal_nan=True)
    bundle = sleeve_returns_bundle(rates, pret, gpr, vix, cfg=cfg)
    assert "scholarly_combo" in bundle and "carry_rank" in bundle
    assert "dollar_tsmom" in bundle


def test_newey_west_tstat_basic():
    # Import from wave script helpers via inline copy of pure functions
    from pathlib import Path
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "scholarly_fx_combo_wave.py"
    spec = importlib.util.spec_from_file_location("combo_wave", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(0)
    x = rng.normal(0.01, 0.02, size=120)
    t_ols = mod.ols_tstat(x)
    t_nw, L = mod.newey_west_tstat(x)
    assert L >= 0
    assert np.isfinite(t_ols) and np.isfinite(t_nw)
    # Strongly positive mean → both t > 0
    assert t_ols > 0 and t_nw > 0
    # Zero series → t ~ 0
    z = np.zeros(50)
    tz, _ = mod.newey_west_tstat(z)
    assert abs(tz) < 1e-8


def test_gpr_event_study_runs():
    from pathlib import Path
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "scholarly_fx_combo_wave.py"
    spec = importlib.util.spec_from_file_location("combo_wave", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    idx = pd.date_range("2020-01-01", periods=400, freq="B", tz="UTC")
    rng = np.random.default_rng(5)
    pret = pd.DataFrame(
        rng.normal(0, 0.005, size=(400, 6)),
        index=idx,
        columns=["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF"],
    )
    gpr = pd.Series(rng.normal(100, 10, 400), index=idx)
    gpr.iloc[50] = 300
    gpr.iloc[150] = 320
    gpr.iloc[250] = 310
    df = mod.gpr_event_study(pret, gpr, pre=3, post=5)
    assert "usd_minus_risk" in df.columns
    assert len(df) == 3 + 5 + 1
