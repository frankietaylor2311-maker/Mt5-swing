"""Tests for OECD MEI building-permits (ODCNPI03) housing-activity FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.fred_building_permits import (
    DEFAULT_PUB_LAG_MONTHS,
    PERMITS_SERIES,
    building_permits_coverage,
    load_building_permits_panel,
    load_us_building_permits,
)
from mt5_swing.strategies.building_permits_fx import (
    BuildingPermitsFxConfig,
    building_permits_factor_returns,
    prepare_building_permits_scores,
    trailing_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_permits_series_map_mapped_and_unmapped():
    assert PERMITS_SERIES["USD"] == "PERMIT"
    assert PERMITS_SERIES["EUR"] == "DEUODCNPI03GYSAM"
    assert PERMITS_SERIES["CAD"] == "CANODCNPI03GYSAM"
    assert PERMITS_SERIES["AUD"] == "AUSODCNPI03GYSAM"
    assert PERMITS_SERIES["NZD"] == "NZLODCNPI03GYSAM"
    assert PERMITS_SERIES["GBP"] is None
    assert PERMITS_SERIES["JPY"] is None
    assert PERMITS_SERIES["CHF"] is None


def test_pub_lag_shifts_known_date():
    """pub_lag_months=2 must move month-start index forward by ~2 months."""
    raw = load_building_permits_panel(pub_lag_months=0, download=True, force=False)
    lagged = load_building_permits_panel(pub_lag_months=2, download=False, force=False)
    assert not raw.empty and not lagged.empty
    r0 = raw["EUR"].dropna().index.min()
    l0 = lagged["EUR"].dropna().index.min()
    delta_days = (l0 - r0).days
    assert 30 <= delta_days <= 90


def test_prepare_scores_high_housing_long_strong():
    """Higher permits YoY → higher high_housing score (growth-channel prior)."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(60, 1.0), np.linspace(1.0, 8.0, 60)])
    cad = np.concatenate([np.full(60, 1.0), np.linspace(1.0, -5.0, 60)])
    panel = pd.DataFrame(
        {
            "USD": np.full(120, 1.0),
            "EUR": eur,
            "CAD": cad,
            "AUD": np.full(120, 1.0),
            "NZD": np.full(120, 1.0),
        },
        index=idx,
    )
    cfg = BuildingPermitsFxConfig(signal_lag=0, z_window=60, min_periods=24)
    scores = prepare_building_permits_scores(panel, cfg=cfg)
    last = scores["high_housing"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["CAD"])
    last_lo = scores["low_housing"].dropna(how="all").iloc[-1]
    assert float(last_lo["CAD"]) > float(last_lo["EUR"])


def test_housing_chg_prefers_acceleration():
    """Rising YoY → positive housing_chg score."""
    idx = pd.date_range("2015-01-01", periods=48, freq="MS", tz="UTC")
    eur_yoy = np.concatenate([np.zeros(12), np.full(24, 2.0), np.full(12, 20.0)])
    cad_yoy = np.concatenate([np.zeros(12), np.full(24, 10.0), np.full(12, 11.0)])
    panel = pd.DataFrame(
        {
            "USD": np.full(48, 1.0),
            "EUR": eur_yoy,
            "CAD": cad_yoy,
            "AUD": np.full(48, 1.0),
            "NZD": np.full(48, 1.0),
        },
        index=idx,
    )
    cfg = BuildingPermitsFxConfig(signal_lag=0, chg_periods=12)
    scores = prepare_building_permits_scores(panel, cfg=cfg)
    last = scores["housing_chg"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["CAD"])


def test_building_permits_no_lookahead():
    """Mutating future permits must not change earlier factor returns."""
    idx = pd.date_range("2015-01-01", periods=2000, freq="B", tz="UTC")
    rng = np.random.default_rng(42)
    close = pd.DataFrame(
        {
            "EURUSD": 1.1 * np.cumprod(1 + rng.normal(0, 0.005, 2000)),
            "GBPUSD": 1.3 * np.cumprod(1 + rng.normal(0, 0.005, 2000)),
            "AUDUSD": 0.75 * np.cumprod(1 + rng.normal(0, 0.006, 2000)),
            "NZDUSD": 0.65 * np.cumprod(1 + rng.normal(0, 0.006, 2000)),
            "USDJPY": 110 * np.cumprod(1 + rng.normal(0, 0.005, 2000)),
            "USDCAD": 1.3 * np.cumprod(1 + rng.normal(0, 0.004, 2000)),
            "USDCHF": 0.95 * np.cumprod(1 + rng.normal(0, 0.004, 2000)),
        },
        index=idx,
    )
    ret = close.pct_change()
    m_idx = pd.date_range("2005-01-01", periods=260, freq="MS", tz="UTC")
    panel = pd.DataFrame(
        {c: rng.normal(1.0, 2.0, 260) for c in ("USD", "EUR", "CAD", "AUD", "NZD")},
        index=m_idx,
    )
    cfg = BuildingPermitsFxConfig(signal_lag=0, cost_bps_side=1.5)
    f1 = building_permits_factor_returns(ret, panel, cfg=cfg)
    assert "high_housing_xs" in f1
    assert "low_housing_xs" in f1
    assert "high_housing_z_xs" in f1
    assert "housing_chg_xs" in f1
    assert "us_housing_stress_fx" in f1
    assert "us_housing_haven_usd" in f1
    assert "housing_ew" in f1

    panel2 = panel.copy()
    panel2.iloc[-6:] = panel2.iloc[-6:] + 15.0
    f2 = building_permits_factor_returns(ret, panel2, cfg=cfg)

    cut = idx[-120]
    for name in ("high_housing_xs", "housing_chg_xs", "us_housing_stress_fx"):
        a = f1[name].loc[:cut].dropna()
        b = f2[name].loc[:cut].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_us_housing_z_fires_on_depressed_growth():
    """Depressed US permits YoY vs recent history → negative z."""
    m_idx = pd.date_range("2000-01-01", periods=240, freq="MS", tz="UTC")
    vals = np.concatenate([np.full(216, 2.0), np.full(24, -8.0)])
    us = pd.Series(vals, index=m_idx, name="USD")
    z = trailing_z(us, lookback=60, min_periods=24)
    assert float(z.dropna().iloc[-1]) < -1.0


def test_loader_smoke_and_coverage():
    panel = load_building_permits_panel(
        pub_lag_months=DEFAULT_PUB_LAG_MONTHS, download=True
    )
    # Mapped: USD/EUR/CAD/AUD/NZD (≥4 foreign + USD)
    assert panel.shape[1] >= 5
    assert "USD" in panel.columns
    assert "EUR" in panel.columns
    assert "CAD" in panel.columns
    assert "AUD" in panel.columns
    assert "NZD" in panel.columns
    for c in ("GBP", "JPY", "CHF"):
        assert c not in panel.columns or panel[c].dropna().empty
    us = load_us_building_permits(download=False)
    assert len(us) > 50
    cov = building_permits_coverage(panel)
    assert len(cov) >= 5
    mapped = cov[cov["n_obs"] > 0]
    assert (mapped["n_obs"] > 0).all()
    assert panel.attrs.get("series_map", {}).get("EUR") == "DEUODCNPI03GYSAM"
    assert panel.attrs.get("series_map", {}).get("USD") == "PERMIT"
    assert panel.attrs.get("unit") == "permits_yoy_pct"
    assert DEFAULT_PUB_LAG_MONTHS == 2
    assert panel.attrs.get("acm_fallback_used") is False
    end = panel.dropna(how="all").index.max()
    assert end.year >= 2025
    unmapped = set(panel.attrs.get("unmapped", []))
    assert {"GBP", "JPY", "CHF"} <= unmapped


def test_high_housing_maps_to_top_score():
    """High permits YoY maps to top high_housing scores."""
    m_idx = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
    eur = np.concatenate([np.full(60, 1.0), np.linspace(1.0, 10.0, 60)])
    cad = np.concatenate([np.full(60, 1.0), np.linspace(1.0, -2.0, 60)])
    panel = pd.DataFrame(
        {
            "USD": np.full(120, 1.0),
            "EUR": eur,
            "CAD": cad,
            "AUD": np.full(120, 1.0),
            "NZD": np.full(120, 1.0),
        },
        index=m_idx,
    )
    cfg = BuildingPermitsFxConfig(signal_lag=0, n_long=2, n_short=2)
    scores = prepare_building_permits_scores(panel, cfg=cfg)
    last = scores["high_housing"].dropna(how="all").iloc[-1].dropna()
    top2 = set(last.nlargest(2).index)
    assert "EUR" in top2


def test_n_long_short_a_priori():
    """Available foreign panel → n_long=n_short=2 a priori."""
    cfg = BuildingPermitsFxConfig()
    assert cfg.n_long == 2
    assert cfg.n_short == 2


def test_usd_permit_yoy_derived_not_levels():
    """USD panel values must be YoY % (not raw PERMIT thousands)."""
    panel = load_building_permits_panel(["USD"], pub_lag_months=0, download=False)
    s = panel["USD"].dropna()
    assert len(s) > 50
    # Raw PERMIT is ~1000–2000; YoY % typically within ±100
    assert float(s.abs().median()) < 80.0
    assert float(s.abs().max()) < 200.0


def test_distinct_from_house_price_series():
    """Building-permits series must not be BIS HPI Q*R628BIS."""
    assert PERMITS_SERIES["USD"] != "QUSR628BIS"
    assert PERMITS_SERIES["EUR"] != "QDER628BIS"
    assert "ODCNPI03" in (PERMITS_SERIES["EUR"] or "")
