"""Tests for CFTC COT positioning FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.cftc_cot import (
    FX_CONTRACT_CODES,
    cot_score_panel,
    dx_series,
    load_cot_panel,
)
from mt5_swing.strategies.cot_positioning_fx import (
    CotPositioningFxConfig,
    cot_positioning_factor_returns,
    prepare_cot_scores,
    trailing_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"


def test_trailing_z_mean_near_zero():
    idx = pd.date_range("2010-01-01", periods=80, freq="W-FRI", tz="UTC")
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"EUR": rng.normal(0, 1, 80)}, index=idx)
    z = trailing_z(x, lookback=52, min_periods=26)
    tail = z.dropna().iloc[-20:]
    assert abs(float(tail["EUR"].mean())) < 0.5


def test_lev_net_long_high_score():
    """Currency with higher lev_net_oi should rank above lower."""
    idx = pd.date_range("2018-01-05", periods=40, freq="W-FRI", tz="UTC")
    lev = pd.DataFrame(
        {
            "EUR": np.linspace(0.1, 0.4, 40),
            "GBP": np.linspace(-0.3, -0.1, 40),
            "AUD": np.zeros(40),
            "CAD": np.zeros(40),
        },
        index=idx,
    )
    cfg = CotPositioningFxConfig(signal_lag=0, n_long=1, n_short=1)
    scores = prepare_cot_scores(lev, cfg=cfg)
    last = scores["cot_lev_net_xs"].dropna(how="all").iloc[-1]
    assert float(last["EUR"]) > float(last["GBP"])
    # MR flips sign of z relative to continuation z
    z_last = scores["cot_lev_z_xs"].dropna(how="all").iloc[-1]
    mr_last = scores["cot_lev_z_mr_xs"].dropna(how="all").iloc[-1]
    assert np.allclose(mr_last.values, -z_last.values, equal_nan=True)


def test_cot_no_lookahead():
    """Mutating future lev_net must not change earlier factor returns."""
    idx = pd.date_range("2016-01-01", periods=1800, freq="B", tz="UTC")
    rng = np.random.default_rng(11)
    close = pd.DataFrame(
        {
            "EURUSD": 1.1 * np.cumprod(1 + rng.normal(0, 0.005, 1800)),
            "GBPUSD": 1.3 * np.cumprod(1 + rng.normal(0, 0.005, 1800)),
            "AUDUSD": 0.75 * np.cumprod(1 + rng.normal(0, 0.006, 1800)),
            "NZDUSD": 0.65 * np.cumprod(1 + rng.normal(0, 0.006, 1800)),
            "USDJPY": 110 * np.cumprod(1 + rng.normal(0, 0.005, 1800)),
            "USDCAD": 1.3 * np.cumprod(1 + rng.normal(0, 0.004, 1800)),
            "USDCHF": 0.95 * np.cumprod(1 + rng.normal(0, 0.004, 1800)),
        },
        index=idx,
    )
    ret = close.pct_change()
    w_idx = pd.date_range("2015-01-02", periods=260, freq="W-FRI", tz="UTC")
    lev = pd.DataFrame(
        {c: rng.normal(0, 0.15, 260) for c in ("EUR", "GBP", "AUD", "NZD", "JPY", "CAD", "CHF")},
        index=w_idx,
    )
    nc = lev * 0.8 + rng.normal(0, 0.05, lev.shape)
    dx = pd.Series(rng.normal(0, 0.1, 260), index=w_idx, name="DX")
    cfg = CotPositioningFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = cot_positioning_factor_returns(ret, lev, nc_net_oi=nc, dx_lev_net_oi=dx, cfg=cfg)
    assert "cot_lev_net_xs" in f1
    assert "cot_lev_chg_xs" in f1
    assert "cot_lev_z_mr_xs" in f1
    assert "cot_noncomm_net_xs" in f1
    assert "cot_dx_usd" in f1
    assert "cot_ew" in f1
    lev2 = lev.copy()
    lev2.iloc[-5:] += 2.0
    f2 = cot_positioning_factor_returns(ret, lev2, nc_net_oi=nc, dx_lev_net_oi=dx, cfg=cfg)
    cut = f1["cot_lev_net_xs"].index[-80]
    a = f1["cot_lev_net_xs"].loc[:cut].dropna()
    b = f2["cot_lev_net_xs"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 200
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


def test_fx_contract_codes_cover_g10_and_dx():
    for ccy in ("EUR", "GBP", "JPY", "CAD", "CHF", "AUD", "NZD", "DX"):
        assert ccy in FX_CONTRACT_CODES


@pytest.mark.skipif(
    not (MACRO / "cftc_cot_fx_panel.csv").exists()
    and not (MACRO / "cftc_tff_fx_raw.csv").exists(),
    reason="CFTC COT cache not present (download in wave script)",
)
def test_load_cot_panel_from_cache():
    panel = load_cot_panel(download=True, force_download=False, release_lag_days=3)
    lev = cot_score_panel(panel, field="lev_net_oi")
    assert "EUR" in lev.columns
    assert "GBP" in lev.columns
    assert len(lev.dropna(how="all")) > 100
    dx = dx_series(panel, field="lev_net_oi")
    assert len(dx.dropna()) > 50
    # known_date should be Friday-ish after Tuesday report (lag=3)
    assert panel.attrs.get("release_lag_days", 3) == 3
