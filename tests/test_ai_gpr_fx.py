"""Tests for AI-GPR bilateral role decompositions FX module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt5_swing.data.ai_gpr import (
    load_ai_gpr_panel,
    load_ai_gpr_series,
    oil_me_vs_non_series,
)
from mt5_swing.strategies.ai_gpr_fx import (
    AiGprFxConfig,
    ai_gpr_factor_returns,
    align_and_z,
    usd_tilt_weights_from_z,
)

MACRO = Path(__file__).resolve().parents[1] / "data" / "macro"
AI_CSV = MACRO / "ai_gpr_daily.csv"


def test_loader_pub_lag_shifts_index(tmp_path: Path):
    """Publication lag must move the usable date forward (no same-day use)."""
    raw = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "THREATS_GPR_AI": [100.0, 110.0, 120.0],
            "ACTS_GPR_AI": [90.0, 95.0, 100.0],
            "GPR_AI": [105.0, 108.0, 112.0],
            "GPR_OIL": [50.0, 55.0, 60.0],
            "GPR_NONOIL": [100.0, 100.0, 100.0],
            "GPR_OIL_MiddleEast": [10.0, 20.0, 30.0],
            "GPR_OIL_THREATS": [40.0, 45.0, 50.0],
        }
    )
    path = tmp_path / "ai_gpr_daily.csv"
    raw.to_csv(path, index=False)

    no_lag = load_ai_gpr_panel(pub_lag_days=0, path=path)
    lagged = load_ai_gpr_panel(pub_lag_days=1, path=path)
    assert no_lag.index.min() == pd.Timestamp("2020-01-01", tz="UTC")
    assert lagged.index.min() == pd.Timestamp("2020-01-02", tz="UTC")
    assert lagged.attrs["pub_lag_days"] == 1
    # Value that was on 2020-01-01 becomes available on 2020-01-02
    assert float(lagged.loc[pd.Timestamp("2020-01-02", tz="UTC"), "THREATS_GPR_AI"]) == 100.0

    s = load_ai_gpr_series("THREATS_GPR_AI", pub_lag_days=1, path=path)
    assert s.name == "THREATS_GPR_AI"
    assert s.attrs["pub_lag_days"] == 1


def test_oil_me_vs_non_diff(tmp_path: Path):
    raw = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02"],
            "GPR_OIL_MiddleEast": [30.0, 40.0],
            "GPR_NONOIL": [10.0, 10.0],
            "THREATS_GPR_AI": [1.0, 1.0],
        }
    )
    path = tmp_path / "ai.csv"
    raw.to_csv(path, index=False)
    panel = load_ai_gpr_panel(
        pub_lag_days=0,
        path=path,
        columns=("GPR_OIL_MiddleEast", "GPR_NONOIL", "THREATS_GPR_AI"),
    )
    d = oil_me_vs_non_series(panel, pub_lag_days=0)
    assert np.isclose(d.iloc[0], 20.0)
    assert np.isclose(d.iloc[1], 30.0)


def test_usd_tilt_fires_on_high_z():
    idx = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    z = pd.Series([0.0, 0.5, 1.5, 2.0, 0.2, -0.5, 1.1, 0.0, 0.0, 0.0], index=idx)
    cfg = AiGprFxConfig(usd_tilt=0.5, z_high=1.0, binary_usd=True)
    cols = ["EURUSD", "USDJPY", "AUDUSD", "USDCAD"]
    w = usd_tilt_weights_from_z(z, cols, cfg=cfg)
    # High z: short risk FX (EURUSD/AUDUSD), long USDJPY/USDCAD
    assert w.loc[idx[2], "EURUSD"] < 0
    assert w.loc[idx[2], "AUDUSD"] < 0
    assert w.loc[idx[2], "USDJPY"] > 0
    assert w.loc[idx[2], "USDCAD"] > 0
    assert w.loc[idx[0], "EURUSD"] == 0.0


def test_align_and_z_signal_lag():
    idx = pd.date_range("2015-01-01", periods=400, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    raw = pd.Series(
        rng.normal(100, 20, 500),
        index=pd.date_range("2014-01-01", periods=500, freq="D", tz="UTC"),
        name="THREATS_GPR_AI",
    )
    cfg = AiGprFxConfig(signal_lag=1, z_window=60, min_periods=30)
    z = align_and_z(raw, idx, cfg=cfg)
    assert z.index.equals(idx)
    assert z.dropna().shape[0] > 50
    assert z.dropna().index.min() > idx.min()


def test_ai_gpr_no_lookahead():
    """Mutating future threats must not change earlier factor returns."""
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
    d_idx = pd.date_range("2014-01-01", periods=2000, freq="D", tz="UTC")
    threats = pd.Series(rng.normal(100, 25, 2000), index=d_idx, name="THREATS_GPR_AI")
    acts = pd.Series(rng.normal(100, 20, 2000), index=d_idx, name="ACTS_GPR_AI")
    gpr_ai = pd.Series(rng.normal(100, 22, 2000), index=d_idx, name="GPR_AI")
    oil = pd.Series(rng.normal(80, 40, 2000), index=d_idx, name="GPR_OIL")
    oil_th = pd.Series(rng.normal(70, 35, 2000), index=d_idx, name="GPR_OIL_THREATS")
    me_vs = pd.Series(rng.normal(0, 30, 2000), index=d_idx, name="OIL_ME_VS_NON")
    m_idx = pd.date_range("2014-01-01", periods=96, freq="MS", tz="UTC")
    rates = pd.DataFrame(
        {
            c: 1.0 + 0.01 * rng.normal(0, 1, 96).cumsum()
            for c in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")
        },
        index=m_idx,
    )
    cfg = AiGprFxConfig(signal_lag=1, cost_bps_side=1.5)
    f1 = ai_gpr_factor_returns(
        ret,
        threats=threats,
        acts=acts,
        gpr_ai=gpr_ai,
        oil_gpr=oil,
        oil_threats=oil_th,
        oil_me_vs_non=me_vs,
        rates=rates,
        cfg=cfg,
    )
    assert "ai_threats_usd" in f1
    assert "ai_acts_usd" in f1
    assert "ai_gpr_usd" in f1
    assert "oil_gpr_usd" in f1
    assert "oil_threats_usd" in f1
    assert "oil_me_vs_non" in f1
    assert "carry_ai_threats_cool" in f1
    assert "ai_ew" in f1

    threats2 = threats.copy()
    threats2.loc["2021-01-01":] = threats2.loc["2021-01-01":] + 80.0
    f2 = ai_gpr_factor_returns(
        ret,
        threats=threats2,
        acts=acts,
        gpr_ai=gpr_ai,
        oil_gpr=oil,
        oil_threats=oil_th,
        oil_me_vs_non=me_vs,
        rates=rates,
        cfg=cfg,
    )
    cut = pd.Timestamp("2019-06-30", tz="UTC")
    a = f1["ai_threats_usd"].loc[:cut].dropna()
    b = f2["ai_threats_usd"].loc[:cut].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 200
    assert np.allclose(a.loc[common].values, b.loc[common].values, equal_nan=True, atol=1e-12)


@pytest.mark.skipif(not AI_CSV.exists(), reason="ai_gpr_daily.csv missing")
def test_load_ai_gpr_from_disk():
    panel = load_ai_gpr_panel(pub_lag_days=1)
    assert len(panel) > 10_000
    assert panel.attrs.get("pub_lag_days") == 1
    assert "THREATS_GPR_AI" in panel.columns
    assert "GPR_OIL_MiddleEast" in panel.columns
    s = load_ai_gpr_series("THREATS_GPR_AI", pub_lag_days=1)
    assert len(s.dropna()) > 10_000


def test_build_role_series_no_gprc():
    """Role constructions must never reference monthly GPRC_* country indexes."""
    if not AI_CSV.exists():
        pytest.skip("ai_gpr_daily.csv missing")
    from mt5_swing.data.ai_gpr import build_role_series

    panel = load_ai_gpr_panel(pub_lag_days=1)
    roles = build_role_series(panel)
    assert "threat_minus_act" in roles
    assert "oil_me_vs_non" in roles
    for k in roles:
        assert "GPRC" not in k.upper()
    common = roles["threat_minus_act"].dropna().index.intersection(
        roles["threats"].dropna().index
    )
    assert len(common) > 100
    expected = roles["threats"].loc[common] - roles["acts"].loc[common]
    assert np.allclose(
        roles["threat_minus_act"].loc[common].values, expected.values, equal_nan=True
    )

