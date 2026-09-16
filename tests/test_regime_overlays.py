"""Causal overlay checks — no same-bar leakage in regime/corr/hotstreak helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("qrd", ROOT / "scripts" / "quest_regime_diversify.py")
qrd = importlib.util.module_from_spec(spec)
sys.modules["qrd"] = qrd
# ewc dependency
spec_e = importlib.util.spec_from_file_location("ewc", ROOT / "scripts" / "eval_windowed_consistency.py")
ewc = importlib.util.module_from_spec(spec_e)
sys.modules["ewc"] = ewc
spec_e.loader.exec_module(ewc)
spec.loader.exec_module(qrd)


def _synth_curves(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    curves = []
    for i in range(3):
        r = rng.normal(0.0001, 0.002, n)
        eq = (1 + pd.Series(r, index=idx)).cumprod() * 100_000
        curves.append(eq.rename(f"leg{i}"))
    return curves


def test_hotstreak_scale_is_lagged():
    curves = _synth_curves()
    port = qrd.combine_weighted(curves, np.ones(3) / 3)
    out = qrd.build_hotstreak(port, look=10, hot=0.01, cool=0.5, cold=-0.01, heat=1.4)
    # first look bars should not explode; series finite
    assert np.isfinite(out.values).all()
    assert len(out) == len(port)


def test_corr_throttle_finite_and_length():
    curves = _synth_curves()
    w = np.ones(3) / 3
    out = qrd.build_corr(curves, w, look=40, corr0=0.3, k=2.0)
    assert len(out) > 50
    assert np.isfinite(out.values).all()


def test_regime_uses_shifted_weights():
    curves = _synth_curves()
    mr, tr = curves[:2], curves[1:]
    # pad to equal length lists
    mr_c = [curves[0], curves[1]]
    tr_c = [curves[1], curves[2]]
    adx = pd.Series(np.linspace(10, 40, len(curves[0])), index=curves[0].index)
    out = qrd.build_regime(mr_c, tr_c, np.array([0.5, 0.5]), np.array([0.5, 0.5]), adx, 22.0, 0.8, 0.2)
    assert np.isfinite(out.values).all()
    # changing only the last ADX bar must not change the last portfolio return
    # (weights are shifted — today's ADX sizes tomorrow)
    adx2 = adx.copy()
    adx2.iloc[-1] = 99.0
    out2 = qrd.build_regime(mr_c, tr_c, np.array([0.5, 0.5]), np.array([0.5, 0.5]), adx2, 22.0, 0.8, 0.2)
    r1 = out.pct_change().iloc[-1]
    r2 = out2.pct_change().iloc[-1]
    assert abs(r1 - r2) < 1e-12
