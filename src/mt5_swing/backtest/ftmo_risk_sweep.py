"""Scale a return series to utilise nearly the full FTMO DD budget.

Prop-challenge ethos: do **not** leave risk headroom on the table. Sweep a
leverage / scale factor on in-sample daily returns until we are *just under*
the binding gate — static max loss 10% of initial **or** daily loss 5% of
initial (FTMO 2-Step style) — whichever binds first. Then re-check OOS /
holdout at that same scale (no OOS retuning).

Gates use strict inequalities matching ``mt5_swing.backtest.metrics.compute_metrics``:
``static_loss < 0.10`` and ``max_daily_dd < 0.05``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.backtest.metrics import compute_metrics

# Stay *just under* the hard FTMO limits (strict < in metrics).
STATIC_CAP = 0.099  # target utilisation vs 10% static
DAILY_CAP = 0.049   # target utilisation vs 5% daily
STATIC_GATE = 0.10
DAILY_GATE = 0.05


@dataclass
class RiskSweepResult:
    scale: float
    binding: str  # "daily" | "static" | "none" | "flat"
    is_total_return: float
    is_static_loss: float
    is_max_daily_dd: float
    is_gates_pass: bool
    oos_total_return: float | None
    oos_static_loss: float | None
    oos_max_daily_dd: float | None
    oos_gates_pass: bool | None
    mean_mo_is: float
    mean_mo_oos: float | None
    n_scales_tried: int

    def as_dict(self) -> dict:
        return {
            "scale": self.scale,
            "binding": self.binding,
            "is_total_return": self.is_total_return,
            "is_static_loss": self.is_static_loss,
            "is_max_daily_dd": self.is_max_daily_dd,
            "is_gates_pass": self.is_gates_pass,
            "oos_total_return": self.oos_total_return,
            "oos_static_loss": self.oos_static_loss,
            "oos_max_daily_dd": self.oos_max_daily_dd,
            "oos_gates_pass": self.oos_gates_pass,
            "mean_mo_is": self.mean_mo_is,
            "mean_mo_oos": self.mean_mo_oos,
            "n_scales_tried": self.n_scales_tried,
        }


def returns_to_equity(r: pd.Series, initial: float = 100_000.0) -> pd.Series:
    return (1.0 + r.fillna(0.0)).cumprod() * float(initial)


def monthly_mean(r: pd.Series) -> float:
    if r.dropna().empty:
        return float("nan")
    eq = (1.0 + r.fillna(0.0)).cumprod()
    m = eq.resample("ME").last().pct_change().dropna()
    return float(m.mean()) if len(m) else float("nan")


def metrics_at_scale(
    r: pd.Series,
    scale: float,
    *,
    initial: float = 100_000.0,
) -> dict:
    eq = returns_to_equity(r * float(scale), initial=initial)
    m = compute_metrics(
        eq,
        max_dd_gate=STATIC_GATE,
        daily_dd_gate=DAILY_GATE,
        periods_per_year=252.0,
        max_loss_mode="static_initial",
        daily_loss_mode="ftmo_initial",
        initial_equity=initial,
    )
    return {
        "scale": float(scale),
        "total_return": float(m.total_return),
        "static_loss": float(m.static_loss_from_initial),
        "max_daily_dd": float(m.max_daily_dd),
        "gates_pass": bool(m.gates_pass),
        "mean_mo": monthly_mean(r * float(scale)),
        "sharpe": float(m.sharpe),
    }


def _binding(static_loss: float, daily_dd: float) -> str:
    # Which constraint is closer to its cap (utilisation view)
    u_s = static_loss / STATIC_CAP if STATIC_CAP > 0 else 0.0
    u_d = daily_dd / DAILY_CAP if DAILY_CAP > 0 else 0.0
    if u_d >= u_s:
        return "daily"
    return "static"


def sweep_scale_to_ftmo_budget(
    r_is: pd.Series,
    r_oos: pd.Series | None = None,
    *,
    initial: float = 100_000.0,
    scale_grid: np.ndarray | None = None,
    target_static: float = STATIC_CAP,
    target_daily: float = DAILY_CAP,
) -> RiskSweepResult:
    """Find largest scale on IS with gates still PASS, prefer near-full utilisation.

    Searches a geometric grid then tightens with binary search so we sit just
    under the binding FTMO limit. OOS is evaluated at the chosen IS scale only
    (no holdout retuning).
    """
    r_is = r_is.dropna()
    if r_is.empty or float(np.nanstd(r_is.values)) == 0.0:
        return RiskSweepResult(
            scale=0.0,
            binding="flat",
            is_total_return=0.0,
            is_static_loss=0.0,
            is_max_daily_dd=0.0,
            is_gates_pass=True,
            oos_total_return=None if r_oos is None else 0.0,
            oos_static_loss=None if r_oos is None else 0.0,
            oos_max_daily_dd=None if r_oos is None else 0.0,
            oos_gates_pass=None if r_oos is None else True,
            mean_mo_is=0.0,
            mean_mo_oos=None if r_oos is None else 0.0,
            n_scales_tried=0,
        )

    if scale_grid is None:
        # Wide grid: unit leverage through aggressive prop sizing
        scale_grid = np.unique(
            np.concatenate(
                [
                    np.geomspace(0.25, 80.0, 48),
                    np.linspace(0.5, 40.0, 40),
                ]
            )
        )

    tried = 0
    best = None
    for sc in scale_grid:
        tried += 1
        st = metrics_at_scale(r_is, sc, initial=initial)
        if st["gates_pass"]:
            if best is None or st["scale"] > best["scale"]:
                best = st

    if best is None:
        # Even tiny scale fails — report scale→0
        st0 = metrics_at_scale(r_is, 0.0, initial=initial)
        return RiskSweepResult(
            scale=0.0,
            binding="none",
            is_total_return=st0["total_return"],
            is_static_loss=st0["static_loss"],
            is_max_daily_dd=st0["max_daily_dd"],
            is_gates_pass=True,
            oos_total_return=None,
            oos_static_loss=None,
            oos_max_daily_dd=None,
            oos_gates_pass=None,
            mean_mo_is=st0["mean_mo"],
            mean_mo_oos=None,
            n_scales_tried=tried,
        )

    # Binary search between best and next failing neighbour for tighter utilisation
    lo = float(best["scale"])
    hi = lo * 1.5
    # find a failing upper bound
    for _ in range(20):
        tried += 1
        st = metrics_at_scale(r_is, hi, initial=initial)
        if st["gates_pass"]:
            lo = hi
            best = st
            hi *= 1.35
        else:
            break
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        tried += 1
        st = metrics_at_scale(r_is, mid, initial=initial)
        if st["gates_pass"]:
            lo = mid
            best = st
        else:
            hi = mid

    # Optional nudge: if still well below both caps, accept (signal may be tiny)
    bind = _binding(best["static_loss"], best["max_daily_dd"])

    oos_tr = oos_sl = oos_dd = oos_gp = oos_mo = None
    if r_oos is not None and len(r_oos.dropna()):
        ot = metrics_at_scale(r_oos, best["scale"], initial=initial)
        oos_tr, oos_sl, oos_dd = ot["total_return"], ot["static_loss"], ot["max_daily_dd"]
        oos_gp, oos_mo = ot["gates_pass"], ot["mean_mo"]

    return RiskSweepResult(
        scale=float(best["scale"]),
        binding=bind,
        is_total_return=float(best["total_return"]),
        is_static_loss=float(best["static_loss"]),
        is_max_daily_dd=float(best["max_daily_dd"]),
        is_gates_pass=bool(best["gates_pass"]),
        oos_total_return=oos_tr,
        oos_static_loss=oos_sl,
        oos_max_daily_dd=oos_dd,
        oos_gates_pass=oos_gp,
        mean_mo_is=float(best["mean_mo"]),
        mean_mo_oos=oos_mo,
        n_scales_tried=int(tried),
    )
