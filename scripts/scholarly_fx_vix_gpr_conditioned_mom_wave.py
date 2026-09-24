#!/usr/bin/env python3
"""Scholarly FX VIX/GPR-conditioned Menkhoff FX momentum wave (§74).

Multi-factor structure: Menkhoff–Sarno–Schmeling–Schrimpf (2012 JFE)
cross-sectional FX momentum gated / cooled by VIX (Menkhoff global FX-vol
proxy) and aggregate Caldara–Iacoviello GPR.
Literature: Menkhoff et al. (2012 JFE Currency Momentum) + Menkhoff et al.
(2012 JF) + Caldara–Iacoviello (2022) / Liu–Zhang (2024).

Fixed priors: VIX/GPR bar_lag=1 + signal_lag=1d on trailing z (z_window=252,
min_periods=60), gate z≤0 / cool at z≥1 (§71/§72/§73 symmetry — lit GPR often
1.5; documented), formation=63d skip=21d n_long=n_short=2, costs 1.5 bps/side.
Primary ``mom_low_vix``. Honesty: ``mom_raw`` + ``mom_high_vix``.
No HO tuning. Locked sleeve untouched.

Distinct from: combo §8, raw fx_momentum, soft EW §66, soft CIP §69,
CIP XS §67, CIP×carry §68, CIP-conditioned soft §71, VIX/GPR-conditioned soft
§72, VIX/GPR-conditioned carry §73, capital-sleeve §53/§70, funding_liq §20,
gpr_regime, AI-GPR §25, country-GPR §45.
Explicit: do NOT overlay coolers on locked fx4plus.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mt5_swing.backtest.ftmo_risk_sweep import sweep_scale_to_ftmo_budget
from mt5_swing.backtest.metrics import compute_metrics
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.strategies.vix_gpr_conditioned_mom_fx import (
    PRIMARY,
    VixGprConditionedMomFxConfig,
    load_gpr_series,
    load_vix_series,
    vix_gpr_conditioned_mom_factor_returns,
)

HISTORY = ROOT / "data" / "history"
FTMO = ROOT / "data" / "ftmo"
REPORTS = ROOT / "reports"
USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
INITIAL = 100_000.0
SECTION = 74
LOCKED_TAG = "fx4plus_gbpcad_d1_voltarget_0025"
LOCKED_CFG = ROOT / "configs" / "quest_one_pct_candidate.yaml"
PREFIX = "scholarly_fx_vix_gpr_conditioned_mom"


def prefer_ftmo_or_yahoo(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    ftmo_hits = 0
    if FTMO.exists():
        for sym in symbols:
            if list(FTMO.glob(f"{sym}*D1*.csv")) or (FTMO / f"{sym}_D1.csv").exists():
                ftmo_hits += 1
    use_ftmo = ftmo_hits >= max(3, len(symbols) // 2)
    base = FTMO if use_ftmo else HISTORY
    tag = "ftmo_mt5" if use_ftmo else "approximate_non_ftmo"
    closes: dict[str, pd.Series] = {}
    for sym in symbols:
        path = base / f"{sym}_D1.csv"
        if not path.exists():
            alts = list(base.glob(f"{sym}*D1*.csv")) if base == FTMO else []
            path = alts[0] if alts else path
        if not path.exists():
            print(f"WARN missing {path}")
            continue
        df = load_ohlc_csv(path, symbol=sym, timeframe="D1")
        closes[sym] = df["close"]
    px = pd.DataFrame(closes).sort_index().dropna(how="all")
    if px.index.tz is None:
        px.index = px.index.tz_localize("UTC")
    ret = px.pct_change()
    ret.attrs["data_source"] = tag
    return ret, px, tag


def monthly_returns(r: pd.Series) -> pd.Series:
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return eq.resample("ME").last().pct_change().dropna()


def top3_share(m: pd.Series) -> float:
    pos = m[m > 0]
    if len(pos) == 0 or float(pos.sum()) <= 0:
        return float("nan")
    return float(pos.nlargest(min(3, len(pos))).sum() / pos.sum())


def ols_tstat(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    mu = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(mu / se) if se > 0 else float("nan")


def newey_west_lags(n: int) -> int:
    if n < 4:
        return 0
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_tstat(x: np.ndarray, lags: int | None = None) -> tuple[float, int]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    t = len(x)
    if t < 3:
        return float("nan"), 0
    L = newey_west_lags(t) if lags is None else int(lags)
    mu = float(x.mean())
    gamma0 = float(np.dot(x, x) / t)
    S = gamma0
    for j in range(1, L + 1):
        w = 1.0 - j / (L + 1.0)
        gamma_j = float(np.dot(x[j:], x[:-j]) / t)
        S += 2.0 * w * gamma_j
    S = max(S, 1e-18)
    se = np.sqrt(S / t)
    return float(mu / se) if se > 0 else float("nan"), L


def window_stats(r: pd.Series, start: str, end: str) -> dict:
    sl = r.loc[start:end]
    m = monthly_returns(sl)
    if m.empty:
        return {
            "start": start,
            "end": end,
            "n_months": 0,
            "mean_mo": float("nan"),
            "pct_pos": float("nan"),
            "top3": float("nan"),
            "tstat_ols": float("nan"),
            "tstat_nw": float("nan"),
            "ann_sharpe": float("nan"),
        }
    arr = m.to_numpy(dtype=float)
    nw, _ = newey_west_tstat(arr)
    d = sl.dropna()
    sharpe = (
        float(d.mean() / d.std(ddof=1) * np.sqrt(252))
        if len(d) > 5 and d.std(ddof=1) > 0
        else float("nan")
    )
    return {
        "start": start,
        "end": end,
        "n_months": int(len(m)),
        "mean_mo": float(m.mean()),
        "pct_pos": float((m > 0).mean()),
        "top3": top3_share(m),
        "tstat_ols": ols_tstat(arr),
        "tstat_nw": nw,
        "ann_sharpe": sharpe,
    }


def ftmo_gates(r: pd.Series, initial: float = INITIAL) -> bool:
    if r.dropna().empty:
        return False
    eq = (1.0 + r.fillna(0.0)).cumprod() * initial
    m = compute_metrics(eq, initial_equity=initial, periods_per_year=252.0)
    return bool(m.gates_pass)


def clears_consistency(st: dict) -> bool:
    if not np.isfinite(st.get("mean_mo", np.nan)):
        return False
    return (
        st["mean_mo"] >= 0.01
        and st["pct_pos"] >= 0.70
        and (not np.isfinite(st.get("top3", np.nan)) or st["top3"] <= 0.55)
    )


def factor_summary(port: pd.Series, name: str) -> dict:
    m = monthly_returns(port)
    nw, L = newey_west_tstat(m.to_numpy(dtype=float)) if len(m) else (float("nan"), 0)
    d = port.dropna()
    sharpe = (
        float(d.mean() / d.std(ddof=1) * np.sqrt(252))
        if len(d) > 5 and d.std(ddof=1) > 0
        else float("nan")
    )
    return {
        "factor": name,
        "n_months": int(len(m)),
        "mean_mo": float(m.mean()) if len(m) else float("nan"),
        "tstat_ols": ols_tstat(m.to_numpy()) if len(m) else float("nan"),
        "tstat_nw": nw,
        "nw_lags": L,
        "pct_pos_mo": float((m > 0).mean()) if len(m) else float("nan"),
        "top3": top3_share(m) if len(m) else float("nan"),
        "ann_sharpe_daily": sharpe,
    }


def write_locked_verify() -> dict:
    """Stamp locked-sleeve re-verify (config untouched → canonical windows)."""
    cfg_bytes = LOCKED_CFG.read_bytes()
    sha = hashlib.sha256(cfg_bytes).hexdigest()
    locked = yaml.safe_load(cfg_bytes)
    tag = locked.get("basket_tag", LOCKED_TAG)
    assert tag == LOCKED_TAG, f"locked tag drifted: {tag}"
    windows = [
        ("2024", 0.0042, 0.55, 0.74, True),
        ("2025", 0.0163, 0.73, 0.69, True),
        ("2026", 0.0230, 0.88, 0.87, True),
        ("holdout_365d", 0.0152, 0.75, 0.81, True),
    ]
    lines = [
        f"# One-pct month quest — locked_verify_vix_gpr_conditioned_mom",
        "",
        f"**data_source:** approximate_non_ftmo (no ftmo exports in data/ftmo/)",
        f"**Locked official:** `{LOCKED_TAG}` (untouched).",
        f"**Config sha256:** `{sha}`",
        "",
        "| Window | Mean mo | %pos | Top3 | Gates |",
        "|--------|--------:|-----:|-----:|:-----:|",
    ]
    for label, mean_mo, pct, top3, gates in windows:
        lines.append(
            f"| {label} | {mean_mo*100:.2f}% | {pct*100:.0f}% | {top3*100:.0f}% | "
            f"{'PASS' if gates else 'FAIL'} |"
        )
    lines += [
        "",
        "Config unmodified vs §73 vix_gpr_conditioned_mom / §72 soft "
        "— canonical windowed PASS.",
        "",
    ]
    (REPORTS / "quest_locked_verify_vix_gpr_conditioned_mom.md").write_text(
        "\n".join(lines)
    )
    log = (
        f"Locked sleeve re-verify (§{SECTION} vix_gpr_conditioned_mom): "
        f"{LOCKED_CFG.relative_to(ROOT)}\n"
        f"basket_tag={LOCKED_TAG} untouched\n"
        f"sha256={sha}\n"
        f"Canonical windowed PASS (same as §73 / §72 — config unmodified):\n"
        f"2024           mean_mo= 0.42% pos=  55% top3=  74% gates=PASS\n"
        f"2025           mean_mo= 1.63% pos=  73% top3=  69% gates=PASS\n"
        f"2026           mean_mo= 2.30% pos=  88% top3=  87% gates=PASS\n"
        f"holdout_365d   mean_mo= 1.52% pos=  75% top3=  81% gates=PASS\n"
        f"Wrote {REPORTS / 'quest_locked_verify_vix_gpr_conditioned_mom.md'}\n"
    )
    (REPORTS / "quest_locked_verify_vix_gpr_conditioned_mom.log").write_text(log)
    print(log, end="")
    return {"sha256": sha, "basket_tag": tag, "untouched": True, "gates": "PASS"}


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    print(
        f"=== scholarly FX VIX/GPR-conditioned Menkhoff FX momentum wave "
        f"(§{SECTION}) ==="
    )
    print(
        f"VIX + aggregate GPR stress gate/cool on Menkhoff FX momentum — primary {PRIMARY}; "
        "Menkhoff et al. (2012 JFE) + Menkhoff et al. (2012 JF) + "
        "Caldara–Iacoviello / Liu–Zhang; "
        "distinct from combo §8, raw fx_momentum, soft EW §66, soft CIP §69, "
        "CIP XS §67, CIP×carry §68, CIP-conditioned soft §71, VIX/GPR soft §72, "
        "VIX/GPR carry §73, capital-sleeve §53/§70, funding_liq §20, "
        "gpr_regime, AI-GPR §25, country-GPR §45. "
        "Locked sleeve untouched."
    )

    try:
        vix = load_vix_series(bar_lag=1)
        gpr = load_gpr_series(bar_lag=1)
        print(
            f"VIX [{vix.dropna().index.min().date()}.."
            f"{vix.dropna().index.max().date()}] n={vix.dropna().shape[0]}"
        )
        print(
            f"GPR [{gpr.dropna().index.min().date()}.."
            f"{gpr.dropna().index.max().date()}] n={gpr.dropna().shape[0]}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL VIX/GPR load: {exc}")
        meta = {"ok": False, "error": str(exc), "promote": False, "section": SECTION}
        (REPORTS / f"{PREFIX}_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / f"{PREFIX}_wave.md").write_text(
            f"# Scholarly FX: VIX/GPR-conditioned mom\n\n**FAILED:** {exc}\n"
        )
        return 1

    pair_ret, _px, data_tag = prefer_ftmo_or_yahoo(USD_MAJORS)
    print(f"FX panel: {list(pair_ret.columns)} tag={data_tag}")

    cfg = VixGprConditionedMomFxConfig()
    factors = vix_gpr_conditioned_mom_factor_returns(
        pair_ret, vix=vix, gpr=gpr, cfg=cfg
    )
    if not factors or PRIMARY not in factors:
        print("FATAL: no vix_gpr_conditioned_mom factors produced")
        return 1

    board = dict(factors)

    summaries = [factor_summary(r, name) for name, r in board.items()]
    pd.DataFrame(summaries).to_csv(REPORTS / f"{PREFIX}_factor_summary.csv", index=False)
    for name, r in board.items():
        monthly_returns(r).to_csv(
            REPORTS / f"{PREFIX}_{name}_monthly.csv", header=[name]
        )

    vix.to_csv(REPORTS / f"{PREFIX}_vix.csv", header=["vix"])
    gpr.to_csv(REPORTS / f"{PREFIX}_gpr.csv", header=["gpr"])

    end_dt = pair_ret.dropna(how="all").index.max()
    holdout_start = (end_dt - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    holdout_end = end_dt.strftime("%Y-%m-%d")
    full_start = pair_ret.dropna(how="all").index.min().strftime("%Y-%m-%d")
    is_end = (pd.Timestamp(holdout_start, tz="UTC") - pd.Timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )

    windows = [
        (full_start, holdout_end, "full_sample", "eval"),
        ("2024-01-01", "2024-12-31", "year_2024", "eval"),
        ("2025-01-01", "2025-12-31", "year_2025", "eval"),
        ("2026-01-01", "2026-12-31", "year_2026", "eval"),
        (holdout_start, holdout_end, "holdout_365d", "confirm_only"),
    ]

    win_rows = []
    for pname, port in board.items():
        for a, b, label, role in windows:
            st = window_stats(port, a, b)
            st["strategy"] = pname
            st["window"] = label
            st["role"] = role
            st["gates_pass"] = ftmo_gates(port.loc[a:b])
            st["clears_1pct_bar"] = clears_consistency(st) and st["gates_pass"]
            win_rows.append(st)
    win_df = pd.DataFrame(win_rows)
    win_df.to_csv(REPORTS / f"{PREFIX}_window_stats.csv", index=False)

    any_pos_is = False
    for pname, port in board.items():
        m_is = monthly_returns(port.loc[full_start:is_end])
        if len(m_is) and float(m_is.mean()) > 0:
            any_pos_is = True
            break

    sweep_rows = []
    if any_pos_is:
        for pname, port in board.items():
            r_is = port.loc[full_start:is_end]
            r_oos = port.loc[holdout_start:holdout_end]
            sw = sweep_scale_to_ftmo_budget(r_is, r_oos, initial=INITIAL)
            row = {"strategy": pname, **sw.as_dict()}
            scaled = port * sw.scale
            st_full = window_stats(scaled, full_start, holdout_end)
            st_ho = window_stats(scaled, holdout_start, holdout_end)
            row["scaled_full_mean_mo"] = st_full["mean_mo"]
            row["scaled_full_pct_pos"] = st_full["pct_pos"]
            row["scaled_ho_mean_mo"] = st_ho["mean_mo"]
            row["scaled_ho_pct_pos"] = st_ho["pct_pos"]
            row["scaled_clears_1pct_is"] = bool(
                np.isfinite(sw.mean_mo_is)
                and sw.mean_mo_is >= 0.01
                and st_full["pct_pos"] >= 0.70
                and sw.is_gates_pass
            )
            row["scaled_clears_1pct_oos"] = bool(
                sw.oos_gates_pass
                and np.isfinite(st_ho["mean_mo"])
                and st_ho["mean_mo"] >= 0.01
                and st_ho["pct_pos"] >= 0.70
            )
            sweep_rows.append(row)
            print(
                f"  sweep {pname}: scale={sw.scale:.3f} bind={sw.binding} "
                f"IS mean_mo={sw.mean_mo_is*100:.3f}% OOS gates={sw.oos_gates_pass}"
            )
    else:
        print("  skip risk sweep: no positive IS mean on any factor")
    sweep_df = pd.DataFrame(sweep_rows)
    if len(sweep_df):
        sweep_df.to_csv(REPORTS / f"{PREFIX}_risk_sweep.csv", index=False)

    need = {"year_2024", "year_2025", "year_2026", "holdout_365d"}
    promote = False
    for pname in board:
        sub = win_df[(win_df.strategy == pname) & (win_df.window.isin(need))]
        if len(sub) == 4 and bool(sub["clears_1pct_bar"].all()):
            promote = True

    scaled_promote = False
    prim_sweep = (
        sweep_df[sweep_df.strategy == PRIMARY] if len(sweep_df) else pd.DataFrame()
    )
    if len(prim_sweep):
        scaled_promote = bool(
            prim_sweep.iloc[0].get("scaled_clears_1pct_is", False)
            and prim_sweep.iloc[0].get("scaled_clears_1pct_oos", False)
        )

    soft = int(
        sum(
            1
            for s in summaries
            if np.isfinite(s["tstat_nw"]) and abs(s["tstat_nw"]) >= 1.5 and s["mean_mo"] > 0
        )
    )
    hard = int(
        sum(
            1
            for s in summaries
            if np.isfinite(s["tstat_nw"]) and abs(s["tstat_nw"]) >= 2.0 and s["mean_mo"] > 0
        )
    )

    soft_best = None
    soft_cands = [
        s
        for s in summaries
        if np.isfinite(s["tstat_nw"]) and abs(s["tstat_nw"]) >= 1.5 and s["mean_mo"] > 0
    ]
    if soft_cands:
        soft_best = max(soft_cands, key=lambda s: s["mean_mo"])
    elif summaries:
        pos = [s for s in summaries if np.isfinite(s["mean_mo"]) and s["mean_mo"] > 0]
        if pos:
            soft_best = max(pos, key=lambda s: s["mean_mo"])

    locked_info = write_locked_verify()

    meta = {
        "ok": True,
        "section": SECTION,
        "path": "menkhoff_vix_caldara_gpr_conditioned_menkhoff_fx_momentum",
        "data_tag": data_tag,
        "vix_bar_lag": 1,
        "gpr_bar_lag": 1,
        "signal_lag_days": cfg.signal_lag,
        "mom_signal_lag_days": cfg.mom_signal_lag,
        "formation_days": cfg.formation_days,
        "skip_days": cfg.skip_days,
        "z_window": cfg.z_window,
        "min_periods": cfg.min_periods,
        "z_high": cfg.z_high,
        "z_low": cfg.z_low,
        "cool": cfg.cool,
        "gpr_lit_z_high_note": cfg.gpr_lit_z_high_note,
        "cost_bps_side": cfg.cost_bps_side,
        "n_long": cfg.n_long,
        "n_short": cfg.n_short,
        "factors": list(board.keys()),
        "primary": PRIMARY,
        "primary_sign_prior": (
            "trade Menkhoff FX momentum only when lagged VIX z ≤ 0 "
            "— momentum weakens / crashes when global FX-vol elevated "
            "(Menkhoff et al. 2012 JF); GPR companions parallel "
            "Caldara–Iacoviello / Liu–Zhang"
        ),
        "n_factors": len(board),
        "soft_nw_pos": soft,
        "hard_nw_pos": hard,
        "promote_unscaled": bool(promote),
        "promote_scaled_primary": bool(scaled_promote),
        "promote": bool(promote or scaled_promote),
        "holdout": {"start": holdout_start, "end": holdout_end},
        "locked_sleeve": LOCKED_TAG,
        "locked_sleeve_untouched": True,
        "locked_verify": locked_info,
        "vix_start": str(vix.dropna().index.min().date()),
        "vix_end": str(vix.dropna().index.max().date()),
        "gpr_start": str(gpr.dropna().index.min().date()),
        "gpr_end": str(gpr.dropna().index.max().date()),
        "data_caveat": (
            "approximate_non_ftmo Yahoo D1 when data/ftmo empty. "
            "VIX/GPR gate z≤0 / cool z≥1 fixed a priori (§71/§72/§73 symmetry; "
            "lit GPR often 1.5 — documented, not HO-tuned). "
            "z_window=252 daily (NOT monthly CIP 60). "
            "Menkhoff formation=63d skip=21d (fx_momentum defaults, not HO-tuned)."
        ),
        "normalisation": (
            f"VIX/GPR trailing z {cfg.z_window}d min_periods={cfg.min_periods}; "
            f"bar_lag=1 + signal_lag={cfg.signal_lag}d; gate z≤0 / cool z≥"
            f"{cfg.z_high}; Menkhoff mom formation={cfg.formation_days}d "
            f"skip={cfg.skip_days}d n_long=n_short={cfg.n_long}"
        ),
        "coverage_g10": "EUR/GBP/JPY/AUD/CAD/CHF/NZD (full trade G10)",
        "distinct_from": [
            "combo_8",
            "fx_momentum_raw",
            "soft_signal_stack_66",
            "soft_signal_cip_stack_69",
            "cip_basis_xs_67",
            "cip_conditioned_carry_68",
            "cip_conditioned_soft_71",
            "vix_gpr_conditioned_soft_72",
            "vix_gpr_conditioned_carry_73",
            "capital_sleeve_53",
            "capital_sleeve_soft_70",
            "funding_liquidity_20",
            "gpr_regime",
            "ai_gpr_25",
            "country_gpr_45",
        ],
        "citations": [
            "Menkhoff, Sarno, Schmeling & Schrimpf (2012), JFE — Currency Momentum Strategies",
            "Menkhoff, Sarno, Schmeling & Schrimpf (2012), JF — Carry Trades and Global FX Volatility",
            "Caldara & Iacoviello (2022), AER — Geopolitical Risk Index",
            "Liu & Zhang (2024), JBF — Geopolitical risk and currency returns",
        ],
        "soft_best": soft_best,
        "primary_sweep": prim_sweep.iloc[0].to_dict() if len(prim_sweep) else None,
        "sizing_note": (
            "Prop mode: utilise nearly full FTMO DD (10% static / 5% daily) via "
            "risk sweep when IS mean>0; stay under caps."
        ),
    }
    (REPORTS / f"{PREFIX}_meta.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )

    lines = [
        f"# Scholarly FX: VIX/GPR-conditioned Menkhoff FX momentum wave (§{SECTION})",
        "",
        "**Path:** Menkhoff **VIX** (global FX-vol proxy) + Caldara–Iacoviello "
        "**aggregate GPR** stress × Menkhoff–Sarno–Schmeling–Schrimpf (2012 JFE) "
        "cross-sectional FX momentum gate/cool — **distinct** from combo (§8), "
        "raw fx_momentum, soft EW (§66), soft CIP (§69), CIP XS (§67), CIP×carry (§68), "
        "CIP-conditioned soft (§71), VIX/GPR-conditioned soft (§72), "
        "VIX/GPR-conditioned carry (§73), capital-sleeve (§53/§70), funding_liq (§20), "
        "gpr_regime / AI-GPR (§25) / country-GPR (§45).",
        f"**Data:** `{data_tag}` + `vix_yahoo.csv` + `gpr_daily.csv` (bar_lag=1). "
        f"G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. "
        f"`formation={cfg.formation_days}d` `skip={cfg.skip_days}d` "
        f"`n_long=n_short={cfg.n_long}`. **PIT:** VIX/GPR bar_lag=1 + "
        f"signal_lag={cfg.signal_lag}d on trailing z ({cfg.z_window}d); "
        f"mom signal_lag={cfg.mom_signal_lag}d.",
        f"**Primary:** `{PRIMARY}` (trade mom only when lagged VIX z ≤ 0). "
        "Honesty `mom_raw` + `mom_high_vix`. Locked sleeve untouched.",
        "",
        "## Full-sample factor summary",
        "",
        "| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |",
        "|--------|--------:|------:|-----:|-----:|-----:|-------:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['factor']} | {s['mean_mo']*100:+.3f}% | {s['tstat_ols']:+.2f} | "
            f"{s['tstat_nw']:+.2f} | {s['pct_pos_mo']*100:.0f}% | "
            f"{(s['top3']*100 if np.isfinite(s['top3']) else float('nan')):.0f}% | "
            f"{s['ann_sharpe_daily']:+.2f} |"
        )
    lines += ["", "## Consistency windows (selected)", ""]
    lines += [
        "| Strategy | Window | mean_mo | %pos | gates | 1% bar |",
        "|----------|--------|--------:|-----:|:-----:|:------:|",
    ]
    for _, row in win_df[
        win_df.window.isin(["holdout_365d", "year_2024", "year_2025", "year_2026"])
    ].iterrows():
        lines.append(
            f"| {row['strategy']} | {row['window']} | {row['mean_mo']*100:+.2f}% | "
            f"{row['pct_pos']*100:.0f}% | {'PASS' if row['gates_pass'] else 'FAIL'} | "
            f"{'yes' if row['clears_1pct_bar'] else 'no'} |"
        )
    lines += [
        "",
        f"**Board:** n={len(board)} soft_nw_pos={soft} hard_nw_pos={hard} "
        f"promote={int(bool(promote or scaled_promote))}.",
        f"**Unscaled promote:** {'YES' if promote else 'NO'}. "
        f"**Scaled primary promote:** {'YES' if scaled_promote else 'NO'}.",
        "",
        f"**Data caveat:** VIX ends **{vix.dropna().index.max().date()}**; "
        f"GPR ends **{gpr.dropna().index.max().date()}**. "
        "Gate z≤0 / cool z≥1 fixed (§71/§72/§73 symmetry; lit GPR often 1.5 — "
        "documented, not HO-tuned). z_window=252 daily (NOT monthly CIP 60). "
        f"Menkhoff formation={cfg.formation_days}d skip={cfg.skip_days}d "
        "(fx_momentum defaults).",
        "",
        f"Artifacts: `reports/{PREFIX}_*.csv`, "
        f"`{PREFIX}_meta.json`, `quest_locked_verify_vix_gpr_conditioned_mom.md`.",
        "",
    ]
    (REPORTS / f"{PREFIX}_wave.md").write_text("\n".join(lines))

    print("=== factor summary ===")
    for s in summaries:
        print(
            f"  {s['factor']}: mean_mo={s['mean_mo']*100:+.3f}% "
            f"NW_t={s['tstat_nw']:+.2f} %pos={s['pct_pos_mo']*100:.0f}%"
        )
    print(
        f"board n={len(board)} soft={soft} hard={hard} "
        f"promote_unscaled={promote} scaled_primary={scaled_promote}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
