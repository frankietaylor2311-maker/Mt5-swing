#!/usr/bin/env python3
"""Scholarly FX Menkhoff realized-vol wave (true G10 FX-RV, not VIX).

Fixed priors: trailing 21d/63d EW |ccy return| RV → z vs 252d; USD tilt on high
z / positive innovations; carry cool & low-vol-only gate. signal_lag=1.
No HO tuning of thresholds. Optional FTMO risk sweep if any +IS mean.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mt5_swing.backtest.ftmo_risk_sweep import sweep_scale_to_ftmo_budget
from mt5_swing.backtest.metrics import compute_metrics
from mt5_swing.data.fred_rates import load_currency_rates
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.strategies.fx_realized_vol import (
    FxRealizedVolConfig,
    build_fxrv_state,
    daily_fx_vol_level,
    fx_realized_vol_factor_returns,
)

HISTORY = ROOT / "data" / "history"
FTMO = ROOT / "data" / "ftmo"
REPORTS = ROOT / "reports"
USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
INITIAL = 100_000.0
SIGNAL_LAG = 1
PRIMARY = "carry_fxrv_cool_21d"


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


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    print("=== scholarly Menkhoff FX realized-vol wave ===")

    pair_ret, pair_close, data_tag = prefer_ftmo_or_yahoo(USD_MAJORS)
    print(f"FX panel: {list(pair_ret.columns)} tag={data_tag} n={len(pair_ret)}")

    try:
        rates = load_currency_rates(pub_lag_months=1, download=False)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN rates load failed ({exc}); carry legs skipped")
        rates = None
    if rates is not None:
        print(f"rates cols={list(rates.columns)} rows={len(rates)}")

    cfg = FxRealizedVolConfig(
        signal_lag=SIGNAL_LAG,
        rv_windows=(21, 63),
        z_window=252,
        z_high=1.0,
        z_low=0.0,
        cool=0.35,
        usd_tilt=0.5,
        cost_bps_side=1.5,
    )
    factors = fx_realized_vol_factor_returns(pair_ret, rates, cfg=cfg)
    if not factors:
        print("FATAL: no FX-RV factors produced")
        return 1

    # Persist RV state series
    level = daily_fx_vol_level(pair_ret)
    level.to_csv(REPORTS / "scholarly_fx_rv_daily_level.csv", header=["fx_vol_level"])
    for w in cfg.rv_windows:
        st = build_fxrv_state(pair_ret, cfg=cfg, window=int(w))
        st["rv"].to_csv(REPORTS / f"scholarly_fx_rv_{int(w)}d.csv", header=[f"rv_{int(w)}d"])
        st["z"].to_csv(REPORTS / f"scholarly_fx_rv_z_{int(w)}d.csv", header=[f"z_{int(w)}d"])
        st["innov"].to_csv(
            REPORTS / f"scholarly_fx_rv_innov_{int(w)}d.csv", header=[f"innov_{int(w)}d"]
        )

    # Correlation of 21d FX-RV vs VIX (if available) — document distinctness
    vix_corr = None
    vix_path = ROOT / "data" / "macro" / "vix_yahoo.csv"
    if vix_path.exists():
        try:
            from mt5_swing.data.macro_uncertainty import load_vix
            from mt5_swing.strategies.gpr_regime import align_macro_to_index

            vix = load_vix(download=False, pub_lag_days=1)
            st21 = build_fxrv_state(pair_ret, cfg=cfg, window=21)
            vix_a = align_macro_to_index(vix, st21["rv"].index)
            common = st21["rv"].dropna().index.intersection(vix_a.dropna().index)
            if len(common) > 100:
                vix_corr = float(st21["rv"].loc[common].corr(vix_a.loc[common]))
                print(f"corr(fx_rv_21d, VIX) = {vix_corr:.3f}")
        except Exception as exc:  # noqa: BLE001
            print(f"WARN VIX corr skipped: {exc}")

    summaries = [factor_summary(r, name) for name, r in factors.items()]
    pd.DataFrame(summaries).to_csv(
        REPORTS / "scholarly_fx_rv_factor_summary.csv", index=False
    )
    for name, r in factors.items():
        monthly_returns(r).to_csv(
            REPORTS / f"scholarly_fx_rv_{name}_monthly.csv", header=[name]
        )

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
    for pname, port in factors.items():
        for a, b, label, role in windows:
            st = window_stats(port, a, b)
            st["strategy"] = pname
            st["window"] = label
            st["role"] = role
            st["gates_pass"] = ftmo_gates(port.loc[a:b])
            st["clears_1pct_bar"] = clears_consistency(st) and st["gates_pass"]
            win_rows.append(st)
    win_df = pd.DataFrame(win_rows)
    win_df.to_csv(REPORTS / "scholarly_fx_rv_window_stats.csv", index=False)

    any_pos_is = False
    for pname, port in factors.items():
        m_is = monthly_returns(port.loc[full_start:is_end])
        if len(m_is) and float(m_is.mean()) > 0:
            any_pos_is = True
            break

    sweep_rows = []
    if any_pos_is:
        for pname, port in factors.items():
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
        sweep_df.to_csv(REPORTS / "scholarly_fx_rv_risk_sweep.csv", index=False)

    need = {"year_2024", "year_2025", "year_2026", "holdout_365d"}
    promote = False
    for pname in factors:
        sub = win_df[(win_df.strategy == pname) & (win_df.window.isin(need))]
        if len(sub) == 4 and bool(sub["clears_1pct_bar"].all()):
            promote = True

    scaled_promote = False
    prim_sweep = (
        sweep_df[sweep_df.strategy == PRIMARY] if len(sweep_df) else pd.DataFrame()
    )
    # Also allow best standalone if primary missing
    if prim_sweep.empty and len(sweep_df):
        # Prefer fxrv_usd_tilt_21d
        alt = sweep_df[sweep_df.strategy == "fxrv_usd_tilt_21d"]
        if len(alt):
            prim_sweep = alt
    if len(prim_sweep):
        scaled_promote = bool(
            prim_sweep.iloc[0].get("scaled_clears_1pct_is", False)
            and prim_sweep.iloc[0].get("scaled_clears_1pct_oos", False)
        )

    meta = {
        "ok": True,
        "data_tag": data_tag,
        "signal_lag": SIGNAL_LAG,
        "rv_windows": list(cfg.rv_windows),
        "z_window": cfg.z_window,
        "z_high": cfg.z_high,
        "z_low": cfg.z_low,
        "cool": cfg.cool,
        "usd_tilt": cfg.usd_tilt,
        "cost_bps_side": cfg.cost_bps_side,
        "factors": list(factors.keys()),
        "primary": PRIMARY,
        "fx_cols": list(pair_ret.columns),
        "rates_cols": list(rates.columns) if rates is not None else [],
        "corr_fxrv21_vix": vix_corr,
        "any_pos_is": any_pos_is,
        "is_end": is_end,
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
        "promote": promote,
        "scaled_promote": scaled_promote,
        "risk_sweep_primary": prim_sweep.iloc[0].to_dict() if len(prim_sweep) else None,
        "refs": [
            "Menkhoff, Sarno, Schmeling & Schrimpf (2012), Carry Trades and Global FX Volatility, JF",
            "True FX RV = EW mean |ccy return|; distinct from equity VIX proxy",
            "Frozen cutoffs z_high=1, cool=0.35 matching scholarly combo priors",
        ],
    }
    (REPORTS / "scholarly_fx_rv_meta.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )

    lines: list[str] = []
    lines.append("# Scholarly FX: Menkhoff global FX realized-vol risk factor\n\n")
    lines.append(
        "**Prior (fixed):** Menkhoff et al. (2012 JF) — global FX volatility is a "
        "priced state variable; high FX vol predicts carry underperformance / "
        "risk-off USD. "
        f"RV windows={list(cfg.rv_windows)}d; z_window={cfg.z_window}; "
        f"z_high={cfg.z_high}; cool={cfg.cool}; signal_lag={SIGNAL_LAG}. "
        "**Not VIX** (already tried). No HO tuning.\n\n"
    )
    lines.append(
        f"**Data:** `{data_tag}` D1 USD majors ({', '.join(pair_ret.columns)}). "
        f"Rates: {', '.join(rates.columns) if rates is not None else 'none'}. "
        f"corr(fx_rv_21d, VIX)={vix_corr if vix_corr is not None else 'n/a'}.\n\n"
    )
    lines.append("## Factor board (full sample, unscaled)\n\n")
    lines.append("| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |\n")
    lines.append("|--------|--------:|------:|-----:|-----:|-----:|-------:|\n")
    for s in summaries:
        lines.append(
            f"| {s['factor']} | {s['mean_mo']*100:.3f}% | {s['tstat_ols']:.2f} | "
            f"{s['tstat_nw']:.2f} | {s['pct_pos_mo']*100:.1f}% | "
            f"{(s['top3']*100 if np.isfinite(s['top3']) else float('nan')):.0f}% | "
            f"{s['ann_sharpe_daily']:.2f} |\n"
        )
    lines.append("\n## Calendar / holdout windows (unscaled)\n\n")
    lines.append(
        "| Strategy | Window | mean_mo | %pos | top3 | t NW | Sharpe | gates | 1% bar | role |\n"
    )
    lines.append(
        "|----------|--------|--------:|-----:|-----:|-----:|-------:|:-----:|:------:|------|\n"
    )
    for _, row in win_df.iterrows():
        t3 = row["top3"] * 100 if np.isfinite(row["top3"]) else float("nan")
        lines.append(
            f"| {row['strategy']} | {row['window']} | {row['mean_mo']*100:.3f}% | "
            f"{row['pct_pos']*100:.1f}% | {t3:.0f}% | {row['tstat_nw']:.2f} | "
            f"{row['ann_sharpe']:.2f} | {'PASS' if row['gates_pass'] else 'FAIL'} | "
            f"{'YES' if row['clears_1pct_bar'] else 'no'} | {row['role']} |\n"
        )

    if len(sweep_df):
        lines.append("\n## FTMO risk sweep (IS → OOS confirm)\n\n")
        lines.append(
            f"IS = [{full_start} … {is_end}]; OOS = [{holdout_start} … {holdout_end}]. "
            "Run because ≥1 factor had positive IS mean.\n\n"
        )
        lines.append(
            "| Strategy | scale | bind | IS mean_mo | IS static | IS daily | IS gate | "
            "OOS mean_mo | OOS gate |\n"
        )
        lines.append(
            "|----------|------:|:----:|-----------:|----------:|---------:|:-------:|"
            "-----------:|:--------:|\n"
        )
        for _, row in sweep_df.iterrows():
            oos_mo = row["mean_mo_oos"] if pd.notna(row["mean_mo_oos"]) else float("nan")
            lines.append(
                f"| {row['strategy']} | {row['scale']:.2f} | {row['binding']} | "
                f"{row['mean_mo_is']*100:.3f}% | {row['is_static_loss']*100:.2f}% | "
                f"{row['is_max_daily_dd']*100:.2f}% | "
                f"{'PASS' if row['is_gates_pass'] else 'FAIL'} | "
                f"{oos_mo*100:.3f}% | "
                f"{'PASS' if row['oos_gates_pass'] else 'FAIL'} |\n"
            )
    else:
        lines.append(
            "\n## FTMO risk sweep\n\n"
            "**Skipped** — no factor with positive full-sample IS mean "
            "(no unused DD budget to invent edge).\n"
        )

    lines.append("\n## Promote / 1%/mo bar\n\n")
    lines.append(
        f"**Unscaled joint promote:** **{'YES' if promote else 'NO'}**. "
        f"**Scaled primary (`{PRIMARY}`) promote:** **{'YES' if scaled_promote else 'NO'}**. "
        "Do not claim 1%/mo unless earned. Locked sleeve untouched; "
        "Yahoo ≠ FTMO MT5; no go-live. Distinct from VIX-only regime (§8).\n\n"
    )
    lines.append(
        "### Honest gaps\n\n"
        "- Yahoo D1 abs-return average ≠ OTC tick FX vol / option-implied FXVIX.\n"
        "- G10 subset only (7 USD majors); no EM carries in the RV basket.\n"
        "- Innovation proxy is RV − trailing mean, not a full ARMA residual.\n"
        "- Menkhoff risk *premium* is about pricing carry crashes — tradable "
        "USD-tilt / cool rules are research proxies, not paper replications.\n"
    )
    lines.append("\nArtifacts: `scholarly_fx_rv_*.csv`, `scholarly_fx_rv_meta.json`.\n")
    (REPORTS / "scholarly_fx_rv_wave.md").write_text("".join(lines))
    print(f"wrote {REPORTS / 'scholarly_fx_rv_wave.md'}")
    print(f"promote={promote} scaled_promote={scaled_promote} any_pos_is={any_pos_is}")
    for s in summaries:
        print(
            f"  {s['factor']}: mean_mo={s['mean_mo']*100:.3f}% "
            f"pos={s['pct_pos_mo']*100:.1f}% sharpe={s['ann_sharpe_daily']:.2f} "
            f"tNW={s['tstat_nw']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
