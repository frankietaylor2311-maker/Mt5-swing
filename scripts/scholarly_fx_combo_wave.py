#!/usr/bin/env python3
"""Scholarly FX combo wave: carry + mom + dollar TSMOM × GPR/VIX regime.

Fixed literature priors — no holdout tuning. signal_lag + FRED/GPR/VIX pub lags.
Reports monthly means, OLS & Newey–West t-stats, %pos, top-3 concentration,
FTMO 10%/5% gates on year/holdout windows, optional GPR top-decile event study.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mt5_swing.backtest.metrics import compute_metrics
from mt5_swing.data.fred_rates import download_default_rate_panel, load_currency_rates
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.data.macro_uncertainty import download_vix, load_gpr, load_vix
from mt5_swing.strategies.gpr_regime import align_macro_to_index
from mt5_swing.strategies.scholarly_combo import ScholarlyComboConfig, sleeve_returns_bundle

HISTORY = ROOT / "data" / "history"
REPORTS = ROOT / "reports"
MACRO = ROOT / "data" / "macro"

USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
RISK_FX = ["AUDUSD", "NZDUSD", "GBPUSD"]  # cyclical / risk-on vs USD
SAFE_USD = ["USDJPY", "USDCHF", "USDCAD"]  # long = long USD
SIGNAL_LAG = 1
BOOT_N = 2000
BOOT_SEED = 42
INITIAL = 100_000.0


def load_pair_panel(symbols: list[str] = USD_MAJORS) -> pd.DataFrame:
    closes = {}
    for sym in symbols:
        path = HISTORY / f"{sym}_D1.csv"
        if not path.exists():
            print(f"WARN missing {path}")
            continue
        df = load_ohlc_csv(path, symbol=sym, timeframe="D1")
        closes[sym] = df["close"]
    px = pd.DataFrame(closes).sort_index().dropna(how="all")
    if px.index.tz is None:
        px.index = px.index.tz_localize("UTC")
    ret = px.pct_change()
    ret.attrs["data_source"] = "approximate_non_ftmo"
    return ret


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
    """Newey–West automatic bandwidth: floor(4 (T/100)^{2/9})."""
    if n < 4:
        return 0
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_tstat(x: np.ndarray, lags: int | None = None) -> tuple[float, int]:
    """HAC t-stat for H0: E[x]=0 using Bartlett kernel Newey–West."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    t = len(x)
    if t < 3:
        return float("nan"), 0
    L = newey_west_lags(t) if lags is None else int(lags)
    mu = float(x.mean())
    # Autocovariances of the series (for mean estimator)
    gamma0 = float(np.dot(x, x) / t)
    S = gamma0
    for j in range(1, L + 1):
        w = 1.0 - j / (L + 1.0)
        gamma_j = float(np.dot(x[j:], x[:-j]) / t)
        S += 2.0 * w * gamma_j
    S = max(S, 1e-18)
    se = np.sqrt(S / t)
    return float(mu / se) if se > 0 else float("nan"), L


def bootstrap_mean_ci(x: np.ndarray, n: int = BOOT_N, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(n)
    for i in range(n):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(np.mean(x)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def window_stats(r: pd.Series, start: str, end: str) -> dict:
    sl = r.loc[start:end]
    m = monthly_returns(sl)
    if m.empty:
        return {
            "start": start,
            "end": end,
            "n_days": int(len(sl)),
            "n_months": 0,
            "mean_mo": float("nan"),
            "pct_pos": float("nan"),
            "top3": float("nan"),
            "tstat_ols": float("nan"),
            "tstat_nw": float("nan"),
            "nw_lags": 0,
            "ann_sharpe": float("nan"),
            "total_ret": float("nan"),
        }
    arr = m.to_numpy(dtype=float)
    nw, L = newey_west_tstat(arr)
    d = sl.dropna()
    sharpe = (
        float(d.mean() / d.std(ddof=1) * np.sqrt(252))
        if len(d) > 5 and d.std(ddof=1) > 0
        else float("nan")
    )
    return {
        "start": start,
        "end": end,
        "n_days": int(len(sl)),
        "n_months": int(len(m)),
        "mean_mo": float(m.mean()),
        "pct_pos": float((m > 0).mean()),
        "top3": top3_share(m),
        "tstat_ols": ols_tstat(arr),
        "tstat_nw": nw,
        "nw_lags": L,
        "ann_sharpe": sharpe,
        "total_ret": float((1.0 + sl.fillna(0)).prod() - 1.0),
    }


def factor_summary(name: str, r: pd.Series) -> dict:
    m = monthly_returns(r)
    arr = m.to_numpy(dtype=float)
    mu, lo, hi = bootstrap_mean_ci(arr)
    nw, L = newey_west_tstat(arr)
    return {
        "factor": name,
        "n_months": int(len(m)),
        "mean_mo": float(m.mean()) if len(m) else float("nan"),
        "std_mo": float(m.std(ddof=1)) if len(m) > 1 else float("nan"),
        "tstat_ols": ols_tstat(arr),
        "tstat_nw": nw,
        "nw_lags": L,
        "boot_mean": mu,
        "boot_ci95_lo": lo,
        "boot_ci95_hi": hi,
        "pct_pos_mo": float((m > 0).mean()) if len(m) else float("nan"),
        "top3": top3_share(m) if len(m) else float("nan"),
        "ann_sharpe_daily": (
            float(r.dropna().mean() / r.dropna().std(ddof=1) * np.sqrt(252))
            if r.dropna().std(ddof=1) > 0
            else float("nan")
        ),
    }


def ftmo_gates(r: pd.Series, initial: float = INITIAL) -> bool:
    if r.dropna().empty:
        return False
    eq = (1.0 + r.fillna(0.0)).cumprod() * initial
    m = compute_metrics(eq, initial_equity=initial, periods_per_year=252.0)
    return bool(m.gates_pass)


def clears_consistency(st: dict, *, mean_floor: float = 0.01, pos_floor: float = 0.70, top3_ceil: float = 0.55) -> bool:
    if not np.isfinite(st.get("mean_mo", np.nan)):
        return False
    return (
        st["mean_mo"] >= mean_floor
        and st["pct_pos"] >= pos_floor
        and (not np.isfinite(st.get("top3", np.nan)) or st["top3"] <= top3_ceil)
    )


def gpr_event_study(
    pair_ret: pd.DataFrame,
    gpr_daily: pd.Series,
    *,
    pre: int = 5,
    post: int = 10,
    decile: float = 0.90,
) -> pd.DataFrame:
    """Average cumulative FX returns around top-decile lagged GPR days.

    Uses GPR known with 1-day lag (already in series if pub_lag applied).
    USD basket = equal-weight SAFE_USD (long USD); risk basket = RISK_FX (long foreign).
    """
    g = align_macro_to_index(gpr_daily, pair_ret.index)
    g_lag = g.shift(1)  # extra bar lag for event definition
    thr = g_lag.quantile(decile)
    events = g_lag[g_lag >= thr].dropna().index
    # Thin events: at most one per rolling week
    kept = []
    last = None
    for dt in events:
        if last is None or (dt - last).days >= 5:
            kept.append(dt)
            last = dt
    usd_cols = [c for c in SAFE_USD if c in pair_ret.columns]
    risk_cols = [c for c in RISK_FX if c in pair_ret.columns]
    # Long-USD return proxy: for USDJPY/USDCHF/USDCAD, +pair return = long USD
    usd_r = pair_ret[usd_cols].mean(axis=1) if usd_cols else pd.Series(0.0, index=pair_ret.index)
    # Risk FX: long foreign vs USD
    risk_r = pair_ret[risk_cols].mean(axis=1) if risk_cols else pd.Series(0.0, index=pair_ret.index)

    rows = []
    for h in range(-pre, post + 1):
        usd_hits = []
        risk_hits = []
        for dt in kept:
            loc = pair_ret.index.get_indexer([dt], method="pad")[0]
            j = loc + h
            if j < 0 or j >= len(pair_ret):
                continue
            # Cumulative from event day 0 to h (or reverse for pre)
            if h >= 0:
                usd_hits.append(float(usd_r.iloc[loc : j + 1].sum()))
                risk_hits.append(float(risk_r.iloc[loc : j + 1].sum()))
            else:
                usd_hits.append(float(usd_r.iloc[j : loc + 1].sum()))
                risk_hits.append(float(risk_r.iloc[j : loc + 1].sum()))
        rows.append(
            {
                "horizon": h,
                "n_events": len(kept),
                "usd_cum_mean": float(np.mean(usd_hits)) if usd_hits else float("nan"),
                "risk_cum_mean": float(np.mean(risk_hits)) if risk_hits else float("nan"),
                "usd_minus_risk": (
                    float(np.mean(usd_hits) - np.mean(risk_hits)) if usd_hits and risk_hits else float("nan")
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["n_events"] = len(kept)
    out.attrs["gpr_threshold"] = float(thr) if np.isfinite(thr) else float("nan")
    return out


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    MACRO.mkdir(parents=True, exist_ok=True)

    print("=== scholarly combo wave: data ===")
    rate_paths = download_default_rate_panel()
    vix_path = download_vix()
    try:
        gpr_m = load_gpr(freq="M", pub_lag_months=1)
        gpr_status = f"OK monthly n={len(gpr_m)} lag=1m"
    except Exception as exc:  # noqa: BLE001
        gpr_m = None
        gpr_status = f"FAIL monthly: {exc}"
    try:
        gpr_d = load_gpr(freq="D", pub_lag_days=1)
        gpr_d_status = f"OK daily n={len(gpr_d)} lag=1d"
    except Exception as exc:  # noqa: BLE001
        gpr_d = None
        gpr_d_status = f"FAIL daily: {exc}"

    rates = load_currency_rates(pub_lag_months=1)
    vix = load_vix(pub_lag_days=1)
    pair_ret = load_pair_panel()
    print(f"FRED={len(rate_paths)} VIX={vix_path} GPR_m={gpr_status} GPR_d={gpr_d_status}")
    print(f"pairs {list(pair_ret.columns)} {pair_ret.index.min().date()}→{pair_ret.index.max().date()}")

    cfg = ScholarlyComboConfig(signal_lag=SIGNAL_LAG)
    # Prefer daily GPR for regime when available; else monthly
    gpr_for_regime = gpr_d if gpr_d is not None and len(gpr_d) else gpr_m
    factors = sleeve_returns_bundle(rates, pair_ret, gpr_for_regime, vix, cfg=cfg)

    summaries = [factor_summary(k, v) for k, v in factors.items()]
    sum_df = pd.DataFrame(summaries)
    sum_df.to_csv(REPORTS / "scholarly_fx_combo_factor_summary.csv", index=False)

    windows = [
        ("2018-01-01", "2023-12-31", "IS_2018_2023", "eval"),
        ("2024-01-01", "2024-12-31", "year_2024", "eval"),
        ("2025-01-01", "2025-12-31", "year_2025", "eval"),
        ("2026-01-01", "2026-12-31", "year_2026", "eval"),
        ("2025-09-17", "2026-09-16", "holdout_365d", "confirm_only"),
    ]

    rows = []
    gate_rows = []
    for fname, series in factors.items():
        for a, b, label, role in windows:
            st = window_stats(series, a, b)
            st["factor"] = fname
            st["window"] = label
            st["role"] = role
            gp = ftmo_gates(series.loc[a:b])
            st["gates_pass"] = gp
            st["clears_1pct_bar"] = clears_consistency(st) and gp
            rows.append(st)
            gate_rows.append(
                {
                    "factor": fname,
                    "window": label,
                    "gates_pass": gp,
                    "mean_mo": st["mean_mo"],
                    "pct_pos": st["pct_pos"],
                    "top3": st["top3"],
                    "clears_1pct_bar": st["clears_1pct_bar"],
                }
            )
    win_df = pd.DataFrame(rows)
    win_df.to_csv(REPORTS / "scholarly_fx_combo_window_stats.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(REPORTS / "scholarly_fx_combo_gates.csv", index=False)

    mo_frames = [monthly_returns(s).rename(k) for k, s in factors.items()]
    pd.concat(mo_frames, axis=1).to_csv(REPORTS / "scholarly_fx_combo_monthly_returns.csv")

    # Event study
    event_df = pd.DataFrame()
    event_note = "skipped (no daily GPR)"
    if gpr_d is not None and len(gpr_d):
        event_df = gpr_event_study(pair_ret, gpr_d)
        event_df.to_csv(REPORTS / "scholarly_fx_gpr_event_study.csv", index=False)
        event_note = (
            f"n_events={event_df.attrs.get('n_events')} "
            f"thr≈{event_df.attrs.get('gpr_threshold'):.2f} "
            f"(top-decile lagged GPR, ≥5d separation)"
        )

    # Consistency board for combo only
    combo_wins = win_df[win_df.factor == "scholarly_combo"]
    any_clear = bool(combo_wins["clears_1pct_bar"].any()) if len(combo_wins) else False
    year_clear = False
    if len(combo_wins):
        need = {"year_2024", "year_2025", "year_2026", "holdout_365d"}
        sub = combo_wins[combo_wins.window.isin(need)]
        year_clear = bool(len(sub) == 4 and sub["clears_1pct_bar"].all())

    # Markdown
    lines: list[str] = []
    lines.append("# Scholarly FX combo wave (carry + mom + dollar TSMOM × GPR/VIX)\n\n")
    lines.append("**data_source:** `approximate_non_ftmo` + FRED rates + VIX + Caldara–Iacoviello GPR\n\n")
    lines.append(
        f"**signal_lag:** {SIGNAL_LAG} | **rates pub_lag:** 1m | **VIX lag:** 1d | "
        f"**GPR:** monthly 1m / daily 1d for regime+events\n\n"
    )
    lines.append(
        "**Priors (fixed, no HO tuning):** EW 1/3 carry / mom / dollar_tsmom; "
        f"carry cool={cfg.carry_cool} when max(z_VIX,z_GPR)≥{cfg.z_high}; "
        f"USD tilt≤{cfg.usd_tilt} on GPR z; mom 63d/skip21; n=2/2.\n\n"
    )
    lines.append("## Full-sample monthly moments\n\n")
    lines.append(
        "| Factor | n_mo | mean_mo | t OLS | t NW | NW L | boot 95% CI | %pos | top3 | Sharpe |\n"
    )
    lines.append("|--------|-----:|--------:|------:|-----:|----:|------------|-----:|-----:|-------:|\n")
    for _, row in sum_df.iterrows():
        ci = (
            f"[{row['boot_ci95_lo']:.4f}, {row['boot_ci95_hi']:.4f}]"
            if np.isfinite(row["boot_ci95_lo"])
            else "n/a"
        )
        t3 = f"{row['top3']*100:.0f}%" if np.isfinite(row["top3"]) else "n/a"
        lines.append(
            f"| {row['factor']} | {row['n_months']} | {row['mean_mo']*100:.3f}% | "
            f"{row['tstat_ols']:.2f} | {row['tstat_nw']:.2f} | {int(row['nw_lags'])} | {ci} | "
            f"{row['pct_pos_mo']*100:.1f}% | {t3} | {row['ann_sharpe_daily']:.2f} |\n"
        )

    lines.append("\n## Calendar / holdout windows\n\n")
    lines.append(
        "| Factor | Window | mean_mo | %pos | top3 | t OLS | t NW | Sharpe | gates | 1% bar | role |\n"
    )
    lines.append(
        "|--------|--------|--------:|-----:|-----:|------:|-----:|-------:|:-----:|:------:|------|\n"
    )
    for _, row in win_df.iterrows():
        mm = row["mean_mo"] * 100 if np.isfinite(row["mean_mo"]) else float("nan")
        pp = row["pct_pos"] * 100 if np.isfinite(row["pct_pos"]) else float("nan")
        t3 = row["top3"] * 100 if np.isfinite(row["top3"]) else float("nan")
        lines.append(
            f"| {row['factor']} | {row['window']} | {mm:.3f}% | {pp:.1f}% | {t3:.0f}% | "
            f"{row['tstat_ols']:.2f} | {row['tstat_nw']:.2f} | {row['ann_sharpe']:.2f} | "
            f"{'PASS' if row['gates_pass'] else 'FAIL'} | "
            f"{'YES' if row['clears_1pct_bar'] else 'no'} | {row['role']} |\n"
        )

    lines.append("\n## GPR top-decile event study (USD vs risk FX)\n\n")
    lines.append(f"{event_note}\n\n")
    if len(event_df):
        lines.append("| h | USD cum | risk FX cum | USD−risk |\n")
        lines.append("|--:|--------:|------------:|---------:|\n")
        for _, row in event_df.iterrows():
            lines.append(
                f"| {int(row['horizon'])} | {row['usd_cum_mean']*100:.3f}% | "
                f"{row['risk_cum_mean']*100:.3f}% | {row['usd_minus_risk']*100:.3f}% |\n"
            )
        # Summarize post-event mean USD-risk at h=5 and h=10
        for h in (5, 10):
            sub = event_df[event_df.horizon == h]
            if len(sub):
                lines.append(
                    f"\nAt h=+{h}: USD−risk ≈ **{float(sub.iloc[0]['usd_minus_risk'])*100:.3f}%** "
                    f"(positive ⇒ USD outperformed risk FX after GPR spike).\n"
                )

    lines.append("\n## Consistency bar vs ~1%/mo FTMO goal\n\n")
    lines.append(
        f"- Any single window clears (≥1% mean, ≥70% pos, top3≤55%, gates PASS)? "
        f"**{'YES' if any_clear else 'NO'}**\n"
    )
    lines.append(
        f"- Joint year_2024+2025+2026+holdout clear for `scholarly_combo`? "
        f"**{'YES' if year_clear else 'NO'}**\n"
    )
    lines.append(
        "- **Do not claim 1%/mo** unless joint clear — this wave does not promote.\n"
        "- Still `approximate_non_ftmo`; need FTMO MT5 CSVs + optional news NLP for next unlock.\n"
    )

    # Combo year snapshot for quest doc
    snap = []
    for wlabel in ("year_2024", "year_2025", "year_2026", "holdout_365d"):
        sub = combo_wins[combo_wins.window == wlabel]
        if len(sub):
            r = sub.iloc[0]
            snap.append(
                {
                    "window": wlabel,
                    "mean_mo": r["mean_mo"],
                    "pct_pos": r["pct_pos"],
                    "top3": r["top3"],
                    "gates": bool(r["gates_pass"]),
                    "clears": bool(r["clears_1pct_bar"]),
                }
            )
    meta = {
        "data_source": "approximate_non_ftmo",
        "signal_lag": SIGNAL_LAG,
        "gpr_status": gpr_status,
        "gpr_daily_status": gpr_d_status,
        "event_note": event_note,
        "any_window_clears_1pct": any_clear,
        "joint_years_holdout_clear": year_clear,
        "promote": False,
        "combo_snapshot": snap,
        "cfg": {
            "w_carry": cfg.w_carry,
            "w_mom": cfg.w_mom,
            "w_dollar": cfg.w_dollar,
            "carry_cool": cfg.carry_cool,
            "z_high": cfg.z_high,
            "usd_tilt": cfg.usd_tilt,
            "mom_formation": cfg.mom_formation_days,
            "mom_skip": cfg.mom_skip_days,
        },
    }
    (REPORTS / "scholarly_fx_combo_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    md_path = REPORTS / "scholarly_fx_combo_wave.md"
    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(sum_df.to_string(index=False))
    print(f"any_clear={any_clear} joint_clear={year_clear} promote=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
