#!/usr/bin/env python3
"""Country-GPR → FX depreciation wave (Caldara–Iacoviello GPRC_*).

Fixed prior: high home-country GPR → depreciate that currency vs USD.
Tests: lagged long-low/short-high sort + local-projection β at h=1,3,6 months.
Also expands/documents FRED OECD rate coverage. No technical overlays; no HO tuning.
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
from mt5_swing.data.fred_rates import (
    download_default_rate_panel,
    rate_panel_coverage,
)
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.data.macro_uncertainty import load_country_gpr
from mt5_swing.strategies.country_gpr_fx import (
    CountryGprFxConfig,
    country_gpr_sort_returns,
    local_projection_panel,
)

HISTORY = ROOT / "data" / "history"
REPORTS = ROOT / "reports"
USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
SIGNAL_LAG = 1
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


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    print("=== country GPR FX wave ===")
    download_default_rate_panel(extended=True)
    cov = rate_panel_coverage(download=False, extended=True)
    cov.to_csv(REPORTS / "scholarly_fx_fred_rate_coverage.csv", index=False)

    cg = load_country_gpr(download=False, pub_lag_months=1)
    pair_ret = load_pair_panel()
    cfg = CountryGprFxConfig(signal_lag=SIGNAL_LAG)
    port = country_gpr_sort_returns(cg, pair_ret, cfg=cfg)
    lp = local_projection_panel(cg, pair_ret, cfg=cfg)
    lp.to_csv(REPORTS / "scholarly_fx_country_gpr_lp.csv", index=False)

    m = monthly_returns(port)
    nw, L = newey_west_tstat(m.to_numpy(dtype=float))
    summary = {
        "factor": "country_gpr_sort",
        "n_months": int(len(m)),
        "mean_mo": float(m.mean()) if len(m) else float("nan"),
        "tstat_ols": ols_tstat(m.to_numpy()),
        "tstat_nw": nw,
        "nw_lags": L,
        "pct_pos_mo": float((m > 0).mean()) if len(m) else float("nan"),
        "top3": top3_share(m) if len(m) else float("nan"),
        "ann_sharpe_daily": (
            float(port.dropna().mean() / port.dropna().std(ddof=1) * np.sqrt(252))
            if port.dropna().std(ddof=1) > 0
            else float("nan")
        ),
        "hypothesis": "long_low_home_GPR_short_high (high GPR → depreciate)",
        "currencies": [c for c in cg.columns if c != "USD"],
        "missing_nzld": True,
    }
    pd.DataFrame([summary]).to_csv(REPORTS / "scholarly_fx_country_gpr_factor_summary.csv", index=False)
    m.to_csv(REPORTS / "scholarly_fx_country_gpr_monthly_returns.csv", header=["country_gpr_sort"])

    windows = [
        ("2018-01-01", "2023-12-31", "IS_2018_2023", "eval"),
        ("2024-01-01", "2024-12-31", "year_2024", "eval"),
        ("2025-01-01", "2025-12-31", "year_2025", "eval"),
        ("2026-01-01", "2026-12-31", "year_2026", "eval"),
        ("2025-09-17", "2026-09-16", "holdout_365d", "confirm_only"),
    ]
    rows = []
    for a, b, label, role in windows:
        st = window_stats(port, a, b)
        st["window"] = label
        st["role"] = role
        st["gates_pass"] = ftmo_gates(port.loc[a:b])
        st["clears_1pct_bar"] = clears_consistency(st) and st["gates_pass"]
        rows.append(st)
    win_df = pd.DataFrame(rows)
    win_df.to_csv(REPORTS / "scholarly_fx_country_gpr_window_stats.csv", index=False)

    any_clear = bool(win_df["clears_1pct_bar"].any())
    need = {"year_2024", "year_2025", "year_2026", "holdout_365d"}
    sub = win_df[win_df.window.isin(need)]
    joint = bool(len(sub) == 4 and sub["clears_1pct_bar"].all())

    # LP sign check
    pooled = lp[(lp.scope == "pooled")]
    hyp_support = {}
    for _, row in pooled.iterrows():
        hyp_support[f"h{int(row['horizon'])}"] = {
            "beta": float(row["beta"]),
            "tstat": float(row["tstat"]),
            "sign_ok": bool(row["beta"] < 0),  # negative = depreciation after high GPR
            "n": int(row["n"]),
        }

    stale = cov[cov.sparse_or_stale]
    lines = []
    lines.append("# Scholarly FX: country-level GPR → FX depreciation\n\n")
    lines.append(
        "**Prior (fixed):** high home-country Caldara–Iacoviello GPR → depreciate "
        "that currency vs USD. Sort = long low-GPR / short high-GPR. "
        f"pub_lag=1m + signal_lag={SIGNAL_LAG}m; z_window={cfg.z_window}m.\n\n"
    )
    lines.append(f"**Data:** `approximate_non_ftmo` Yahoo D1 + GPRC_* from monthly export. ")
    lines.append("**NZD gap:** no `GPRC_NZL` in 44-country file.\n\n")
    lines.append("## Lagged sort portfolio\n\n")
    lines.append(
        f"| mean_mo | t OLS | t NW | %pos | top3 | Sharpe |\n"
        f"|--------:|------:|-----:|-----:|-----:|-------:|\n"
        f"| {summary['mean_mo']*100:.3f}% | {summary['tstat_ols']:.2f} | "
        f"{summary['tstat_nw']:.2f} | {summary['pct_pos_mo']*100:.1f}% | "
        f"{summary['top3']*100:.0f}% | {summary['ann_sharpe_daily']:.2f} |\n\n"
    )
    lines.append("## Calendar / holdout windows\n\n")
    lines.append("| Window | mean_mo | %pos | top3 | t NW | gates | 1% bar | role |\n")
    lines.append("|--------|--------:|-----:|-----:|-----:|:-----:|:------:|------|\n")
    for _, row in win_df.iterrows():
        lines.append(
            f"| {row['window']} | {row['mean_mo']*100:.3f}% | {row['pct_pos']*100:.1f}% | "
            f"{row['top3']*100:.0f}% | {row['tstat_nw']:.2f} | "
            f"{'PASS' if row['gates_pass'] else 'FAIL'} | "
            f"{'YES' if row['clears_1pct_bar'] else 'no'} | {row['role']} |\n"
        )
    lines.append("\n## Local projections (FX cumret on lagged home GPR z)\n\n")
    lines.append("Hypothesis support if **β < 0** (high GPR → subsequent depreciation).\n\n")
    lines.append("| Scope | h | beta | t | n | sign_ok |\n")
    lines.append("|-------|--:|-----:|--:|--:|:------:|\n")
    for _, row in lp.sort_values(["scope", "currency", "horizon"]).iterrows():
        if row["scope"] == "pooled" or row["currency"] in ("EUR", "JPY", "GBP", "AUD", "CAD", "CHF"):
            lines.append(
                f"| {row['currency']} | {int(row['horizon'])} | {row['beta']*100:.3f}%/σ | "
                f"{row['tstat']:.2f} | {int(row['n'])} | "
                f"{'yes' if row['beta'] < 0 else 'no'} |\n"
            )
    lines.append("\n## FRED OECD rate coverage (sparse / missing)\n\n")
    lines.append(
        f"Extended panel: {len(cov[cov.status=='ok'])} series OK; "
        f"{int(stale.shape[0])} sparse/stale/missing flagged.\n\n"
    )
    lines.append("| CCY | series | end | months_lag | note |\n")
    lines.append("|-----|--------|-----|----------:|------|\n")
    for _, row in stale.iterrows():
        lines.append(
            f"| {row['currency']} | {row['series_id']} | {row['end']} | "
            f"{row['months_since_end']} | {row['note'] or row['status']} |\n"
        )
    lines.append("\n## Consistency vs ~1%/mo\n\n")
    lines.append(f"- Any window clears? **{'YES' if any_clear else 'NO'}**\n")
    lines.append(f"- Joint years+holdout? **{'YES' if joint else 'NO'}**\n")
    lines.append("- **Do not claim 1%/mo.** Still `approximate_non_ftmo`.\n")
    lines.append(
        "- **Gaps:** news NLP not wired; FTMO MT5 CSVs not present; "
        "NZD country GPR missing; SEK FRED discontinued ~2020.\n"
    )

    md = REPORTS / "scholarly_fx_country_gpr_wave.md"
    md.write_text("".join(lines), encoding="utf-8")
    meta = {
        "data_source": "approximate_non_ftmo",
        "signal_lag": SIGNAL_LAG,
        "pub_lag_months": 1,
        "any_window_clears_1pct": any_clear,
        "joint_years_holdout_clear": joint,
        "promote": False,
        "lp_pooled": hyp_support,
        "summary": summary,
        "fred_stale_count": int(stale.shape[0]),
        "gaps": ["news_nlp", "ftmo_csvs", "GPRC_NZL", "SEK_fred_discontinued"],
    }
    (REPORTS / "scholarly_fx_country_gpr_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote {md}")
    print(f"mean_mo={summary['mean_mo']*100:.3f}% t_nw={summary['tstat_nw']:.2f} "
          f"any_clear={any_clear} joint={joint}")
    print("LP pooled:", hyp_support)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
