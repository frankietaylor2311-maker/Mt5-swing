#!/usr/bin/env python3
"""Scholarly FX macro-differential wave (Dahlquist-style; free FRED only).

Fixed priors: lagged CPI/IP/UR differentials vs USD → G10 FX long/short sorts.
pub lags in loaders (CPI/UR=1m, IP=2m) + signal_lag≥1 month. No HO tuning.
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
from mt5_swing.data.fred_macro_diff import (
    DEFAULT_PUB_LAGS,
    load_cpi_yoy_panel,
    load_ip_yoy_panel,
    load_ur_panel,
    macro_diff_coverage,
    macro_differentials_vs_usd,
)
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.strategies.macro_diff_fx import MacroDiffFxConfig, macro_diff_factor_returns

HISTORY = ROOT / "data" / "history"
FTMO = ROOT / "data" / "ftmo"
REPORTS = ROOT / "reports"
USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
INITIAL = 100_000.0
SIGNAL_LAG = 1


def prefer_ftmo_or_yahoo(symbols: list[str]) -> tuple[pd.DataFrame, str]:
    ftmo_hits = 0
    if FTMO.exists():
        for sym in symbols:
            if list(FTMO.glob(f"{sym}*D1*.csv")) or (FTMO / f"{sym}_D1.csv").exists():
                ftmo_hits += 1
    use_ftmo = ftmo_hits >= max(3, len(symbols) // 2)
    base = FTMO if use_ftmo else HISTORY
    tag = "ftmo_mt5" if use_ftmo else "approximate_non_ftmo"
    closes = {}
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
    return ret, tag


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
    print("=== scholarly macro-diff FX wave (Dahlquist-style FRED) ===")

    try:
        cpi = load_cpi_yoy_panel(download=True)
        ip = load_ip_yoy_panel(download=True)
        ur = load_ur_panel(download=True)
        cpi_d = macro_differentials_vs_usd(cpi)
        ip_d = macro_differentials_vs_usd(ip) if "USD" in ip.columns else ip
        ur_d = macro_differentials_vs_usd(ur) if "USD" in ur.columns else ur
        cov = macro_diff_coverage()
        cov.to_csv(REPORTS / "scholarly_fx_macro_diff_coverage.csv", index=False)
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL macro load: {exc}")
        meta = {"ok": False, "error": str(exc), "promote": False}
        (REPORTS / "scholarly_fx_macro_diff_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / "scholarly_fx_macro_diff_wave.md").write_text(
            f"# Scholarly FX: macro differentials\n\n**FAILED:** {exc}\n"
        )
        return 1

    pair_ret, data_tag = prefer_ftmo_or_yahoo(USD_MAJORS)
    print(f"FX panel: {list(pair_ret.columns)} tag={data_tag}")
    print(f"cpi_diff cols={list(cpi_d.columns)} ip={list(ip_d.columns)} ur={list(ur_d.columns)}")

    cfg = MacroDiffFxConfig(signal_lag=SIGNAL_LAG)
    factors = macro_diff_factor_returns(pair_ret, cpi_d, ip_d, ur_d, cfg=cfg)
    if not factors:
        print("FATAL: no macro factors produced")
        return 1

    summaries = [factor_summary(r, name) for name, r in factors.items()]
    pd.DataFrame(summaries).to_csv(REPORTS / "scholarly_fx_macro_diff_factor_summary.csv", index=False)
    for name, r in factors.items():
        monthly_returns(r).to_csv(
            REPORTS / f"scholarly_fx_macro_diff_{name}_monthly.csv", header=[name]
        )

    end_dt = pair_ret.dropna(how="all").index.max()
    holdout_start = (end_dt - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    holdout_end = end_dt.strftime("%Y-%m-%d")
    full_start = pair_ret.dropna(how="all").index.min().strftime("%Y-%m-%d")
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
    win_df.to_csv(REPORTS / "scholarly_fx_macro_diff_window_stats.csv", index=False)

    need = {"year_2024", "year_2025", "year_2026", "holdout_365d"}
    promote = False
    for pname in factors:
        sub = win_df[(win_df.strategy == pname) & (win_df.window.isin(need))]
        if len(sub) == 4 and bool(sub["clears_1pct_bar"].all()):
            promote = True

    meta = {
        "ok": True,
        "data_tag": data_tag,
        "signal_lag_months": SIGNAL_LAG,
        "pub_lags": DEFAULT_PUB_LAGS,
        "n_long": cfg.n_long,
        "n_short": cfg.n_short,
        "factors": list(factors.keys()),
        "cpi_cols": list(cpi_d.columns),
        "ip_cols": list(ip_d.columns),
        "ur_cols": list(ur_d.columns),
        "promote": promote,
        "refs": [
            "Dahlquist & Hasseltoft — macro differentials and currency risk premia",
            "PPP / inflation differential: high relative inflation → short foreign",
            "Activity (IP): high relative growth → long foreign",
            "Labour slack (UR): high relative unemployment → short foreign",
        ],
    }
    (REPORTS / "scholarly_fx_macro_diff_meta.json").write_text(json.dumps(meta, indent=2))

    lines: list[str] = []
    lines.append("# Scholarly FX: macro differentials (Dahlquist-style FRED)\n\n")
    lines.append(
        "**Prior (fixed):** lagged CPI/IP/UR differentials vs USD sort G10 FX. "
        f"pub_lags={DEFAULT_PUB_LAGS} + signal_lag={SIGNAL_LAG}m; n_long={cfg.n_long}, "
        f"n_short={cfg.n_short}.\n\n"
    )
    lines.append(f"**Data:** `{data_tag}` D1 FX + free FRED macro panels under `data/macro/`.\n\n")
    lines.append("## Factor board (full sample)\n\n")
    lines.append("| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |\n")
    lines.append("|--------|--------:|------:|-----:|-----:|-----:|-------:|\n")
    for s in summaries:
        lines.append(
            f"| {s['factor']} | {s['mean_mo']*100:.3f}% | {s['tstat_ols']:.2f} | "
            f"{s['tstat_nw']:.2f} | {s['pct_pos_mo']*100:.1f}% | "
            f"{(s['top3']*100 if np.isfinite(s['top3']) else float('nan')):.0f}% | "
            f"{s['ann_sharpe_daily']:.2f} |\n"
        )
    lines.append("\n## Calendar / holdout windows\n\n")
    lines.append("| Strategy | Window | mean_mo | %pos | top3 | t NW | Sharpe | gates | 1% bar | role |\n")
    lines.append("|----------|--------|--------:|-----:|-----:|-----:|-------:|:-----:|:------:|------|\n")
    for _, row in win_df.iterrows():
        t3 = row["top3"] * 100 if np.isfinite(row["top3"]) else float("nan")
        lines.append(
            f"| {row['strategy']} | {row['window']} | {row['mean_mo']*100:.3f}% | "
            f"{row['pct_pos']*100:.1f}% | {t3:.0f}% | {row['tstat_nw']:.2f} | "
            f"{row['ann_sharpe']:.2f} | {'PASS' if row['gates_pass'] else 'FAIL'} | "
            f"{'YES' if row['clears_1pct_bar'] else 'no'} | {row['role']} |\n"
        )
    lines.append("\n## Promote / 1%/mo bar\n\n")
    lines.append(
        f"**Joint promote:** **{'YES' if promote else 'NO'}**. "
        "No overlay on locked sleeve; Yahoo/FRED ≠ FTMO MT5; no go-live.\n"
    )
    (REPORTS / "scholarly_fx_macro_diff_wave.md").write_text("".join(lines))
    print(f"wrote {REPORTS / 'scholarly_fx_macro_diff_wave.md'}")
    print(f"promote={promote}")
    for s in summaries:
        print(
            f"  {s['factor']}: mean_mo={s['mean_mo']*100:.3f}% "
            f"pos={s['pct_pos_mo']*100:.1f}% sharpe={s['ann_sharpe_daily']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
