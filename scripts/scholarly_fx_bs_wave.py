#!/usr/bin/env python3
"""Scholarly FX Balassa–Samuelson / productivity wave (free FRED IP + CPI).

Fixed priors: productivity-adjusted real-FX misalignment + relative IP sorts.
CPI pub_lag=1m, IP pub_lag=2m + signal_lag=1m; lookbacks 60m/120m. No HO tuning.
Prop sizing: optional IS risk sweep if any positive IS mean; OOS confirm.
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
from mt5_swing.data.fred_macro_diff import (
    DEFAULT_PUB_LAGS,
    load_cpi_level_panel,
    load_ip_level_panel,
    relative_productivity_vs_usd,
)
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.strategies.balassa_samuelson_fx import (
    BalassaSamuelsonFxConfig,
    balassa_samuelson_factor_returns,
    bs_gap_scores,
    monthly_nominal_fx,
)
from mt5_swing.strategies.ppp_real_fx import real_fx_panel

HISTORY = ROOT / "data" / "history"
FTMO = ROOT / "data" / "ftmo"
REPORTS = ROOT / "reports"
USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
INITIAL = 100_000.0
SIGNAL_LAG = 1
PUB_LAG_CPI = DEFAULT_PUB_LAGS["cpi"]
PUB_LAG_IP = DEFAULT_PUB_LAGS["ip"]
PRIMARY = "bs_gap_60m"


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
    print("=== scholarly Balassa–Samuelson / productivity FX wave ===")

    try:
        cpi = load_cpi_level_panel(download=True, pub_lag_months=PUB_LAG_CPI)
        ip = load_ip_level_panel(download=True, pub_lag_months=PUB_LAG_IP)
        rel_prod = relative_productivity_vs_usd(ip)
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL macro load: {exc}")
        meta = {"ok": False, "error": str(exc), "promote": False}
        (REPORTS / "scholarly_fx_bs_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / "scholarly_fx_bs_wave.md").write_text(
            f"# Scholarly FX: Balassa–Samuelson\n\n**FAILED:** {exc}\n"
        )
        return 1

    pair_ret, pair_close, data_tag = prefer_ftmo_or_yahoo(USD_MAJORS)
    print(f"FX panel: {list(pair_ret.columns)} tag={data_tag}")
    print(
        f"cpi={list(cpi.columns)} ip={list(ip.columns)} "
        f"rel_prod={list(rel_prod.columns)} "
        f"missing_ip={ip.attrs.get('missing')}"
    )

    cfg = BalassaSamuelsonFxConfig(signal_lag=SIGNAL_LAG)
    factors = balassa_samuelson_factor_returns(
        pair_close, pair_ret, cpi, rel_prod, cfg=cfg
    )
    if not factors:
        print("FATAL: no BS factors produced")
        return 1

    fx_m = monthly_nominal_fx(pair_close)
    q = real_fx_panel(fx_m, cpi)
    q.to_csv(REPORTS / "scholarly_fx_bs_real_fx_monthly.csv")
    rel_prod.to_csv(REPORTS / "scholarly_fx_bs_rel_prod_monthly.csv")
    gap60 = bs_gap_scores(q, rel_prod, cfg=cfg, lookback=60)
    if not gap60.empty:
        gap60.to_csv(REPORTS / "scholarly_fx_bs_gap60_scores.csv")

    # Coverage table
    cov_rows = []
    for ccy in sorted(set(list(cpi.columns) + list(ip.columns) + ["AUD", "NZD", "CHF"])):
        cov_rows.append(
            {
                "currency": ccy,
                "has_cpi": ccy in cpi.columns,
                "has_ip": ccy in ip.columns,
                "in_bs_sort": ccy in rel_prod.columns and ccy != "USD",
                "cpi_note": (cpi.attrs.get("notes") or {}).get(ccy, ""),
                "ip_note": (ip.attrs.get("notes") or {}).get(ccy, ""),
            }
        )
    pd.DataFrame(cov_rows).to_csv(REPORTS / "scholarly_fx_bs_coverage.csv", index=False)

    summaries = [factor_summary(r, name) for name, r in factors.items()]
    pd.DataFrame(summaries).to_csv(REPORTS / "scholarly_fx_bs_factor_summary.csv", index=False)
    for name, r in factors.items():
        monthly_returns(r).to_csv(REPORTS / f"scholarly_fx_bs_{name}_monthly.csv", header=[name])

    end_dt = pair_ret.dropna(how="all").index.max()
    holdout_start = (end_dt - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    holdout_end = end_dt.strftime("%Y-%m-%d")
    full_start = pair_ret.dropna(how="all").index.min().strftime("%Y-%m-%d")
    is_end = (pd.Timestamp(holdout_start, tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

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
    win_df.to_csv(REPORTS / "scholarly_fx_bs_window_stats.csv", index=False)

    # Risk sweep only if any factor has positive full-sample IS mean
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
        sweep_df.to_csv(REPORTS / "scholarly_fx_bs_risk_sweep.csv", index=False)

    need = {"year_2024", "year_2025", "year_2026", "holdout_365d"}
    promote = False
    for pname in factors:
        sub = win_df[(win_df.strategy == pname) & (win_df.window.isin(need))]
        if len(sub) == 4 and bool(sub["clears_1pct_bar"].all()):
            promote = True

    scaled_promote = False
    prim_sweep = sweep_df[sweep_df.strategy == PRIMARY] if len(sweep_df) else pd.DataFrame()
    if len(prim_sweep):
        scaled_promote = bool(
            prim_sweep.iloc[0].get("scaled_clears_1pct_is", False)
            and prim_sweep.iloc[0].get("scaled_clears_1pct_oos", False)
        )

    meta = {
        "ok": True,
        "data_tag": data_tag,
        "signal_lag_months": SIGNAL_LAG,
        "pub_lag_cpi": PUB_LAG_CPI,
        "pub_lag_ip": PUB_LAG_IP,
        "lookbacks": list(cfg.lookbacks),
        "min_periods": cfg.min_periods,
        "n_long": cfg.n_long,
        "n_short": cfg.n_short,
        "factors": list(factors.keys()),
        "primary": PRIMARY,
        "cpi_cols": list(cpi.columns),
        "ip_cols": list(ip.columns),
        "rel_prod_cols": list(rel_prod.columns),
        "ip_missing": ip.attrs.get("missing", []),
        "any_pos_is": any_pos_is,
        "is_end": is_end,
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
        "promote": promote,
        "scaled_promote": scaled_promote,
        "risk_sweep_primary": prim_sweep.iloc[0].to_dict() if len(prim_sweep) else None,
        "refs": [
            "Balassa (1964); Samuelson (1964) — productivity & relative prices",
            "Chong, Jordà & Taylor (2012), The Harrod–Balassa–Samuelson Hypothesis, IER",
            "Ricci, Milesi-Ferretti & Lee — real exchange rates and fundamentals",
            "IP as free tradable-productivity proxy (AUD/NZD/CHF sparse on FRED)",
        ],
    }
    (REPORTS / "scholarly_fx_bs_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    lines: list[str] = []
    lines.append("# Scholarly FX: Balassa–Samuelson / productivity-adjusted real FX\n\n")
    lines.append(
        "**Prior (fixed):** HBS — relative productivity co-moves with real FX; "
        "residuals / gaps are productivity-adjusted value. "
        f"CPI pub_lag={PUB_LAG_CPI}m, IP pub_lag={PUB_LAG_IP}m + signal_lag={SIGNAL_LAG}m; "
        f"lookbacks={list(cfg.lookbacks)}m. No HO tuning.\n\n"
    )
    lines.append(
        f"**Data:** `{data_tag}` D1 FX + FRED CPI levels ({', '.join(cpi.columns)}) + "
        f"IP levels ({', '.join(ip.columns)}). "
        f"IP missing: {ip.attrs.get('missing')}.\n\n"
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
        "Yahoo/FRED ≠ FTMO MT5; no go-live.\n\n"
    )
    lines.append(
        "### Honest gaps\n\n"
        "- IP is a **manufacturing/output proxy**, not TFP or tradable labour productivity.\n"
        "- AUD/NZD/CHF lack clean FRED IP → thin cross-section (EUR/GBP/JPY/CAD).\n"
        "- EUR IP series sparse after ~2023 on FRED.\n"
        "- HBS is a **long-run** relative-price relation; monthly prop hit-rate is different.\n"
    )
    lines.append("\nArtifacts: `scholarly_fx_bs_*.csv`, `scholarly_fx_bs_meta.json`.\n")
    (REPORTS / "scholarly_fx_bs_wave.md").write_text("".join(lines))
    print(f"wrote {REPORTS / 'scholarly_fx_bs_wave.md'}")
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
