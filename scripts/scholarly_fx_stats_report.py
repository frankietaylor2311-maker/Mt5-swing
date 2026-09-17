#!/usr/bin/env python3
"""Scholarly FX factor stats: carry, momentum, dollar, GPR/VIX regime.

Walk-forward / calendar-year windows only. Fixed literature priors — no holdout
tuning. signal_lag enforced. Data tag: approximate_non_ftmo (+ FRED / GPR / VIX).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mt5_swing.data.fred_rates import download_default_rate_panel, load_currency_rates
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.data.macro_uncertainty import download_vix, load_gpr, load_vix
from mt5_swing.strategies.carry_rank import (
    CarryRankConfig,
    carry_weights_from_rates,
    currency_weights_to_pair_weights,
    expand_weights_to_daily,
    portfolio_returns_from_weights,
)
from mt5_swing.strategies.fx_momentum import (
    FxMomentumBasket,
    FxMomentumConfig,
    dollar_factor_returns,
)
from mt5_swing.strategies.gpr_regime import GprRegimeConfig, apply_regime_to_returns

HISTORY = ROOT / "data" / "history"
REPORTS = ROOT / "reports"
MACRO = ROOT / "data" / "macro"

USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
SIGNAL_LAG = 1
BOOT_N = 2000
BOOT_SEED = 42


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
    # Compound daily → month-end
    eq = (1.0 + r.fillna(0.0)).cumprod()
    m = eq.resample("ME").last().pct_change().dropna()
    return m


def tstat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    mu = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(mu / se) if se > 0 else float("nan")


def bootstrap_mean_ci(x: np.ndarray, n: int = BOOT_N, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
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
            "ann_sharpe": float("nan"),
            "total_ret": float("nan"),
        }
    mean_mo = float(m.mean())
    pct_pos = float((m > 0).mean())
    # ann sharpe from daily
    d = sl.dropna()
    sharpe = float(d.mean() / d.std(ddof=1) * np.sqrt(252)) if len(d) > 5 and d.std(ddof=1) > 0 else float("nan")
    total = float((1.0 + sl.fillna(0)).prod() - 1.0)
    return {
        "start": start,
        "end": end,
        "n_days": int(len(sl)),
        "n_months": int(len(m)),
        "mean_mo": mean_mo,
        "pct_pos": pct_pos,
        "ann_sharpe": sharpe,
        "total_ret": total,
    }


def factor_summary(name: str, r: pd.Series) -> dict:
    m = monthly_returns(r)
    arr = m.to_numpy(dtype=float)
    mu, lo, hi = bootstrap_mean_ci(arr)
    return {
        "factor": name,
        "n_months": int(len(m)),
        "mean_mo": float(m.mean()) if len(m) else float("nan"),
        "std_mo": float(m.std(ddof=1)) if len(m) > 1 else float("nan"),
        "tstat_mo": tstat(arr),
        "boot_mean": mu,
        "boot_ci95_lo": lo,
        "boot_ci95_hi": hi,
        "pct_pos_mo": float((m > 0).mean()) if len(m) else float("nan"),
        "ann_sharpe_daily": (
            float(r.dropna().mean() / r.dropna().std(ddof=1) * np.sqrt(252))
            if r.dropna().std(ddof=1) > 0
            else float("nan")
        ),
    }


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    MACRO.mkdir(parents=True, exist_ok=True)

    print("=== data refresh ===")
    rate_paths = download_default_rate_panel()
    vix_path = download_vix()
    try:
        gpr = load_gpr(freq="M", pub_lag_months=1)
        gpr_status = f"OK monthly n={len(gpr)} lag=1m [{gpr.index.min().date()} → {gpr.index.max().date()}]"
    except Exception as exc:  # noqa: BLE001
        gpr = None
        gpr_status = f"STUB/FAIL: {exc}"

    rates = load_currency_rates(pub_lag_months=1)
    vix = load_vix(pub_lag_days=1)
    pair_ret = load_pair_panel()

    print(f"FRED series cached: {len(rate_paths)}")
    print(f"Rates panel: {rates.shape} pub_lag_months={rates.attrs.get('pub_lag_months')}")
    print(f"VIX: n={len(vix)} path={vix_path}")
    print(f"GPR: {gpr_status}")
    print(f"Pair returns D1: {pair_ret.shape} {pair_ret.index.min().date()} → {pair_ret.index.max().date()}")

    # --- Carry ---
    cfg_c = CarryRankConfig(n_long=2, n_short=2, signal_lag=SIGNAL_LAG)
    ccy_w = carry_weights_from_rates(rates, cfg=cfg_c)
    pair_w = currency_weights_to_pair_weights(ccy_w)
    daily_w = expand_weights_to_daily(pair_w, pair_ret.index, signal_lag=SIGNAL_LAG)
    # Align columns
    common = [c for c in daily_w.columns if c in pair_ret.columns]
    carry_r = portfolio_returns_from_weights(daily_w[common], pair_ret[common])
    carry_r.name = "carry_rank"

    # --- Momentum ---
    mom = FxMomentumBasket(
        formation_days=63,
        skip_days=21,
        n_long=2,
        n_short=2,
        signal_lag=SIGNAL_LAG,
    )
    mom_r = mom.portfolio_returns(pair_ret[common] if set(common) <= set(pair_ret.columns) else pair_ret)
    mom_r.name = "fx_momentum"

    # --- Dollar factor ---
    dol_r = dollar_factor_returns(pair_ret)
    # lag signal interpretation: dollar is a contemporaneous factor; for "tradable"
    # version use sign of trailing month mean with lag
    dol_sig = np.sign(dol_r.rolling(21, min_periods=10).mean().shift(SIGNAL_LAG))
    dol_tradable = (dol_sig * dol_r).rename("dollar_tsmom")

    # --- Regime overlays (fixed priors) ---
    cfg_g = GprRegimeConfig(signal_lag=SIGNAL_LAG, cool=0.35, z_high=1.0, usd_tilt=0.0)
    carry_reg = apply_regime_to_returns(carry_r, gpr, vix, cfg=cfg_g)
    mom_reg = apply_regime_to_returns(mom_r, gpr, vix, cfg=cfg_g)

    factors = {
        "carry_rank": carry_r,
        "fx_momentum": mom_r,
        "dollar_factor_avg": dol_r,
        "dollar_tsmom": dol_tradable,
        "carry_gpr_vix": carry_reg,
        "mom_gpr_vix": mom_reg,
    }

    summaries = [factor_summary(k, v) for k, v in factors.items()]
    sum_df = pd.DataFrame(summaries)
    sum_df.to_csv(REPORTS / "scholarly_fx_factor_summary.csv", index=False)

    windows = [
        ("2018-01-01", "2023-12-31", "IS_2018_2023"),
        ("2024-01-01", "2024-12-31", "year_2024"),
        ("2025-01-01", "2025-12-31", "year_2025"),
        ("2026-01-01", "2026-12-31", "year_2026"),
        ("2025-09-17", "2026-09-16", "holdout_365d_confirm_only"),
    ]
    rows = []
    for fname, series in factors.items():
        for a, b, label in windows:
            st = window_stats(series, a, b)
            st["factor"] = fname
            st["window"] = label
            st["role"] = "confirm_only" if "holdout" in label else "eval"
            rows.append(st)
    win_df = pd.DataFrame(rows)
    win_df.to_csv(REPORTS / "scholarly_fx_window_stats.csv", index=False)

    # Monthly distributions
    mo_frames = []
    for fname, series in factors.items():
        m = monthly_returns(series)
        mo_frames.append(m.rename(fname))
    mo_df = pd.concat(mo_frames, axis=1)
    mo_df.to_csv(REPORTS / "scholarly_fx_monthly_returns.csv")

    # Gates proxy: crude equity from 100k, static 10% / daily 5% check on daily curve
    def gates_pass(r: pd.Series) -> bool:
        if r.dropna().empty:
            return False
        eq = (1.0 + r.fillna(0)).cumprod() * 100_000.0
        static = (100_000.0 - float(eq.min())) / 100_000.0
        # daily loss from day-open vs initial (FTMO-ish)
        day = eq.resample("1D").last().dropna()
        if len(day) < 2:
            return static < 0.10
        day_open = day.shift(1)
        daily_loss = ((day_open - day) / 100_000.0).max()
        return bool(static < 0.10 and (pd.isna(daily_loss) or daily_loss < 0.05))

    gate_rows = []
    for fname, series in factors.items():
        for a, b, label in windows:
            sl = series.loc[a:b]
            gate_rows.append(
                {
                    "factor": fname,
                    "window": label,
                    "gates_pass": gates_pass(sl),
                    "mean_mo": window_stats(series, a, b)["mean_mo"],
                }
            )
    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(REPORTS / "scholarly_fx_gates.csv", index=False)

    # Markdown report
    lines = []
    lines.append("# Scholarly FX factor stats (v1)\n")
    lines.append("**data_source:** `approximate_non_ftmo` (Yahoo D1) + FRED rates + VIX + Caldara–Iacoviello GPR\n")
    lines.append(f"**signal_lag:** {SIGNAL_LAG} | **pub_lag rates:** 1 month | **VIX lag:** 1 day | **GPR lag:** 1 month\n")
    lines.append("**No holdout tuning** — fixed literature priors (n_long=n_short=2, mom 63d/skip21, regime z_high=1 cool=0.35).\n")
    lines.append("\n## Data landed\n")
    lines.append(f"- FRED immediate-rate CSVs: {len(rate_paths)} series under `data/macro/fred_*.csv`\n")
    lines.append(f"- Rates panel shape `{rates.shape}` currencies {list(rates.columns)}\n")
    lines.append(f"- VIX: `{vix_path.name}` n={len(vix)}\n")
    lines.append(f"- GPR: {gpr_status}\n")
    lines.append(f"- FX D1 pairs: {list(pair_ret.columns)} bars≈{len(pair_ret)}\n")
    lines.append("\n## Full-sample monthly factor moments\n")
    lines.append("| Factor | n_mo | mean_mo | t-stat | boot 95% CI | %pos | ann Sharpe (d) |\n")
    lines.append("|--------|-----:|--------:|-------:|------------|-----:|---------------:|\n")
    for _, row in sum_df.iterrows():
        ci = f"[{row['boot_ci95_lo']:.4f}, {row['boot_ci95_hi']:.4f}]" if np.isfinite(row["boot_ci95_lo"]) else "n/a"
        lines.append(
            f"| {row['factor']} | {row['n_months']} | {row['mean_mo']*100:.3f}% | {row['tstat_mo']:.2f} | {ci} | "
            f"{row['pct_pos_mo']*100:.1f}% | {row['ann_sharpe_daily']:.2f} |\n"
        )
    lines.append("\n## Calendar / walk-forward windows (mean monthly)\n")
    lines.append("| Factor | Window | mean_mo | %pos | Sharpe | gates | role |\n")
    lines.append("|--------|--------|--------:|-----:|-------:|:-----:|------|\n")
    for _, row in win_df.iterrows():
        g = gate_df[(gate_df.factor == row["factor"]) & (gate_df.window == row["window"])]["gates_pass"]
        gp = bool(g.iloc[0]) if len(g) else False
        mm = row["mean_mo"] * 100 if np.isfinite(row["mean_mo"]) else float("nan")
        pp = row["pct_pos"] * 100 if np.isfinite(row["pct_pos"]) else float("nan")
        lines.append(
            f"| {row['factor']} | {row['window']} | {mm:.3f}% | {pp:.1f}% | {row['ann_sharpe']:.2f} | "
            f"{'PASS' if gp else 'FAIL'} | {row['role']} |\n"
        )
    lines.append("\n## Honest read vs 1%/mo FTMO goal\n")
    lines.append(
        "- Literature carry/momentum/dollar premia are **real but small** after costs; "
        "full-sample mean_mo above is **not** a claim of stable ≥1%/month.\n"
        "- Year windows will often miss 1% mean and/or 70% positive months — that is expected.\n"
        "- Overlay hunt on technical sleeves is **deprecated**; this line is the active research path.\n"
        "- Still **approximate_non_ftmo** — never golive without FTMO CSVs.\n"
    )
    lines.append("\nArtifacts: `scholarly_fx_factor_summary.csv`, `scholarly_fx_window_stats.csv`, "
                 "`scholarly_fx_monthly_returns.csv`, `scholarly_fx_gates.csv`.\n")

    md_path = REPORTS / "scholarly_fx_stats_v1.md"
    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(sum_df.to_string(index=False))
    meta = {
        "data_source": "approximate_non_ftmo",
        "signal_lag": SIGNAL_LAG,
        "gpr_status": gpr_status,
        "n_fred": len(rate_paths),
        "pairs": list(pair_ret.columns),
    }
    (REPORTS / "scholarly_fx_stats_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
