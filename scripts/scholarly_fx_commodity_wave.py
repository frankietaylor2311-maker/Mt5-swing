#!/usr/bin/env python3
"""Scholarly FX commodity-currency wave (Chen–Rogoff–Rossi spirit).

Fixed priors: lagged commodity momentum → AUD/CAD/NZD FX.
No holdout tuning. signal_lag≥1 + commodity pub_lag_days=1.
Data: approximate_non_ftmo (Yahoo) unless FTMO CSVs appear under data/ftmo/.
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
from mt5_swing.data.commodity_prices import (
    COUNTRY_COMMODITY_MAP,
    download_commodity_panel,
    load_commodity_panel,
)
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.strategies.commodity_fx import (
    CommodityFxConfig,
    commodity_country_ts_returns,
    commodity_local_projection_panel,
    commodity_xs_basket_returns,
)

HISTORY = ROOT / "data" / "history"
FTMO = ROOT / "data" / "ftmo"
REPORTS = ROOT / "reports"
USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
INITIAL = 100_000.0
SIGNAL_LAG = 1
PUB_LAG = 1


def prefer_ftmo_or_yahoo(symbols: list[str]) -> tuple[pd.DataFrame, str]:
    """Prefer FTMO CSVs under data/ftmo/ if present; else Yahoo history."""
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
            # try glob under ftmo
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
    print("=== scholarly commodity-currency wave (Chen–Rogoff–Rossi) ===")

    download_ok = True
    download_err = None
    try:
        cpath = download_commodity_panel(force=False)
        print(f"commodity panel: {cpath}")
    except Exception as exc:  # noqa: BLE001
        download_ok = False
        download_err = str(exc)
        print(f"WARN commodity download: {exc}")

    try:
        panel = load_commodity_panel(download=False, pub_lag_days=PUB_LAG)
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: cannot load commodity panel: {exc}")
        meta = {
            "ok": False,
            "error": str(exc),
            "download_ok": download_ok,
            "download_err": download_err,
            "promote": False,
        }
        (REPORTS / "scholarly_fx_commodity_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / "scholarly_fx_commodity_wave.md").write_text(
            "# Scholarly FX: commodity currencies\n\n"
            f"**FAILED:** commodity panel unavailable ({exc}). "
            "No fabricated metrics. Retry download or cache Yahoo CL=F/HG=F/GC=F.\n"
        )
        return 1

    pair_ret, data_tag = prefer_ftmo_or_yahoo(USD_MAJORS)
    print(f"FX panel: {list(pair_ret.columns)} tag={data_tag}")
    print(f"commodity cols={list(panel.columns)} basket={panel.attrs.get('basket_source')}")

    cfg = CommodityFxConfig(signal_lag=SIGNAL_LAG)
    port_ts = commodity_country_ts_returns(panel, pair_ret, cfg=cfg)
    port_xs = commodity_xs_basket_returns(panel, pair_ret, cfg=cfg)
    lp = commodity_local_projection_panel(panel, pair_ret, cfg=cfg)
    lp.to_csv(REPORTS / "scholarly_fx_commodity_lp.csv", index=False)

    summaries = [
        factor_summary(port_ts, "commodity_country_ts"),
        factor_summary(port_xs, "commodity_xs_basket"),
    ]
    sum_df = pd.DataFrame(summaries)
    sum_df.to_csv(REPORTS / "scholarly_fx_commodity_factor_summary.csv", index=False)
    monthly_returns(port_ts).to_csv(
        REPORTS / "scholarly_fx_commodity_country_ts_monthly.csv", header=["commodity_country_ts"]
    )
    monthly_returns(port_xs).to_csv(
        REPORTS / "scholarly_fx_commodity_xs_basket_monthly.csv", header=["commodity_xs_basket"]
    )

    # Full sample + calendar windows + holdout last 365d from latest data
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
    for port, pname in ((port_ts, "commodity_country_ts"), (port_xs, "commodity_xs_basket")):
        for a, b, label, role in windows:
            st = window_stats(port, a, b)
            st["strategy"] = pname
            st["window"] = label
            st["role"] = role
            st["gates_pass"] = ftmo_gates(port.loc[a:b])
            st["clears_1pct_bar"] = clears_consistency(st) and st["gates_pass"]
            win_rows.append(st)
    win_df = pd.DataFrame(win_rows)
    win_df.to_csv(REPORTS / "scholarly_fx_commodity_window_stats.csv", index=False)

    # Promote only if both strategies jointly clear recent windows (strict — unlikely)
    need = {"year_2024", "year_2025", "year_2026", "holdout_365d"}
    promote = False
    for pname in ("commodity_country_ts", "commodity_xs_basket"):
        sub = win_df[(win_df.strategy == pname) & (win_df.window.isin(need))]
        if len(sub) == 4 and bool(sub["clears_1pct_bar"].all()):
            promote = True

    # LP sign check (prior: beta > 0)
    pooled = lp[(lp.scope == "pooled") & (lp.regressor == "commodity_mom")] if len(lp) else lp
    hyp_support = {}
    for _, row in pooled.iterrows():
        hyp_support[f"h{int(row['horizon'])}"] = {
            "beta": float(row["beta"]),
            "tstat": float(row["tstat"]),
            "sign_ok": bool(row["beta"] > 0),
            "n": int(row["n"]),
        }

    meta = {
        "ok": True,
        "data_tag": data_tag,
        "pub_lag_days": PUB_LAG,
        "signal_lag": SIGNAL_LAG,
        "formation_days": cfg.formation_days,
        "skip_days": cfg.skip_days,
        "cost_bps_per_side": cfg.cost_bps_per_side,
        "country_map": COUNTRY_COMMODITY_MAP,
        "basket_source": panel.attrs.get("basket_source"),
        "commodity_columns": list(panel.columns),
        "download_ok": download_ok,
        "download_err": download_err,
        "hyp_support_lp": hyp_support,
        "promote": promote,
        "refs": [
            "Chen, Rogoff & Rossi (2010), Can Exchange Rates Forecast Commodity Prices?, QJE",
            "Cashin, Céspedes & Sahay — commodity currencies",
            "Amano & van Norden — oil and the Canadian dollar",
        ],
    }
    (REPORTS / "scholarly_fx_commodity_meta.json").write_text(json.dumps(meta, indent=2))

    # Markdown board
    lines: list[str] = []
    lines.append("# Scholarly FX: commodity currencies (Chen–Rogoff–Rossi)\n\n")
    lines.append(
        "**Prior (fixed):** lagged commodity price momentum co-moves with / forecasts "
        "commodity-currency FX (AUD, CAD, NZD). "
        f"pub_lag={PUB_LAG}d + signal_lag={SIGNAL_LAG}d; formation={cfg.formation_days}d, "
        f"skip={cfg.skip_days}d; costs={cfg.cost_bps_per_side} bps/side.\n\n"
    )
    lines.append(
        f"**Data:** `{data_tag}` Yahoo D1 FX + Yahoo commodity futures/ETF "
        f"(`CL=F`, `HG=F`, `GC=F`, basket=`{panel.attrs.get('basket_source')}`). "
        "Futures ≠ spot export baskets (CRR limitation).\n\n"
    )
    lines.append("**Map:** AUD→copper, CAD→oil, NZD→basket (no free dairy proxy).\n\n")
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
    lines.append("\n## Local projections (diagnostic)\n\n")
    lines.append("Prior: β > 0 (commodity ↑ → commodity FX appreciates vs USD).\n\n")
    if len(lp):
        lines.append("| scope | ccy | regressor | h | β | t | n | sign_ok |\n")
        lines.append("|-------|-----|-----------|--:|--:|--:|--:|:-------:|\n")
        for _, row in lp.iterrows():
            b = row.get("beta", float("nan"))
            sign_ok = bool(np.isfinite(b) and b > 0)
            lines.append(
                f"| {row.get('scope')} | {row.get('currency')} | {row.get('regressor')} | "
                f"{int(row['horizon'])} | {b:.5f} | {row.get('tstat', float('nan')):.2f} | "
                f"{int(row.get('n', 0))} | {'yes' if sign_ok else 'no'} |\n"
            )
    else:
        lines.append("_LP empty (insufficient overlap)._\n")

    lines.append("\n## Promote / 1%/mo bar\n\n")
    lines.append(
        f"**Joint promote:** **{'YES' if promote else 'NO'}**. "
        "Consistency clear requires mean_mo≥1%, %pos≥70%, top3≤55%, and FTMO gates — "
        "not earned unless tables above show YES.\n\n"
    )
    lines.append(
        "### What this does NOT support\n\n"
        "- No claim that commodity-FX alone delivers FTMO ~1%/mo consistency.\n"
        "- No go-live; Yahoo futures ≠ FTMO MT5; no overlay on locked sleeve.\n"
        "- CRR often study commodity *prices* forecasted by FX — we test the "
        "tradable commodity→FX direction with lags; results are sample-specific.\n"
    )
    lines.append(
        "\nArtifacts: `scholarly_fx_commodity_*.csv`, `scholarly_fx_commodity_meta.json`.\n"
    )
    (REPORTS / "scholarly_fx_commodity_wave.md").write_text("".join(lines))
    print(f"wrote {REPORTS / 'scholarly_fx_commodity_wave.md'}")
    print(f"promote={promote}")
    for s in summaries:
        print(
            f"  {s['factor']}: mean_mo={s['mean_mo']*100:.3f}% "
            f"pos={s['pct_pos_mo']*100:.1f}% sharpe={s['ann_sharpe_daily']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
