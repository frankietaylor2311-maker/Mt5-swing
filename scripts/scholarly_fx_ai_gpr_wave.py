#!/usr/bin/env python3
"""Scholarly FX bilateral AI-GPR role decompositions wave (Caldara–Iacoviello).

Threats / acts / oil-region roles → USD haven + commodity-FX pressure.
Distinct from aggregate GPR regime (§8), country-GPRC sorts (§9), news events
(§10), and CRR/ToT commodity FX.

Fixed priors: pub_lag_days=1 + signal_lag=1; trailing z 252d; binary z≥1 USD
tilt; costs 1.5 bps/side. No HO tuning. Optional IS risk sweep if any positive
IS mean. Locked sleeve untouched — no cooler overlays on fx4plus.
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
from mt5_swing.data.ai_gpr import (
    load_ai_gpr_panel,
    oil_me_vs_non_series,
)
from mt5_swing.data.fred_rates import load_currency_rates
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.strategies.ai_gpr_fx import AiGprFxConfig, ai_gpr_factor_returns

HISTORY = ROOT / "data" / "history"
FTMO = ROOT / "data" / "ftmo"
REPORTS = ROOT / "reports"
USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
INITIAL = 100_000.0
PRIMARY = "ai_threats_usd"
PUB_LAG = 1


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
    print("=== scholarly FX bilateral AI-GPR role decompositions wave ===")
    print(
        "Caldara–Iacoviello AI-GPR threats/acts / oil-region roles — "
        "distinct from aggregate GPR, GPRC_* sorts, news events, CRR/ToT. "
        "Locked sleeve untouched."
    )

    try:
        panel = load_ai_gpr_panel(pub_lag_days=PUB_LAG)
        me_vs = oil_me_vs_non_series(panel, pub_lag_days=PUB_LAG)
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL AI-GPR load: {exc}")
        meta = {"ok": False, "error": str(exc), "promote": False}
        (REPORTS / "scholarly_fx_ai_gpr_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / "scholarly_fx_ai_gpr_wave.md").write_text(
            f"# Scholarly FX: AI-GPR roles\n\n**FAILED:** {exc}\n"
        )
        return 1

    threats = panel["THREATS_GPR_AI"].dropna()
    acts = panel["ACTS_GPR_AI"].dropna()
    gpr_ai = panel["GPR_AI"].dropna()
    oil_gpr = panel["GPR_OIL"].dropna() if "GPR_OIL" in panel.columns else None
    oil_threats = (
        panel["GPR_OIL_THREATS"].dropna() if "GPR_OIL_THREATS" in panel.columns else None
    )
    print(
        f"AI-GPR panel n={len(panel)} "
        f"[{panel.index.min().date()}..{panel.index.max().date()}] "
        f"pub_lag={PUB_LAG}d threats n={len(threats)} oil n={len(oil_gpr) if oil_gpr is not None else 0}"
    )
    panel.to_csv(REPORTS / "scholarly_fx_ai_gpr_panel.csv")
    me_vs.to_frame().to_csv(REPORTS / "scholarly_fx_ai_gpr_oil_me_vs_non.csv")

    pair_ret, _px, data_tag = prefer_ftmo_or_yahoo(USD_MAJORS)
    print(f"FX panel: {list(pair_ret.columns)} tag={data_tag}")

    try:
        rates = load_currency_rates(pub_lag_months=1, download=True)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN rates: {exc}; carry_ai_threats_cool skipped")
        rates = None

    cfg = AiGprFxConfig()
    factors = ai_gpr_factor_returns(
        pair_ret,
        threats=threats,
        acts=acts,
        gpr_ai=gpr_ai,
        oil_gpr=oil_gpr,
        oil_threats=oil_threats,
        oil_me_vs_non=me_vs,
        rates=rates,
        cfg=cfg,
    )
    if not factors:
        print("FATAL: no AI-GPR factors produced")
        return 1

    board = {k: v for k, v in factors.items() if k != "carry_raw"}

    summaries = [factor_summary(r, name) for name, r in board.items()]
    pd.DataFrame(summaries).to_csv(
        REPORTS / "scholarly_fx_ai_gpr_factor_summary.csv", index=False
    )
    for name, r in board.items():
        monthly_returns(r).to_csv(
            REPORTS / f"scholarly_fx_ai_gpr_{name}_monthly.csv", header=[name]
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
    win_df.to_csv(REPORTS / "scholarly_fx_ai_gpr_window_stats.csv", index=False)

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
        sweep_df.to_csv(REPORTS / "scholarly_fx_ai_gpr_risk_sweep.csv", index=False)

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

    meta = {
        "ok": True,
        "path": "ai_gpr_bilateral_role_decompositions",
        "data_tag": data_tag,
        "pub_lag_days": PUB_LAG,
        "signal_lag": cfg.signal_lag,
        "z_window": cfg.z_window,
        "z_high": cfg.z_high,
        "cool": cfg.cool,
        "usd_tilt": cfg.usd_tilt,
        "binary_usd": cfg.binary_usd,
        "cost_bps_side": cfg.cost_bps_side,
        "factors": list(board.keys()),
        "primary": PRIMARY,
        "n_factors": len(board),
        "soft_nw_pos": soft,
        "hard_nw_pos": hard,
        "promote_unscaled": bool(promote),
        "promote_scaled_primary": bool(scaled_promote),
        "promote": bool(promote or scaled_promote),
        "holdout": {"start": holdout_start, "end": holdout_end},
        "locked_sleeve": "fx4plus_gbpcad_d1_voltarget_0025",
        "locked_sleeve_untouched": True,
        "distinct_from": [
            "aggregate_GPR_regime",
            "country_GPRC_sorts",
            "news_event_intensity",
            "commodity_CRR",
            "commodity_ToT",
        ],
        "citations": [
            "Caldara & Iacoviello (2022), Measuring Geopolitical Risk, AER",
            "Caldara–Iacoviello AI-GPR / threats-acts / oil-region role indices "
            "(matteoiacoviello.com/gpr.htm)",
        ],
        "series": {
            "THREATS_GPR_AI": {
                "n": len(threats),
                "end": str(threats.index.max().date()) if len(threats) else None,
            },
            "GPR_OIL_end": str(oil_gpr.index.max().date()) if oil_gpr is not None and len(oil_gpr) else None,
            "panel_end": str(panel.index.max().date()),
        },
    }
    (REPORTS / "scholarly_fx_ai_gpr_meta.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )

    lines = [
        "# Scholarly FX: bilateral AI-GPR role decompositions wave",
        "",
        "**Path:** Caldara–Iacoviello AI-GPR threats/acts / oil-region roles — "
        "**distinct** from aggregate GPR (§8), country-GPRC sorts (§9), news "
        "events (§10), and CRR/ToT commodity FX.",
        f"**Data:** `{data_tag}` + free `ai_gpr_daily.csv`. "
        f"**PIT:** pub_lag={PUB_LAG}d + signal_lag={cfg.signal_lag}d; "
        f"trailing z={cfg.z_window}d; binary z≥{cfg.z_high}; costs "
        f"{cfg.cost_bps_side} bps/side.",
        f"**Primary:** `{PRIMARY}` (high threats z → long USD / short risk FX). "
        "Locked sleeve untouched (no cooler overlay).",
        "",
        f"**Board:** n={len(board)} soft_nw_pos={soft} hard_nw_pos={hard} "
        f"promote={int(promote or scaled_promote)}",
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
        f"**Unscaled promote:** {'YES' if promote else 'NO'}. "
        f"**Scaled primary promote:** {'YES' if scaled_promote else 'NO'}.",
        "",
        "Artifacts: `reports/scholarly_fx_ai_gpr_*.csv`, "
        "`scholarly_fx_ai_gpr_meta.json`.",
        "",
    ]
    (REPORTS / "scholarly_fx_ai_gpr_wave.md").write_text("\n".join(lines))

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
