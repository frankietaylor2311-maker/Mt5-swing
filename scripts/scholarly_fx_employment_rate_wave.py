#!/usr/bin/env python3
"""Scholarly FX OECD/FRED employment-rate (LREM64TT) FX differential wave (§57).

Fixed priors: FRED OECD MEI LREM64TT employment rate ages 15–64 (%)
(USD/JPY/CAD/AUD *M156S; GBP/CHF/NZD *Q156S; EUR=DEQ156S Germany
proxy — EZQ156S stale ~2022-10), pub_lag_months=2 + signal_lag=1m +
signal_lag_days=1. Primary high_er_xs (labour-strength / engagement stock —
high relative ER → appreciate). Scores on **ER level %**. No HO
tuning. Locked sleeve untouched. Distinct from employment persons §41,
UR §54, LFP §56, wages §55, ULC/LP/CU.
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
from mt5_swing.data.fred_employment_rate import (
    DEFAULT_PUB_LAG_MONTHS,
    load_employment_rate_panel,
    load_us_employment_rate,
    employment_rate_coverage,
)
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.strategies.employment_rate_fx import (
    EmploymentRateFxConfig,
    employment_rate_factor_returns,
)

HISTORY = ROOT / "data" / "history"
FTMO = ROOT / "data" / "ftmo"
REPORTS = ROOT / "reports"
USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
INITIAL = 100_000.0
PRIMARY = "high_er_xs"
PUB_LAG_M = DEFAULT_PUB_LAG_MONTHS
SECTION = 57


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
    print(f"=== scholarly FX OECD/FRED employment-rate wave (§{SECTION}) ===")
    print(
        "FRED LREM64TT employment-rate levels (pub_lag=2) — high_er XS; labour-"
        "strength/engagement-stock prior; full G10; EUR=DE proxy (EZQ stale); "
        "distinct from employment persons §41, UR §54, LFP §56, wages §55. "
        "Locked sleeve untouched."
    )

    try:
        lp = load_employment_rate_panel(pub_lag_months=PUB_LAG_M, download=True, force=False)
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL employment_rate load: {exc}")
        meta = {"ok": False, "error": str(exc), "promote": False, "section": SECTION}
        (REPORTS / "scholarly_fx_employment_rate_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / "scholarly_fx_employment_rate_wave.md").write_text(
            f"# Scholarly FX: OECD/FRED employment-rate\n\n**FAILED:** {exc}\n"
        )
        return 1

    if lp.empty or lp.shape[1] < 4:
        msg = (
            f"employment_rate panel too thin: cols={list(lp.columns)}. "
            "Fallback candidates: FTMO CSV re-score or other free scholarly structure."
        )
        print(f"FATAL: {msg}")
        meta = {
            "ok": False,
            "error": msg,
            "promote": False,
            "section": SECTION,
            "fallback": "ftmo_csv_rescore_or_other",
        }
        (REPORTS / "scholarly_fx_employment_rate_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / "scholarly_fx_employment_rate_wave.md").write_text(
            f"# Scholarly FX: OECD/FRED employment-rate\n\n**FAILED data:** {msg}\n"
        )
        return 1

    cov = employment_rate_coverage(lp)
    cov.to_csv(REPORTS / "scholarly_fx_employment_rate_coverage.csv", index=False)
    lp.to_csv(REPORTS / "scholarly_fx_employment_rate_panel.csv")
    print(
        f"employment_rate panel cols={list(lp.columns)} "
        f"[{lp.dropna(how='all').index.min().date()}.."
        f"{lp.dropna(how='all').index.max().date()}] "
        f"pub_lag_m={PUB_LAG_M}"
    )
    print(cov.to_string(index=False))
    print(f"Series source note: {lp.attrs.get('source')}")
    print(f"Score basis: {lp.attrs.get('score_basis')} unit={lp.attrs.get('unit')}")
    print(f"Failed/alt documented: {lp.attrs.get('failed_or_alt')}")
    print(f"Stale: {lp.attrs.get('stale')} note={lp.attrs.get('stale_note')}")
    print(f"Unmapped: {lp.attrs.get('unmapped')}")

    us_er = None
    try:
        us_er = load_us_employment_rate(download=True, force=False)
        print(
            f"US LREM64TTUSM156S [{us_er.dropna().index.min().date()}.."
            f"{us_er.dropna().index.max().date()}] n={us_er.dropna().shape[0]} "
            f"series={us_er.attrs.get('series_id')}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARN US employment_rate load failed: {exc}")
        us_er = None

    pair_ret, _px, data_tag = prefer_ftmo_or_yahoo(USD_MAJORS)
    print(f"FX panel: {list(pair_ret.columns)} tag={data_tag}")

    cfg = EmploymentRateFxConfig()
    factors = employment_rate_factor_returns(
        pair_ret,
        lp,
        cfg=cfg,
        us_er_override=us_er,
    )
    if not factors:
        print("FATAL: no employment_rate factors produced")
        return 1

    board = dict(factors)
    summaries = [factor_summary(r, name) for name, r in board.items()]
    pd.DataFrame(summaries).to_csv(
        REPORTS / "scholarly_fx_employment_rate_factor_summary.csv", index=False
    )
    for name, r in board.items():
        monthly_returns(r).to_csv(
            REPORTS / f"scholarly_fx_employment_rate_{name}_monthly.csv", header=[name]
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
    win_df.to_csv(REPORTS / "scholarly_fx_employment_rate_window_stats.csv", index=False)

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
        sweep_df.to_csv(REPORTS / "scholarly_fx_employment_rate_risk_sweep.csv", index=False)

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

    meta = {
        "ok": True,
        "section": SECTION,
        "path": "oecd_mei_employment_rate_lrem64tt",
        "data_tag": data_tag,
        "pub_lag_months": PUB_LAG_M,
        "signal_lag_months": cfg.signal_lag,
        "signal_lag_days": 1,
        "z_window": cfg.z_window,
        "chg_periods": cfg.chg_periods,
        "score_unit": "er_level_pct",
        "z_low": cfg.z_low,
        "usd_tilt": cfg.usd_tilt,
        "cost_bps_side": cfg.cost_bps_side,
        "n_long": cfg.n_long,
        "n_short": cfg.n_short,
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
        "normalisation": (
            "FRED OECD MEI LREM64TT employment rate ages 15–64 (%) "
            "(USD/JPY/CAD/AUD *M156S; GBP/CHF/NZD *Q156S; "
            "EUR=LREM64TTDEQ156S Germany proxy — EZQ156S stale ~2022-10); "
            "XS on ER level %; also 5y z / +Δ12 of level boarded; "
            "high_er primary (labour-strength/engagement stock → appreciate); "
            "n_long=n_short=2 on available foreign panel (full G10)"
        ),
        "eur_mapping": "LREM64TTDEQ156S Germany proxy (EA EZQ156S stale ~2022-10)",
        "coverage_g10": "USD/EUR/GBP/JPY/AUD/CAD/CHF/NZD all mapped (full G10)",
        "series_map": lp.attrs.get("series_map", {}),
        "score_basis": "er_level_pct",
        "distinct_from": [
            "employment_persons_41",
            "unemployment_rate_54",
            "lfp_activity_56",
            "wage_earnings_55",
            "ulc_unit_labour_cost_48",
            "labour_prod_49",
            "cap_util_50",
            "cpi_52",
            "ppi_51",
            "industrial_production_42",
            "macro_diff_EW_CPI_IP_UR_blend",
            "oecd_cli_37",
            "oecd_cci_38",
            "oecd_bci_39",
            "wui_40",
            "retail_sales_46",
        ],
        "citations": [
            "Dahlquist & Hasseltoft (2020), JFE — economic momentum labour companion",
            "Employment rate (employed/population) as labour-strength / engagement stock",
            "OECD MEI LREM64TT employment rate ages 15–64 (%)",
            "Honesty: low ER / labour-disengagement debtor premium",
        ],
        "coverage": cov.to_dict(orient="records"),
        "unmapped": lp.attrs.get("unmapped", []),
        "failed_or_alt": lp.attrs.get("failed_or_alt", {}),
        "stale": lp.attrs.get("stale", False),
        "stale_note": lp.attrs.get("stale_note", ""),
        "soft_best": soft_best,
        "primary_sweep": prim_sweep.iloc[0].to_dict() if len(prim_sweep) else None,
        "sizing_note": (
            "Prop mode: utilise nearly full FTMO DD (10% static / 5% daily) via "
            "risk sweep when IS mean>0; stay under caps."
        ),
    }
    (REPORTS / "scholarly_fx_employment_rate_meta.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )

    lines = [
        f"# Scholarly FX: OECD/FRED employment-rate differential wave (§{SECTION})",
        "",
        "**Path:** FRED OECD MEI **employment-rate levels** "
        "(`LREM64TT*`; full G10; EUR=Germany DEQ proxy) — ER level XS — "
        "**distinct** from employment persons (§41), UR levels (§54), LFP (§56), wages (§55), "
        "ULC (§48), LP (§49), CU (§50).",
        f"**Data:** `{data_tag}` + free FRED LREM64TT levels after pub_lag. "
        f"EUR EA EZQ156S stale ~2022-10 → DEQ156S Germany proxy; full G10 mapped. "
        f"`n_long=n_short=2` on available foreign panel. "
        f"**PIT:** pub_lag_months={PUB_LAG_M} (employment-rate lag a priori; mixed M+Q) + "
        f"signal_lag_months={cfg.signal_lag} + 1d weight lag. "
        f"Score basis = **employment-rate level %**.",
        f"**Primary:** `{PRIMARY}` (long high relative ER / short low — "
        "labour-strength/engagement stock → appreciate). Locked sleeve untouched.",
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
        "Artifacts: `reports/scholarly_fx_employment_rate_*.csv`, "
        "`scholarly_fx_employment_rate_meta.json`.",
        "",
    ]
    (REPORTS / "scholarly_fx_employment_rate_wave.md").write_text("\n".join(lines))

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
