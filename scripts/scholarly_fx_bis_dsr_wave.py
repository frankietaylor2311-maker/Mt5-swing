#!/usr/bin/env python3
"""Scholarly FX BIS debt-service ratio (DSR / PNFS) wave (§47).

Fixed priors: BIS SDMX WS_DSR Q..P PNFS debt-service ratio (quarterly →
monthly after pub_lag_months=5) + signal_lag=1m + signal_lag_days=1.
Primary high_dsr_xs (high relative DSR → FX stress / risk premia;
Drehmann–Juselius / BIS early-warning). Scores on **DSR level %**.
FRED Q*PNFDSR / BISDSR* all 404 — live via free BIS SDMX. EUR=DE proxy; NZD gap.
No HO tuning. Locked sleeve untouched.
Distinct from credit/GDP §32, gov debt §29, fiscal §28, funding-liq §20,
IG OAS §36, CA/TB, REER, money, reserves, retail/IP/emp.
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
from mt5_swing.data.fred_bis_dsr import (
    DEFAULT_PUB_LAG_MONTHS,
    bis_dsr_coverage,
    load_bis_dsr_panel,
    load_us_bis_dsr,
)
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.strategies.bis_dsr_fx import (
    BisDsrFxConfig,
    bis_dsr_factor_returns,
)

HISTORY = ROOT / "data" / "history"
FTMO = ROOT / "data" / "ftmo"
REPORTS = ROOT / "reports"
USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
INITIAL = 100_000.0
PRIMARY = "high_dsr_xs"
PUB_LAG_M = DEFAULT_PUB_LAG_MONTHS
SECTION = 47


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
    print(f"=== scholarly FX BIS debt-service (DSR / PNFS) wave (§{SECTION}) ===")
    print(
        "BIS SDMX WS_DSR Q..P PNFS DSR (pub_lag=5) — high DSR XS risk-premia; "
        "Drehmann–Juselius. FRED DSR 404. EUR=DE; NZD gap. Distinct from "
        "credit/GDP §32, debt §29, fiscal §28. Locked sleeve untouched."
    )

    try:
        dsr = load_bis_dsr_panel(pub_lag_months=PUB_LAG_M, download=True, force=False)
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL DSR load: {exc}")
        meta = {"ok": False, "error": str(exc), "promote": False, "section": SECTION}
        (REPORTS / "scholarly_fx_bis_dsr_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / "scholarly_fx_bis_dsr_wave.md").write_text(
            f"# Scholarly FX: BIS DSR\n\n**FAILED:** {exc}\n"
        )
        return 1

    if dsr.empty or dsr.shape[1] < 4:
        msg = (
            f"DSR panel too thin: cols={list(dsr.columns)}. "
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
        (REPORTS / "scholarly_fx_bis_dsr_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / "scholarly_fx_bis_dsr_wave.md").write_text(
            f"# Scholarly FX: BIS DSR\n\n**FAILED data:** {msg}\n"
        )
        return 1

    cov = bis_dsr_coverage(dsr)
    cov.to_csv(REPORTS / "scholarly_fx_bis_dsr_coverage.csv", index=False)
    dsr.to_csv(REPORTS / "scholarly_fx_bis_dsr_panel.csv")
    print(
        f"DSR panel cols={list(dsr.columns)} "
        f"[{dsr.dropna(how='all').index.min().date()}.."
        f"{dsr.dropna(how='all').index.max().date()}] "
        f"pub_lag_m={PUB_LAG_M}"
    )
    print(cov.to_string(index=False))
    print(f"Series source note: {dsr.attrs.get('source')}")
    print(f"Score basis: {dsr.attrs.get('score_basis')} unit={dsr.attrs.get('unit')}")
    print(f"Failed/alt documented: {dsr.attrs.get('failed_or_alt')}")

    us_dsr = None
    try:
        us_dsr = load_us_bis_dsr(download=True, force=False)
        print(
            f"US BIS DSR [{us_dsr.dropna().index.min().date()}.."
            f"{us_dsr.dropna().index.max().date()}] n={us_dsr.dropna().shape[0]} "
            f"series={us_dsr.attrs.get('series_id')}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARN US DSR load failed: {exc}")
        us_dsr = None

    pair_ret, _px, data_tag = prefer_ftmo_or_yahoo(USD_MAJORS)
    print(f"FX panel: {list(pair_ret.columns)} tag={data_tag}")

    cfg = BisDsrFxConfig()
    factors = bis_dsr_factor_returns(
        pair_ret,
        dsr,
        cfg=cfg,
        us_dsr_override=us_dsr,
    )
    if not factors:
        print("FATAL: no DSR factors produced")
        return 1

    board = dict(factors)
    summaries = [factor_summary(r, name) for name, r in board.items()]
    pd.DataFrame(summaries).to_csv(
        REPORTS / "scholarly_fx_bis_dsr_factor_summary.csv", index=False
    )
    for name, r in board.items():
        monthly_returns(r).to_csv(
            REPORTS / f"scholarly_fx_bis_dsr_{name}_monthly.csv", header=[name]
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
    win_df.to_csv(REPORTS / "scholarly_fx_bis_dsr_window_stats.csv", index=False)

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
        sweep_df.to_csv(REPORTS / "scholarly_fx_bis_dsr_risk_sweep.csv", index=False)

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
        "path": "bis_ws_dsr_pnfs",
        "data_tag": data_tag,
        "pub_lag_months": PUB_LAG_M,
        "signal_lag_months": cfg.signal_lag,
        "signal_lag_days": 1,
        "z_window": cfg.z_window,
        "chg_periods": cfg.chg_periods,
        "score_unit": "dsr_pct_of_income",
        "z_high": cfg.z_high,
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
            "BIS SDMX WS_DSR Q..P PNFS DSR % after pub_lag=5; XS on DSR level; "
            "also 5y z / Δ12 boarded; high_dsr primary (Drehmann–Juselius risk premia); "
            "n_long=n_short=2 on mapped foreign (NZD unmapped)"
        ),
        "eur_mapping": "BIS WS_DSR Q.DE.P Germany proxy (XM/EA euro-area keys missing; FR available unused)",
        "coverage_g10": "USD/GBP/JPY/CAD/AUD/CHF + EUR=DE mapped; NZD unmapped (NZ absent from WS_DSR)",
        "series_map": dsr.attrs.get("series_map", {}),
        "score_basis": "pnfs_dsr_level_pct",
        "distinct_from": [
            "macro_diff_EW_CPI_IP_UR_blend",
            "oecd_cli_37",
            "oecd_cci_38",
            "oecd_bci_39",
            "wui_40",
            "epu_tpu",
            "gpr_country",
            "AI_GPR",
            "ig_oas_BAMLC0A0CM_36",
            "CA_GDP_21",
            "trade_balance_30",
            "fiscal_28",
            "debt_gdp_29",
            "bis_reer_31",
            "bis_credit_gap_32",
            "money_growth_33",
            "reserves_34",
            "house_price_35",
            "equity_diff",
            "employment_41",
            "industrial_production_42",
            "building_permits_43",
            "retail_sales_46",
            "acm_tp_44",
            "country_gpr_bilateral_45",
            "commodity",
            "ig_oas_36",
        ],
        "citations": [
            "Drehmann & Juselius (BIS): debt-service ratio early-warning vs stock leverage",
            "Drehmann et al. (2015): DSR = interest + amortisation / income",
            "Primary: high relative PNFS DSR → FX stress / risk premia",
            "Honesty: low DSR lean balance-sheet",
        ],
        "coverage": cov.to_dict(orient="records"),
        "unmapped": dsr.attrs.get("unmapped", []),
        "failed_or_alt": dsr.attrs.get("failed_or_alt", {}),
        "soft_best": soft_best,
        "primary_sweep": prim_sweep.iloc[0].to_dict() if len(prim_sweep) else None,
    }
    (REPORTS / "scholarly_fx_bis_dsr_meta.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )

    lines = [
        f"# Scholarly FX: BIS debt-service (DSR / PNFS) differential wave (§{SECTION})",
        "",
        "**Path:** BIS SDMX `WS_DSR` Q..P private non-financial debt-service ratio "
        "(FRED Q*PNFDSR / BISDSR* **404**) — DSR level XS — **distinct** from "
        "credit/GDP stock (§32), gov debt (§29), fiscal (§28), funding-liq (§20), "
        "IG OAS (§36), CA/TB, REER, money, reserves, retail/IP/emp.",
        f"**Data:** `{data_tag}` + free BIS SDMX WS_DSR (quarterly → monthly after "
        f"pub_lag). EUR = `Q.DE.P` (Germany proxy; XM/EA missing). NZD unmapped. "
        f"`n_long=n_short=2` a priori. "
        f"**PIT:** pub_lag_months={PUB_LAG_M} (BIS quarterly a priori) + "
        f"signal_lag_months={cfg.signal_lag} + 1d weight lag. "
        f"Score basis = **PNFS DSR %**.",
        f"**Primary:** `{PRIMARY}` (long high relative DSR / short low — "
        "Drehmann–Juselius risk premia). Locked sleeve untouched.",
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
        "Artifacts: `reports/scholarly_fx_bis_dsr_*.csv`, "
        "`scholarly_fx_bis_dsr_meta.json`.",
        "",
    ]
    (REPORTS / "scholarly_fx_bis_dsr_wave.md").write_text("\n".join(lines))

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
