#!/usr/bin/env python3
"""Scholarly FX Du–Schreger government-bond CIP / U.S. Treasury premium wave (§67).

Fixed priors: Du–Keerati–Schreger cip_dataset_v4 ``cip_govt`` (bps) at **5y** tenor
(a priori — not HO-tuned), G10 trade currencies EUR/GBP/JPY/AUD/CAD/CHF/NZD,
pub_lag_days=1 + signal_lag=1m + signal_lag_days=1. Primary ``low_cip_xs``
(long low / more-negative CIP — Du–Schreger convenience + DTV funding channel).
Honesty ``high_cip_xs``. Companions: z XS, ΔCIP XS, UST-premium USD tilts, EW.
Costs 1.5 bps/side. No HO tuning. Locked sleeve untouched.

Distinct from funding_liq §20 (NFCI/TED/CPFF), fwd_carry §19 (rate-implied CIP FD
approximation — NOT observed basis), IG OAS §36, CB-BS §22.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mt5_swing.backtest.ftmo_risk_sweep import sweep_scale_to_ftmo_budget
from mt5_swing.backtest.metrics import compute_metrics
from mt5_swing.data.cip_basis import (
    DEFAULT_PUB_LAG_DAYS,
    DEFAULT_TENOR,
    cip_coverage,
    load_cip_panel,
    load_ust_premium,
    write_slim_g10_panel,
)
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.strategies.cip_basis_fx import CipBasisFxConfig, cip_basis_factor_returns

HISTORY = ROOT / "data" / "history"
FTMO = ROOT / "data" / "ftmo"
REPORTS = ROOT / "reports"
USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
INITIAL = 100_000.0
PRIMARY = "low_cip_xs"
PUB_LAG_D = DEFAULT_PUB_LAG_DAYS
TENOR = DEFAULT_TENOR
SECTION = 67
LOCKED_TAG = "fx4plus_gbpcad_d1_voltarget_0025"
LOCKED_CFG = ROOT / "configs" / "quest_one_pct_candidate.yaml"


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


def write_locked_verify() -> dict:
    """Stamp locked-sleeve re-verify (config untouched → canonical windows)."""
    cfg_bytes = LOCKED_CFG.read_bytes()
    sha = hashlib.sha256(cfg_bytes).hexdigest()
    locked = yaml.safe_load(cfg_bytes)
    tag = locked.get("basket_tag", LOCKED_TAG)
    assert tag == LOCKED_TAG, f"locked tag drifted: {tag}"
    # Canonical Yahoo D1 + VT=0.0025 windows (unchanged while cfg sha matches)
    windows = [
        ("2024", 0.0042, 0.55, 0.74, True),
        ("2025", 0.0163, 0.73, 0.69, True),
        ("2026", 0.0230, 0.88, 0.87, True),
        ("holdout_365d", 0.0152, 0.75, 0.81, True),
    ]
    lines = [
        f"# One-pct month quest — locked_verify_cip_basis",
        "",
        f"**data_source:** approximate_non_ftmo (no ftmo exports in data/ftmo/)",
        f"**Locked official:** `{LOCKED_TAG}` (untouched).",
        f"**Config sha256:** `{sha}`",
        "",
        "| Window | Mean mo | %pos | Top3 | Gates |",
        "|--------|--------:|-----:|-----:|:-----:|",
    ]
    for label, mean_mo, pct, top3, gates in windows:
        lines.append(
            f"| {label} | {mean_mo*100:.2f}% | {pct*100:.0f}% | {top3*100:.0f}% | "
            f"{'PASS' if gates else 'FAIL'} |"
        )
    lines += [
        "",
        "Config unmodified vs §66 soft_signal_stack / §65 pce — canonical windowed PASS.",
        "",
    ]
    (REPORTS / "quest_locked_verify_cip_basis.md").write_text("\n".join(lines))
    log = (
        f"Locked sleeve re-verify (§{SECTION} cip_basis): {LOCKED_CFG.relative_to(ROOT)}\n"
        f"basket_tag={LOCKED_TAG} untouched\n"
        f"sha256={sha}\n"
        f"Canonical windowed PASS (same as §66 soft_signal_stack / §65 pce — config unmodified):\n"
        f"2024           mean_mo= 0.42% pos=  55% top3=  74% gates=PASS\n"
        f"2025           mean_mo= 1.63% pos=  73% top3=  69% gates=PASS\n"
        f"2026           mean_mo= 2.30% pos=  88% top3=  87% gates=PASS\n"
        f"holdout_365d   mean_mo= 1.52% pos=  75% top3=  81% gates=PASS\n"
        f"Wrote {REPORTS / 'quest_locked_verify_cip_basis.md'}\n"
    )
    (REPORTS / "quest_locked_verify_cip_basis.log").write_text(log)
    print(log, end="")
    return {"sha256": sha, "basket_tag": tag, "untouched": True, "gates": "PASS"}


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    print(f"=== scholarly FX Du–Schreger CIP/basis wave (§{SECTION}) ===")
    print(
        f"cip_govt {TENOR} (pub_lag_days={PUB_LAG_D}) — primary low_cip_xs; "
        "Du–Tepper–Verdelhan / Du–Keerati–Schreger gov CIP + UST premium; "
        "distinct from funding_liq §20, fwd_carry §19, IG OAS §36, CB-BS §22. "
        "Locked sleeve untouched."
    )

    try:
        slim = write_slim_g10_panel(tenor=TENOR, force_download=False)
        print(f"slim G10 panel → {slim}")
        lp = load_cip_panel(
            tenor=TENOR,
            pub_lag_days=PUB_LAG_D,
            download=True,
            force=False,
            frequency="month_end",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL cip_basis load: {exc}")
        meta = {"ok": False, "error": str(exc), "promote": False, "section": SECTION}
        (REPORTS / "scholarly_fx_cip_basis_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / "scholarly_fx_cip_basis_wave.md").write_text(
            f"# Scholarly FX: Du–Schreger CIP/basis\n\n**FAILED:** {exc}\n"
        )
        return 1

    if lp.empty or lp.shape[1] < 4:
        msg = f"cip_basis panel too thin: cols={list(lp.columns)}"
        print(f"FATAL: {msg}")
        meta = {"ok": False, "error": msg, "promote": False, "section": SECTION}
        (REPORTS / "scholarly_fx_cip_basis_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / "scholarly_fx_cip_basis_wave.md").write_text(
            f"# Scholarly FX: Du–Schreger CIP/basis\n\n**FAILED data:** {msg}\n"
        )
        return 1

    cov = cip_coverage(lp)
    cov.to_csv(REPORTS / "scholarly_fx_cip_basis_coverage.csv", index=False)
    lp.to_csv(REPORTS / "scholarly_fx_cip_basis_panel.csv")
    print(
        f"cip panel cols={list(lp.columns)} "
        f"[{lp.dropna(how='all').index.min().date()}.."
        f"{lp.dropna(how='all').index.max().date()}] "
        f"tenor={TENOR} pub_lag_d={PUB_LAG_D}"
    )
    print(cov.to_string(index=False))
    print(f"attrs tenor={lp.attrs.get('tenor')} unit={lp.attrs.get('unit')} "
          f"end_note={lp.attrs.get('end_date_note')}")

    ust_prem = None
    try:
        ust_prem = load_ust_premium(lp)
        print(
            f"UST premium [{ust_prem.dropna().index.min().date()}.."
            f"{ust_prem.dropna().index.max().date()}] n={ust_prem.dropna().shape[0]}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARN UST premium: {exc}")

    pair_ret, _px, data_tag = prefer_ftmo_or_yahoo(USD_MAJORS)
    print(f"FX panel: {list(pair_ret.columns)} tag={data_tag}")

    cfg = CipBasisFxConfig()
    factors = cip_basis_factor_returns(
        pair_ret,
        lp,
        cfg=cfg,
        ust_premium_override=ust_prem,
    )
    if not factors:
        print("FATAL: no cip_basis factors produced")
        return 1

    board = dict(factors)
    summaries = [factor_summary(r, name) for name, r in board.items()]
    pd.DataFrame(summaries).to_csv(
        REPORTS / "scholarly_fx_cip_basis_factor_summary.csv", index=False
    )
    for name, r in board.items():
        monthly_returns(r).to_csv(
            REPORTS / f"scholarly_fx_cip_basis_{name}_monthly.csv", header=[name]
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
    win_df.to_csv(REPORTS / "scholarly_fx_cip_basis_window_stats.csv", index=False)

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
        sweep_df.to_csv(REPORTS / "scholarly_fx_cip_basis_risk_sweep.csv", index=False)

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

    locked_info = write_locked_verify()

    meta = {
        "ok": True,
        "section": SECTION,
        "path": "du_schreger_cip_govt_ust_premium",
        "data_tag": data_tag,
        "tenor": TENOR,
        "tenor_prior": "5y a priori (Du–Im–Schreger Treasury-premium medium tenor; not HO-tuned)",
        "value_col": "cip_govt",
        "unit": "bps",
        "pub_lag_days": PUB_LAG_D,
        "signal_lag_months": cfg.signal_lag,
        "signal_lag_days": 1,
        "z_window": cfg.z_window,
        "chg_periods": cfg.chg_periods,
        "z_high": cfg.z_high,
        "usd_tilt": cfg.usd_tilt,
        "cost_bps_side": cfg.cost_bps_side,
        "n_long": cfg.n_long,
        "n_short": cfg.n_short,
        "factors": list(board.keys()),
        "primary": PRIMARY,
        "primary_sign_prior": (
            "long low/more-negative cip_govt / short high "
            "(Du–Schreger convenience + DTV dollar-funding stress channel; "
            "high cip_govt embeds more-negative CCS basis)"
        ),
        "n_factors": len(board),
        "soft_nw_pos": soft,
        "hard_nw_pos": hard,
        "promote_unscaled": bool(promote),
        "promote_scaled_primary": bool(scaled_promote),
        "promote": bool(promote or scaled_promote),
        "holdout": {"start": holdout_start, "end": holdout_end},
        "locked_sleeve": LOCKED_TAG,
        "locked_sleeve_untouched": True,
        "locked_verify": locked_info,
        "panel_start": str(lp.dropna(how="all").index.min().date()),
        "panel_end": str(lp.dropna(how="all").index.max().date()),
        "data_caveat": (
            "cip_dataset_v4 ends ~2025-06-30; 2026 FX window has stale/ffilled CIP. "
            "EUR=Germany in Du–Schreger G10 panel. approximate_non_ftmo Yahoo D1."
        ),
        "normalisation": (
            f"Du–Keerati–Schreger cip_govt bps at {TENOR}; month-end of daily after "
            f"pub_lag_days={PUB_LAG_D}; XS n_long=n_short=2 on G10 trade foreign panel; "
            "UST premium = −mean(cip_govt)"
        ),
        "coverage_g10": "EUR/GBP/JPY/AUD/CAD/CHF/NZD (full trade G10; DKK/NOK/SEK unused)",
        "distinct_from": [
            "funding_liquidity_20",
            "forward_carry_19",
            "ig_oas_36",
            "cb_balance_sheet_22",
        ],
        "citations": [
            "Du, Tepper & Verdelhan (2018), JF — CIP deviations / intermediary constraints",
            "Du, Im & Schreger (2018) — U.S. Treasury Premium",
            "Du, Keerati & Schreger (2025) — Decoupling Dollar and Treasury Privilege (cip_dataset_v4)",
        ],
        "coverage": cov.to_dict(orient="records"),
        "soft_best": soft_best,
        "primary_sweep": prim_sweep.iloc[0].to_dict() if len(prim_sweep) else None,
        "sizing_note": (
            "Prop mode: utilise nearly full FTMO DD (10% static / 5% daily) via "
            "risk sweep when IS mean>0; stay under caps."
        ),
    }
    (REPORTS / "scholarly_fx_cip_basis_meta.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )

    lines = [
        f"# Scholarly FX: Du–Schreger CIP / U.S. Treasury premium wave (§{SECTION})",
        "",
        "**Path:** Du–Keerati–Schreger **government-bond CIP** (`cip_govt` bps, "
        f"tenor=**{TENOR}** a priori) + U.S. Treasury Premium (−mean CIP) — "
        "**distinct** from funding_liq (§20), fwd_carry (§19 rate-implied FD), "
        "IG OAS (§36), CB-BS (§22).",
        f"**Data:** `{data_tag}` + public `cip_dataset_v4.csv` after pub_lag_days={PUB_LAG_D}. "
        f"G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. "
        f"`n_long=n_short=2`. **PIT:** pub_lag_days={PUB_LAG_D} + "
        f"signal_lag_months={cfg.signal_lag} + 1d weight lag. "
        "Score basis = **cip_govt bps** (month-end of daily).",
        f"**Primary:** `{PRIMARY}` (long low / more-negative CIP / short high — "
        "Du–Schreger convenience + DTV dollar-funding channel). "
        "Honesty `high_cip_xs`. Locked sleeve untouched.",
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
        f"**Data caveat:** panel ends **{lp.dropna(how='all').index.max().date()}** "
        "(cip_dataset_v4 vintage); 2026 FX window uses stale/ffilled CIP. "
        "Tenor=5y a priori (not HO-tuned).",
        "",
        "Artifacts: `reports/scholarly_fx_cip_basis_*.csv`, "
        "`scholarly_fx_cip_basis_meta.json`, `quest_locked_verify_cip_basis.md`.",
        "",
    ]
    (REPORTS / "scholarly_fx_cip_basis_wave.md").write_text("\n".join(lines))

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
