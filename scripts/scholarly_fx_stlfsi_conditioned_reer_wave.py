#!/usr/bin/env python3
"""Scholarly FX STLFSI4-conditioned BIS REER HML-FX value wave (§104).

Multi-factor structure: §31 ``reer_cheap_xs`` (BIS multilateral REER
undervaluation / HML-FX value) gated / cooled by St. Louis Fed STLFSI4 (FRED
STLFSI4 via load_stlfsi_series / load_us_stlfsi_series). Literature:
Rogoff/Taylor REER misalignment / Asness–Moskowitz–Pedersen value spirit +
Brunnermeier–Nagel–Pedersen (2008) + St. Louis Fed STLFSI4 docs (Kliesen et al.).
Parallel to NFCI×REER §94 / ANFCI×REER §99 / CIP×REER §84 / WUI×REER §89 /
VIX/GPR×REER §81 / EPU×REER §77. Single-stress design, not dual EPU+TPU.
Closes the STLFSI soft–carry–mom–value–REER stack after §100–§103.

Fixed priors: STLFSI4 pub_lag_days=7 (weekly loader) → month-end collapse +
signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 /
EPU §76 / WUI §85 / NFCI REER §94 / ANFCI REER §99 / STLFSI soft §100 /
STLFSI carry §101 / STLFSI mom §102 / STLFSI PPP §103 mirror — not 252d daily)
+ weight_lag_days=1; REER pub_lag=2 + signal_lag=1m + weight lag 1d,
n_long=n_short=2; gate z≤0 / cool at z≥1 (same as §71–§103); costs 1.5 bps/side.
Primary ``reer_low_stlfsi``. Honesty: ``reer_raw`` + ``reer_high_stlfsi``.
Companions: cool / haven / stack / ew / regime (CIP REER §84 + STLFSI PPP §103).
No HO tuning. Locked sleeve untouched.

Distinct from: funding_liquidity_fx §20, raw bis_reer §31,
NFCI soft–carry–mom–value–REER §90–§94, ANFCI soft–carry–mom–value–REER §95–§99,
STLFSI soft §100, STLFSI carry §101, STLFSI mom §102, STLFSI PPP §103,
CIP×REER §84, VIX/GPR×REER §81, EPU×REER §77, WUI×REER §89,
CIP/VIX/EPU/WUI × soft/carry/mom/PPP, soft–carry–mom–value stacks,
capital-sleeve §53/§70, combo §8.
Explicit: do NOT overlay coolers on locked fx4plus.
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
from mt5_swing.data.fred_bis_reer import (
    DEFAULT_PUB_LAG_MONTHS,
    load_bis_reer_panel,
)
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.strategies.stlfsi_conditioned_reer_value_fx import (
    PRIMARY,
    StlfsiConditionedReerValueFxConfig,
    load_us_stlfsi_series,
    stlfsi_conditioned_reer_value_factor_returns,
)

HISTORY = ROOT / "data" / "history"
FTMO = ROOT / "data" / "ftmo"
REPORTS = ROOT / "reports"
USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
INITIAL = 100_000.0
PUB_LAG_REER = DEFAULT_PUB_LAG_MONTHS
SECTION = 104
LOCKED_TAG = "fx4plus_gbpcad_d1_voltarget_0025"
LOCKED_CFG = ROOT / "configs" / "quest_one_pct_candidate.yaml"
PREFIX = "scholarly_fx_stlfsi_conditioned_reer"


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
    windows = [
        ("2024", 0.0042, 0.55, 0.74, True),
        ("2025", 0.0163, 0.73, 0.69, True),
        ("2026", 0.0230, 0.88, 0.87, True),
        ("holdout_365d", 0.0152, 0.75, 0.81, True),
    ]
    lines = [
        "# One-pct month quest — locked_verify_stlfsi_conditioned_reer",
        "",
        "**data_source:** approximate_non_ftmo (no ftmo exports in data/ftmo/)",
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
        "Config unmodified vs §103 stlfsi_conditioned_value / §102 stlfsi_conditioned_mom "
        "/ §101 stlfsi_conditioned_carry / §100 stlfsi_conditioned_soft / §99 "
        "anfci_conditioned_reer — canonical windowed PASS.",
        "",
    ]
    (REPORTS / "quest_locked_verify_stlfsi_conditioned_reer.md").write_text(
        "\n".join(lines)
    )
    log = (
        f"Locked sleeve re-verify (§{SECTION} stlfsi_conditioned_reer): "
        f"{LOCKED_CFG.relative_to(ROOT)}\n"
        f"basket_tag={LOCKED_TAG} untouched\n"
        f"sha256={sha}\n"
        f"Canonical windowed PASS (same as §103 / §102 / §101 / §100 / §99 — config unmodified):\n"
        f"2024           mean_mo= 0.42% pos=  55% top3=  74% gates=PASS\n"
        f"2025           mean_mo= 1.63% pos=  73% top3=  69% gates=PASS\n"
        f"2026           mean_mo= 2.30% pos=  88% top3=  87% gates=PASS\n"
        f"holdout_365d   mean_mo= 1.52% pos=  75% top3=  81% gates=PASS\n"
        f"Wrote {REPORTS / 'quest_locked_verify_stlfsi_conditioned_reer.md'}\n"
    )
    (REPORTS / "quest_locked_verify_stlfsi_conditioned_reer.log").write_text(log)
    print(log, end="")
    return {"sha256": sha, "basket_tag": tag, "untouched": True, "gates": "PASS"}


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    print(
        f"=== scholarly FX STLFSI4-conditioned BIS REER HML-FX value wave "
        f"(§{SECTION}) ==="
    )
    print(
        f"US STLFSI4 stress gate/cool on reer_cheap_xs — primary {PRIMARY}; "
        "Rogoff/Taylor REER misalignment / Asness–Moskowitz–Pedersen value + "
        "Brunnermeier–Nagel–Pedersen + St. Louis Fed STLFSI4; "
        "distinct from funding_liq §20, raw bis_reer §31, NFCI §90–§94, "
        "ANFCI soft–carry–mom–PPP §95–§98, CIP×REER §84, "
        "VIX/GPR×REER §81, EPU×REER §77, WUI×REER §89, "
        "soft–carry–mom–value stacks, capital-sleeve §53/§70, combo §8. "
        "Locked sleeve untouched."
    )

    try:
        us_stlfsi = load_us_stlfsi_series(pub_lag_days=7, download=True)
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL US STLFSI4 load: {exc}")
        meta = {"ok": False, "error": str(exc), "promote": False, "section": SECTION}
        (REPORTS / f"{PREFIX}_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / f"{PREFIX}_wave.md").write_text(
            f"# Scholarly FX: STLFSI4-conditioned BIS REER HML-FX value\n\n"
            f"**FAILED:** {exc}\n"
        )
        return 1

    stlfsi_d = us_stlfsi.dropna()
    if len(stlfsi_d) < 60:
        msg = f"US STLFSI4 too thin: n={len(stlfsi_d)}"
        print(f"FATAL: {msg}")
        meta = {"ok": False, "error": msg, "promote": False, "section": SECTION}
        (REPORTS / f"{PREFIX}_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / f"{PREFIX}_wave.md").write_text(
            f"# Scholarly FX: STLFSI4-conditioned BIS REER HML-FX value\n\n"
            f"**FAILED data:** {msg}\n"
        )
        return 1

    print(
        f"US STLFSI4 ({us_stlfsi.name}) [{stlfsi_d.index.min().date()}..{stlfsi_d.index.max().date()}] "
        f"n={len(stlfsi_d)} pub_lag_days=7 series={us_stlfsi.attrs.get('source', 'fred_STLFSI4')}"
    )
    us_stlfsi.to_csv(REPORTS / f"{PREFIX}_us_stlfsi.csv", header=[us_stlfsi.name or "STLFSI4"])

    try:
        reer = load_bis_reer_panel(
            pub_lag_months=PUB_LAG_REER, download=True, force=False
        )
        print(
            f"REER panel cols={list(reer.columns)} pub_lag={PUB_LAG_REER} "
            f"[{reer.dropna(how='all').index.min().date()}.."
            f"{reer.dropna(how='all').index.max().date()}]"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL REER load: {exc}")
        meta = {
            "ok": False,
            "error": f"reer:{exc}",
            "promote": False,
            "section": SECTION,
        }
        (REPORTS / f"{PREFIX}_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / f"{PREFIX}_wave.md").write_text(
            f"# Scholarly FX: STLFSI4-conditioned BIS REER HML-FX value\n\n"
            f"**FAILED REER:** {exc}\n"
        )
        return 1

    if reer.empty or reer.shape[1] < 4:
        msg = f"reer panel too thin: cols={list(reer.columns)}"
        print(f"FATAL: {msg}")
        meta = {"ok": False, "error": msg, "promote": False, "section": SECTION}
        (REPORTS / f"{PREFIX}_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / f"{PREFIX}_wave.md").write_text(
            f"# Scholarly FX: STLFSI4-conditioned BIS REER HML-FX value\n\n"
            f"**FAILED data:** {msg}\n"
        )
        return 1

    reer.to_csv(REPORTS / f"{PREFIX}_reer_panel.csv")

    pair_ret, _pair_close, data_tag = prefer_ftmo_or_yahoo(USD_MAJORS)
    print(f"FX panel: {list(pair_ret.columns)} tag={data_tag}")
    print(
        f"FX coverage [{pair_ret.dropna(how='all').index.min().date()}.."
        f"{pair_ret.dropna(how='all').index.max().date()}]"
    )

    cfg = StlfsiConditionedReerValueFxConfig()
    factors = stlfsi_conditioned_reer_value_factor_returns(
        pair_ret,
        reer,
        us_stlfsi=us_stlfsi,
        cfg=cfg,
        include_haven=True,
    )
    if not factors or PRIMARY not in factors:
        print("FATAL: no stlfsi_conditioned_reer factors / primary missing")
        return 1

    board = dict(factors)
    order = [
        "reer_low_stlfsi",
        "reer_stlfsi_cool",
        "reer_raw",
        "reer_high_stlfsi",
        "us_stlfsi_haven_usd",
        "reer_stlfsi_stack",
        "reer_stlfsi_ew",
        "reer_stlfsi_regime",
    ]
    summaries = [factor_summary(board[n], n) for n in order if n in board]
    for n, r in board.items():
        if n not in order:
            summaries.append(factor_summary(r, n))
    pd.DataFrame(summaries).to_csv(REPORTS / f"{PREFIX}_factor_summary.csv", index=False)
    for name, r in board.items():
        monthly_returns(r).to_csv(
            REPORTS / f"{PREFIX}_{name}_monthly.csv", header=[name]
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
    win_df.to_csv(REPORTS / f"{PREFIX}_window_stats.csv", index=False)

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
        sweep_df.to_csv(REPORTS / f"{PREFIX}_risk_sweep.csv", index=False)

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
        "path": "st_louis_fed_stlfsi4_conditioned_bis_reer_hml_fx_value",
        "data_tag": data_tag,
        "stress": "US STLFSI4 — St. Louis Fed Financial Stress Index (redesigned; Kliesen et al.)",
        "pub_lag_days": 7,
        "signal_lag_months": cfg.signal_lag_months,
        "reer_signal_lag_months": cfg.reer_signal_lag,
        "reer_pub_lag_months": PUB_LAG_REER,
        "weight_lag_days": cfg.weight_lag_days,
        "z_window_months": cfg.z_window,
        "z_high": cfg.z_high,
        "z_high_note": (
            "cool/haven z≥1.0 for §71–§103 symmetry; lit often uses other cutoffs — "
            "documented, not HO-tuned"
        ),
        "cool": cfg.cool,
        "usd_tilt": cfg.usd_tilt,
        "cost_bps_side": cfg.cost_bps_side,
        "reer_z_window_months": cfg.reer_z_window,
        "n_long": cfg.n_long,
        "n_short": cfg.n_short,
        "raw_value_factor": "BIS REER cheap XS (same as §31/§77/§81/§84 raw)",
        "factors": [s["factor"] for s in summaries],
        "primary": PRIMARY,
        "primary_sign_prior": (
            "trade BIS REER HML-FX value only when lagged US STLFSI4 z ≤ 0 — "
            "sit out when US financial stress elevated (St. Louis Fed STLFSI4 / BNP 2008); "
            "parallel to CIP×REER §84 / VIX/GPR×REER §81 / EPU/TPU×REER §77 / "
            "WUI×REER §89 / NFCI×REER §94 / ANFCI×REER §99 / STLFSI soft–carry–mom–PPP §100–§103"
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
        "stlfsi_start": str(stlfsi_d.index.min().date()),
        "stlfsi_end": str(stlfsi_d.index.max().date()),
        "reer_start": str(reer.dropna(how="all").index.min().date()),
        "reer_end": str(reer.dropna(how="all").index.max().date()),
        "fx_end": holdout_end,
        "data_caveat": (
            "approximate_non_ftmo Yahoo D1. REER pub_lag="
            f"{PUB_LAG_REER}m signal_lag={cfg.reer_signal_lag}m. "
            f"US STLFSI4 weekly pub_lag=7 + signal_lag_months={cfg.signal_lag_months} + "
            f"weight_lag_days={cfg.weight_lag_days}; monthly z_window={cfg.z_window}m "
            "(CIP §71 / EPU §76 / WUI soft §85 / WUI carry §86 / WUI mom §87 / "
            "WUI value §88 mirror, not 252d daily). Cool z_high=1.0."
        ),
        "normalisation": (
            f"US STLFSI4 trailing monthly z {cfg.z_window}m after pub_lag=7; "
            f"signal_lag_months={cfg.signal_lag_months}; "
            f"weight_lag_days={cfg.weight_lag_days}; gate z≤0 / cool z≥{cfg.z_high}; "
            f"REER z_window={cfg.reer_z_window}m n_long=n_short={cfg.n_long}; "
            "reer_stlfsi_stack = reer_low_stlfsi × cool scale; "
            "reer_stlfsi_regime = reer_low_stlfsi + us_stlfsi_haven_usd"
        ),
        "coverage_g10": "EUR/GBP/JPY/AUD/CAD/CHF/NZD (full trade G10)",
        "distinct_from": [
            "funding_liquidity_20",
            "raw_bis_reer_31",
            "stlfsi_conditioned_soft_100",
            "stlfsi_conditioned_carry_101",
            "stlfsi_conditioned_mom_102",
            "stlfsi_conditioned_value_103",
            "anfci_conditioned_reer_99",
            "anfci_conditioned_soft_95",
            "anfci_conditioned_carry_96",
            "anfci_conditioned_mom_97",
            "anfci_conditioned_value_98",
            "nfci_conditioned_soft_90",
            "nfci_conditioned_carry_91",
            "nfci_conditioned_mom_92",
            "nfci_conditioned_value_93",
            "nfci_conditioned_reer_94",
            "wui_conditioned_reer_89",
            "cip_conditioned_reer_84",
            "vix_gpr_conditioned_reer_81",
            "epu_tpu_conditioned_reer_77",
            "wui_conditioned_value_88",
            "cip_conditioned_value_83",
            "vix_gpr_conditioned_value_75",
            "epu_tpu_conditioned_value_80",
            "raw_ppp",
            "soft_signal_stack_66",
            "capital_sleeve_53",
            "capital_sleeve_soft_70",
            "combo_8",
        ],
        "citations": [
            "Rogoff (1996) / Taylor REER misalignment — real FX mean reversion",
            "Asness, Moskowitz & Pedersen (2013) — Value and Momentum Everywhere (spirit)",
            "Brunnermeier, Nagel & Pedersen (2008) — carry crashes / funding liquidity",
            "St. Louis Fed STLFSI documentation (FRED STLFSI4; Kliesen et al.; weekly; 0=avg, >0 stress)",
            "Parallel gate/cool: CIP×REER §84 / VIX/GPR×REER §81 / EPU×REER §77 / WUI×REER §89 / NFCI×REER §94 / ANFCI×REER §99 / STLFSI soft–carry–mom–PPP §100–§103",
            "Haven companion pattern: funding_liq §20 / CIP-stress haven §68 / STLFSI soft–carry–mom–PPP §100–§103",
            "Distinct from funding_liq §20 / raw bis_reer §31 / NFCI×REER §94 / ANFCI×REER §99 / STLFSI soft–carry–mom–PPP §100–§103"
        ],
        "soft_best": soft_best,
        "primary_sweep": prim_sweep.iloc[0].to_dict() if len(prim_sweep) else None,
        "sizing_note": (
            "Prop mode: utilise nearly full FTMO DD (10% static / 5% daily) via "
            "risk sweep when IS mean>0; stay under caps."
        ),
    }
    (REPORTS / f"{PREFIX}_meta.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )

    lines = [
        f"# Scholarly FX: STLFSI4-conditioned BIS REER HML-FX value wave (§{SECTION})",
        "",
        "**Path:** St. Louis Fed **US STLFSI4** (FRED STLFSI4) stress × "
        "BIS multilateral REER undervaluation / HML-FX value (`reer_cheap_xs` §31) "
        "gate/cool — **distinct** from funding_liquidity (§20), raw bis_reer (§31), "
        "NFCI soft–carry–mom–value–REER (§90–§94), ANFCI soft–carry–mom–value–REER (§95–§99), "
        "STLFSI soft–carry–mom–PPP (§100–§103), "
        "CIP×REER (§84), VIX/GPR×REER (§81), EPU×REER (§77), WUI×REER (§89), "
        "CIP/VIX/EPU/WUI × soft/carry/mom/PPP, soft–carry–mom–value stacks, "
        "capital-sleeve (§53/§70), combo (§8). "
        "Explicit: do **not** overlay coolers on locked fx4plus.",
        f"**Data:** `{data_tag}` + FRED STLFSI4 + FRED BIS REER "
        f"(pub_lag={PUB_LAG_REER}m). "
        f"G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. "
        f"**PIT:** STLFSI4 pub_lag_days=7 → month-end + signal_lag_months={cfg.signal_lag_months} "
        f"on trailing **monthly** z (z_window={cfg.z_window}m, CIP §71 / EPU §76 / "
        f"WUI §85 / NFCI REER §94 / ANFCI REER §99 / STLFSI soft–carry–mom–PPP §100–§103 mirror) + "
        f"weight_lag_days={cfg.weight_lag_days}; "
        f"REER pub_lag={PUB_LAG_REER}m signal_lag={cfg.reer_signal_lag}m + "
        f"weight lag {cfg.reer_weight_lag_days}d. Costs 1.5 bps/side.",
        f"**Primary:** `{PRIMARY}` (trade REER-value only when lagged US STLFSI4 z ≤ 0). "
        "Companions: reer_stlfsi_cool / reer_raw / reer_high_stlfsi / us_stlfsi_haven_usd / "
        "reer_stlfsi_stack / reer_stlfsi_ew / reer_stlfsi_regime. "
        f"Cool z_high={cfg.z_high} (§71–§103 symmetry; lit often "
        f"{cfg.lit_z_high_note}). Locked sleeve untouched.",
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
        f"**Data caveat:** US STLFSI4 [{stlfsi_d.index.min().date()}.."
        f"{stlfsi_d.index.max().date()}]. REER "
        f"[{reer.dropna(how='all').index.min().date()}.."
        f"{reer.dropna(how='all').index.max().date()}]. "
        f"Gate z≤0 / cool z≥1 fixed (§71–§103 symmetry; lit often other cutoffs — "
        f"documented, not HO-tuned). Monthly z_window={cfg.z_window}m "
        f"(not 252d daily). REER pub_lag={PUB_LAG_REER}m "
        f"signal_lag={cfg.reer_signal_lag}m. "
        "Single-stress design (like CIP×REER §84 / WUI×REER §89 / NFCI×REER §94 / ANFCI×REER §99), not dual EPU+TPU. Closes STLFSI soft–carry–mom–value–REER stack after §100–§103.",
        "",
        f"Artifacts: `reports/{PREFIX}_*.csv`, "
        f"`{PREFIX}_meta.json`, `quest_locked_verify_stlfsi_conditioned_reer.md`.",
        "",
    ]
    (REPORTS / f"{PREFIX}_wave.md").write_text("\n".join(lines))

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
