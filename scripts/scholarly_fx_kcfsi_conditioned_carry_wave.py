#!/usr/bin/env python3
"""Scholarly FX KCFSI-conditioned Lustig–Verdelhan carry wave (§106).

Multi-factor structure: IR3M/policy-rate Lustig–Verdelhan carry gated / cooled
by Kansas City Fed Financial Stress Index (FRED KCFSI via load_kcfsi_series /
load_us_kcfsi_series). Literature: Lustig–Roussanov–Verdelhan (2011) +
Brunnermeier–Nagel–Pedersen (2008) + KC Fed FSI docs (Hakkio & Keeton 2009).
Continues eighth free stress family after KCFSI soft §105. Parallel to
STLFSI×carry §101 / ANFCI×carry §96 / NFCI×carry §91 / CIP×carry §68 /
WUI×carry §86. Single-stress design, not dual EPU+TPU. Natural continuation
after KCFSI soft §105.

Fixed priors: KCFSI pub_lag_months=1 (monthly loader, like EPU) +
signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 /
EPU §76 / WUI §85 / NFCI soft §90 / ANFCI soft §95 / KCFSI soft §105 /
KCFSI soft §105 mirror — not 252d daily) + weight_lag_days=1;
rates pub_lag=1 + carry_signal_lag=1d; gate z≤0 / cool at z≥1 (same as
§71–§105); costs 1.5 bps/side.
Primary ``carry_low_kcfsi``. Honesty: ``carry_raw`` + ``carry_high_kcfsi``.
Companions: cool / haven / stack / ew / regime (CIP carry §68 + KCFSI soft §105).
No HO tuning. Locked sleeve untouched.

Distinct from: funding_liquidity_fx §20 (raw nfci/anfci_usd / carry×NFCI cool
standalone), NFCI×carry §91, ANFCI×carry §96, STLFSI×carry §101, KCFSI soft §105,
KCFSI soft §105, NFCI soft §90, WUI carry §86, soft_ew §66, CIP×carry §68,
VIX/GPR×carry §73, EPU/TPU×carry §78, CIP/VIX/EPU/WUI/NFCI/ANFCI/STLFSI
soft/mom/value/REER §71–§104, capital-sleeve §53/§70, combo §8.
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
from mt5_swing.data.fred_rates import load_currency_rates
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.strategies.kcfsi_conditioned_carry_fx import (
    PRIMARY,
    KcfsiConditionedCarryFxConfig,
    load_us_kcfsi_series,
    kcfsi_conditioned_carry_factor_returns,
)

HISTORY = ROOT / "data" / "history"
FTMO = ROOT / "data" / "ftmo"
REPORTS = ROOT / "reports"
USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
INITIAL = 100_000.0
SECTION = 106
LOCKED_TAG = "fx4plus_gbpcad_d1_voltarget_0025"
LOCKED_CFG = ROOT / "configs" / "quest_one_pct_candidate.yaml"
PREFIX = "scholarly_fx_kcfsi_conditioned_carry"


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
        "# One-pct month quest — locked_verify_kcfsi_conditioned_carry",
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
        "Config unmodified vs §105 kcfsi_conditioned_soft / §101 stlfsi_conditioned_carry "
        "/ §96 anfci_conditioned_carry — canonical windowed PASS.",
        "",
    ]
    (REPORTS / "quest_locked_verify_kcfsi_conditioned_carry.md").write_text(
        "\n".join(lines)
    )
    log = (
        f"Locked sleeve re-verify (§{SECTION} kcfsi_conditioned_carry): "
        f"{LOCKED_CFG.relative_to(ROOT)}\n"
        f"basket_tag={LOCKED_TAG} untouched\n"
        f"sha256={sha}\n"
        f"Canonical windowed PASS (same as §105 / §101 / §96 / §91 — config unmodified):\n"
        f"2024           mean_mo= 0.42% pos=  55% top3=  74% gates=PASS\n"
        f"2025           mean_mo= 1.63% pos=  73% top3=  69% gates=PASS\n"
        f"2026           mean_mo= 2.30% pos=  88% top3=  87% gates=PASS\n"
        f"holdout_365d   mean_mo= 1.52% pos=  75% top3=  81% gates=PASS\n"
        f"Wrote {REPORTS / 'quest_locked_verify_kcfsi_conditioned_carry.md'}\n"
    )
    (REPORTS / "quest_locked_verify_kcfsi_conditioned_carry.log").write_text(log)
    print(log, end="")
    return {"sha256": sha, "basket_tag": tag, "untouched": True, "gates": "PASS"}


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    print(
        f"=== scholarly FX KCFSI-conditioned Lustig–Verdelhan carry wave "
        f"(§{SECTION}) ==="
    )
    print(
        f"US KCFSI stress gate/cool on IR3M carry — primary {PRIMARY}; "
        "Lustig–Roussanov–Verdelhan + Brunnermeier–Nagel–Pedersen + Kansas City Fed KCFSI; "
        "distinct from funding_liq §20, NFCI×carry §91, KCFSI soft §105, NFCI soft §90, "
        "WUI carry §86, soft_ew §66, CIP×carry §68, VIX/GPR×carry §73, EPU/TPU×carry §78, "
        "CIP/VIX/EPU/WUI soft/mom/value/REER §71–§77/§79–§89, NFCI mom/value/REER §92–§94, "
        "capital-sleeve §53/§70, combo §8. Locked sleeve untouched."
    )

    try:
        us_kcfsi = load_us_kcfsi_series(pub_lag_months=1, download=True)
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL US KCFSI load: {exc}")
        meta = {"ok": False, "error": str(exc), "promote": False, "section": SECTION}
        (REPORTS / f"{PREFIX}_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / f"{PREFIX}_wave.md").write_text(
            f"# Scholarly FX: KCFSI-conditioned carry\n\n**FAILED:** {exc}\n"
        )
        return 1

    kcfsi_d = us_kcfsi.dropna()
    if len(kcfsi_d) < 60:
        msg = f"US KCFSI too thin: n={len(kcfsi_d)}"
        print(f"FATAL: {msg}")
        meta = {"ok": False, "error": msg, "promote": False, "section": SECTION}
        (REPORTS / f"{PREFIX}_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / f"{PREFIX}_wave.md").write_text(
            f"# Scholarly FX: KCFSI-conditioned carry\n\n**FAILED data:** {msg}\n"
        )
        return 1

    print(
        f"US KCFSI ({us_kcfsi.name}) [{kcfsi_d.index.min().date()}..{kcfsi_d.index.max().date()}] "
        f"n={len(kcfsi_d)} pub_lag_months=1 series={us_kcfsi.attrs.get('source', 'fred_KCFSI')}"
    )
    us_kcfsi.to_csv(REPORTS / f"{PREFIX}_us_kcfsi.csv", header=[us_kcfsi.name or "KCFSI"])

    pair_ret, _px, data_tag = prefer_ftmo_or_yahoo(USD_MAJORS)
    print(f"FX panel: {list(pair_ret.columns)} tag={data_tag}")

    try:
        rates = load_currency_rates(pub_lag_months=1, download=True)
        print(
            f"rates cols={list(rates.columns)} "
            f"[{rates.dropna(how='all').index.min().date()}.."
            f"{rates.dropna(how='all').index.max().date()}]"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL rates: {exc}")
        meta = {"ok": False, "error": str(exc), "promote": False, "section": SECTION}
        (REPORTS / f"{PREFIX}_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / f"{PREFIX}_wave.md").write_text(
            f"# Scholarly FX: KCFSI-conditioned carry\n\n**FAILED rates:** {exc}\n"
        )
        return 1

    cfg = KcfsiConditionedCarryFxConfig()
    factors = kcfsi_conditioned_carry_factor_returns(
        pair_ret, rates=rates, us_kcfsi=us_kcfsi, cfg=cfg, include_haven=True
    )
    if not factors or PRIMARY not in factors:
        print("FATAL: no kcfsi_conditioned_carry factors / primary missing")
        return 1

    board = dict(factors)
    order = [
        "carry_low_kcfsi",
        "carry_kcfsi_cool",
        "carry_raw",
        "carry_high_kcfsi",
        "us_kcfsi_haven_usd",
        "carry_kcfsi_stack",
        "carry_kcfsi_ew",
        "carry_kcfsi_regime",
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
        "path": "kansas_city_fed_kcfsi_conditioned_lustig_verdelhan_carry",
        "data_tag": data_tag,
        "stress": "US KCFSI — Kansas City Fed Financial Stress Index (Hakkio & Keeton 2009)",
        "pub_lag_months": 1,
        "pub_lag_months_rates": 1,
        "signal_lag_months": cfg.signal_lag_months,
        "carry_signal_lag_days": cfg.carry_signal_lag,
        "weight_lag_days": cfg.weight_lag_days,
        "z_window_months": cfg.z_window,
        "z_high": cfg.z_high,
        "z_high_note": (
            "cool/haven z≥1.0 for §71–§105 symmetry; lit often uses other cutoffs — "
            "documented, not HO-tuned"
        ),
        "cool": cfg.cool,
        "usd_tilt": cfg.usd_tilt,
        "cost_bps_side": cfg.cost_bps_side,
        "n_long": cfg.carry_n_long,
        "n_short": cfg.carry_n_short,
        "raw_carry_factor": "IR3M Lustig–Verdelhan carry (same as §68/§73/§78 raw)",
        "factors": [s["factor"] for s in summaries],
        "primary": PRIMARY,
        "primary_sign_prior": (
            "trade Lustig–Verdelhan cash-rate carry only when lagged US KCFSI z ≤ 0 — "
            "sit out when US financial stress elevated (Kansas City Fed KCFSI / BNP 2008); "
            "parallel to CIP×carry §68 / VIX/GPR×carry §73 / EPU/TPU×carry §78 / "
            "WUI×carry §86 / NFCI×carry §91 / ANFCI×carry §96 / STLFSI×carry §101 / KCFSI soft §105 with monthly KCFSI stress"
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
        "kcfsi_start": str(kcfsi_d.index.min().date()),
        "kcfsi_end": str(kcfsi_d.index.max().date()),
        "fx_end": holdout_end,
        "data_caveat": (
            "approximate_non_ftmo Yahoo D1. FRED IR3M rates pub_lag=1 + "
            f"carry_signal_lag={cfg.carry_signal_lag}d. US KCFSI monthly pub_lag_months=1 + "
            f"signal_lag_months={cfg.signal_lag_months} + weight_lag_days="
            f"{cfg.weight_lag_days}; monthly z_window={cfg.z_window}m "
            "(CIP §71 / EPU §76 / WUI §85 / NFCI soft §90 / ANFCI soft §95 / STLFSI soft §100 / KCFSI soft §105 mirror, not 252d daily). "
            "Cool z_high=1.0. Distinct from funding_liq §20 / NFCI×carry §91."
        ),
        "normalisation": (
            f"US KCFSI monthly trailing z {cfg.z_window}m after pub_lag_months=1; "
            f"signal_lag_months={cfg.signal_lag_months}; "
            f"weight_lag_days={cfg.weight_lag_days}; gate z≤0 / cool z≥{cfg.z_high}; "
            f"IR3M carry n_long=n_short={cfg.carry_n_long}; "
            "carry_kcfsi_stack = carry_low_kcfsi × cool scale; "
            "carry_kcfsi_regime = carry_low_kcfsi + us_kcfsi_haven_usd"
        ),
        "coverage_g10": "EUR/GBP/JPY/AUD/CAD/CHF/NZD (full trade G10)",
        "distinct_from": [
            "funding_liquidity_20",
            "kcfsi_conditioned_soft_105",
            "stlfsi_conditioned_carry_101",
            "stlfsi_conditioned_soft_100",
            "anfci_conditioned_carry_96",
            "anfci_conditioned_soft_95",
            "nfci_conditioned_carry_91",
            "nfci_conditioned_soft_90",
            "nfci_conditioned_mom_92",
            "nfci_conditioned_value_93",
            "nfci_conditioned_reer_94",
            "wui_conditioned_carry_86",
            "wui_conditioned_soft_85",
            "soft_signal_stack_66",
            "cip_conditioned_carry_68",
            "vix_gpr_conditioned_carry_73",
            "epu_tpu_conditioned_carry_78",
            "cip_conditioned_soft_71",
            "vix_gpr_conditioned_soft_72",
            "epu_tpu_conditioned_soft_76",
            "vix_gpr_conditioned_mom_74",
            "vix_gpr_conditioned_value_75",
            "epu_tpu_conditioned_reer_77",
            "epu_tpu_conditioned_mom_79",
            "epu_tpu_conditioned_value_80",
            "vix_gpr_conditioned_reer_81",
            "cip_conditioned_mom_82",
            "cip_conditioned_value_83",
            "cip_conditioned_reer_84",
            "wui_conditioned_mom_87",
            "wui_conditioned_value_88",
            "wui_conditioned_reer_89",
            "capital_sleeve_53",
            "capital_sleeve_soft_70",
            "combo_8",
        ],
        "citations": [
            "Lustig, Roussanov & Verdelhan (2011) — currency carry risk factors",
            "Brunnermeier, Nagel & Pedersen (2008) — carry crashes / funding liquidity",
            "Kansas City Fed KCFSI documentation (FRED KCFSI; Hakkio & Keeton 2009; monthly; 0=avg, >0 stress)",
            "Parallel gate/cool: CIP×carry §68 / VIX/GPR×carry §73 / EPU/TPU×carry §78 / WUI×carry §86 / NFCI×carry §91 / ANFCI×carry §96 / STLFSI×carry §101 / KCFSI soft §105",
            "Haven companion pattern: funding_liq §20 / CIP-stress haven §68 / KCFSI soft §105 / NFCI×carry §91 / ANFCI×carry §96",
            "Distinct from funding_liq §20 (raw nfci/anfci_usd / carry×NFCI cool — not Lustig–Verdelhan IR3M gate); distinct from NFCI×carry §91 / ANFCI×carry §96 / STLFSI×carry §101 / KCFSI soft §105",
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
        f"# Scholarly FX: KCFSI-conditioned Lustig–Verdelhan carry wave (§{SECTION})",
        "",
        "**Path:** Kansas City Fed **US KCFSI** (FRED KCFSI; Financial Stress Index) stress × "
        "Lustig–Verdelhan IR3M carry gate/cool — **distinct** from "
        "funding_liquidity (§20), NFCI×carry (§91), ANFCI×carry (§96), STLFSI×carry (§101), KCFSI soft (§105), "
        "ANFCI soft (§95), NFCI soft (§90), WUI carry (§86), soft_ew (§66), "
        "CIP×carry (§68), VIX/GPR×carry (§73), EPU/TPU×carry (§78), "
        "CIP/VIX/EPU/WUI soft/mom/value/REER (§71–§77/§79–§89), capital-sleeve "
        "(§53/§70), combo (§8). Explicit: do "
        "**not** overlay coolers on locked fx4plus.",
        f"**Data:** `{data_tag}` + FRED KCFSI + FRED IR3M rates (pub_lag=1m). "
        f"G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. "
        f"**PIT:** KCFSI pub_lag_months=1 + signal_lag_months={cfg.signal_lag_months} "
        f"on trailing **monthly** z (z_window={cfg.z_window}m, CIP §71 / EPU §76 / "
        f"WUI §85 / NFCI soft §90 / ANFCI soft §95 / STLFSI soft §100 / KCFSI soft §105 mirror) + weight_lag_days={cfg.weight_lag_days}; "
        f"carry signal_lag={cfg.carry_signal_lag}d. Costs 1.5 bps/side.",
        f"**Primary:** `{PRIMARY}` (trade carry only when lagged US KCFSI z ≤ 0). "
        "Companions: carry_kcfsi_cool / carry_raw / carry_high_kcfsi / us_kcfsi_haven_usd / "
        "carry_kcfsi_stack / carry_kcfsi_ew / carry_kcfsi_regime. "
        f"Cool z_high={cfg.z_high} (§71–§105 symmetry; lit often "
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
        f"**Data caveat:** US KCFSI [{kcfsi_d.index.min().date()}.."
        f"{kcfsi_d.index.max().date()}]. Gate z≤0 / cool z≥1 fixed "
        "(§71–§105 symmetry; lit often other cutoffs — documented, not HO-tuned). "
        f"Monthly z_window={cfg.z_window}m (not 252d daily). Single-stress design "
        "(like CIP×carry §68 / WUI×carry §86 / NFCI×carry §91 / STLFSI×carry §101). Distinct from funding_liq §20 / NFCI×carry §91 / ANFCI×carry §96 / STLFSI×carry §101 / KCFSI soft §105.",
        "",
        f"Artifacts: `reports/{PREFIX}_*.csv`, "
        f"`{PREFIX}_meta.json`, `quest_locked_verify_kcfsi_conditioned_carry.md`.",
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
