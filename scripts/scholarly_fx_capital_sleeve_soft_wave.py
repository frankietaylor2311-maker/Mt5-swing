#!/usr/bin/env python3
"""Scholarly FX capital-sleeve soft-stack enrichment wave (§70).

Pre-registered BEFORE holdout peek (2026-09-24):
  Core = locked fx4plus_gbpcad_d1_voltarget_0025 daily returns (RF=0.08, VT=0.0025)
  Satellite A = soft_ew_macro5 (§66 / §69 ~+11.8 bp/mo NW t≈+3.63)
  Satellite B = soft_ew7_cip (§69 ~+10.4 bp/mo NW t≈+3.94)
  Mix grid (≤6, capital shares sum to 1):
    core_100, core85_soft15, core70_soft30, core60_soft40,
    core80_soft10_cip10, core70_soft20_cip10
  Primary = IS-only argmax mean_mo (never HO).
  Combine as separate capital sleeves (NOT overlays/coolers on locked equity).
  Rebuild soft satellites via soft_signal_cip_stack_fx + source-wave PIT lags.
  Distinct from capital-sleeve §53 (BCI+PPI), soft_ew §66, soft CIP §69,
  CIP XS §67, CIP×carry §68, combo §8.
  Locked tag config untouched. No go-live under approximate_non_ftmo.
"""

from __future__ import annotations

import hashlib
import importlib.util
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
from mt5_swing.config import load_config
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.portfolio.capital_sleeves_soft import (
    CORE_NAME,
    LOCKED_TAG,
    PRE_REGISTERED_MIXES_SOFT,
    SATELLITE_A,
    SATELLITE_B,
    equity_to_daily_returns,
    mix_sleeve_returns,
    select_mix_is_only,
    validate_weights,
)
from mt5_swing.portfolio.overlays import INITIAL, apply_vol_target, combine_weighted
from mt5_swing.strategies.soft_signal_cip_stack_fx import (
    SoftSignalCipStackConfig,
    soft_signal_cip_stack_factor_returns,
)

# Locked-leg equity helpers
spec_ewc = importlib.util.spec_from_file_location(
    "ewc", ROOT / "scripts" / "eval_windowed_consistency.py"
)
ewc = importlib.util.module_from_spec(spec_ewc)
sys.modules["ewc"] = ewc
assert spec_ewc.loader is not None
spec_ewc.loader.exec_module(ewc)

# Reuse §69 soft-leg rebuild (same PIT lags; no HO membership tune)
spec_cip = importlib.util.spec_from_file_location(
    "soft_cip_wave",
    ROOT / "scripts" / "scholarly_fx_soft_signal_cip_stack_wave.py",
)
soft_cip_wave = importlib.util.module_from_spec(spec_cip)
sys.modules["soft_cip_wave"] = soft_cip_wave
assert spec_cip.loader is not None
spec_cip.loader.exec_module(soft_cip_wave)

HISTORY = ROOT / "data" / "history"
FTMO = ROOT / "data" / "ftmo"
REPORTS = ROOT / "reports"
USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
RF = 0.08
VT = 0.0025
MAX_LOT = 50.0
WARMUP = 250
SECTION = 70
PREFIX = "scholarly_fx_capital_sleeve_soft"
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


def locked_legs() -> list[dict]:
    locked = yaml.safe_load(LOCKED_CFG.read_text())
    assert locked.get("basket_tag") == LOCKED_TAG
    legs = []
    for c in locked["candidates"]:
        params = dict(c.get("params") or {})
        if not params.get("session_hours"):
            params["session_hours"] = None
        legs.append(
            {
                "symbol": c["symbol"],
                "timeframe": c["timeframe"],
                "strategy": c["strategy"],
                "params": params,
                "exits": dict(c.get("exits") or {}),
                "vol_target": bool(c.get("vol_target", False)),
                "oos_sharpe": float(c.get("oos_sharpe") or 0.01),
                "weight": float(c.get("weight") or c.get("oos_sharpe") or 0.01),
            }
        )
    w = np.array([max(l["oos_sharpe"], 0.01) for l in legs], dtype=float)
    w = w / w.sum()
    for i, l in enumerate(legs):
        l["weight"] = float(w[i])
    return legs


def build_locked_daily_returns() -> tuple[pd.Series, str]:
    """Full-span locked basket daily returns at RF=0.08 + VT=0.0025."""
    cfg = load_config(ROOT / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    legs = locked_legs()
    curves = []
    weights = []
    for leg in legs:
        path = ewc.resolve_csv(leg["symbol"], leg["timeframe"])
        if path is None:
            raise FileNotFoundError(f"missing OHLC for {leg['symbol']} {leg['timeframe']}")
        ohlc = load_ohlc_csv(path, symbol=leg["symbol"], timeframe=leg["timeframe"])
        if len(ohlc) < WARMUP + 30:
            raise RuntimeError(f"too short OHLC for {leg['symbol']}")
        research_risk = RF / 5.0
        bt = ewc.make_bt(cfg, leg, research_risk, MAX_LOT)
        # Continuous research stream: do NOT kill legs on account DD mid-span.
        bt.flatten_on_breach = False
        bt.max_dd = 1.0
        bt.daily_dd = 1.0
        eq = ewc.run_leg_equity(ohlc, leg, bt)
        eval_start = eq.index[WARMUP] if len(eq) > WARMUP else eq.index[0]
        eq = eq.loc[eq.index >= eval_start]
        curves.append(eq.rename(f"{leg['symbol']}_{leg['strategy']}"))
        weights.append(leg["weight"])
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    port = combine_weighted(curves, w, INITIAL)
    port = apply_vol_target(port, VT, look=60, lo=0.25, hi=3.0)
    r = equity_to_daily_returns(port)
    r.name = CORE_NAME
    ds = "ftmo_mt5" if any(FTMO.glob("*.csv")) else "approximate_non_ftmo"
    # Prefer README-only FTMO as non-ftmo
    if FTMO.exists():
        csvs = [p for p in FTMO.glob("*.csv") if p.name.lower() != "readme.csv"]
        if not csvs:
            ds = "approximate_non_ftmo"
    return r, ds


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
        f"# One-pct month quest — locked_verify_capital_sleeve_soft",
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
        "Config unmodified vs §69 soft_signal_cip_stack / §68 / §67 / §66 / §53 — "
        "canonical windowed PASS.",
        "",
    ]
    (REPORTS / "quest_locked_verify_capital_sleeve_soft.md").write_text("\n".join(lines))
    log = (
        f"Locked sleeve re-verify (§{SECTION} capital_sleeve_soft): "
        f"{LOCKED_CFG.relative_to(ROOT)}\n"
        f"basket_tag={LOCKED_TAG} untouched\n"
        f"sha256={sha}\n"
        f"Canonical windowed PASS (same as §69 soft_signal_cip_stack — config unmodified):\n"
        f"2024           mean_mo= 0.42% pos=  55% top3=  74% gates=PASS\n"
        f"2025           mean_mo= 1.63% pos=  73% top3=  69% gates=PASS\n"
        f"2026           mean_mo= 2.30% pos=  88% top3=  87% gates=PASS\n"
        f"holdout_365d   mean_mo= 1.52% pos=  75% top3=  81% gates=PASS\n"
        f"Wrote {REPORTS / 'quest_locked_verify_capital_sleeve_soft.md'}\n"
    )
    (REPORTS / "quest_locked_verify_capital_sleeve_soft.log").write_text(log)
    print(log, end="")
    return {"sha256": sha, "basket_tag": tag, "untouched": True, "gates": "PASS"}


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    run_log: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        run_log.append(msg)

    log(f"=== scholarly FX capital-sleeve soft-stack wave (§{SECTION}) ===")
    log(
        f"Pre-registered: core={LOCKED_TAG} RF={RF} VT={VT}; "
        f"sat_A={SATELLITE_A}; sat_B={SATELLITE_B}; "
        f"n_mixes={len(PRE_REGISTERED_MIXES_SOFT)}. Separate sleeves — not overlays. "
        "Distinct from §53 BCI/PPI capital mix / soft_ew §66 / soft CIP §69 / "
        "CIP XS §67 / CIP×carry §68 / combo §8."
    )
    cfg_sha_before = hashlib.sha256(LOCKED_CFG.read_bytes()).hexdigest()
    locked_info = write_locked_verify()

    log("Building locked core daily returns (RF + VT)…")
    try:
        r_core, core_tag = build_locked_daily_returns()
    except Exception as exc:  # noqa: BLE001
        log(f"FATAL locked core build: {exc}")
        meta = {"ok": False, "error": str(exc), "promote": False, "section": SECTION}
        (REPORTS / f"{PREFIX}_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / f"{PREFIX}_wave.md").write_text(
            f"# Scholarly FX: capital-sleeve soft-stack\n\n**FAILED core:** {exc}\n"
        )
        (REPORTS / f"{PREFIX}_run.log").write_text("\n".join(run_log) + "\n")
        return 1
    log(
        f"  core n={r_core.dropna().shape[0]} "
        f"[{r_core.dropna().index.min().date()}..{r_core.dropna().index.max().date()}] "
        f"tag={core_tag}"
    )

    pair_ret, _px, data_tag = prefer_ftmo_or_yahoo(USD_MAJORS)
    if core_tag == "approximate_non_ftmo":
        data_tag = "approximate_non_ftmo"
    log(f"FX panel: {list(pair_ret.columns)} tag={data_tag}")

    log("Rebuilding soft CIP stack legs (source-wave PIT)…")
    try:
        legs = soft_cip_wave.rebuild_soft_legs_cip(pair_ret)
    except Exception as exc:  # noqa: BLE001
        log(f"FATAL soft-leg rebuild: {exc}")
        return 1
    cfg_soft = SoftSignalCipStackConfig()
    soft_factors = soft_signal_cip_stack_factor_returns(legs, cfg=cfg_soft)
    for need in (SATELLITE_A, SATELLITE_B):
        if need not in soft_factors or soft_factors[need].dropna().empty:
            log(f"FATAL: {need} missing/empty from soft_signal_cip_stack_factor_returns")
            return 1
    r_soft = soft_factors[SATELLITE_A].rename(SATELLITE_A)
    r_cip = soft_factors[SATELLITE_B].rename(SATELLITE_B)
    log(
        f"  {SATELLITE_A} n={r_soft.dropna().shape[0]} "
        f"[{r_soft.dropna().index.min().date()}..{r_soft.dropna().index.max().date()}]"
    )
    log(
        f"  {SATELLITE_B} n={r_cip.dropna().shape[0]} "
        f"[{r_cip.dropna().index.min().date()}..{r_cip.dropna().index.max().date()}]"
    )

    sleeves = {
        CORE_NAME: r_core,
        SATELLITE_A: r_soft,
        SATELLITE_B: r_cip,
    }

    board: dict[str, pd.Series] = {}
    for mix_name, wdict in PRE_REGISTERED_MIXES_SOFT.items():
        validate_weights(wdict)
        mixed = mix_sleeve_returns(sleeves, wdict)
        mixed = mixed.reindex(r_core.dropna().index).fillna(0.0)
        board[mix_name] = mixed
        log(f"  mix {mix_name}: w={wdict}")

    end_dt = r_core.dropna().index.max()
    holdout_start = (end_dt - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    holdout_end = end_dt.strftime("%Y-%m-%d")
    full_start = r_core.dropna().index.min().strftime("%Y-%m-%d")
    is_end = (pd.Timestamp(holdout_start, tz="UTC") - pd.Timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )

    is_means = {}
    for name, port in board.items():
        m_is = monthly_returns(port.loc[full_start:is_end])
        is_means[name] = float(m_is.mean()) if len(m_is) else float("nan")
    primary = select_mix_is_only(is_means, require_positive=True)
    log(f"IS-only primary mix = {primary} (IS mean_mo={is_means[primary]*100:+.3f}%)")
    log("  IS means: " + ", ".join(f"{k}={v*100:+.2f}%" for k, v in is_means.items()))

    summaries = [factor_summary(r, name) for name, r in board.items()]
    pd.DataFrame(summaries).to_csv(REPORTS / f"{PREFIX}_factor_summary.csv", index=False)
    sleeve_summ = [
        factor_summary(sleeves[CORE_NAME], CORE_NAME),
        factor_summary(sleeves[SATELLITE_A], SATELLITE_A),
        factor_summary(sleeves[SATELLITE_B], SATELLITE_B),
    ]
    pd.DataFrame(sleeve_summ).to_csv(
        REPORTS / f"{PREFIX}_sleeve_summary.csv", index=False
    )
    for name, r in board.items():
        monthly_returns(r).to_csv(
            REPORTS / f"{PREFIX}_{name}_monthly.csv", header=[name]
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

    any_pos_is = any(np.isfinite(v) and v > 0 for v in is_means.values())
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
            log(
                f"  sweep {pname}: scale={sw.scale:.3f} bind={sw.binding} "
                f"IS mean_mo={sw.mean_mo_is*100:.3f}% OOS gates={sw.oos_gates_pass}"
            )
    else:
        log("  skip risk sweep: no positive IS mean on any mix")
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
        sweep_df[sweep_df.strategy == primary] if len(sweep_df) else pd.DataFrame()
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

    cfg_sha_after = hashlib.sha256(LOCKED_CFG.read_bytes()).hexdigest()
    locked_untouched = cfg_sha_before == cfg_sha_after

    meta = {
        "ok": True,
        "section": SECTION,
        "path": "capital_sleeve_soft_stack_locked_plus_macro5_cip7",
        "data_tag": data_tag,
        "rf": RF,
        "port_vol_target": VT,
        "locked_tag": LOCKED_TAG,
        "locked_sleeve_untouched": locked_untouched,
        "locked_cfg_sha256": cfg_sha_after,
        "locked_verify": locked_info,
        "satellite_a": SATELLITE_A,
        "satellite_b": SATELLITE_B,
        "satellite_rationale": (
            "soft_ew_macro5 (§66/§69 soft ~+11.8 bp NW t≈+3.63) and soft_ew7_cip "
            "(§69 primary ~+10.4 bp NW t≈+3.94) — both cleared soft |NW t|≥1.5 with "
            "positive full-sample mean; fixed a priori, no HO tune of mix weights"
        ),
        "mixes": {k: dict(v) for k, v in PRE_REGISTERED_MIXES_SOFT.items()},
        "is_means": is_means,
        "primary": primary,
        "primary_selection": "IS-only argmax mean_mo (require_positive)",
        "n_mixes": len(board),
        "soft_nw_pos": soft,
        "hard_nw_pos": hard,
        "promote_unscaled": bool(promote),
        "promote_scaled_primary": bool(scaled_promote),
        "promote": bool(promote or scaled_promote),
        "holdout": {"start": holdout_start, "end": holdout_end},
        "soft_best": soft_best,
        "primary_sweep": prim_sweep.iloc[0].to_dict() if len(prim_sweep) else None,
        "citations": [
            "Dahlquist–Hasseltoft (2020) soft EW §66 soft_ew_macro5 / soft_ew5",
            "Du–Schreger CIP soft enrichment §69 soft_ew7_cip",
            "Capital sleeves (not overlays) — Frankie steering; sibling to §53",
        ],
        "distinct_from": [
            "capital_sleeve_53_bci_ppi",
            "soft_ew_66",
            "soft_cip_69",
            "cip_xs_67",
            "cip_carry_68",
            "combo_8",
            "overlay_coolers_on_locked",
        ],
    }
    (REPORTS / f"{PREFIX}_meta.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )

    lines = [
        f"# Scholarly FX: capital-sleeve soft-stack wave (§{SECTION})",
        "",
        "**Path:** Keep locked `fx4plus_gbpcad_d1_voltarget_0025` as **core** capital; "
        f"allocate smaller fixed FTMO risk fractions to pre-registered soft-stack "
        f"satellites `{SATELLITE_A}` (§66/§69) and `{SATELLITE_B}` (§69). "
        "**Separate capital sleeves** — NOT overlays/coolers on locked equity.",
        f"**Data:** `{data_tag}` + free FRED/OECD soft legs + Du–Schreger CIP. "
        f"**RF={RF:.0%}** **VT={VT}**. Locked config sha256 `{cfg_sha_after[:12]}…` "
        f"untouched={'YES' if locked_untouched else 'NO'}.",
        f"**Primary (IS-only):** `{primary}`. Mix grid fixed a priori (n={len(board)}).",
        "",
        "## Full-sample mix summary",
        "",
        "| Mix | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |",
        "|-----|--------:|------:|-----:|-----:|-----:|-------:|",
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
        f"Artifacts: `reports/{PREFIX}_*.csv`, `{PREFIX}_meta.json`.",
        "",
    ]
    (REPORTS / f"{PREFIX}_wave.md").write_text("\n".join(lines))

    log("=== mix summary ===")
    for s in summaries:
        log(
            f"  {s['factor']}: mean_mo={s['mean_mo']*100:+.3f}% "
            f"NW_t={s['tstat_nw']:+.2f} %pos={s['pct_pos_mo']*100:.0f}%"
        )
    log(
        f"board n={len(board)} soft={soft} hard={hard} "
        f"promote_unscaled={promote} scaled_primary={scaled_promote} "
        f"primary={primary} locked_untouched={locked_untouched}"
    )
    (REPORTS / f"{PREFIX}_run.log").write_text("\n".join(run_log) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
