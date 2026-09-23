#!/usr/bin/env python3
"""Scholarly FX capital-sleeve mix wave (§53).

Pre-registered BEFORE holdout peek (2026-09-24):
  Core = locked fx4plus_gbpcad_d1_voltarget_0025 daily returns (RF=0.08, VT=0.0025)
  Satellite A = bci_chg_xs (§39 best scholarly seed ~+14.8 bp/mo NW t≈+2.34)
  Satellite B = high_ppi_xs (§51 hard board ~+11.6 bp/mo NW t≈+2.00; cross-family
                vs same-family high_bci_z_xs — fixed a priori, not HO-tuned)
  Mix grid (≤6, capital shares sum to 1):
    core_100, core85_bci15, core70_bci30, core60_bci40,
    core80_bci10_ppi10, core70_bci20_ppi10
  Primary = IS-only argmax mean_mo (never HO).
  Combine as separate capital sleeves (NOT overlays/coolers on locked equity).
  Sweep FTMO scale on combined series when IS mean>0.
  Locked tag config untouched. No go-live under approximate_non_ftmo.
"""

from __future__ import annotations

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
from mt5_swing.data.fred_oecd_bci import (
    DEFAULT_PUB_LAG_MONTHS as BCI_PUB_LAG,
    load_oecd_bci_panel,
    bci_coverage,
)
from mt5_swing.data.fred_ppi import (
    DEFAULT_PUB_LAG_MONTHS as PPI_PUB_LAG,
    load_ppi_panel,
    load_us_ppi,
    ppi_coverage,
)
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.portfolio.capital_sleeves import (
    CORE_NAME,
    LOCKED_TAG,
    PRE_REGISTERED_MIXES,
    SATELLITE_A,
    SATELLITE_B,
    equity_to_daily_returns,
    mix_sleeve_returns,
    select_mix_is_only,
    validate_weights,
)
from mt5_swing.portfolio.overlays import INITIAL, apply_vol_target, combine_weighted
from mt5_swing.strategies.oecd_bci_fx import OecdBciFxConfig, oecd_bci_factor_returns
from mt5_swing.strategies.ppi_fx import PpiFxConfig, ppi_factor_returns

# Import eval_windowed_consistency helpers for locked leg equity
spec = importlib.util.spec_from_file_location(
    "ewc", ROOT / "scripts" / "eval_windowed_consistency.py"
)
ewc = importlib.util.module_from_spec(spec)
sys.modules["ewc"] = ewc
assert spec.loader is not None
spec.loader.exec_module(ewc)

HISTORY = ROOT / "data" / "history"
FTMO = ROOT / "data" / "ftmo"
REPORTS = ROOT / "reports"
USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
RF = 0.08
VT = 0.0025
MAX_LOT = 50.0
WARMUP = 250
SECTION = 53
# Primary name filled after IS-only selection; placeholder for meta
PRIMARY_FALLBACK = "core85_bci15"


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
    locked = yaml.safe_load(
        (ROOT / "configs" / "quest_one_pct_candidate.yaml").read_text()
    )
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
        # Windowed locked-verify resets each calendar window; full-span flatten
        # would zero post-breach years and invalidate capital-sleeve mixing.
        # FTMO 10%/5% gates are applied on the *combined* mix series instead.
        bt.flatten_on_breach = False
        bt.max_dd = 1.0
        bt.daily_dd = 1.0
        eq = ewc.run_leg_equity(ohlc, leg, bt)
        # Drop warmup period for evaluation
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
    return r, ds


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    print(f"=== scholarly FX capital-sleeve mix wave (§{SECTION}) ===")
    print(
        f"Pre-registered: core={LOCKED_TAG} RF={RF} VT={VT}; "
        f"sat_A={SATELLITE_A}; sat_B={SATELLITE_B}; "
        f"n_mixes={len(PRE_REGISTERED_MIXES)}. Separate sleeves — not overlays."
    )
    # Confirm locked config present and will not be rewritten
    cfg_path = ROOT / "configs" / "quest_one_pct_candidate.yaml"
    cfg_sha_before = __import__("hashlib").sha256(cfg_path.read_bytes()).hexdigest()

    print("Building locked core daily returns (RF + VT)…")
    try:
        r_core, core_tag = build_locked_daily_returns()
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL locked core build: {exc}")
        meta = {"ok": False, "error": str(exc), "promote": False, "section": SECTION}
        (REPORTS / "scholarly_fx_capital_sleeve_mix_meta.json").write_text(
            json.dumps(meta, indent=2)
        )
        (REPORTS / "scholarly_fx_capital_sleeve_mix_wave.md").write_text(
            f"# Scholarly FX: capital-sleeve mix\n\n**FAILED core:** {exc}\n"
        )
        return 1
    print(
        f"  core n={r_core.dropna().shape[0]} "
        f"[{r_core.dropna().index.min().date()}..{r_core.dropna().index.max().date()}] "
        f"tag={core_tag}"
    )

    pair_ret, _px, data_tag = prefer_ftmo_or_yahoo(USD_MAJORS)
    # Prefer non-ftmo tag consistency with core if no FTMO
    if core_tag == "approximate_non_ftmo":
        data_tag = "approximate_non_ftmo"
    print(f"FX panel: {list(pair_ret.columns)} tag={data_tag}")

    # --- BCI satellite A ---
    try:
        bci_panel = load_oecd_bci_panel(
            pub_lag_months=BCI_PUB_LAG, download=True, force=False
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL BCI load: {exc}")
        return 1
    cov_bci = bci_coverage(bci_panel)
    cov_bci.to_csv(REPORTS / "scholarly_fx_capital_sleeve_mix_bci_coverage.csv", index=False)
    bci_cfg = OecdBciFxConfig()
    bci_factors = oecd_bci_factor_returns(pair_ret, bci_panel, cfg=bci_cfg)
    if SATELLITE_A not in bci_factors:
        print(f"FATAL: {SATELLITE_A} missing from BCI factors {list(bci_factors)}")
        return 1
    r_bci = bci_factors[SATELLITE_A].rename(SATELLITE_A)
    print(
        f"  {SATELLITE_A} n={r_bci.dropna().shape[0]} "
        f"[{r_bci.dropna().index.min().date()}..{r_bci.dropna().index.max().date()}]"
    )

    # --- PPI satellite B ---
    try:
        ppi_panel = load_ppi_panel(pub_lag_months=PPI_PUB_LAG, download=True, force=False)
        us_ppi = load_us_ppi(download=True, force=False)
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL PPI load: {exc}")
        return 1
    cov_ppi = ppi_coverage(ppi_panel)
    cov_ppi.to_csv(REPORTS / "scholarly_fx_capital_sleeve_mix_ppi_coverage.csv", index=False)
    ppi_cfg = PpiFxConfig()
    ppi_factors = ppi_factor_returns(
        pair_ret, ppi_panel, cfg=ppi_cfg, us_ppi_override=us_ppi
    )
    if SATELLITE_B not in ppi_factors:
        print(f"FATAL: {SATELLITE_B} missing from PPI factors {list(ppi_factors)}")
        return 1
    r_ppi = ppi_factors[SATELLITE_B].rename(SATELLITE_B)
    print(
        f"  {SATELLITE_B} n={r_ppi.dropna().shape[0]} "
        f"[{r_ppi.dropna().index.min().date()}..{r_ppi.dropna().index.max().date()}]"
    )

    sleeves = {
        CORE_NAME: r_core,
        SATELLITE_A: r_bci,
        SATELLITE_B: r_ppi,
    }

    # Align mix evaluation to core span (satellites may be shorter)
    board: dict[str, pd.Series] = {}
    for mix_name, wdict in PRE_REGISTERED_MIXES.items():
        validate_weights(wdict)  # a priori sanity
        mixed = mix_sleeve_returns(sleeves, wdict)
        # Restrict to days where core has data (capital always allocated)
        mixed = mixed.reindex(r_core.dropna().index).fillna(0.0)
        board[mix_name] = mixed
        print(f"  mix {mix_name}: w={wdict}")

    # Windows
    end_dt = r_core.dropna().index.max()
    holdout_start = (end_dt - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    holdout_end = end_dt.strftime("%Y-%m-%d")
    full_start = r_core.dropna().index.min().strftime("%Y-%m-%d")
    is_end = (pd.Timestamp(holdout_start, tz="UTC") - pd.Timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )

    # IS-only primary selection (BEFORE using holdout for promote)
    is_means = {}
    for name, port in board.items():
        m_is = monthly_returns(port.loc[full_start:is_end])
        is_means[name] = float(m_is.mean()) if len(m_is) else float("nan")
    primary = select_mix_is_only(is_means, require_positive=True)
    print(f"IS-only primary mix = {primary} (IS mean_mo={is_means[primary]*100:+.3f}%)")
    print("  IS means: " + ", ".join(f"{k}={v*100:+.2f}%" for k, v in is_means.items()))

    summaries = [factor_summary(r, name) for name, r in board.items()]
    pd.DataFrame(summaries).to_csv(
        REPORTS / "scholarly_fx_capital_sleeve_mix_factor_summary.csv", index=False
    )
    # Also sleeve-level coverage summary
    sleeve_summ = [
        factor_summary(sleeves[CORE_NAME], CORE_NAME),
        factor_summary(sleeves[SATELLITE_A], SATELLITE_A),
        factor_summary(sleeves[SATELLITE_B], SATELLITE_B),
    ]
    pd.DataFrame(sleeve_summ).to_csv(
        REPORTS / "scholarly_fx_capital_sleeve_mix_sleeve_summary.csv", index=False
    )
    for name, r in board.items():
        monthly_returns(r).to_csv(
            REPORTS / f"scholarly_fx_capital_sleeve_mix_{name}_monthly.csv",
            header=[name],
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
    win_df.to_csv(
        REPORTS / "scholarly_fx_capital_sleeve_mix_window_stats.csv", index=False
    )

    any_pos_is = any(
        np.isfinite(v) and v > 0 for v in is_means.values()
    )
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
        print("  skip risk sweep: no positive IS mean on any mix")
    sweep_df = pd.DataFrame(sweep_rows)
    if len(sweep_df):
        sweep_df.to_csv(
            REPORTS / "scholarly_fx_capital_sleeve_mix_risk_sweep.csv", index=False
        )

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

    cfg_sha_after = __import__("hashlib").sha256(cfg_path.read_bytes()).hexdigest()
    locked_untouched = cfg_sha_before == cfg_sha_after

    meta = {
        "ok": True,
        "section": SECTION,
        "path": "capital_sleeve_mix_locked_plus_bci_ppi",
        "data_tag": data_tag,
        "rf": RF,
        "port_vol_target": VT,
        "locked_tag": LOCKED_TAG,
        "locked_sleeve_untouched": locked_untouched,
        "locked_cfg_sha256": cfg_sha_after,
        "satellite_a": SATELLITE_A,
        "satellite_b": SATELLITE_B,
        "satellite_b_rationale": (
            "high_ppi_xs pre-registered over high_bci_z_xs for cross-family "
            "diversification (§51 hard NW t≈+2.00); fixed a priori, no HO tune"
        ),
        "mixes": {k: dict(v) for k, v in PRE_REGISTERED_MIXES.items()},
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
        "bci_pub_lag_months": BCI_PUB_LAG,
        "ppi_pub_lag_months": PPI_PUB_LAG,
        "soft_best": soft_best,
        "primary_sweep": prim_sweep.iloc[0].to_dict() if len(prim_sweep) else None,
        "citations": [
            "OECD BCI §39 bci_chg_xs soft/hard seed",
            "OECD PPI §51 high_ppi_xs hard board (Dahlquist–Hasseltoft)",
            "Capital sleeves (not overlays) — Frankie steering 2026-09",
        ],
        "distinct_from": [
            "overlay_coolers_on_locked",
            "oecd_bci_39_alone",
            "oecd_ppi_51_alone",
            "scholarly_combo_equal_weight_sleeves",
        ],
    }
    (REPORTS / "scholarly_fx_capital_sleeve_mix_meta.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )

    lines = [
        f"# Scholarly FX: capital-sleeve mix wave (§{SECTION})",
        "",
        "**Path:** Keep locked `fx4plus_gbpcad_d1_voltarget_0025` as **core** capital; "
        f"allocate smaller fixed FTMO risk fractions to pre-registered scholarly satellites "
        f"`{SATELLITE_A}` (§39) and `{SATELLITE_B}` (§51). **Separate capital sleeves** — "
        "NOT overlays/coolers on locked equity.",
        f"**Data:** `{data_tag}` + free FRED OECD BCI / PPI. "
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
        "Artifacts: `reports/scholarly_fx_capital_sleeve_mix_*.csv`, "
        "`scholarly_fx_capital_sleeve_mix_meta.json`.",
        "",
    ]
    (REPORTS / "scholarly_fx_capital_sleeve_mix_wave.md").write_text("\n".join(lines))

    print("=== mix summary ===")
    for s in summaries:
        print(
            f"  {s['factor']}: mean_mo={s['mean_mo']*100:+.3f}% "
            f"NW_t={s['tstat_nw']:+.2f} %pos={s['pct_pos_mo']*100:.0f}%"
        )
    print(
        f"board n={len(board)} soft={soft} hard={hard} "
        f"promote_unscaled={promote} scaled_primary={scaled_promote} "
        f"primary={primary} locked_untouched={locked_untouched}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
