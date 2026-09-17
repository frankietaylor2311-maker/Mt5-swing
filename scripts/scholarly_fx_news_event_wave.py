#!/usr/bin/env python3
"""Scholarly FX news/event wave: geopolitics intensity → USD vs EUR/GBP/JPY/AUD.

1. Resolve intensity: prefer GDELT DOC TimelineVol; if blocked/short → GPR daily
   spike proxy (documented).
2. Event study: high-intensity days vs random controls; USD-vs-ccy + basket.
3. Lagged long-USD rule only if event−control edge clears fixed prior gate;
   year/holdout with costs + FTMO gates. No holdout tuning. Do not claim 1%/mo.
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
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.data.news_events import resolve_news_intensity
from mt5_swing.strategies.news_event_fx import (
    NewsEventStudyConfig,
    lagged_usd_event_strategy_returns,
    run_news_event_study,
    usd_basket_returns,
    usd_vs_ccy_returns,
)

HISTORY = ROOT / "data" / "history"
REPORTS = ROOT / "reports"
USD_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
INITIAL = 100_000.0


def load_pair_panel(symbols: list[str] = USD_PAIRS) -> pd.DataFrame:
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


def ftmo_gates(r: pd.Series, initial: float = INITIAL) -> bool:
    if r.dropna().empty:
        return False
    eq = (1.0 + r.fillna(0.0)).cumprod() * initial
    m = compute_metrics(eq, initial_equity=initial, periods_per_year=252.0)
    return bool(m.gates_pass)


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
            "gates_pass": False,
            "clears_1pct": False,
        }
    arr = m.to_numpy(dtype=float)
    nw, _ = newey_west_tstat(arr)
    mean_mo = float(m.mean())
    pct_pos = float((m > 0).mean())
    t3 = top3_share(m)
    return {
        "start": start,
        "end": end,
        "n_months": int(len(m)),
        "mean_mo": mean_mo,
        "pct_pos": pct_pos,
        "top3": t3,
        "tstat_ols": ols_tstat(arr),
        "tstat_nw": nw,
        "gates_pass": ftmo_gates(sl),
        "clears_1pct": bool(mean_mo >= 0.01 and pct_pos >= 0.70 and (not np.isfinite(t3) or t3 <= 0.55)),
    }


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    print("=== scholarly news/event wave: resolve intensity ===")
    resolved = resolve_news_intensity(prefer_gdelt=True, try_rss=True, force=False)
    intensity = resolved.series
    print(f"source={resolved.meta.source} available={resolved.meta.available} n={resolved.meta.n_obs}")
    print(f"note={resolved.meta.note}")
    for a in resolved.attempted:
        print(f"  attempted {a.source}: available={a.available} n={a.n_obs} | {a.note[:120]}")

    pair_ret = load_pair_panel()
    if pair_ret.empty or pair_ret.dropna(how="all").shape[0] < 200:
        print("ERROR: insufficient FX history")
        return 1

    cfg = NewsEventStudyConfig(
        pre=5,
        post=10,
        decile=0.90,
        min_gap_days=5,
        extra_lag=1,
        signal_horizon=5,
        min_edge=0.0,
        min_t=1.0,
        hold_days=5,
        signal_lag=1,
        cost_bps_per_side=1.5,
    )
    print("=== event study ===")
    study = run_news_event_study(pair_ret, intensity, cfg=cfg)
    signal = study["signal"]
    meta = study["meta"]
    print(f"n_events={meta['n_events']} n_controls={meta['n_controls']} thr={meta['threshold']:.4g}")
    print(f"signal_gate trade={signal['trade']} diff={signal['diff']:.6f} t={signal['tstat']:.3f} | {signal['reason']}")

    # Persist per-horizon summaries
    frames = []
    for name, df in study["summaries"].items():
        d = df.copy()
        d.insert(0, "series", name)
        frames.append(d)
    summary_all = pd.concat(frames, ignore_index=True)
    summary_all.to_csv(REPORTS / "scholarly_fx_news_event_study.csv", index=False)
    study["events"].to_csv(REPORTS / "scholarly_fx_news_event_dates.csv", index=False)

    # Snapshot at key horizons
    snap_rows = []
    for name, df in study["summaries"].items():
        for h in (0, 5, 10):
            row = df.loc[df["horizon"] == h]
            if row.empty:
                continue
            r = row.iloc[0].to_dict()
            r["series"] = name
            snap_rows.append(r)
    snap = pd.DataFrame(snap_rows)
    snap.to_csv(REPORTS / "scholarly_fx_news_event_snapshots.csv", index=False)

    # Trading rule (only if gate passes; also record force=False path)
    port = lagged_usd_event_strategy_returns(pair_ret, intensity, cfg=cfg, force_trade=None)
    trade_on = bool(port.attrs.get("trade_enabled", False))

    # Always also compute conditional path for honesty table if gate failed
    port_forced = lagged_usd_event_strategy_returns(pair_ret, intensity, cfg=cfg, force_trade=True)

    end = pair_ret.index.max()
    windows = [
        ("full", str(pair_ret.index.min().date()), str(end.date())),
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026", "2026-01-01", str(end.date())),
        ("holdout_365d", str((end - pd.Timedelta(days=365)).date()), str(end.date())),
    ]

    win_rows = []
    for label, a, b in windows:
        series = port if trade_on else port * 0.0
        st = window_stats(series, a, b)
        st["window"] = label
        st["strategy"] = "news_usd_event_gated"
        st["trade_enabled"] = trade_on
        win_rows.append(st)
        # diagnostic: forced rule (not for promotion)
        stf = window_stats(port_forced, a, b)
        stf["window"] = label
        stf["strategy"] = "news_usd_event_forced_diag"
        stf["trade_enabled"] = True
        win_rows.append(stf)

    win_df = pd.DataFrame(win_rows)
    win_df.to_csv(REPORTS / "scholarly_fx_news_event_windows.csv", index=False)

    # Monthly returns artifact
    m_gate = monthly_returns(port if trade_on else port * 0.0)
    m_force = monthly_returns(port_forced)
    pd.DataFrame({"gated": m_gate, "forced_diag": m_force}).to_csv(
        REPORTS / "scholarly_fx_news_event_monthly_returns.csv"
    )

    # Markdown board
    lines: list[str] = []
    lines.append("# Scholarly FX news/event wave\n")
    lines.append(f"**Date:** 2026-09-17 BST\n")
    lines.append(f"**Intensity source:** `{resolved.meta.source}`\n")
    lines.append(f"**Note:** {resolved.meta.note}\n")
    lines.append(f"**Data:** `approximate_non_ftmo` Yahoo D1 for {USD_PAIRS}. ")
    lines.append(f"Costs: {cfg.cost_bps_per_side} bps/side on turnover. `signal_lag={cfg.signal_lag}`, ")
    lines.append(f"hold={cfg.hold_days}d, top-decile thinned events.\n\n")

    lines.append("## Feed attempts\n\n")
    lines.append("| source | available | n_obs | note |\n|---|---|---:|---|\n")
    for a in resolved.attempted:
        note = a.note.replace("|", "/").replace("\n", " ")[:160]
        lines.append(f"| {a.source} | {a.available} | {a.n_obs} | {note} |\n")
    lines.append("\n")

    lines.append("## Event vs control (USD strengthens = +)\n\n")
    lines.append(f"Events: **{meta['n_events']}** (≥{cfg.decile:.0%}ile lagged intensity / top-decile, ≥{cfg.min_gap_days}d apart). ")
    lines.append(f"Controls: **{meta['n_controls']}** random non-event days.\n\n")
    lines.append("| series | h | event_cum | control_cum | diff | t |\n|---|---:|---:|---:|---:|---:|\n")
    for name in list(cfg.currencies) + ["USD_BASKET"]:
        df = study["summaries"][name]
        for h in (0, 5, 10):
            row = df.loc[df["horizon"] == h]
            if row.empty:
                continue
            r = row.iloc[0]
            lines.append(
                f"| {name} | {h} | {100*r['event_cum_mean']:.3f}% | {100*r['control_cum_mean']:.3f}% | "
                f"{100*r['diff_event_minus_control']:.3f}% | {r['tstat_diff']:.2f} |\n"
            )
    lines.append("\n")

    lines.append("## Signal gate (fixed prior)\n\n")
    lines.append(
        f"At h=+{signal['horizon']}: diff={100*signal['diff']:.3f}%, t={signal['tstat']:.2f} → "
        f"**trade={'YES' if signal['trade'] else 'NO'}** ({signal['reason']}).\n\n"
    )

    lines.append("## Lagged USD rule (costs on)\n\n")
    if not trade_on:
        lines.append(
            "Gate **failed** → gated strategy returns are flat (no trades). "
            "Forced diagnostic below is **not** for promotion.\n\n"
        )
    lines.append("| window | strategy | mean_mo | %pos | top3 | t_nw | gates | clears 1%/70%/top3? |\n")
    lines.append("|---|---|---:|---:|---:|---:|:---:|:---:|\n")
    for _, r in win_df.iterrows():
        lines.append(
            f"| {r['window']} | {r['strategy']} | {100*r['mean_mo']:.3f}% | {100*r['pct_pos']:.1f}% | "
            f"{100*r['top3'] if np.isfinite(r['top3']) else float('nan'):.1f}% | {r['tstat_nw']:.2f} | "
            f"{'PASS' if r['gates_pass'] else 'FAIL'} | {'yes' if r['clears_1pct'] else '**no**'} |\n"
        )
    lines.append("\n")

    joint = trade_on and any(
        win_df[(win_df.strategy == "news_usd_event_gated") & (win_df.window == w)]["clears_1pct"].any()
        for w in ("2024", "2025", "2026", "holdout_365d")
    )
    lines.append(f"**Joint clear / promote:** **{'YES' if joint else 'NO'}.** ")
    lines.append("Do **not** claim 1%/mo.\n\n")

    lines.append("## Paid NLP?\n\n")
    lines.append(
        "**Yes for full news NLP.** Free GDELT DOC is rate-limited / short-history; "
        "RSS counts are recent-only; GPR spike is a newspaper-share *proxy*, not signed "
        "entity/sentiment NLP. A paid API (RavenPack, Refinitiv News Analytics, Bloomberg, "
        "or GDELT Cloud keyed history) would be needed for multi-year multilingual NLP panels.\n\n"
    )

    lines.append("## Artifacts\n\n")
    lines.append(
        "- `scholarly_fx_news_event_study.csv`, `scholarly_fx_news_event_snapshots.csv`, "
        "`scholarly_fx_news_event_windows.csv`, `scholarly_fx_news_event_monthly_returns.csv`, "
        "`scholarly_fx_news_event_meta.json`\n"
    )

    (REPORTS / "scholarly_fx_news_event_wave.md").write_text("".join(lines), encoding="utf-8")

    meta_json = {
        "intensity_source": resolved.meta.source,
        "intensity_note": resolved.meta.note,
        "intensity_n": resolved.meta.n_obs,
        "paid_nlp_needed": True,
        "attempted": [
            {
                "source": a.source,
                "available": a.available,
                "n_obs": a.n_obs,
                "note": a.note,
            }
            for a in resolved.attempted
        ],
        "n_events": meta["n_events"],
        "n_controls": meta["n_controls"],
        "threshold": meta["threshold"],
        "signal": signal,
        "trade_enabled": trade_on,
        "cfg": {
            "decile": cfg.decile,
            "hold_days": cfg.hold_days,
            "signal_lag": cfg.signal_lag,
            "cost_bps_per_side": cfg.cost_bps_per_side,
            "signal_horizon": cfg.signal_horizon,
            "min_t": cfg.min_t,
        },
        "data_source": "approximate_non_ftmo",
        "joint_clear": bool(joint),
        "claim_1pct_mo": False,
    }
    (REPORTS / "scholarly_fx_news_event_meta.json").write_text(
        json.dumps(meta_json, indent=2, default=str), encoding="utf-8"
    )
    print("=== wrote reports/scholarly_fx_news_event_* ===")
    print(f"joint_clear={joint} trade_enabled={trade_on}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
