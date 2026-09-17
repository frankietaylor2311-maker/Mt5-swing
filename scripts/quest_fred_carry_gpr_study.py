#!/usr/bin/env python3
"""FRED OECD cross-sectional carry + Caldara–Iacoviello GPR/VIX regime study.

Literature priors only (Lustig–Roussanov–Verdelhan; Menkhoff et al. 2012 carry &
momentum; Caldara–Iacoviello 2022 GPR). Fixed n_long=n_short=2 — NO grids.
Does NOT touch locked fx4plus_gbpcad_d1_voltarget_0025 overlays.

Data: approximate_non_ftmo Yahoo D1 + cached FRED/GPR/VIX. Not go-live.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mt5_swing.backtest.metrics import compute_metrics
from mt5_swing.data.fred_rates import load_currency_rates
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.data.macro_uncertainty import load_gpr, load_vix
from mt5_swing.strategies.carry_rank import (
    CarryRankConfig,
    carry_weights_from_rates,
    currency_weights_to_pair_weights,
    expand_weights_to_daily,
    portfolio_returns_from_weights,
)
from mt5_swing.strategies.fred_carry import FredCarry
from mt5_swing.strategies.fx_momentum import FxMomentumBasket, FxMomentumConfig
from mt5_swing.strategies.gpr_regime import GprRegimeConfig, apply_regime_to_weights

HISTORY = ROOT / "data" / "history"
REPORTS = ROOT / "reports"
MACRO = ROOT / "data" / "macro"

USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
CURRENCIES = ["USD", "EUR", "GBP", "AUD", "NZD", "JPY", "CAD", "CHF"]
INITIAL = 10_000.0
SIGNAL_LAG = 1
N_LONG = 2
N_SHORT = 2
DATA_SOURCE = "approximate_non_ftmo"


def load_pair_closes(symbols: list[str] = USD_MAJORS) -> pd.DataFrame:
    closes: dict[str, pd.Series] = {}
    for sym in symbols:
        path = HISTORY / f"{sym}_D1.csv"
        if not path.exists():
            print(f"SKIP missing {path.name}", flush=True)
            continue
        df = load_ohlc_csv(path, symbol=sym, timeframe="D1")
        closes[sym] = df["close"]
    if not closes:
        raise FileNotFoundError("No USD-major D1 history under data/history/")
    px = pd.DataFrame(closes).sort_index().dropna(how="all")
    if px.index.tz is None:
        px.index = px.index.tz_localize("UTC")
    else:
        px.index = px.index.tz_convert("UTC")
    return px


def load_rates_ffilled(*, pub_lag_months: int = 1) -> pd.DataFrame:
    """FRED OECD immediate rates with publication lag + ffill for sparse tails.

    EUR / CHF / NZD FRED series end earlier than USD/GBP/JPY/AUD/CAD. After the
    mandatory ``pub_lag_months`` shift, trailing months are NaN. We **forward-fill**
    the last known lagged print so the cross-section remains rankable (stale-rate
    caveat documented in the report). No backward-fill; leading NaNs stay NaN.
    """
    rates = load_currency_rates(
        CURRENCIES,
        freq="M",
        pub_lag_months=pub_lag_months,
        download=False,
    )
    before = rates.isna().sum().to_dict()
    rates = rates.ffill()
    rates.attrs["ffill_sparse_after_lag"] = True
    rates.attrs["ffill_na_before"] = before
    rates.attrs["ffill_note"] = (
        "Forward-filled trailing NaNs on sparse EUR/CHF/NZD (and any early gaps) "
        "AFTER publication lag; leading NaNs unchanged."
    )
    return rates


def monthly_returns(r: pd.Series) -> pd.Series:
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return eq.resample("ME").last().pct_change().dropna()


def equity_from_returns(r: pd.Series, initial: float = INITIAL) -> pd.Series:
    return (1.0 + r.fillna(0.0)).cumprod() * float(initial)


def window_slice(r: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return r.loc[(r.index >= start) & (r.index <= end)]


def eval_window(name: str, r: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    sl = window_slice(r, start, end)
    if sl.empty or sl.notna().sum() < 5:
        return {
            "sleeve": name,
            "window": str(getattr(start, "year", start)),
            "label": "",
            "start": str(start.date()),
            "end": str(end.date()),
            "n_days": int(len(sl)),
            "n_months": 0,
            "mean_mo": float("nan"),
            "pct_pos_months": float("nan"),
            "total_ret": float("nan"),
            "static_dd": float("nan"),
            "max_dd_p2t": float("nan"),
            "max_daily_dd": float("nan"),
            "gates_pass": False,
            "ann_sharpe": float("nan"),
        }
    eq = equity_from_returns(sl, INITIAL)
    m = monthly_returns(sl)
    metrics = compute_metrics(
        eq,
        trades=None,
        max_dd_gate=0.10,
        daily_dd_gate=0.05,
        periods_per_year=252.0,
        daily_tz="Europe/Prague",
        initial_equity=INITIAL,
    )
    d = sl.dropna()
    sharpe = (
        float(d.mean() / d.std(ddof=1) * np.sqrt(252))
        if len(d) > 5 and d.std(ddof=1) > 0
        else float("nan")
    )
    return {
        "sleeve": name,
        "window": "",
        "label": "",
        "start": str(start.date()),
        "end": str(end.date()),
        "n_days": int(len(sl)),
        "n_months": int(len(m)),
        "mean_mo": float(m.mean()) if len(m) else float("nan"),
        "pct_pos_months": float((m > 0).mean()) if len(m) else float("nan"),
        "total_ret": float(metrics.total_return),
        "static_dd": float(metrics.static_loss_from_initial),
        "max_dd_p2t": float(metrics.max_drawdown),
        "max_daily_dd": float(metrics.max_daily_dd),
        "gates_pass": bool(metrics.gates_pass),
        "ann_sharpe": sharpe,
    }


def calendar_windows(index: pd.DatetimeIndex) -> list[tuple[str, str, pd.Timestamp, pd.Timestamp]]:
    """Named research windows with clear labels (not holdout-tuned)."""
    end = index.max()
    holdout_start = end - pd.Timedelta(days=365)
    wins: list[tuple[str, str, pd.Timestamp, pd.Timestamp]] = []
    for year, lab in (
        (2024, "calendar_2024 (research/WF)"),
        (2025, "calendar_2025 (research/WF)"),
        (2026, "calendar_2026_YTD (partial / mixed)"),
    ):
        s = pd.Timestamp(f"{year}-01-01", tz="UTC")
        e = pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC")
        if year == 2026:
            e = end
        if e < index.min() or s > end:
            continue
        s = max(s, index.min())
        e = min(e, end)
        wins.append((str(year), lab, s, e))
    wins.append(
        (
            "holdout_365d",
            "pure holdout last ~365 calendar days (no tuning)",
            holdout_start,
            end,
        )
    )
    return wins


def optional_fred_carry_sleeve(px: pd.DataFrame) -> pd.Series | None:
    """Equal-weight single-pair FredCarry signals → simple return sleeve (cheap)."""
    rets = []
    for sym in px.columns:
        try:
            df = pd.DataFrame({"close": px[sym], "signal_close": px[sym]})
            df.attrs["symbol"] = sym
            strat = FredCarry(
                symbol=sym,
                min_diff=0.25,
                monthly_lag=1,
                require_trend_agree=True,
                trend_fast=48,
                trend_slow=120,
            )
            sig = strat.generate_signals(df)
            # Position known prior bar → today's return
            r = px[sym].pct_change().fillna(0.0) * sig.shift(1).fillna(0).astype(float)
            rets.append(r.rename(sym))
        except Exception as exc:  # noqa: BLE001 — optional sleeve
            print(f"fred_carry skip {sym}: {exc}", flush=True)
    if not rets:
        return None
    panel = pd.concat(rets, axis=1)
    port = panel.mean(axis=1)
    port.name = "fred_carry_ew"
    return port


def build_sleeves(px: pd.DataFrame) -> dict[str, pd.Series]:
    pair_ret = px.pct_change()
    rates = load_rates_ffilled(pub_lag_months=1)
    # Restrict rates to overlap with price history (with warmup for ranking)
    rates = rates.loc[rates.index >= (px.index.min() - pd.DateOffset(months=3))]

    cfg = CarryRankConfig(n_long=N_LONG, n_short=N_SHORT, signal_lag=SIGNAL_LAG)
    ccy_w = carry_weights_from_rates(rates, cfg=cfg)
    pair_w = currency_weights_to_pair_weights(ccy_w)
    # Drop pairs we do not have prices for
    pair_w = pair_w[[c for c in pair_w.columns if c in pair_ret.columns]]
    daily_w = expand_weights_to_daily(pair_w, pair_ret.index, signal_lag=cfg.signal_lag)
    carry_r = portfolio_returns_from_weights(daily_w, pair_ret)
    carry_r.name = "carry_rank"

    gpr = load_gpr(freq="D", download=False, pub_lag_days=1)
    vix = load_vix(download=False, pub_lag_days=1)
    regime_cfg = GprRegimeConfig(
        z_window=252,
        z_high=1.0,
        z_low=0.0,
        cool=0.35,
        signal_lag=1,
        use_gpr=True,
        use_vix=True,
        usd_tilt=0.15,
        min_periods=60,
    )
    daily_w_reg = apply_regime_to_weights(daily_w, gpr, vix, cfg=regime_cfg)
    carry_gpr_r = portfolio_returns_from_weights(daily_w_reg, pair_ret)
    carry_gpr_r.name = "carry_rank_gpr_vix"

    mom_cfg = FxMomentumConfig(
        formation_days=63,
        skip_days=21,
        n_long=N_LONG,
        n_short=N_SHORT,
        signal_lag=SIGNAL_LAG,
    )
    mom = FxMomentumBasket(
        formation_days=mom_cfg.formation_days,
        skip_days=mom_cfg.skip_days,
        n_long=mom_cfg.n_long,
        n_short=mom_cfg.n_short,
        signal_lag=mom_cfg.signal_lag,
    )
    mom_r = mom.portfolio_returns(pair_ret[list(px.columns)])
    mom_r.name = "fx_momentum"

    sleeves: dict[str, pd.Series] = {
        "carry_rank": carry_r,
        "carry_rank_gpr_vix": carry_gpr_r,
        "fx_momentum": mom_r,
    }
    fc = optional_fred_carry_sleeve(px)
    if fc is not None:
        sleeves["fred_carry_ew"] = fc
    return sleeves


def promote_decision(rows: list[dict]) -> tuple[bool, str]:
    """Honest FTMO-consistency bar: ~1% mean month + high hit-rate + gates.

    Literature carry/momentum alone are not expected to clear this; default NO.
    """
    by = {}
    for r in rows:
        by.setdefault(r["sleeve"], {})[r["window"]] = r
    reasons = []
    any_ok = False
    for sleeve, wins in by.items():
        ho = wins.get("holdout_365d")
        y24 = wins.get("2024")
        y25 = wins.get("2025")
        if not ho:
            continue
        ok = (
            bool(ho.get("gates_pass"))
            and float(ho.get("mean_mo") or 0) >= 0.009
            and float(ho.get("pct_pos_months") or 0) >= 0.65
            and (y24 is None or (float(y24.get("mean_mo") or 0) >= 0.0 and bool(y24.get("gates_pass"))))
            and (y25 is None or (float(y25.get("mean_mo") or 0) >= 0.0 and bool(y25.get("gates_pass"))))
        )
        if ok:
            any_ok = True
            reasons.append(f"{sleeve}: holdout clears soft 1%/mo-ish bar")
        else:
            reasons.append(
                f"{sleeve}: NO — HO mean_mo={ho.get('mean_mo')} pos={ho.get('pct_pos_months')} "
                f"gates={ho.get('gates_pass')}"
            )
    if not any_ok:
        return False, "; ".join(reasons) + (
            ". Literature FX carry/momentum premia ≠ FTMO ~1%/month consistency; "
            "do not promote on approximate_non_ftmo."
        )
    return True, "; ".join(reasons)


def fmt_pct(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{100.0 * float(x):.2f}%"


def write_report(rows: list[dict], sleeves: dict[str, pd.Series], rates: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = REPORTS / "quest_fred_carry_gpr_study.csv"
    md_path = REPORTS / "quest_fred_carry_gpr_study.md"
    df.to_csv(csv_path, index=False)

    promote, promote_reason = promote_decision(rows)
    na_before = rates.attrs.get("ffill_na_before", {})

    lines = [
        "# Quest: FRED OECD carry + GPR/VIX regime (+ FX momentum)",
        "",
        f"**Generated:** {pd.Timestamp.now(tz='UTC').tz_convert('Europe/London'):%Y-%m-%d %H:%M %Z}  ",
        f"**Data source:** `{DATA_SOURCE}` (Yahoo D1) + FRED OECD immediate rates + "
        "Caldara–Iacoviello GPR + Yahoo VIX  ",
        f"**Initial equity:** {INITIAL:,.0f}  ",
        "**Priors (fixed, no grid):** `n_long=n_short=2`, monthly rebalance, "
        f"`signal_lag={SIGNAL_LAG}`, FRED `pub_lag_months=1`, GPR/VIX lag 1 day, "
        "momentum formation=63 / skip=21  ",
        "**Locked sleeve:** not modified (`fx4plus_gbpcad_d1_voltarget_0025` untouched)  ",
        f"**Promote:** **{'YES' if promote else 'NO'}** — {promote_reason}",
        "",
        "## Literature",
        "",
        "- Lustig, Roussanov & Verdelhan — cross-sectional FX carry / dollar–HML factors",
        "- Menkhoff, Sarno, Schmeling & Schrimpf (2012a) *JF* — carry & global FX volatility",
        "- Menkhoff et al. (2012b) *JFE* — currency momentum",
        "- Caldara & Iacoviello (2022) *AER* — geopolitical risk (GPR)",
        "",
        "## Point-in-time / ffill note",
        "",
        "Rates are loaded with `pub_lag_months=1` then **forward-filled** so sparse",
        "trailing NaNs on EUR / CHF / NZD (FRED series end earlier than USD/GBP/etc.)",
        "do not empty the cross-section. Leading NaNs are not back-filled.",
        f"NaN counts before ffill (full panel): `{na_before}`.",
        "Stale last-prints for CHF/NZD/EUR after their final FRED observation are a",
        "documented limitation — not an invitation to invent rates.",
        "",
        "## Windows",
        "",
        "| Window key | Label |",
        "|------------|-------|",
    ]
    # Deduce labels from first sleeve's rows order
    seen = set()
    for r in rows:
        key = r["window"]
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"| `{key}` | {r['label']} |")

    lines += [
        "",
        "## Results by sleeve × window",
        "",
        "| Sleeve | Window | Mean mo | %pos mo | Total ret | Static DD | Daily DD | Gates | Sharpe |",
        "|--------|--------|---------|---------|-----------|-----------|----------|-------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['sleeve']} | {r['window']} | {fmt_pct(r['mean_mo'])} | "
            f"{fmt_pct(r['pct_pos_months'])} | {fmt_pct(r['total_ret'])} | "
            f"{fmt_pct(r['static_dd'])} | {fmt_pct(r['max_daily_dd'])} | "
            f"{'PASS' if r['gates_pass'] else 'FAIL'} | "
            f"{r['ann_sharpe'] if np.isfinite(r['ann_sharpe']) else float('nan'):.2f} |"
        )

    lines += [
        "",
        "## Sleeve coverage",
        "",
    ]
    for name, s in sleeves.items():
        lines.append(
            f"- `{name}`: {s.dropna().index.min().date()} → {s.dropna().index.max().date()} "
            f"({s.notna().sum()} days)"
        )

    lines += [
        "",
        "## Interpretation / go-live",
        "",
        "- This is a **research replication** of scholarly FX factors on approximate_non_ftmo data.",
        "- Academic carry/momentum Sharpe is typically well below what a stable ≥1%/month",
        "  FTMO hit-rate implies; crash risk in risk-off remains.",
        "- GPR/VIX cooling may dampen stress periods; it is **not** insurance.",
        "- **Do not go live** without FTMO MT5 history exports matching challenge symbols/feed.",
        "",
        f"CSV: `{csv_path.name}`",
        "",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)
    print(f"PROMOTE={promote}", flush=True)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    MACRO.mkdir(parents=True, exist_ok=True)

    print("=== load D1 USD majors ===", flush=True)
    px = load_pair_closes()
    print(f"pairs={list(px.columns)} bars={len(px)} "
          f"{px.index.min().date()}→{px.index.max().date()}", flush=True)

    print("=== build sleeves (carry / carry+GPR-VIX / momentum / optional fred_carry) ===", flush=True)
    rates = load_rates_ffilled(pub_lag_months=1)
    sleeves = build_sleeves(px)

    wins = calendar_windows(px.index)
    rows: list[dict] = []
    for sleeve_name, r in sleeves.items():
        print(f"--- sleeve {sleeve_name} ---", flush=True)
        for key, label, start, end in wins:
            row = eval_window(sleeve_name, r, start, end)
            row["window"] = key
            row["label"] = label
            rows.append(row)
            print(
                f"  {key:14s} mean_mo={fmt_pct(row['mean_mo']):>8s} "
                f"pos={fmt_pct(row['pct_pos_months']):>7s} "
                f"staticDD={fmt_pct(row['static_dd']):>7s} "
                f"gates={'PASS' if row['gates_pass'] else 'FAIL'}",
                flush=True,
            )

    write_report(rows, sleeves, rates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
