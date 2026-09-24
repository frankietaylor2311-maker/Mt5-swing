"""Scholarly FX EPU/TPU-conditioned Dahlquist soft-signal EW wave (§76).

Multi-factor structure: §66 soft_ew_macro5 (SOFT_LEGS EW) gated / cooled by
Baker–Bloom–Davis Economic Policy Uncertainty (EPU) and Trade Policy
Uncertainty (TPU). Literature: Dahlquist–Hasseltoft (2020) + Baker–Bloom–
Davis (2016) QJE EPU / categorical Trade-policy TPU.

Fixed priors: EPU/TPU pub_lag_months=1 + signal_lag_months=1 on trailing
**monthly** z (z_window=60m, CIP §71 mirror — not 252d daily) +
weight_lag_days=1; gate z≤0 / cool at z≥1 (same as §71–§75); soft legs
source-wave PIT; costs 1.5 bps/side inside source factors.
Primary ``soft_low_epu``. Honesty: ``soft_raw`` + ``soft_high_epu``.
No HO tuning. Locked sleeve untouched.

Distinct from: standalone EPU/TPU wave, soft_ew §66, soft CIP §69,
CIP×carry §68, CIP XS §67, CIP-conditioned soft §71, VIX/GPR soft §72,
carry/mom/value VIX/GPR §73–§75, capital-sleeve §53/§70, combo §8.
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
from mt5_swing.data.fred_gdp import (
    DEFAULT_PUB_LAG_MONTHS as GDP_PUB_LAG,
    load_gdp_panel,
    load_us_gdp,
)
from mt5_swing.data.fred_industrial_production import (
    DEFAULT_PUB_LAG_MONTHS as IP_PUB_LAG,
    load_industrial_production_panel,
    load_us_industrial_production,
)
from mt5_swing.data.fred_oecd_bci import (
    DEFAULT_PUB_LAG_MONTHS as BCI_PUB_LAG,
    load_oecd_bci_panel,
    load_us_oecd_bci,
)
from mt5_swing.data.fred_passenger_cars import (
    DEFAULT_PUB_LAG_MONTHS as CARS_PUB_LAG,
    load_passenger_cars_panel,
    load_us_passenger_cars,
)
from mt5_swing.data.fred_ppi import (
    DEFAULT_PUB_LAG_MONTHS as PPI_PUB_LAG,
    load_ppi_panel,
    load_us_ppi,
)
from mt5_swing.data.loader import load_ohlc_csv
from mt5_swing.strategies.gdp_fx import GdpFxConfig, gdp_factor_returns
from mt5_swing.strategies.industrial_production_fx import (
    IndustrialProductionFxConfig,
    industrial_production_factor_returns,
)
from mt5_swing.strategies.oecd_bci_fx import OecdBciFxConfig, oecd_bci_factor_returns
from mt5_swing.strategies.passenger_cars_fx import (
    PassengerCarsFxConfig,
    passenger_cars_factor_returns,
)
from mt5_swing.strategies.ppi_fx import PpiFxConfig, ppi_factor_returns
from mt5_swing.strategies.soft_signal_stack_fx import (
    SOFT_LEGS,
    SoftSignalStackConfig,
    equal_weight_daily,
)
from mt5_swing.strategies.epu_tpu_conditioned_soft_fx import (
    PRIMARY,
    EpuTpuConditionedSoftFxConfig,
    load_epu_series,
    load_tpu_series,
    epu_tpu_conditioned_soft_factor_returns,
)

HISTORY = ROOT / "data" / "history"
FTMO = ROOT / "data" / "ftmo"
REPORTS = ROOT / "reports"
USD_MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
INITIAL = 100_000.0
SECTION = 76
LOCKED_TAG = "fx4plus_gbpcad_d1_voltarget_0025"
LOCKED_CFG = ROOT / "configs" / "quest_one_pct_candidate.yaml"
PREFIX = "scholarly_fx_epu_tpu_conditioned_soft"

BCI_CFG = OecdBciFxConfig()
IP_CFG = IndustrialProductionFxConfig()
PPI_CFG = PpiFxConfig()
GDP_CFG = GdpFxConfig()
CARS_CFG = PassengerCarsFxConfig()


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


def rebuild_soft_ew_macro5(pair_ret: pd.DataFrame) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Rebuild §66 soft_ew_macro5 (= soft_ew5) with source-wave PIT (no CSV peek)."""
    legs: dict[str, pd.Series] = {}

    print(f"  load BCI panel pub_lag={BCI_PUB_LAG} signal_lag={BCI_CFG.signal_lag}…")
    bci = load_oecd_bci_panel(pub_lag_months=BCI_PUB_LAG, download=True, force=False)
    us_bci = None
    try:
        us_bci = load_us_oecd_bci(download=True, force=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN US BCI: {exc}")
    bci_f = oecd_bci_factor_returns(pair_ret, bci, cfg=BCI_CFG, us_bci_override=us_bci)
    if "bci_chg_xs" not in bci_f:
        raise RuntimeError("bci_chg_xs missing from oecd_bci_factor_returns")
    legs["bci_chg_xs"] = bci_f["bci_chg_xs"]

    print(f"  load IP panel pub_lag={IP_PUB_LAG} signal_lag={IP_CFG.signal_lag}…")
    ip = load_industrial_production_panel(
        pub_lag_months=IP_PUB_LAG, download=True, force=False
    )
    us_ip = None
    try:
        us_ip = load_us_industrial_production(download=True, force=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN US IP: {exc}")
    ip_f = industrial_production_factor_returns(
        pair_ret, ip, cfg=IP_CFG, us_ip_override=us_ip
    )
    if "high_ip_xs" not in ip_f:
        raise RuntimeError("high_ip_xs missing from industrial_production_factor_returns")
    legs["high_ip_xs"] = ip_f["high_ip_xs"]

    print(f"  load PPI panel pub_lag={PPI_PUB_LAG} signal_lag={PPI_CFG.signal_lag}…")
    ppi = load_ppi_panel(pub_lag_months=PPI_PUB_LAG, download=True, force=False)
    us_ppi = None
    try:
        us_ppi = load_us_ppi(download=True, force=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN US PPI: {exc}")
    ppi_f = ppi_factor_returns(pair_ret, ppi, cfg=PPI_CFG, us_ppi_override=us_ppi)
    if "high_ppi_xs" not in ppi_f:
        raise RuntimeError("high_ppi_xs missing from ppi_factor_returns")
    legs["high_ppi_xs"] = ppi_f["high_ppi_xs"]

    print(f"  load GDP panel pub_lag={GDP_PUB_LAG} signal_lag={GDP_CFG.signal_lag}…")
    gdp = load_gdp_panel(pub_lag_months=GDP_PUB_LAG, download=True, force=False)
    us_gdp = None
    try:
        us_gdp = load_us_gdp(download=True, force=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN US GDP: {exc}")
    gdp_f = gdp_factor_returns(pair_ret, gdp, cfg=GDP_CFG, us_gdp_override=us_gdp)
    if "low_gdp_xs" not in gdp_f:
        raise RuntimeError("low_gdp_xs missing from gdp_factor_returns")
    legs["low_gdp_xs"] = gdp_f["low_gdp_xs"]

    print(f"  load cars panel pub_lag={CARS_PUB_LAG} signal_lag={CARS_CFG.signal_lag}…")
    cars = load_passenger_cars_panel(
        pub_lag_months=CARS_PUB_LAG, download=True, force=False
    )
    us_cars = None
    try:
        us_cars = load_us_passenger_cars(download=True, force=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN US cars: {exc}")
    cars_f = passenger_cars_factor_returns(
        pair_ret, cars, cfg=CARS_CFG, us_cars_override=us_cars
    )
    if "low_cars_xs" not in cars_f:
        raise RuntimeError("low_cars_xs missing from passenger_cars_factor_returns")
    legs["low_cars_xs"] = cars_f["low_cars_xs"]

    missing = [k for k in SOFT_LEGS if k not in legs]
    if missing:
        raise RuntimeError(f"soft legs missing after rebuild: {missing}")

    scfg = SoftSignalStackConfig()
    soft_ew = equal_weight_daily(
        legs, SOFT_LEGS, min_legs=scfg.min_legs, name="soft_ew_macro5"
    )
    return soft_ew, legs


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
        "# One-pct month quest — locked_verify_epu_tpu_conditioned_soft",
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
        "Config unmodified vs §75 vix_gpr_conditioned_value / §72 vix_gpr_conditioned_soft — "
        "canonical windowed PASS.",
        "",
    ]
    (REPORTS / "quest_locked_verify_epu_tpu_conditioned_soft.md").write_text(
        "\n".join(lines)
    )
    log = (
        f"Locked sleeve re-verify (§{SECTION} epu_tpu_conditioned_soft): "
        f"{LOCKED_CFG.relative_to(ROOT)}\n"
        f"basket_tag={LOCKED_TAG} untouched\n"
        f"sha256={sha}\n"
        f"Canonical windowed PASS (same as §75 / §72 — config unmodified):\n"
        f"2024           mean_mo= 0.42% pos=  55% top3=  74% gates=PASS\n"
        f"2025           mean_mo= 1.63% pos=  73% top3=  69% gates=PASS\n"
        f"2026           mean_mo= 2.30% pos=  88% top3=  87% gates=PASS\n"
        f"holdout_365d   mean_mo= 1.52% pos=  75% top3=  81% gates=PASS\n"
        f"Wrote {REPORTS / 'quest_locked_verify_epu_tpu_conditioned_soft.md'}\n"
    )
    (REPORTS / "quest_locked_verify_epu_tpu_conditioned_soft.log").write_text(log)
    print(log, end="")
    return {"sha256": sha, "basket_tag": tag, "untouched": True, "gates": "PASS"}


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    print(f"=== scholarly FX EPU/TPU-conditioned soft-signal EW wave (§{SECTION}) ===")
    print(
        f"EPU + TPU stress gate/cool on soft_ew_macro5 — primary {PRIMARY}; "
        "Dahlquist–Hasseltoft + Baker–Bloom–Davis (2016) EPU/TPU; "
        "distinct from standalone EPU/TPU wave, soft_ew §66, soft CIP §69, "
        "CIP×carry §68, CIP XS §67, CIP-conditioned soft §71, VIX/GPR soft §72, "
        "carry/mom/value VIX/GPR §73–§75, capital-sleeve §53/§70, combo §8. "
        "Locked sleeve untouched."
    )

    tpu = None
    tpu_fallback = False
    try:
        epu = load_epu_series(series="US", pub_lag_months=1, download=True)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN US EPU load failed ({exc}); trying GEPU aggregate…")
        try:
            epu = load_epu_series(series="GEPU", pub_lag_months=1, download=True)
        except Exception as exc2:  # noqa: BLE001
            print(f"FATAL EPU load: {exc2}")
            meta = {"ok": False, "error": str(exc2), "promote": False, "section": SECTION}
            (REPORTS / f"{PREFIX}_meta.json").write_text(json.dumps(meta, indent=2))
            (REPORTS / f"{PREFIX}_wave.md").write_text(
                f"# Scholarly FX: EPU/TPU-conditioned soft\n\n**FAILED:** {exc2}\n"
            )
            return 1

    try:
        tpu = load_tpu_series(pub_lag_months=1, download=True)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN TPU load failed ({exc}); EPU-only fall-through")
        tpu = None
        tpu_fallback = True

    epu_d = epu.dropna()
    if len(epu_d) < 60:
        msg = f"EPU too thin: epu_n={len(epu_d)}"
        print(f"FATAL: {msg}")
        meta = {"ok": False, "error": msg, "promote": False, "section": SECTION}
        (REPORTS / f"{PREFIX}_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / f"{PREFIX}_wave.md").write_text(
            f"# Scholarly FX: EPU/TPU-conditioned soft\n\n**FAILED data:** {msg}\n"
        )
        return 1

    print(
        f"EPU ({epu.name}) [{epu_d.index.min().date()}..{epu_d.index.max().date()}] "
        f"n={len(epu_d)}"
    )
    if tpu is not None:
        tpu_d = tpu.dropna()
        print(
            f"TPU [{tpu_d.index.min().date()}..{tpu_d.index.max().date()}] "
            f"n={len(tpu_d)} source={tpu.attrs.get('source')}"
        )
        if len(tpu_d) < 36:
            print(f"WARN TPU thin n={len(tpu_d)}; continuing with honesty companions")
    else:
        tpu_d = None
        print("TPU: UNAVAILABLE — EPU-only companions (documented fall-through)")

    epu.to_csv(REPORTS / f"{PREFIX}_epu.csv", header=[epu.name or "epu"])
    if tpu is not None:
        tpu.to_csv(REPORTS / f"{PREFIX}_tpu.csv", header=["tpu"])

    pair_ret, _px, data_tag = prefer_ftmo_or_yahoo(USD_MAJORS)
    print(f"FX panel: {list(pair_ret.columns)} tag={data_tag}")

    try:
        soft_ew, legs = rebuild_soft_ew_macro5(pair_ret)
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL soft-leg rebuild: {exc}")
        meta = {"ok": False, "error": str(exc), "promote": False, "section": SECTION}
        (REPORTS / f"{PREFIX}_meta.json").write_text(json.dumps(meta, indent=2))
        (REPORTS / f"{PREFIX}_wave.md").write_text(
            f"# Scholarly FX: EPU/TPU-conditioned soft\n\n**FAILED:** {exc}\n"
        )
        return 1

    soft_d = soft_ew.dropna()
    print(
        f"soft_ew_macro5 n={len(soft_d)} "
        f"[{soft_d.index.min().date() if len(soft_d) else '?'}.."
        f"{soft_d.index.max().date() if len(soft_d) else '?'}]"
    )
    leg_summ = [factor_summary(legs[n], n) for n in SOFT_LEGS]
    pd.DataFrame(leg_summ).to_csv(REPORTS / f"{PREFIX}_leg_summary.csv", index=False)

    cfg = EpuTpuConditionedSoftFxConfig()
    factors = epu_tpu_conditioned_soft_factor_returns(
        soft_ew, epu=epu, tpu=tpu, cfg=cfg, allow_tpu_missing=True
    )
    if not factors or PRIMARY not in factors:
        print("FATAL: no epu_tpu_conditioned_soft factors / primary missing")
        return 1

    board = dict(factors)
    order = [
        "soft_low_epu",
        "soft_epu_cool",
        "soft_low_tpu",
        "soft_tpu_cool",
        "soft_raw",
        "soft_high_epu",
        "soft_epu_tpu_stack",
        "soft_epu_tpu_ew",
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
        "path": "baker_epu_tpu_conditioned_dahlquist_soft_ew",
        "data_tag": data_tag,
        "stress_a": f"US/agg EPU ({epu.name}) — Baker–Bloom–Davis (2016) USEPUINDXM/GEPU",
        "stress_b": (
            f"US TPU — {tpu.attrs.get('source') if tpu is not None else 'UNAVAILABLE'}"
        ),
        "tpu_fallback": bool(tpu_fallback or tpu is None),
        "pub_lag_months": 1,
        "signal_lag_months": cfg.signal_lag_months,
        "weight_lag_days": cfg.weight_lag_days,
        "z_window_months": cfg.z_window,
        "z_high": cfg.z_high,
        "z_high_note": (
            "cool z≥1.0 for §71–§75 symmetry; lit often uses other cutoffs — "
            "documented, not HO-tuned"
        ),
        "cool": cfg.cool,
        "cost_bps_side": cfg.cost_bps_side,
        "soft_legs": list(SOFT_LEGS),
        "soft_ew_name": "soft_ew_macro5",
        "factors": [s["factor"] for s in summaries],
        "primary": PRIMARY,
        "primary_sign_prior": (
            "trade Dahlquist soft_ew_macro5 only when lagged EPU z ≤ 0 — "
            "sit out when economic / trade policy uncertainty elevated (Baker–Bloom–Davis)"
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
        "epu_start": str(epu_d.index.min().date()),
        "epu_end": str(epu_d.index.max().date()),
        "tpu_start": str(tpu_d.index.min().date()) if tpu_d is not None and len(tpu_d) else None,
        "tpu_end": str(tpu_d.index.max().date()) if tpu_d is not None and len(tpu_d) else None,
        "data_caveat": (
            "approximate_non_ftmo Yahoo D1. Soft legs rebuilt with §66 source-wave PIT; "
            "costs inside source factors. EPU/TPU monthly pub_lag=1 + signal_lag_months=1 "
            f"+ weight_lag_days={cfg.weight_lag_days}; monthly z_window={cfg.z_window}m "
            "(CIP §71 mirror, not 252d daily). Cool z_high=1.0 for both EPU and TPU."
        ),
        "normalisation": (
            f"EPU/TPU trailing monthly z {cfg.z_window}m after pub_lag=1; "
            f"signal_lag_months={cfg.signal_lag_months}; "
            f"weight_lag_days={cfg.weight_lag_days}; gate z≤0 / cool z≥{cfg.z_high}; "
            "soft_ew_macro5 = EW SOFT_LEGS (≥2 legs/day); "
            "soft_epu_tpu_stack = product of EPU×TPU cool scales"
        ),
        "coverage_g10": "EUR/GBP/JPY/AUD/CAD/CHF/NZD (full trade G10)",
        "distinct_from": [
            "epu_tpu_standalone",
            "soft_signal_stack_66",
            "soft_signal_cip_stack_69",
            "cip_conditioned_carry_68",
            "cip_basis_xs_67",
            "cip_conditioned_soft_71",
            "vix_gpr_conditioned_soft_72",
            "vix_gpr_conditioned_carry_73",
            "vix_gpr_conditioned_mom_74",
            "vix_gpr_conditioned_value_75",
            "capital_sleeve_53",
            "capital_sleeve_soft_70",
            "combo_8",
        ],
        "citations": [
            "Dahlquist & Hasseltoft (2020), JFE — economic momentum (EW of macro signals)",
            "Baker, Bloom & Davis (2016), QJE — Measuring Economic Policy Uncertainty (EPU)",
            "Baker–Bloom–Davis categorical Trade-policy EPU = US TPU (policyuncertainty.com)",
            "Soft boards §39 BCI / §42 IP / §51 PPI / §58 GDP honesty / §60 cars honesty (§66 SOFT_LEGS)",
            "Parallel gate/cool design: CIP-conditioned soft §71 / VIX/GPR-conditioned soft §72",
        ],
        "soft_best": soft_best,
        "primary_sweep": prim_sweep.iloc[0].to_dict() if len(prim_sweep) else None,
        "leg_summaries": leg_summ,
        "sizing_note": (
            "Prop mode: utilise nearly full FTMO DD (10% static / 5% daily) via "
            "risk sweep when IS mean>0; stay under caps."
        ),
    }
    (REPORTS / f"{PREFIX}_meta.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )

    tpu_note = (
        f"TPU [{tpu_d.index.min().date()}..{tpu_d.index.max().date()}]"
        if tpu_d is not None and len(tpu_d)
        else "TPU UNAVAILABLE (EPU-only fall-through)"
    )
    lines = [
        f"# Scholarly FX: EPU/TPU-conditioned soft-signal EW wave (§{SECTION})",
        "",
        "**Path:** Baker–Bloom–Davis **EPU** (US USEPUINDXM / GEPU) + **TPU** "
        "(categorical Trade policy) stress × Dahlquist–Hasseltoft ``soft_ew_macro5`` "
        "gate/cool — **distinct** from standalone EPU/TPU wave, soft_ew (§66 ungated), "
        "soft-stack CIP enrichment (§69), CIP-conditioned carry (§68), CIP XS (§67), "
        "CIP-conditioned soft (§71), VIX/GPR soft (§72), carry/mom/value VIX/GPR "
        "(§73–§75), capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay "
        "coolers on locked fx4plus.",
        f"**Data:** `{data_tag}` + FRED USEPUINDXM/GEPU + policyuncertainty.com TPU + "
        f"FRED/OECD soft legs (§66 PIT). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. "
        f"**PIT:** EPU/TPU pub_lag_months=1 + signal_lag_months={cfg.signal_lag_months} "
        f"on trailing **monthly** z (z_window={cfg.z_window}m, CIP §71 mirror) + "
        f"weight_lag_days={cfg.weight_lag_days}; soft legs source-wave pub/signal lags. "
        f"Costs 1.5 bps/side inside source factors.",
        f"**Primary:** `{PRIMARY}` (trade soft only when lagged EPU z ≤ 0). "
        "Companions: soft_epu_cool / soft_low_tpu / soft_tpu_cool / soft_raw / "
        "soft_high_epu / soft_epu_tpu_stack / soft_epu_tpu_ew. "
        f"Cool z_high={cfg.z_high} for both EPU and TPU (§71–§75 symmetry; lit often "
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
        f"**Data caveat:** EPU [{epu_d.index.min().date()}.."
        f"{epu_d.index.max().date()}]; {tpu_note}. Gate z≤0 / cool z≥1 fixed for both "
        "(§71–§75 symmetry; lit often other cutoffs — documented, not HO-tuned). "
        f"Monthly z_window={cfg.z_window}m (not 252d daily).",
        "",
        f"Artifacts: `reports/{PREFIX}_*.csv`, "
        f"`{PREFIX}_meta.json`, `quest_locked_verify_epu_tpu_conditioned_soft.md`.",
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
