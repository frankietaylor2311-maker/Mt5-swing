# Consistency @ 7.5% risk — preferred FX4 (OOS-Sharpe weights)

**Locked basket** from `configs/best_interim_approximate.yaml`  
**Legs:** USDCHF H4 `bbands_reversion`, GBPUSD H4 `breakout_donchian`, CADJPY H4 `mean_reversion_regime`, AUDCAD H4 `mean_reversion_regime`  
**Settings:** `risk_fraction=0.075`, `max_lot=50` (so ATR sizing actually scales), weights ∝ OOS Sharpe from config, exits/vol_target per leg from config. **No param retune.**

**Method:** independent calendar / holdout windows with **250-bar warmup** each (not full-span equity slicing — that path goes flat and understates period P&L). Gates = FTMO 2-Step style (static loss &lt;10%, daily loss &lt;5%).

---

## 1) Calendar-year / holdout slices

| Period | Return | Peak-to-trough DD | Gates | Overlap label |
|--------|-------:|------------------:|:-----:|---------------|
| **2024** (full) | **+2.67%** | **5.63%** | **PASS** | research / WF overlap — **not virgin OOS** |
| **2025** (full) | **+11.73%** | **2.87%** | **PASS** | mixed: H1 research/WF, H2 → holdout |
| **2026 YTD** (~Jan–Sep) | **+14.56%** | **3.48%** | **PASS** | pure holdout (partial year) |
| **Holdout 365d** | **+2.18%** | **4.76%** | **PASS** | pure holdout — **official confirmation** |

Static / daily loss figures were not separately tabulated in the parent windowed run; all listed periods **PASS** both gates. Sharpe and trade counts were not reported for these independent windows.

### Half-year returns (same method)

| Half | Return | Label |
|------|-------:|-------|
| 2024 H1 | +0.09% | research / WF |
| 2024 H2 | +5.97% | research / WF |
| 2025 H1 | +6.65% | research / WF |
| 2025 H2 | +5.38% | mixed → holdout |

**2023 stub:** H4 history starts ~2023-11-30 — too short for a meaningful calendar-year slice after warmup; omitted.

**Overlap legend**

- **research / WF overlap:** params were selected on pre-holdout bars; 2024 and much of early–mid 2025 are **in-sample relative to that fit** (walk-forward OOS folds exist inside research, but the calendar year is not a virgin holdout).
- **pure holdout:** last 365 days of available H4 (and 2026 YTD) — confirmation only; never used to pick params or weights.

---

## 2) Monthly consistency

### (a) Holdout 365d @ 7.5%

| Stat | Value |
|------|------:|
| Mean monthly return | +0.17% |
| Median | +0.06% |
| Stdev | 0.99% |
| % positive months | 62% |
| Best month | +2.17% (2026-08) |
| Worst month | −1.59% (2026-01) |
| Top-3 months share of total gains | ≈76% |

**Verdict:** returns are **somewhat bursty** — a small number of strong months (top 3 ≈ three-quarters of positive P&L) dominate; median near zero with a modest positive mean and ~62% up months. Not a smooth grind.

### (b) Full available H4 span (~2023-11 → 2026-09)

Full-span single-equity slicing was **not** used for consistency (goes flat / size-capped artifacts). Prefer the **independent year/half windows** above as the multi-year picture: positive in every full/partial year tested at 7.5% with gates PASS; 2025 and 2026 YTD stronger than 2024 and the official holdout year.

---

## 3) Caveats (honest)

1. **Yahoo `approximate_non_ftmo`** — spreads/commission/session differ from FTMO MT5; not go-live evidence.
2. **H4 depth only ~2.8 years** (from ~Nov 2023) — multi-year sample is thin; 2023 has no usable full year after warmup.
3. **Params fit on pre-holdout** — 2024 through mid-2025 overlap research/WF and are **not virgin out-of-sample**. Only the official **holdout 365d** (and 2026 YTD inside it) is clean confirmation.
4. **Preferred holdout consistency number is +2.18%** (independent window + 250-bar warmup). An earlier **~+16.8%** holdout figure was **no-warmup / holdout-only** and is **not** the preferred consistency number for this report.

---

## Summary for parent

At **rf=7.5% / max_lot=50**, locked FX4 (OOS-Sharpe weights): all year windows and holdout **PASS** gates. Calendar: 2024 +2.67% (p2t 5.63%), 2025 +11.73% (2.87%), 2026YTD +14.56% (3.48%), holdout_365d **+2.18%** (4.76%). Holdout months somewhat bursty (top-3 ≈76% of gains; 62% positive). Use holdout +2.18% not the no-warmup +16.8%. Yahoo≈; short H4 history; 2024–mid2025 not virgin OOS.
