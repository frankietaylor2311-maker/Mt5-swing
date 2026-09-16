# One-percent per month quest — status

**Data:** `approximate_non_ftmo` (Yahoo via yfinance). **No `data/ftmo/` exports — never golive.**
**Gates:** FTMO 2-Step static max loss 10%, daily 5% (Europe/Prague). `signal_lag=1`. IS-only grids. Holdout never for selection.

Session: 2026-09-16 17:54 UTC+01:00 Europe/London (wave: **monthly-consistency-first** — vol/equity/month-aware flatten + dual-year diversify + H1).

## Scoring rubric this wave (primary)

1. High **% positive months** (prefer ≥70–75%)
2. Lower **top-3 gain concentration** (prefer ≤~50–55%; reject if HO top3 worsens vs locked)
3. Mean monthly **~1% stable** across 2024, 2025, 2026, holdout — **2024 cannot stay at 0.42%**
4. Prefer **lower monthly variance / less lumpy** mean over higher bursty mean
5. **Reject** candidates that boost mean via bursts even if gates PASS
6. **No leverage increase** (RF fixed at 8%; new flatten overlays scale ≤1)

## Target vs result

| Criterion | Target | Locked candidate | Met? |
|-----------|--------|------------------|:----:|
| Mean monthly on eval windows | ≥ ~1.0% | Holdout **+1.52%**; 2025 **+1.63%**; 2026 **+2.30%**; roll12_m6 **+1.13%**; **2024 +0.42%** | **Mostly — 2024 still short** |
| % positive months | ≥ ~70–75% | Holdout **75%**; 2026 **88%**; 2025 **73%**; **2024 55%** | 2024 fails consistency |
| Top-3 months share of gains | ≲ 50–55% | Holdout **~81%**; 2024 **74%** | **No — still bursty** |
| Gates | PASS | All listed windows **PASS** | **Yes** |
| Multi-window, warmup, fixed params | required | Independent windows + 250-bar warmup; params frozen | **Yes** |

**Verdict:** Official locked tag **unchanged**. Consistency overlays can lift 2024 %pos 55%→73% and trim top3 a bit, but they **cut mean further below 1%**. Dual-year (≥0.8%/mo on **both** 2024 and 2025) add-ons: **zero passers**. Soft explores (+AUDJPY) help 2024 %pos but hurt 2025 %pos. **Target not met.**

## Locked candidate (unchanged — still official)

**Tag:** `fx4plus_gbpcad_d1_voltarget_0025`  
**Config:** `configs/quest_one_pct_candidate.yaml`

Verified this wave (`QUEST_CFG=... RISK_FRACTION=0.08 WEIGHTS=oos_sharpe PORT_VOL_TARGET=0.0025`):

| Window | Return | Mean mo | %pos | Top3 | Gates | P2T |
|--------|-------:|--------:|-----:|-----:|:-----:|----:|
| 2024 | +5.38% | **0.42%** | 55% | 74% | PASS | 7.4% |
| 2025 | +22.04% | **1.63%** | 73% | 69% | PASS | 3.9% |
| 2026 YTD | +20.03% | **2.30%** | 88% | 87% | PASS | 6.4% |
| holdout_365d | +21.55% | **1.52%** | 75% | 81% | PASS | 6.8% |
| roll12_m6 | +16.45% | **1.13%** | 67% | 80% | PASS | 3.9% |

## Wave: consistency-first flatten + dual-year + H1 (new)

Script: `scripts/quest_vol_month_diversify.py`. Overlays in `src/mt5_swing/portfolio/overlays.py`:
`apply_month_aware_scale`, `apply_equity_curve_target`, `apply_runup_throttle` (all causal, scale≤1).

### Overlay / clip results (locked legs, RF=8%)

| Idea | 2024 mo / %pos / top3 | 2025 mo / %pos | HO mo / %pos / top3 | Notes |
|------|----------------------:|---------------:|--------------------:|-------|
| baseline_vt0025 | 0.42 / 55 / 74 | 1.63 / 73 | 1.52 / 75 / 81 | official |
| vt0025 clip_hi=1.5 | 0.25 / **73** / **67** | 0.82 / 73 | 1.00 / 75 / 80 | best consistency lift; mean too low |
| vt0025 clip_hi=2.0 | 0.28 / 64 / 71 | 1.09 / 73 | 1.27 / **83** / 79 | HO %pos up; 2024 still <<1% |
| vt0025 clip_hi=1.0 | 0.18 / 73 / 67 | 0.54 / 73 | 0.72 / 75 / 82 | over-flattened |
| eq_only target_mo_vol=0.6–1.0% | ≤0.17 / 73 / ~65 | ≤0.49 | ≤0.56 | IS consistency score winners; kill mean |
| month-aware only | ~0.16–0.18 / 73 / 67 | ~0.5 | ~0.69 | same tradeoff |
| month-aware **on top of** vt | often worsens %pos/top3 | — | — | rejected (bursty after throttle) |
| consistency_stack | 0.10 / 73 / 68 | — | — | too weak mean |

**Promote overlays?** **No** — none reach ≥1% mean on 2024 while keeping multi-year stability; flatteners trade mean for %pos.

### Dual-year diversifier screen

Require dual-confirm + corr≤0.55 + **2024 AND 2025 mean_mo ≥ 0.8%** + gates before add.

**Passers: 0 / 12.** Classic pattern: GBPJPY H4 squeeze 2024 +1.42% / 2025 **−1.20%**; EURCHF 2024 −0.47% / 2025 +1.35%; EURGBP 2024 +0.71% / 2025 −0.17%.

Soft explore (not promote path; both years positive but <<0.8%):

| Basket | 2024 mo/%pos/top3 | 2025 mo/%pos | HO mo/%pos/top3 |
|--------|------------------:|-------------:|----------------:|
| +AUDJPY D1 MR | 0.46 / **73** / 71 | 1.60 / **64** | 1.54 / 83 / 80 |
| +NZDCAD H4 MR | 0.42 / 55 / 76 | 1.64 / 64 | 1.14 / 67 / 83 |
| +both | 0.46 / 73 / 77 | 1.56 / 64 | 1.12 / 67 / 80 |

+AUDJPY helps 2024 %pos but **regresses 2025 %pos 73→64** — rejected under consistency rubric.

### H1 probe

Downloaded H1 (Yahoo ~730d closed bars) for EURUSD/GBPUSD/USDJPY/EURJPY/AUDUSD/USDCHF. Fixed a-priori presets produced near-zero trades / flat equity on most combos at RF=8% (history starts ~2023-11; 2024 calendar incomplete). **No H1 leg promoted.**

Details: `reports/quest_vol_month_diversify.md`, `reports/quest_dualyear_screen.csv`, `reports/quest_soft_dualyear_explore.csv`, `configs/quest_vol_month_selected.json`.

## Ideas tried (cumulative)

| Idea | Result |
|------|--------|
| New strats TSMOM/vol-breakout/carry/Kalman/ORB | Weak OOS; many holdout fails; carry ~0 trades |
| Two-leg pairs MR | Negative once real spreads+ATR sizing |
| FX5/FX6 H4 & mix @8% | HO ~0.5–0.8%/mo; weak 2024 |
| Risk-parity / higher risk 10–12% | Flatten or non-monotonic vs kill-switch |
| Clip vol-scale 1.5–2.0 | **%pos↑ on 2024** but mean↓ — not enough for 1% |
| Regime sleeves | 2024↑ HO↓ — rejected |
| Corr / hotstreak / ADX tilt | Marginal vs locked; not promoted |
| 2024-positive dual add-only | 2024↑ 2025↓ — rejected |
| D1/H4 clock + max-k recycle | 2024→1.18%/mo but HO top3 worse — not promoted |
| **Month-aware / equity-curve / runup flatten (this wave)** | Consistency↑ mean↓ — rejected |
| **Dual-year ≥0.8% add-ons (this wave)** | **0 passers** |
| **Soft +AUDJPY (this wave)** | 2024 %pos↑ 2025 %pos↓ — rejected |
| **H1 Yahoo probe (this wave)** | Flat / no edge — rejected |

## Gap remaining

1. **2024 mean still ~0.4%/mo** on official locked; flatteners that fix 2024 %pos cannot reach 1% mean without new uncorrelated edge.
2. **Burstiness** — locked HO top3 ~81%; need ≤55% without killing mean (clip_hi=1.5 only trims to ~67–80%).
3. **No dual-year diversifier** survives 2024∧2025 ≥0.8%/mo.
4. **No FTMO MT5 exports** — all `approximate_non_ftmo`.
5. Next: (a) build legs optimized for **monthly hit-rate** on IS (not OOS Sharpe alone); (b) purged multi-year panel with explicit top3/%pos constraints in WF; (c) FTMO exports when available.

## Process / tests

- pytest: `tests/test_month_equity_overlays.py`, `tests/test_clock_recycle.py` (+ prior suite).
- `signal_lag=1`; IS grids only; holdout confirmation.
- Overlay hyperparams scored on **2024 only** with consistency-first rubric; holdout never for selection.
- RF forced to **8%** (locked); no leverage hike.
