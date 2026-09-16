# One-percent per month quest — status

**Data:** `approximate_non_ftmo` (Yahoo via yfinance). **No `data/ftmo/` exports — never golive.**
**Gates:** FTMO 2-Step static max loss 10%, daily 5% (Europe/Prague). `signal_lag=1`. IS-only grids. Holdout never for selection.

Session: 2026-09-16 18:00 BST Europe/London (wave: **smooth-return portfolio + IS min(year mean_mo) WF**).

## Scoring rubric this wave (primary)

1. High **% positive months** (prefer ≥70–75%)
2. Lower **top-3 gain concentration** (prefer ≤~50–55%; reject if HO top3 worsens vs locked)
3. Mean monthly **~1% stable** across 2024, 2025, 2026, holdout — **2024 cannot stay at 0.42%**
4. Prefer **lower monthly variance / less lumpy** mean over higher bursty mean
5. **Reject** candidates that boost mean via bursts even if gates PASS
6. **No leverage increase** (RF fixed at 8%; per-leg risk = RF/n_legs)

**Selection objective (this wave):** maximize `min(IS year mean_mo)` for IS windows `{2024, 2025_IS}` subject to %pos≥70% and top3≤55% (soft≤70%). Holdout / full 2025–2026 confirmation only.

## Target vs result

| Criterion | Target | Locked candidate | Met? |
|-----------|--------|------------------|:----:|
| Mean monthly on eval windows | ≥ ~1.0% | Holdout **+1.52%**; 2025 **+1.63%**; 2026 **+2.30%**; roll12_m6 **+1.13%**; **2024 +0.42%** | **Mostly — 2024 still short** |
| % positive months | ≥ ~70–75% | Holdout **75%**; 2026 **88%**; 2025 **73%**; **2024 55%** | 2024 fails consistency |
| Top-3 months share of gains | ≲ 50–55% | Holdout **~81%**; 2024 **74%** | **No — still bursty** |
| Gates | PASS | All listed windows **PASS** | **Yes** |
| Multi-window, warmup, fixed params | required | Independent windows + 250-bar warmup; params frozen | **Yes** |

**Verdict:** Official locked tag **unchanged**. Smooth many-leg IS search (528 VT/weight combos, 104 soft constraint passers, **0 hard top3≤55% passers**) lifts 2024 toward ~0.8%/mo with better %pos on some lock+diversifier baskets, but **none** clear ≥1% mean on **each** of 2024/2025/2026/holdout with ≥70% pos. Metals not dual-confirm — skipped. **Target not met.**

## Locked candidate (unchanged — still official)

**Tag:** `fx4plus_gbpcad_d1_voltarget_0025`  
**Config:** `configs/quest_one_pct_candidate.yaml`

Verified this wave (`RF=0.08`, `PORT_VOL_TARGET=0.0025`, weights oos_sharpe):

| Window | Return | Mean mo | %pos | Top3 | Gates | P2T |
|--------|-------:|--------:|-----:|-----:|:-----:|----:|
| 2024 | +5.38% | **0.42%** | 55% | 74% | PASS | ~7% |
| 2025 | +22.04% | **1.63%** | 73% | 69% | PASS | ~4% |
| 2026 YTD | +20.03% | **2.30%** | 88% | 87% | PASS | ~6% |
| holdout_365d | +21.55% | **1.52%** | 75% | 81% | PASS | ~7% |
| roll12_m6 | +16.45% | **1.13%** | 67% | 80% | PASS | ~4% |

## Wave: smooth-return from scratch + IS min(year) WF (new)

Script: `scripts/quest_smooth_return_wf.py`. Helpers: `src/mt5_swing/portfolio/smooth_select.py`.

Design:
1. Pool of 18 MR/breakout OOS-gate legs (prefer dual-confirm); hard per-leg risk `RF/n`.
2. Decorrelated / dual-top / MR-only / lock+extra / seeded baskets (22 unique symbol sets).
3. Grid: equal|oos_sharpe × VT∈{0.002…0.005} × clip_hi∈{2,3} → 528 IS evals.
4. Score = `min(mean_mo on 2024, 2025_IS)` with %pos/top3 constraints; holdout never in score.
5. Metals only if dual-confirm **and** improve 2024 without hurting 2025 %pos.

### Best IS soft passer (not promoted)

`lock_plus4|oos_sharpe|vt0.0035|hi3.0` — locked 5 + EURUSD/GBPJPY/NZDCAD/EURCHF.

| Window | Mean mo | %pos | Top3 | Gates |
|--------|--------:|-----:|-----:|:-----:|
| 2024 | **0.81%** | **73%** | **55%** | PASS |
| 2025 | 0.80% | **64%** | 63% | PASS |
| 2026 | 1.33% | 75% | 84% | PASS |
| holdout_365d | 0.63% | **50%** | 84% | PASS |

Closer on 2024 mean/%pos/top3 vs locked, but **2025 %pos and HO %pos regress** — rejected under consistency rubric. IS raw min_mo ≈0.81% < 1%.

### Other confirmation notes

| Idea | 2024 mo/%pos/top3 | 2025 mo/%pos | HO mo/%pos | Note |
|------|------------------:|-------------:|-----------:|------|
| baseline_vt0025 | 0.42 / 55 / 74 | 1.63 / 73 | 1.52 / 75 | official |
| lock_plus4 vt0035 | 0.81 / 73 / 55 | 0.80 / 64 | 0.63 / 50 | best IS soft; HO %pos fail |
| lock_plus3 vt005 | **1.11** / 73 / 56 | **0.64** / 73 | 0.83 / 50 | 2024≥1% but 2025 mean & HO %pos fail |
| pure smooth n8 | ~0.28 / 82 / 59 | ~0.64 / 64 | ~0.57 / 58 | consistency↑ mean↓ |

**Hard top3≤55% on both IS windows:** **0 / 528**. Soft (top3≤70%) passers: **104**.

**Metals:** XAUUSD H4 breakout_donchian OOS-ok but **not dual-confirm** → not eligible for promote path. No index CFDs in history.

**Promote?** **No** — official tag unchanged.

Details: `reports/quest_smooth_return_wf.md`, `reports/quest_smooth_is_board.csv`, `reports/quest_smooth_promote.csv`, `configs/quest_smooth_selected.json`.

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
| Month-aware / equity-curve / runup flatten | Consistency↑ mean↓ — rejected |
| Dual-year ≥0.8% add-ons | **0 passers** |
| Soft +AUDJPY | 2024 %pos↑ 2025 %pos↓ — rejected |
| H1 Yahoo probe | Flat / no edge — rejected |
| **Smooth many-leg + IS min(year) WF (this wave)** | Best IS ~0.81%/mo; HO %pos collapses — **rejected** |
| **Metals dual-confirm (this wave)** | **None eligible** |

## Gap remaining / irreducible Yahoo limits

1. **2024 mean still ~0.4%/mo** on official locked; lock+diversifier IS search reaches ~0.8% with better 2024 %pos but **cannot hold ≥70% pos on holdout**.
2. **Burstiness** — locked HO top3 ~81%; hard top3≤55% on both IS years: **zero** passers in 528-grid.
3. **No dual-year diversifier** that is both strong and sign-stable 2024∧2025 under consistency gates.
4. **No dual-confirm metals**; **no index history** (US30/NAS100/SPX) in repo for extra uncorrelated sleeves.
5. **No FTMO MT5 exports** — all `approximate_non_ftmo`. Spreads/swap/sessions/Yahoo FX ≠ FTMO CFD book.
6. **What FTMO `data/ftmo/` would unlock:** true spreads/commission/swap, session filters, deeper H1/M15, index CFDs if offered, and any go-live path (`ftmo_mt5_export`).

## Process / tests

- pytest: `tests/test_smooth_select.py` (+ prior overlay/clock/lookahead suite).
- `signal_lag=1`; IS grids only (`2024`, `2025_IS` to holdout_start); holdout confirmation.
- RF forced to **8%** (locked); per-leg = RF/n; no leverage hike.
