# One-percent per month quest — status

**Data:** `approximate_non_ftmo` (Yahoo via yfinance). **No `data/ftmo/` exports — never golive.**
**Gates:** FTMO 2-Step static max loss 10%, daily 5% (Europe/Prague). `signal_lag=1`. IS-only grids. Holdout never for selection.

Session: 2026-09-17 00:52 BST (wave: **causal daily loss-streak cool** on locked sleeve ± frozen HO-robust wrebal / MTD).

## Scoring rubric this wave (primary)

1. High **% positive months** (prefer ≥70–75%)
2. Lower **top-3 gain concentration** (prefer ≤~50–55%; reject if HO top3 worsens vs locked)
3. Mean monthly **~1% stable** across 2024, 2025, 2026, holdout — **2024 cannot stay at 0.42%**
4. Prefer **lower monthly variance / less lumpy** mean over higher bursty mean
5. **Reject** candidates that boost mean via bursts even if gates PASS
6. **No leverage increase** (RF fixed at 8%; overlay `hi≤1`)

**Selection objective:** maximize `min(IS year mean_mo)` for `{2024, 2025_IS}` subject to %pos≥70% and top3≤55% (soft≤70%; soft also mean≥~0.95%). Holdout / full 2025–2026 **confirmation only** — promote only if holdout **also** clears ≥1% mean and ≥70% pos **and** each of 2024/2025/2026 clears the same.

## Target vs result

| Criterion | Target | Locked candidate | Met? |
|-----------|--------|------------------|:----:|
| Mean monthly on eval windows | ≥ ~1.0% | Holdout **+1.52%**; 2025 **+1.63%**; 2026 **+2.30%**; roll12_m6 **+1.13%**; **2024 +0.42%** | **Mostly — 2024 still short** |
| % positive months | ≥ ~70–75% | Holdout **75%**; 2026 **88%**; 2025 **73%**; **2024 55%** | 2024 fails consistency |
| Top-3 months share of gains | ≲ 50–55% | Holdout **~81%**; 2024 **74%** | **No — still bursty** |
| Gates | PASS | All listed windows **PASS** | **Yes** |
| Multi-window, warmup, fixed params | required | Independent windows + 250-bar warmup; params frozen | **Yes** |

**Verdict:** Official locked tag **unchanged**. Loss-streak wave: board **n=30**, soft=**4**, hard=**0**, promote=**0**. Best soft `wrebal+mtd_t0.015_a0.5` (HO %pos 67%; lstreak nested nearly inert). Lstreak alone lifts 2024 %pos at aggressive n=2 but tanks HO %pos. **Promote: NO.** Still blocked on FTMO CSVs.

## Locked candidate (unchanged — still official)

**Tag:** `fx4plus_gbpcad_d1_voltarget_0025`  
**Config:** `configs/quest_one_pct_candidate.yaml`

Re-verified this wave (`RF=0.08`, `PORT_VOL_TARGET=0.0025`, weights oos_sharpe; `scripts/eval_windowed_consistency.py` → `quest_locked_verify_keepalive_0052`):

| Window | Return | Mean mo | %pos | Top3 | Gates | P2T |
|--------|-------:|--------:|-----:|-----:|:-----:|----:|
| 2024 | +5.38% | **0.42%** | 55% | 74% | PASS | ~7% |
| 2025 | +22.04% | **1.63%** | 73% | 69% | PASS | ~4% |
| 2026 YTD | +20.0%+ | **2.30%** | 88% | 87% | PASS | ~6% |
| holdout_365d | +21.5%+ | **1.52%** | 75% | 81% | PASS | ~7% |
| roll12_m6 | +16.4%+ | **1.13%** | 67% | 80% | PASS | ~4% |

## Wave: daily loss-streak cool (this)

Scripts/helpers:
- `scripts/quest_loss_streak_consistency_wf.py`
- `apply_daily_loss_streak_cool` in `overlays.py` (daily consec. downs; lag-1; hi≤1)
- Soft mean floor 0.0095 both IS windows; HO never for selection

### Board

| Family | Notes |
|--------|-------|
| baseline / lstreak alone | 16-cell grid — no soft; aggressive n=2 hurts HO %pos |
| frozen_wrebal ± lstreak | is_fail |
| frozen_wrebal_mtd ± lstreak | **4 soft** — all holdout_fail (HO %pos 67%) |
| is_blend | soft_mean_fail (2024 < 0.95%) |

**Totals:** soft=**4** hard=**0** promote=**0** (board n=30).

### Best soft (not promote) — `wrebal+mtd_t0.015_a0.5_frozen`

| Window | Mean mo | %pos | Top3 | Role |
|--------|--------:|-----:|-----:|------|
| 2024 | **1.18%** | **82%** | **45%** | IS — soft OK |
| 2025_IS | **1.74%** | **75%** | **64%** | IS — hard top3 fail |
| holdout | **1.69%** | **67%** | **71%** | confirm — holdout_fail |

**Promote: NO.** Locked tag unchanged.

---

## Prior session log (preserved)

# One-percent per month quest — status

**Data:** `approximate_non_ftmo` (Yahoo via yfinance). **No `data/ftmo/` exports — never golive.**
**Gates:** FTMO 2-Step static max loss 10%, daily 5% (Europe/Prague). `signal_lag=1`. IS-only grids. Holdout never for selection.

Session: 2026-09-17 00:30 BST (wave: **causal trailing downside-vol scale** on locked sleeve ± frozen HO-robust wrebal / MTD).

## Scoring rubric this wave (primary)

1. High **% positive months** (prefer ≥70–75%)
2. Lower **top-3 gain concentration** (prefer ≤~50–55%; reject if HO top3 worsens vs locked)
3. Mean monthly **~1% stable** across 2024, 2025, 2026, holdout — **2024 cannot stay at 0.42%**
4. Prefer **lower monthly variance / less lumpy** mean over higher bursty mean
5. **Reject** candidates that boost mean via bursts even if gates PASS
6. **No leverage increase** (RF fixed at 8%; overlay `hi≤1`)

**Selection objective:** maximize `min(IS year mean_mo)` for `{2024, 2025_IS}` subject to %pos≥70% and top3≤55% (soft≤70%; soft also mean≥~0.95%). Holdout / full 2025–2026 **confirmation only** — promote only if holdout **also** clears ≥1% mean and ≥70% pos **and** each of 2024/2025/2026 clears the same.

## Target vs result

| Criterion | Target | Locked candidate | Met? |
|-----------|--------|------------------|:----:|
| Mean monthly on eval windows | ≥ ~1.0% | Holdout **+1.52%**; 2025 **+1.63%**; 2026 **+2.30%**; roll12_m6 **+1.13%**; **2024 +0.42%** | **Mostly — 2024 still short** |
| % positive months | ≥ ~70–75% | Holdout **75%**; 2026 **88%**; 2025 **73%**; **2024 55%** | 2024 fails consistency |
| Top-3 months share of gains | ≲ 50–55% | Holdout **~81%**; 2024 **74%** | **No — still bursty** |
| Gates | PASS | All listed windows **PASS** | **Yes** |
| Multi-window, warmup, fixed params | required | Independent windows + 250-bar warmup; params frozen | **Yes** |

**Verdict:** Official locked tag **unchanged**. Downside-vol wave: board **n=26**, soft=**4**, hard=**0**, promote=**0**. Best soft `wrebal+mtd_t0.015_a0.5+ddown_lb42_t0.004` (HO %pos 67%, nearly identical to prior-wave best soft). Ddown alone on locked does not lift 2024. **Promote: NO.** Still blocked on FTMO CSVs.

## Locked candidate (unchanged — still official)

**Tag:** `fx4plus_gbpcad_d1_voltarget_0025`  
**Config:** `configs/quest_one_pct_candidate.yaml`

Re-verified this wave (`RF=0.08`, `PORT_VOL_TARGET=0.0025`, weights oos_sharpe; `scripts/eval_windowed_consistency.py` → `quest_locked_verify_keepalive_0030`):

| Window | Return | Mean mo | %pos | Top3 | Gates | P2T |
|--------|-------:|--------:|-----:|-----:|:-----:|----:|
| 2024 | +5.38% | **0.42%** | 55% | 74% | PASS | ~7% |
| 2025 | +22.04% | **1.63%** | 73% | 69% | PASS | ~4% |
| 2026 YTD | +20.0%+ | **2.30%** | 88% | 87% | PASS | ~6% |
| holdout_365d | +21.5%+ | **1.52%** | 75% | 81% | PASS | ~7% |
| roll12_m6 | +16.4%+ | **1.13%** | 67% | 80% | PASS | ~4% |

## Wave: trailing downside-vol scale (this)

Scripts/helpers:
- `scripts/quest_ddown_vol_consistency_wf.py`
- `apply_trailing_downside_vol_scale` in `overlays.py` (daily downside std; lag-1; hi≤1)
- Soft mean floor 0.0095 both IS windows; HO never for selection

### Board

| Family | Notes |
|--------|-------|
| baseline / ddown alone | 12-cell grid — no soft; 2024 stuck ~0.42% |
| frozen_wrebal ± ddown | soft_mean or holdout_fail |
| frozen_wrebal_mtd ± ddown | **4 soft** — all holdout_fail (HO %pos 67%) |
| is_blend | soft_mean_fail (2024 < 0.95%) |

**Totals:** soft=**4** hard=**0** promote=**0** (board n=26).

### Best soft (not promote) — `wrebal+mtd_t0.015_a0.5_frozen+ddown_lb42_t0.004`

| Window | Mean mo | %pos | Top3 | Role |
|--------|--------:|-----:|-----:|------|
| 2024 | **1.19%** | **82%** | **45%** | IS — soft OK |
| 2025_IS | **1.74%** | **75%** | **64%** | IS — hard top3 fail |
| holdout | **1.61%** | **67%** | **71%** | confirm — holdout_fail |

**Promote: NO.** Locked tag unchanged.

---

## Prior session log (preserved)

# One-percent per month quest — status

**Data:** `approximate_non_ftmo` (Yahoo via yfinance). **No `data/ftmo/` exports — never golive.**
**Gates:** FTMO 2-Step static max loss 10%, daily 5% (Europe/Prague). `signal_lag=1`. IS-only grids. Holdout never for selection.

Session: 2026-09-17 00:00 BST (wave: **causal prior-month win throttle + IS return-blend** on locked sleeve ± frozen HO-robust wrebal).

## Scoring rubric this wave (primary)

1. High **% positive months** (prefer ≥70–75%)
2. Lower **top-3 gain concentration** (prefer ≤~50–55%; reject if HO top3 worsens vs locked)
3. Mean monthly **~1% stable** across 2024, 2025, 2026, holdout — **2024 cannot stay at 0.42%**
4. Prefer **lower monthly variance / less lumpy** mean over higher bursty mean
5. **Reject** candidates that boost mean via bursts even if gates PASS
6. **No leverage increase** (RF fixed at 8%; overlay `hi≤1`)

**Selection objective:** maximize `min(IS year mean_mo)` for `{2024, 2025_IS}` subject to %pos≥70% and top3≤55% (soft≤70%; soft also mean≥~1%). Holdout / full 2025–2026 **confirmation only** — promote only if holdout **also** clears ≥1% mean and ≥70% pos **and** each of 2024/2025/2026 clears the same.

## Target vs result

| Criterion | Target | Locked candidate | Met? |
|-----------|--------|------------------|:----:|
| Mean monthly on eval windows | ≥ ~1.0% | Holdout **+1.52%**; 2025 **+1.63%**; 2026 **+2.30%**; roll12_m6 **+1.13%**; **2024 +0.42%** | **Mostly — 2024 still short** |
| % positive months | ≥ ~70–75% | Holdout **75%**; 2026 **88%**; 2025 **73%**; **2024 55%** | 2024 fails consistency |
| Top-3 months share of gains | ≲ 50–55% | Holdout **~81%**; 2024 **74%** | **No — still bursty** |
| Gates | PASS | All listed windows **PASS** | **Yes** |
| Multi-window, warmup, fixed params | required | Independent windows + 250-bar warmup; params frozen | **Yes** |

**Verdict:** Official locked tag **unchanged**. Prior-win + blend wave: board **n=21**, soft=**4**, hard=**0**, promote=**0**. Best soft remains `wrebal+mtd_t0.015_a0.5` (HO %pos 67%). Blends that lift HO %pos to 75% crush 2024 mean below soft floor. **Promote: NO.** Still blocked on FTMO CSVs.

## Locked candidate (unchanged — still official)

**Tag:** `fx4plus_gbpcad_d1_voltarget_0025`  
**Config:** `configs/quest_one_pct_candidate.yaml`

Re-verified this wave (`RF=0.08`, `PORT_VOL_TARGET=0.0025`, weights oos_sharpe; `scripts/eval_windowed_consistency.py` → `quest_locked_verify_keepalive_0000`):

| Window | Return | Mean mo | %pos | Top3 | Gates | P2T |
|--------|-------:|--------:|-----:|-----:|:-----:|----:|
| 2024 | +5.38% | **0.42%** | 55% | 74% | PASS | ~7% |
| 2025 | +22.04% | **1.63%** | 73% | 69% | PASS | ~4% |
| 2026 YTD | +20.0%+ | **2.30%** | 88% | 87% | PASS | ~6% |
| holdout_365d | +21.5%+ | **1.52%** | 75% | 81% | PASS | ~7% |
| roll12_m6 | +16.4%+ | **1.13%** | 67% | 80% | PASS | ~4% |

## Wave: prior-month win throttle + IS blend (this)

Scripts/helpers:
- `scripts/quest_prior_win_blend_consistency_wf.py`
- `apply_prior_month_win_throttle` in `overlays.py` (completed M-1; lag-1; hi≤1)
- Daily-return blend `(1-α)*locked + α*overlay` with α nested on 2024
- Soft mean floor 0.0095 both IS windows; HO never for selection

### Board

| Family | Notes |
|--------|-------|
| baseline / win / mtd / wrebal | win alone no soft; mtd soft_mean or is_fail |
| frozen_wrebal_mtd | **3 soft** incl. +win — all holdout_fail (HO %pos 67%) |
| is_blend | α=0.70 vs mtd soft at 2024 0.95%; win-blends HO %pos 75% but soft_mean_fail |

**Totals:** soft=**4** hard=**0** promote=**0** (board n=21).

### Best soft (not promote) — `wrebal+mtd_t0.015_a0.5_frozen`

| Window | Mean mo | %pos | Top3 | Role |
|--------|--------:|-----:|-----:|------|
| 2024 | **1.18%** | **82%** | **45%** | IS — soft OK |
| 2025_IS | **1.74%** | **75%** | **64%** | IS — hard top3 fail |
| holdout | **1.69%** | **67%** | **71%** | confirm — holdout_fail |

**Promote: NO.** Locked tag unchanged.

---

## Prior session log (preserved)

# One-percent per month quest — status

**Data:** `approximate_non_ftmo` (Yahoo via yfinance). **No `data/ftmo/` exports — never golive.**
**Gates:** FTMO 2-Step static max loss 10%, daily 5% (Europe/Prague). `signal_lag=1`. IS-only grids. Holdout never for selection.

Session: 2026-09-16 23:36 BST (wave: **causal trailing gain-concentration dampen** on locked sleeve ± frozen HO-robust wrebal).

## Scoring rubric this wave (primary)

1. High **% positive months** (prefer ≥70–75%)
2. Lower **top-3 gain concentration** (prefer ≤~50–55%; reject if HO top3 worsens vs locked)
3. Mean monthly **~1% stable** across 2024, 2025, 2026, holdout — **2024 cannot stay at 0.42%**
4. Prefer **lower monthly variance / less lumpy** mean over higher bursty mean
5. **Reject** candidates that boost mean via bursts even if gates PASS
6. **No leverage increase** (RF fixed at 8%; overlay `hi≤1`)

**Selection objective:** maximize `min(IS year mean_mo)` for `{2024, 2025_IS}` subject to %pos≥70% and top3≤55% (soft≤70%; soft also mean≥~1%). Holdout / full 2025–2026 **confirmation only** — promote only if holdout **also** clears ≥1% mean and ≥70% pos **and** each of 2024/2025/2026 clears the same.

## Target vs result

| Criterion | Target | Locked candidate | Met? |
|-----------|--------|------------------|:----:|
| Mean monthly on eval windows | ≥ ~1.0% | Holdout **+1.52%**; 2025 **+1.63%**; 2026 **+2.30%**; roll12_m6 **+1.13%**; **2024 +0.42%** | **Mostly — 2024 still short** |
| % positive months | ≥ ~70–75% | Holdout **75%**; 2026 **88%**; 2025 **73%**; **2024 55%** | 2024 fails consistency |
| Top-3 months share of gains | ≲ 50–55% | Holdout **~81%**; 2024 **74%** | **No — still bursty** |
| Gates | PASS | All listed windows **PASS** | **Yes** |
| Multi-window, warmup, fixed params | required | Independent windows + 250-bar warmup; params frozen | **Yes** |

**Verdict:** Official locked tag **unchanged**. Conc-dampen wave: board **n=8**, soft=**4**, hard=**0**, promote=**0**. Primary thresh≤0.70 always-fires on 2024 (top3-of-pos≈0.88–1.0). Best soft remains prior `wrebal+mtd_t0.015_a0.5` (HO %pos 67%). Soft `wrebal+mtd+conc` lifts HO %pos to **75%** but HO top3 **70.7%** and hard IS top3 fail. **Promote: NO.** Still blocked on FTMO CSVs.

## Locked candidate (unchanged — still official)

**Tag:** `fx4plus_gbpcad_d1_voltarget_0025`  
**Config:** `configs/quest_one_pct_candidate.yaml`

Re-verified this wave (`RF=0.08`, `PORT_VOL_TARGET=0.0025`, weights oos_sharpe; `scripts/eval_windowed_consistency.py` → `quest_locked_verify_keepalive_2330`):

| Window | Return | Mean mo | %pos | Top3 | Gates | P2T |
|--------|-------:|--------:|-----:|-----:|:-----:|----:|
| 2024 | +5.38% | **0.42%** | 55% | 74% | PASS | ~7% |
| 2025 | +22.04% | **1.63%** | 73% | 69% | PASS | ~4% |
| 2026 YTD | +20.0%+ | **2.30%** | 88% | 87% | PASS | ~6% |
| holdout_365d | +21.5%+ | **1.52%** | 75% | 81% | PASS | ~7% |
| roll12_m6 | +16.4%+ | **1.13%** | 67% | 80% | PASS | ~4% |

## Wave: trailing gain-concentration dampen (this)

Scripts/helpers:
- `scripts/quest_conc_dampen_consistency_wf.py` (importlib reuse of refine helpers + frozen wrebal from `quest_ho_robust_selected.json`)
- `apply_trailing_gain_concentration_dampen` in `overlays.py` (completed months; <2 positives → conc=1.0; lag-1; hi≤1)
- Nested pick on 2024; HO never for selection; soft mean floor 0.0095 both IS windows
- Grid: lb∈{4,6,8}, thresh∈{0.55..0.70}+mild{0.80,0.85,0.90}, cool∈{0.35,0.5,0.65}, K=3

### Board

| Family | N | Soft IS | Hard IS | Promote | Notes |
|--------|--:|--------:|--------:|--------:|-------|
| baseline | 1 | 0 | 0 | 0 | 2024 0.42%/55%/74% |
| conc_dampen | 1 | 0 | 0 | 0 | best lb8 th0.9 cs0.65; mean crushed |
| frozen_wrebal | 1 | 0 | 0 | 0 | soft_mean_fail (2024 0.68%) |
| frozen_wrebal_conc | 1 | 0 | 0 | 0 | soft_mean_fail |
| frozen_wrebal_mtd | 2 | **2** | 0 | 0 | **best soft**; holdout_fail |
| frozen_wrebal_mtd_conc | 1 | **1** | 0 | 0 | HO %pos 75% but top3 70.7% |
| mtd_gain_clip_prior | 1 | **1** | 0 | 0 | year_clear_fail |

**Totals:** soft=**4** hard=**0** promote=**0** (board n=8).

### Best soft (not promote) — `wrebal_…+mtd_t0.015_a0.5_frozen`

| Window | Mean mo | %pos | Top3 | Role |
|--------|--------:|-----:|-----:|------|
| 2024 | **1.18%** | **82%** | **45%** | IS — soft OK; hard OK this window |
| 2025_IS | **1.74%** | **75%** | **64%** | IS — soft OK; hard top3 fail |
| 2025 (cal) | **1.11%** | 73% | 61% | confirm |
| 2026 | **2.41%** | 75% | 80% | confirm |
| holdout | **1.69%** | **67%** | **71%** | confirm — holdout_fail |

### Soft near-miss — `wrebal+mtd+conc_lb6_th0.85_cs0.65`

| Window | Mean mo | %pos | Top3 | Role |
|--------|--------:|-----:|-----:|------|
| 2024 | **1.06%** | **82%** | **47%** | IS soft |
| 2025_IS | **1.30%** | **75%** | **61%** | hard top3 fail |
| holdout | **1.11%** | **75%** | **71%** | HO %pos OK; top3 70.7%>70% soft |

**Promote: NO.** Locked tag unchanged.

---


## Prior session log (preserved)

# One-percent per month quest — status

**Data:** `approximate_non_ftmo` (Yahoo via yfinance). **No `data/ftmo/` exports — never golive.**
**Gates:** FTMO 2-Step static max loss 10%, daily 5% (Europe/Prague). `signal_lag=1`. IS-only grids. Holdout never for selection.

Session: 2026-09-16 23:00 BST Europe/London (wave: **causal MTD loss-halt + after-loss throttle** on locked sleeve).

## Scoring rubric this wave (primary)

1. High **% positive months** (prefer ≥70–75%)
2. Lower **top-3 gain concentration** (prefer ≤~50–55%; reject if HO top3 worsens vs locked)
3. Mean monthly **~1% stable** across 2024, 2025, 2026, holdout — **2024 cannot stay at 0.42%**
4. Prefer **lower monthly variance / less lumpy** mean over higher bursty mean
5. **Reject** candidates that boost mean via bursts even if gates PASS
6. **No leverage increase** (RF fixed at 8%; overlay `hi≤1`)

**Selection objective:** maximize `min(IS year mean_mo)` for `{2024, 2025_IS}` subject to %pos≥70% and top3≤55% (soft≤70%; soft also mean≥~1%). Holdout / full 2025–2026 **confirmation only** — promote only if holdout **also** clears ≥1% mean and ≥70% pos **and** each of 2024/2025/2026 clears the same.

## Target vs result

| Criterion | Target | Locked candidate | Met? |
|-----------|--------|------------------|:----:|
| Mean monthly on eval windows | ≥ ~1.0% | Holdout **+1.52%**; 2025 **+1.63%**; 2026 **+2.30%**; roll12_m6 **+1.13%**; **2024 +0.42%** | **Mostly — 2024 still short** |
| % positive months | ≥ ~70–75% | Holdout **75%**; 2026 **88%**; 2025 **73%**; **2024 55%** | 2024 fails consistency |
| Top-3 months share of gains | ≲ 50–55% | Holdout **~81%**; 2024 **74%** | **No — still bursty** |
| Gates | PASS | All listed windows **PASS** | **Yes** |
| Multi-window, warmup, fixed params | required | Independent windows + 250-bar warmup; params frozen | **Yes** |

**Verdict:** Official locked tag **unchanged**. Loss-halt wave produced **4** soft IS / **0** hard / **0** promote. Best soft `lock_mtd_t0.015_a0.0+lh_t0.02_ah0.0` (2024 **1.05%/73%/52%**; HO **0.89%/67%/74%**) — hard top3 fail + holdout_fail. Alone loss-halt / after-loss did not soft-pass. **Promote: NO.** Still blocked on FTMO CSVs.

## Locked candidate (unchanged — still official)

**Tag:** `fx4plus_gbpcad_d1_voltarget_0025`  
**Config:** `configs/quest_one_pct_candidate.yaml`

Re-verified this wave (`RF=0.08`, `PORT_VOL_TARGET=0.0025`, weights oos_sharpe; `scripts/eval_windowed_consistency.py` → `quest_locked_verify_keepalive_2300`):

| Window | Return | Mean mo | %pos | Top3 | Gates | P2T |
|--------|-------:|--------:|-----:|-----:|:-----:|----:|
| 2024 | +5.38% | **0.42%** | 55% | 74% | PASS | ~7% |
| 2025 | +22.04% | **1.63%** | 73% | 69% | PASS | ~4% |
| 2026 YTD | +20.0%+ | **2.30%** | 88% | 87% | PASS | ~6% |
| holdout_365d | +21.5%+ | **1.52%** | 75% | 81% | PASS | ~7% |
| roll12_m6 | +16.4%+ | **1.13%** | 67% | 80% | PASS | ~4% |

## Wave: causal MTD loss-halt + after-loss throttle (this)

Scripts/helpers:
- `scripts/quest_loss_halt_consistency_wf.py` (new; importlib reuse of refine helpers)
- `apply_mtd_loss_halt` / `apply_after_loss_throttle` in `overlays.py`
- Nested pick on 2024; HO never for selection; soft mean floor 0.0095 both IS windows

### Board

| Family | N | Soft IS | Hard IS | Promote | Notes |
|--------|--:|--------:|--------:|--------:|-------|
| baseline | 1 | 0 | 0 | 0 | 2024 0.42%/55%/74% |
| mtd_loss_halt | 3 | 0 | 0 | 0 | best τ=0.02 ah=0; 2024 %pos still 55% |
| after_loss_throttle | 1 | 0 | 0 | 0 | after_loss=0.25; IS fail |
| combo_lh_al | 1 | 0 | 0 | 0 | IS fail |
| combo_lh_mtd | 1 | **1** | 0 | 0 | holdout_fail |
| combo_lh_mtd_prior | 1 | **1** | 0 | 0 | twin of nested |
| combo_al_mtd | 1 | 0 | 0 | 0 | IS fail |
| combo_al_mtd_prior | 1 | 0 | 0 | 0 | soft_mean_fail |
| mtd_gain_clip_prior | 1 | **1** | 0 | 0 | year_clear_fail (2025 cal) |
| combo_prior_mtd_lh | 1 | **1** | 0 | 0 | **best soft**; holdout_fail |

**Totals:** soft=**4** hard=**0** promote=**0** (board n=12).

### Best soft (not promote) — `lock_mtd_t0.015_a0.0+lh_t0.02_ah0.0`

| Window | Mean mo | %pos | Top3 | Role |
|--------|--------:|-----:|-----:|------|
| 2024 | **1.05%** | **73%** | **52%** | IS — soft OK; hard OK this window |
| 2025_IS | **1.37%** | **75%** | **59%** | IS — soft OK; hard top3 fail (>55%) |
| 2025 (cal) | **0.79%** | 73% | 56% | confirm — below 1% |
| 2026 | **1.89%** | 88% | 81% | confirm |
| holdout | **0.89%** | **67%** | **74%** | confirm — holdout_fail |

**Promote: NO.** Locked tag unchanged.

---

## Wave: HO-robust consistency (this)

Scripts/helpers:
- `scripts/quest_ho_robust_consistency_wf.py` (new; importlib reuse of refine + eval helpers)
- `apply_monthly_budget_vt` / `apply_daily_budget_vt` / `apply_equity_tsmom` from `equity_tsmom.py`
- Nested pick on 2024; weights max min(2024,2025_IS); HO never for selection

### Board

| Family | N | Soft IS | Hard IS | Promote | Notes |
|--------|--:|--------:|--------:|--------:|-------|
| baseline | 1 | 0 | 0 | 0 | 2024 0.42%/55%/74% |
| monthly_budget_vt | 1 | 0 | 0 | 0 | IS fail |
| daily_budget_vt | 1 | 0 | 0 | 0 | IS fail |
| equity_tsmom | 1 | 0 | 0 | 0 | IS fail |
| weight_rebalance | 1 | 0 | 0 | 0 | soft_mean_fail (2024 0.69%) |
| wrebal_plus_mtd | 1 | **1** | 0 | 0 | **best soft**; holdout_fail |
| wrebal_plus_monthly_budget | 1 | 0 | 0 | 0 | soft_mean_fail |
| wrebal_plus_daily_budget | 1 | 0 | 0 | 0 | soft_mean_fail |
| wrebal_combo_two | 1 | **1** | 0 | 0 | mb+mtd; holdout_fail |

**Totals:** soft=**2** hard=**0** promote=**0** (board n=9; weight grid 207 unique).

### Best soft (not promote) — `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5`

| Window | Mean mo | %pos | Top3 | Role |
|--------|--------:|-----:|-----:|------|
| 2024 | **1.20%** | **82%** | **45%** | IS — soft OK; hard OK on this window |
| 2025_IS | **1.74%** | **75%** | **64%** | IS — soft OK; hard top3 fail (>55%) |
| 2025 (cal) | **1.11%** | 73% | 61% | confirm |
| 2026 | **2.41%** | 75% | 80% | confirm |
| holdout | **1.69%** | **67%** | **71%** | confirm — holdout_fail |

**Promote: NO.** Locked tag unchanged.

---

## Wave: consistency overlay refine (this)

Scripts/helpers:
- `scripts/quest_consistency_overlay_refine_wf.py` (new)
- Existing overlays in `src/mt5_swing/portfolio/overlays.py` (no new functions)
- Nested pick on 2024 / IS dual for weights; HO never for selection

### Design

1. Rebuild locked basket (VT=0.0025, warmup=250, RF=0.08).
2. Grid causal overlays: finer MTD τ×after; month_aware; runup; equity_curve_target (hi=1); combos of ≤2.
3. IS-only simplex/Dirichlet weight perturbations around oos_sharpe; freeze best max min(2024,2025_IS) with %pos≥70% top3 soft≤70%.
4. Promote only soft+hard IS **and** year_clear (2024/2025/2026 ≥1% & ≥70% pos) **and** holdout clears.

### Board

| Family | N | Soft IS | Hard IS | Promote | Notes |
|--------|--:|--------:|--------:|--------:|-------|
| baseline locked | 1 | 0 | 0 | 0 | 2024 0.42%/55%/74% |
| mtd_gain_clip | 4 | **4** | 0 | 0 | best τ=0.015 a=0.0; year_clear_fail on cal 2025 |
| month_aware / runup / eqtarget | 3 | 0 | 0 | 0 | alone insufficient for soft dual-year |
| combo_two | 5 | **2** | 0 | 0 | MTD+runup soft; year_clear_fail |
| weight_rebalance | 2 | **1** | 0 | 0 | soft; HO %pos 67% |
| weight_plus_mtd | 1 | **1** | 0 | 0 | **best soft**; holdout_fail |
| weight_plus_month | 1 | 0 | 0 | 0 | IS fail |

**Totals:** soft=**8** hard=**0** promote=**0** (board n=17).

### Best soft (not promote) — `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5`

Weights IS-chosen (boost USDCHF/CADJPY/GBPCAD, cut GBPUSD); MTD τ=0.015 after=0.5 re-picked on reweighted 2024.

| Window | Mean mo | %pos | Top3 | Role |
|--------|--------:|-----:|-----:|------|
| 2024 | **1.20%** | **82%** | **45%** | IS — soft OK; hard OK on this window |
| 2025_IS | **1.74%** | **75%** | **64%** | IS — soft OK; hard top3 fail (>55%) |
| 2025 (cal) | **1.11%** | 73% | 61% | confirm — clears ≥1% mean |
| 2026 | **2.41%** | 75% | 80% | confirm |
| holdout | **1.69%** | **67%** | **71%** | confirm — **%pos&top3 fail** → holdout_fail |

**Promote: NO.** Locked tag unchanged.

---

## Wave: D1 multi-year consistency repair + MTD gain-clip (this)

Scripts/helpers:
- `scripts/quest_d1_consistency_repair_wf.py`
- `src/mt5_swing/portfolio/overlays.py` — `apply_mtd_gain_clip` (causal intra-month MTD; hi≤1)
- Screen: unused D1 OOS-passers from `reports/walk_forward_summary.csv` (not exact locked legs)

### Design

1. **D1 pool:** OOS gates+profitable, sharpe>0.35; prefer unused symbols; RF fixed 8%; signal_lag=1.
2. **Addon sleeves:** lock + single/pair D1 at sleeve ∈ {15…30%}; score only `{2024, 2025_IS}`; HO/2025/2026 confirm only.
3. **MTD gain-clip:** tau chosen on **2024 only**; after_clip ∈ {0,0.25,0.5}; freeze; confirm.

### Board

| Family | N | Soft IS | Hard IS | Promote | Notes |
|--------|--:|--------:|--------:|--------:|-------|
| baseline locked | 1 | 0 | 0 | 0 | 2024 still 0.42%/55% |
| lock+D1 addon | 28 | **0** | 0 | 0 | Best `lock+EURUSD_bbands_s30` min_is_mo 0.42%; 2024 %pos 64% |
| lock+D1 pair | 20 | **0** | 0 | 0 | 10 pair trials; no soft |
| D1 alternate baskets | 8 | **0** | 0 | 0 | Solo D1 edges tiny (~0.02–0.18%/mo) |
| MTD gain-clip | 1 | **1** | **0** | **0** | soft IS; year_clear_fail |
| lock+addon+clip | 2 | 0 | 0 | 0 | soft regresses vs clip-alone |

**Totals:** soft=**1** hard=**0** promote=**0** (board n=60). D1 pool screened=24; solo dual-year positive IS=7 (low corr to locked).

### Best soft (not promote) — `lock_mtdclip_t0.015_a0.0`

Tau=0.015, after_clip=0.0 chosen on 2024 only.

| Window | Mean mo | %pos | Top3 | Role |
|--------|--------:|-----:|-----:|------|
| 2024 | **0.99%** | **73%** | **52%** | IS — soft OK; mean <1.00% for year-clear |
| 2025_IS | **1.45%** | **75%** | **59%** | IS — soft OK; hard top3 fail (>55%) |
| 2025 (cal) | **0.89%** | 73% | 56% | confirm — **mean <1%** → year_clear_fail |
| 2026 | 1.91% | 88% | 77% | confirm |
| holdout_365d | 1.37% | 75% | 66% | confirm — would clear HO soft top3≤70% |

**Promote?** **No** — hard top3 fail; calendar 2025 mean 0.89%; 2024 mean 0.99% just shy of 1%. Locked tag unchanged.

### Why D1 add-ons missed soft

Unused D1 OOS-passers at basket-scale risk print dual-year positive means but **low %pos** (often 25–55% on 2024). Blending into locked lifts 2024 mean only marginally (0.42%→~0.42%) and rarely reaches %pos≥70%. Nearest addon miss: `lock+EURUSD_bbands_reversion_s30` (2024 0.42%/64% pos; 2025_IS 1.72%/75%).

Details: `reports/quest_d1_consistency_repair.md`, `reports/quest_d1_consistency_board.csv`, `reports/quest_d1_consistency_solo.csv`, `reports/quest_d1_consistency_pool.csv`, `configs/quest_d1_consistency_selected.json`, `reports/quest_locked_verify_keepalive_2100.md`.

## Wave: intraday→swing + cross-asset stack + missing-month (prior)

Scripts/helpers:
- `scripts/quest_intraday_crossasset_stack_wf.py`
- `src/mt5_swing/portfolio/intraday_stack.py` — IS monthly scale, missing-month gates, swing exits
- `src/mt5_swing/data/download.py` — **M15** Yahoo probe (~60d) + H1

### Design (no pairs proxy)

1. **Intraday→swing:** download H1 (12 symbols) + M15 probe; a-priori swing holds (`max_hold_bars` / ATR exits); `signal_lag=1`; closed bars only.
2. **Cross-asset dual-confirm:** expand FX with H4+D1 OOS passers; metals only if dual — **XAU not dual** (H4 only).
3. **Stack + IS monthly PnL target:** decorrelated add-ons; `is_scale_for_monthly_target` from IS vol only (`hi≤1`); freeze for confirm.
4. **Missing-month diversifiers:** must **strictly** raise 2024 **and** 2025_IS %pos vs locked before HO check.

### Board

| Family | N | Soft IS | Hard IS | Best IS min_mo | Notes |
|--------|--:|--------:|--------:|---------------:|-------|
| H1 swing single-leg | 60 | 0 dual-year+ | — | — | Mostly flat/neg; **0** dual-year gate+positive |
| M15 Yahoo | 3 sym | n/a | — | — | **~60d only** (2026-06→now); no 2024/2025 |
| dual-confirm stacks | 10 | **0** | **0** | **1.03%** (`lock+dual2`) | soft fail: 2025_IS %pos **62.5%** |
| missing-month div | 10 | 1 IS %pos gate | 0 promote | 0.30% 2024 | Only USDJPY lifts both %pos; mean collapses |
| metals dual | 0 | — | — | — | XAUUSD H4-only |

### Nearest miss (not eligible — IS soft fail + HO top3)

`lock+dual2` = locked + USDJPY D1 `hybrid_regime` + EURUSD D1 `cci_reversion`

| Window | Mean mo | %pos | Top3 | Role |
|--------|--------:|-----:|-----:|------|
| 2024 | **1.03%** | 73% | 69% | IS |
| 2025_IS | **1.09%** | **62.5%** | 77% | IS — **%pos fail** |
| 2025 (cal) | 1.07% | 82% | 61% | overlaps HO — not for selection |
| 2026 | 1.69% | 75% | 83% | confirm |
| holdout_365d | 1.26% | 75% | **73%** | confirm — top3 vs soft 70% → **holdout_fail** |

**Promote?** **No** — 2025_IS consistency fail; HO top3; 0 soft IS. Locked tag unchanged.

Details: `reports/quest_intraday_crossasset_stack.md`, `reports/quest_intraday_stack_board.csv`, `reports/quest_intraday_stack_promote.csv`, `reports/quest_h1_swing_screen.csv`, `reports/quest_missing_month_diversifiers.csv`, `configs/quest_intraday_stack_selected.json`.

## Wave: real two-leg pairs vs Δz proxy (prior)

Scripts/helpers:
- `scripts/quest_pairs_real_two_leg_wf.py`
- `src/mt5_swing/portfolio/pairs_residual.py` — `backtest_two_leg_spread`, `diagnose_proxy_vs_residual`
- Prior ATR/independent-leg basket (`pairs_two_leg_basket.py`) diagnosed as **bug+edge-death**: ATR stops fought z-exits; legs sized independently (hedge broken). Fixed path uses z-exit only + beta hedge lots.

### Nested IS (no HO/2026 peek)

1. Fit OLS beta on **2024 only**
2. Optimize knobs on 2024, validate on **2025_IS**
3. Score = `min(2024, 2025_IS mean_mo)` with %pos / top3; freeze; confirm HO+2026 only after

Knobs tried (IS-safe): entry z ∈ {1.5…2.5}, exit z ∈ {0.2,0.3,0.5}, risk_frac ≤ RF/n (≤1%), max pairs 2–4, sleeve 0.55/1.0, win 40/60, sizing `z_vol` | `unit_residual`.

### Diagnosis

| Mode | Sizing | 2024 mean_mo | Notes |
|------|--------|-------------:|-------|
| Δz proxy | z_vol map | ~1.28% | Research only — **not tradable** |
| Real two-leg | z_vol (match proxy economics) | ~0.89% single-pair | Costs eat most of proxy edge |
| Real two-leg | unit_residual | ~0.01% | Honest notional; amp≈**180×** shows proxy illusion |

### Board

| Family | N | Soft IS | Hard IS | Best IS min_mo |
|--------|--:|--------:|--------:|---------------:|
| real two-leg (nested) | **1080** | **0** | **0** | **0.51%** (n2 z_vol) |

### Best REAL candidate (confirm — promote gate)

`real_n3_z_vol` entry=2.0 exit=0.3 risk=… sleeve=1.0 (EURAUD/AUDUSD, AUDUSD/NZDUSD, GBPCAD/USDCAD; betas frozen on 2024)

| Window | Mean mo | %pos | Top3 | Gates |
|--------|--------:|-----:|-----:|:-----:|
| 2024 | **0.75%** | 73% | 71% | PASS |
| 2025 | **0.58%** | 73% | 58% | PASS |
| 2026 | **0.48%** | 50% | 83% | PASS |
| holdout_365d | **0.30%** | 50% | 59% | PASS |

Proxy reference on **same** frozen sleeve (not eligible): ~1.33–1.40%/mo all windows — proves gap is fills/costs/sizing, not window luck.

**Promote?** **No** — confirm_min_mo **0.30%** ≪ 1%; 2026/HO %pos fail; 0 soft IS passers. **Pairs proxy path closed.**

Details: `reports/quest_pairs_real_two_leg.md`, `reports/quest_pairs_real_board.csv`, `reports/quest_pairs_real_promote.csv`, `reports/quest_pairs_proxy_diagnosis.csv`, `configs/quest_pairs_real_selected.json`.

## Wave: equity TSMOM + monthly-budget VT + pairs residual (new)

Scripts/helpers:
- `scripts/quest_equity_tsmom_pairs_wf.py`
- `src/mt5_swing/portfolio/equity_tsmom.py`
- `src/mt5_swing/portfolio/pairs_residual.py`

### Design (not another same-board basket grid)

1. **Equity-curve TSMOM** on locked diversified sleeve (lagged lookback return → scale next bar; `hi≤1`) then/or **causal monthly-budget VT** (~1%/mo → daily vol map; flattener `hi≤1`).
2. **Pairs/spread residual MR** on cointegrated FX (IS OLS hedge + rolling z); hard risk caps (`RF/n`, bar-return clip); monthly distribution score.
3. **D1 history expand** via yfinance to ~15y for pairs symbols; **skip locked D1 (GBPCAD)**; H4 Yahoo 1h still ~730d — irreducible.
4. Score = `min(2024, 2025_IS mean_mo)` with %pos/top3; holdout confirmation only.

### Board summary

| Family | N | Soft IS | Hard IS | Best IS min_mo |
|--------|--:|--------:|--------:|---------------:|
| equity overlays | 30 | 5 | 0 | ~0.42% (budget on locked VT; %pos fail) / soft ~0.18% |
| pairs residual (Δz proxy) | 48 | 36 | 0 | **~1.05%** |
| blend lock+pairs | 126 | 77 | 0 | ~0.96% |
| **Total** | **204** | **118** | **0** | |

### Best IS soft (holdout excluded from score)

`pairs_n3` entry=2.0 sleeve=1.0 (EURUSD/GBPUSD, EURCHF/USDCHF, EURUSD/USDCHF; win=40) — IS min_mo **1.01%**, max top3 **62%**, min %pos **75%**.

### Confirmation (promote gate)

| Idea | 2024 mo/%pos | 2025 | 2026 | HO mo/%pos | Promote | Reason |
|------|-------------:|-----:|-----:|-----------:|:-------:|--------|
| pairs_n3 e2.0 | 1.35/100 | 1.12/91 | **0.88**/88 | **0.95**/92 | no | HO&2026 &lt;1% |
| pairs_n4 e1.5 | 1.79/100 | 1.01/100 | 1.07/100 | **0.99**/100 | no | HO 0.99% &lt;1% |
| pairs_n2 e2.0 | 1.47/100 | 1.06/91 | **0.98**/88 | 1.06/92 | no | 2026 &lt;1% |
| eq budget_mo / tsmom | ≤0.18 soft | — | — | — | no | mean collapse |
| locked baseline | 0.42/55 | 1.63/73 | 2.30/88 | 1.52/75 | — | official |

**Promote?** **No** — official tag unchanged.

### Evidence: impossible (this structure) on approximate_non_ftmo

1. **Equity TSMOM / monthly-budget VT** on the locked sleeve raises 2024 %pos only when they **cut mean far below 1%** (soft passers ≈0.17–0.18%/mo). On locked VT base they keep ~0.42% but fail %pos. No overlay clears dual-year ≥1% with ≥70% pos.
2. **Pairs Δz residual proxy** can print ~1%/mo IS with excellent %pos / lower top3, but:
   - Holdout or calendar-2026 consistently lands **0.88–0.99%** — just under the ≥1% joint gate.
   - Proxy maps `risk_frac · Δz` → return; **prior real two-leg ATR/spread basket was negative** (`pairs_two_leg_quest`) — so proxy **cannot** be promoted without FTMO-realistic fills.
3. **Blends** lock+pairs sit between (~0.75–0.96% IS min) — inherit lock 2024 weakness or pairs HO shortfall.
4. **H4 cannot be extended** past Yahoo ~730d 1h; D1 expand helps pairs fit history but does not unlock dual-year ≥1% under promote rules.
5. **0 / 204** hard top3≤55% on both IS windows.

**Conclusion:** Under no-look-ahead, no RF hike, consistency &gt; peak mean, and holdout-as-confirmation-only, **clean ≥1% mean monthly on EACH of 2024, 2025, 2026, holdout with ≥70% pos is not achieved** on `approximate_non_ftmo` with this wave’s structures. Closest approaches fail either HO/2026 by a few bp or rely on a non-tradable Δz proxy.

Details: `reports/quest_equity_tsmom_pairs_wf.md`, `reports/quest_equity_tsmom_pairs_board.csv`, `reports/quest_equity_tsmom_pairs_promote.csv`, `configs/quest_equity_tsmom_pairs_selected.json`.

## Ideas tried (cumulative)

| Idea | Result |
|------|--------|
| New strats TSMOM/vol-breakout/carry/Kalman/ORB | Weak OOS; many holdout fails; carry ~0 trades |
| Two-leg pairs MR (real fills) | Negative once real spreads+ATR sizing |
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
| Smooth many-leg + IS min(year) WF | Best IS ~0.81%/mo; HO %pos collapses — rejected |
| Metals dual-confirm | None eligible |
| **Equity TSMOM + monthly-budget VT (this wave)** | Soft ~0.18%/mo; mean collapse — **rejected** |
| **Pairs residual Δz + hard caps (this wave)** | IS ~1.0%+ / high %pos; HO or 2026 &lt;1%; proxy≠real fills — **rejected** |
| **Blend lock+pairs (this wave)** | IS min &lt;1% — **rejected** |
| **D1 yfinance expand (this wave)** | ~15y D1 for pairs; H4 still capped; locked GBPCAD preserved |
|| **Intraday H1 swing (this wave)** | 60 legs; **0** dual-year gate+positive — **rejected** |
|| **M15 Yahoo probe (this wave)** | ~60d only; no multi-year calendars — **probe only** |
|| **Dual-confirm FX stacks + IS mo scale (this wave)** | Best `lock+dual2` IS min **1.03%** but 2025_IS %pos 62.5%; HO top3 fail — **rejected** |
|| **Missing-month diversifiers (this wave)** | 1/10 strict dual-year %pos lift (USDJPY); 2024 mean 0.30% — **rejected** |
|| **Metals dual-confirm (this wave)** | Still none (XAU H4-only) |
| **Real two-leg z-exit + beta hedge (prior)** | 1080 nested-IS cands; best confirm_min **0.30%**; proxy ~1.3% on same sleeve — **proxy illusory; stop promoting pairs** |
| **Prior ATR two-leg (re-diagnosed)** | Negative: ATR exits fought MR + independent leg sizing broke hedge — design bug, not sole cause of death |

## Evidence: Yahoo approx cannot support the ≥1%/mo joint goal

Under **no look-ahead**, **no OOS tune**, **no RF hike** (8%), **consistency first** (≥70% pos months, lower top3), and **holdout confirmation only**, clean **≥1% mean monthly on EACH of 2024, 2025, 2026, holdout** is **not achieved** on `approximate_non_ftmo`.

### What failed (this pivot + prior)

| Approach | Outcome |
|----------|---------|
| Locked `fx4plus_gbpcad_d1_voltarget_0025` | 2024 **0.42%**/55% pos; HO top3 ~81% — official but short of joint goal |
| Pairs Δz proxy | Illusory **~180×** vs unit-notional; **do not promote** |
| Real two-leg pairs | Best confirm_min **0.30%/mo**; 0 soft IS / 1080 |
| Equity TSMOM / monthly-budget VT | Mean collapse (~0.18%) when %pos rises |
| Smooth many-leg / clock recycle / regime | HO %pos or top3 regressions |
| **H1 swing (Yahoo ~730d)** | **0**/60 dual-year gate+positive; costs eat intraday MR |
| **M15 Yahoo** | **~60 calendar days** — cannot score 2024/2025/2026 |
| **Dual-confirm stacks** | Nearest `lock+dual2` calendar ≥1% but **2025_IS %pos 62.5%** + HO top3 fail |
| **Missing-month diversifiers** | 1/10 strict dual %pos lift; 2024 mean falls to ~0.30% |
| Metals / indices | No dual-confirm XAU; no US30/NAS100/SPX history in repo |

### Irreducible Yahoo limits

1. **H4** from 1h resample ≈ **730d** — cannot extend multi-year H4 via yfinance.
2. **H1** ≈ **730d** from ~2023-11 — incomplete early-2024; diversity probe only.
3. **M15** ≈ **60d** — probe-only; useless for multi-year IS/HO.
4. Yahoo FX ≠ FTMO CFD spreads/swap/sessions/commission; reports stay `approximate_non_ftmo`.
5. No broker index CFD history for dual-confirm cross-asset.

### Required unlock: FTMO MT5 history

**`data/ftmo/` exports from the Windows FTMO MT5 terminal** (History Center / script), tagged `ftmo_mt5_export`, are required to continue honestly toward the joint ≥1%/mo goal and any go-live path:

- True spreads, commission, swap, session filters
- Deeper **H1/M15** (and H4/D1) multi-year books
- Index/commodity CFDs if offered on the FTMO account
- Any candidate that could clear FTMO 2-Step gates on broker-real fills

Until then: **locked tag stays** `fx4plus_gbpcad_d1_voltarget_0025`; **promote = NO**; research on Yahoo is exhausted for this objective.

## Process / tests

- pytest: **67 passed** (`tests/test_intraday_stack.py` + prior suite) — `reports/quest_intraday_stack_pytest.log`.
- `signal_lag=1`; IS windows `{2024, 2025_IS}`; holdout/2026 confirmation only; no OOS tune.
- RF forced to **8%**; IS monthly scale `hi≤1`; no leverage hike; FTMO gates on confirm.
- Pairs proxy path **closed**; this wave did not re-promote pairs.
