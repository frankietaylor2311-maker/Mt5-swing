# Consecutive win-day cool consistency wave

**Data:** `approximate_non_ftmo`  **RF:** 0.08  **VT:** 0.0025  **Locked:** `fx4plus_gbpcad_d1_voltarget_0025`
**Board:** n=29 soft=**4** hard=**0** promote=**0**

Win-streak = consecutive positive daily returns; lag-1 day; cool if streak>=n; hi≤1.
Distinct from rolling hit-rate (window mean) and loss-streak (cools on downs).

## Board (top by score)

| Idea | Soft | Hard | Promote | 2024 mo/pos/top3 | 2025_IS | HO | Note |
|------|:----:|:----:|:-------:|------------------|---------|----|------|
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen+` | True | False | False | 1.19%/82%/45% | 1.74%/75%/64% | 1.69%/67%/71% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen` | True | False | False | 1.18%/82%/45% | 1.74%/75%/64% | 1.69%/67%/71% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.65_frozen` | True | False | False | 1.03%/82%/48% | 1.95%/75%/66% | 1.70%/67%/73% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.65_frozen` | True | False | False | 1.03%/82%/48% | 1.95%/75%/66% | 1.70%/67%/73% | holdout_fail |
| `baseline_locked` | False | False | False | 0.42%/55%/74% | 2.33%/75%/70% | 1.52%/75%/81% | is_fail |
| `lock+wstreak_n4_c0.5` | False | False | False | 0.42%/55%/71% | 2.27%/75%/71% | 1.63%/75%/78% | is_fail |
| `lock+wstreak_n4_c0.65` | False | False | False | 0.42%/55%/72% | 2.29%/75%/71% | 1.60%/75%/79% | is_fail |
| `lock+wstreak_n5_c0.35` | False | False | False | 0.44%/45%/73% | 2.33%/75%/70% | 1.55%/75%/81% | is_fail |
| `lock+wstreak_n5_c0.5` | False | False | False | 0.44%/45%/74% | 2.33%/75%/70% | 1.54%/75%/81% | is_fail |
| `lock+wstreak_n3_c0.35` | False | False | False | 0.37%/55%/74% | 2.28%/75%/73% | 1.75%/83%/79% | is_fail |
| `lock+wstreak_n3_c0.5` | False | False | False | 0.38%/55%/74% | 2.29%/75%/73% | 1.70%/75%/79% | is_fail |
| `lock+wstreak_n3_c0.65` | False | False | False | 0.39%/55%/74% | 2.30%/75%/72% | 1.65%/75%/80% | is_fail |
| `lock+wstreak_n4_c0.35` | False | False | False | 0.42%/55%/70% | 2.25%/75%/72% | 1.66%/75%/77% | is_fail |
| `lock+wstreak_n6_c0.65` | False | False | False | 0.42%/55%/74% | 2.33%/75%/70% | 1.56%/75%/81% | is_fail |
| `lock+wstreak_n6_c0.5` | False | False | False | 0.43%/55%/74% | 2.33%/75%/70% | 1.57%/75%/81% | is_fail |

## Best soft (not auto-promote unless promote=True)

- **idea:** `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen+wstreak_n8_c0.35`
- 2024 **1.19%** / 82% / 45%
- 2025_IS **1.74%** / 75% / 64%
- HO **1.69%** / 67% / 71% note=holdout_fail

**Promote: NO. Locked tag unchanged unless promote.**
