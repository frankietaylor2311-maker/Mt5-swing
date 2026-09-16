# Prior-month win throttle + IS blend consistency wave

**Data:** `approximate_non_ftmo`  **RF:** 0.08  **VT:** 0.0025  **Locked:** `fx4plus_gbpcad_d1_voltarget_0025`
**Board:** n=21 soft=**4** hard=**0** promote=**0**

## Board (top by score)

| Idea | Soft | Hard | Promote | 2024 mo/pos/top3 | 2025_IS | HO | Note |
|------|:----:|:----:|:-------:|------------------|---------|----|------|
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen` | True | False | False | 1.18%/82%/45% | 1.74%/75%/64% | 1.69%/67%/71% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen+` | True | False | False | 1.11%/82%/45% | 1.66%/75%/68% | 1.56%/67%/73% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.65_frozen` | True | False | False | 1.03%/82%/48% | 1.95%/75%/66% | 1.70%/67%/73% | holdout_fail |
| `blend_a0.70_lock_vs_wrebal_0.328_0.029_0.241_0.178_0.224+mtd` | True | False | False | 0.95%/82%/51% | 1.91%/75%/66% | 1.64%/67%/74% | holdout_fail |
| `baseline_locked` | False | False | False | 0.42%/55%/74% | 2.33%/75%/70% | 1.52%/75%/81% | is_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224_frozen` | False | False | False | 0.68%/82%/59% | 2.44%/75%/70% | 1.72%/67%/77% | soft_mean_fail |
| `lock+mtd_t0.015_a0.65` | False | False | False | 0.62%/73%/63% | 2.02%/75%/67% | 1.47%/75%/75% | soft_mean_fail |
| `lock+mtd_t0.015_a0.5` | False | False | False | 0.70%/73%/59% | 1.89%/75%/65% | 1.45%/75%/73% | soft_mean_fail |
| `lock+win_t0.02_cs0.75` | False | False | False | 0.40%/55%/76% | 2.04%/75%/71% | 1.50%/75%/82% | is_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.65_frozen` | False | False | False | 0.93%/82%/46% | 1.83%/75%/71% | 1.54%/67%/75% | is_fail |
| `blend_a0.40_lock_vs_wrebal_0.328_0.029_0.241_0.178_0.224+mtd` | False | False | False | 0.73%/73%/57% | 2.09%/75%/68% | 1.59%/67%/77% | soft_mean_fail |
| `blend_a0.55_lock_vs_wrebal_0.328_0.029_0.241_0.178_0.224+mtd` | False | False | False | 0.84%/73%/53% | 2.00%/75%/67% | 1.61%/67%/76% | soft_mean_fail |

## Best soft (not auto-promote unless promote=True)

- **idea:** `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen`
- 2024 **1.18%** / 82% / 45%
- 2025_IS **1.74%** / 75% / 64%
- HO **1.69%** / 67% / 71% note=holdout_fail

**Promote: NO. Locked tag unchanged unless promote.**
