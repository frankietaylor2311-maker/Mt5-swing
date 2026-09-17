# Rolling hit-rate cool consistency wave

**Data:** `approximate_non_ftmo`  **RF:** 0.08  **VT:** 0.0025  **Locked:** `fx4plus_gbpcad_d1_voltarget_0025`
**Board:** n=20 soft=**3** hard=**0** promote=**0**

Hit-rate = mean(r_d > 0) over lookback; lag-1 day; cool if >high or <low; hi≤1.

## Board (top by score)

| Idea | Soft | Hard | Promote | 2024 mo/pos/top3 | 2025_IS | HO | Note |
|------|:----:|:----:|:-------:|------------------|---------|----|------|
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen` | True | False | False | 1.18%/82%/45% | 1.74%/75%/64% | 1.69%/67%/71% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen+` | True | False | False | 1.05%/82%/43% | 1.67%/75%/63% | 1.69%/67%/71% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.65_frozen` | True | False | False | 1.03%/82%/48% | 1.95%/75%/66% | 1.70%/67%/73% | holdout_fail |
| `baseline_locked` | False | False | False | 0.42%/55%/74% | 2.33%/75%/70% | 1.52%/75%/81% | is_fail |
| `lock+hitrate_lb21_hi0.6_lo0.3_c0.35` | False | False | False | 0.76%/64%/70% | 2.21%/88%/73% | 1.52%/67%/84% | is_fail |
| `lock+hitrate_lb21_hi0.6_lo0.4_c0.5` | False | False | False | 0.63%/64%/69% | 1.87%/75%/73% | 1.76%/83%/80% | is_fail |
| `lock+hitrate_lb21_hi0.6_lo0.4_c0.35` | False | False | False | 0.70%/64%/68% | 1.73%/75%/80% | 1.83%/75%/79% | is_fail |
| `lock+hitrate_lb21_hi0.7_lo0.4_c0.35` | False | False | False | 0.49%/64%/70% | 1.59%/75%/77% | 1.81%/83%/77% | is_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224_frozen` | False | False | False | 0.68%/82%/59% | 2.44%/75%/70% | 1.72%/67%/77% | soft_mean_fail |
| `lock+hitrate_lb21_hi0.55_lo0.4_c0.5` | False | False | False | 0.48%/73%/77% | 1.75%/75%/73% | 1.51%/83%/75% | is_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+hitrate_lb42_hi0.6_lo0.` | False | False | False | 0.69%/82%/55% | 2.26%/75%/74% | 1.68%/67%/77% | is_fail |
| `lock+hitrate_lb21_hi0.7_lo0.4_c0.5` | False | False | False | 0.47%/64%/71% | 1.76%/75%/71% | 1.74%/83%/79% | is_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+hitrate_lb63_hi0.55_lo0` | False | False | False | 0.91%/91%/56% | 2.46%/75%/70% | 1.75%/67%/76% | soft_mean_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.65_frozen` | False | False | False | 0.86%/82%/42% | 1.83%/75%/66% | 1.71%/67%/73% | soft_mean_fail |
| `blend_a0.40_lock_vs_wrebal_0.328_0.029_0.241_0.178_0.224+mtd` | False | False | False | 0.67%/73%/57% | 2.07%/75%/68% | 1.59%/75%/77% | soft_mean_fail |

## Best soft (not auto-promote unless promote=True)

- **idea:** `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen`
- 2024 **1.18%** / 82% / 45%
- 2025_IS **1.74%** / 75% / 64%
- HO **1.69%** / 67% / 71% note=holdout_fail

**Promote: NO. Locked tag unchanged unless promote.**
