# Trailing downside-vol scale consistency wave

**Data:** `approximate_non_ftmo`  **RF:** 0.08  **VT:** 0.0025  **Locked:** `fx4plus_gbpcad_d1_voltarget_0025`
**Board:** n=26 soft=**4** hard=**0** promote=**0**

## Board (top by score)

| Idea | Soft | Hard | Promote | 2024 mo/pos/top3 | 2025_IS | HO | Note |
|------|:----:|:----:|:-------:|------------------|---------|----|------|
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen+` | True | False | False | 1.19%/82%/45% | 1.74%/75%/64% | 1.61%/67%/71% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen` | True | False | False | 1.18%/82%/45% | 1.74%/75%/64% | 1.69%/67%/71% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.65_frozen` | True | False | False | 1.03%/82%/48% | 1.95%/75%/66% | 1.63%/67%/73% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.65_frozen` | True | False | False | 1.03%/82%/48% | 1.95%/75%/66% | 1.70%/67%/73% | holdout_fail |
| `lock+ddown_lb42_t0.008` | False | False | False | 0.42%/55%/74% | 2.33%/75%/70% | 1.52%/75%/81% | is_fail |
| `lock+ddown_lb63_t0.004` | False | False | False | 0.38%/55%/77% | 2.33%/75%/70% | 1.44%/75%/82% | is_fail |
| `lock+ddown_lb42_t0.005` | False | False | False | 0.42%/55%/74% | 2.33%/75%/70% | 1.46%/75%/81% | is_fail |
| `lock+ddown_lb42_t0.006` | False | False | False | 0.42%/55%/74% | 2.33%/75%/70% | 1.49%/75%/81% | is_fail |
| `baseline_locked` | False | False | False | 0.42%/55%/74% | 2.33%/75%/70% | 1.52%/75%/81% | is_fail |
| `lock+ddown_lb42_t0.004` | False | False | False | 0.38%/55%/75% | 2.33%/75%/70% | 1.37%/67%/81% | is_fail |
| `lock+ddown_lb84_t0.004` | False | False | False | 0.42%/55%/76% | 2.33%/75%/70% | 1.48%/75%/82% | is_fail |
| `lock+ddown_lb63_t0.008` | False | False | False | 0.42%/55%/74% | 2.33%/75%/70% | 1.52%/75%/81% | is_fail |
| `lock+ddown_lb63_t0.006` | False | False | False | 0.42%/55%/74% | 2.33%/75%/70% | 1.52%/75%/81% | is_fail |
| `lock+ddown_lb63_t0.005` | False | False | False | 0.42%/55%/74% | 2.33%/75%/70% | 1.52%/75%/81% | is_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224_frozen` | False | False | False | 0.68%/82%/59% | 2.44%/75%/70% | 1.72%/67%/77% | soft_mean_fail |

## Best soft (not auto-promote unless promote=True)

- **idea:** `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen+ddown_lb42_t0.004`
- 2024 **1.19%** / 82% / 45%
- 2025_IS **1.74%** / 75% / 64%
- HO **1.61%** / 67% / 71% note=holdout_fail

**Promote: NO. Locked tag unchanged unless promote.**
