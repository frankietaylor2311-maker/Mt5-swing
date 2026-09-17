# Peak-proximity cool consistency wave

**Data:** `approximate_non_ftmo`  **RF:** 0.08  **VT:** 0.0025  **Locked:** `fx4plus_gbpcad_d1_voltarget_0025`
**Board:** n=41 soft=**5** hard=**0** promote=**0**

Peak-proximity = lag-1 equity within eps of rolling N-bar high → cool; hi≤1.
Distinct from runup throttle (trailing return) and win-streak / hit-rate / Sharpe cools.

## Board (top by score)

| Idea | Soft | Hard | Promote | 2024 mo/pos/top3 | 2025_IS | HO | Note |
|------|:----:|:----:|:-------:|------------------|---------|----|------|
| `wrebal_0.328_0.029_0.241_0.178_0.224+peakprox_lb63_e0.01_c0.` | True | False | False | 1.27%/82%/49% | 1.28%/88%/63% | 1.45%/75%/60% | year_clear_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen` | True | False | False | 1.18%/82%/45% | 1.74%/75%/64% | 1.69%/67%/71% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen+` | True | False | False | 0.99%/82%/41% | 1.16%/75%/62% | 1.47%/75%/62% | year_clear_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.65_frozen` | True | False | False | 1.03%/82%/48% | 1.95%/75%/66% | 1.70%/67%/73% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+peakprox_lb63_e0.01_c0.` | True | False | False | 1.00%/82%/48% | 1.81%/75%/69% | 1.58%/75%/68% | year_clear_fail |
| `lock+peakprox_lb21_e0.005_c0.35` | False | False | False | 0.44%/64%/68% | 1.50%/75%/63% | 1.02%/75%/61% | is_fail |
| `lock+peakprox_lb21_e0.005_c0.5` | False | False | False | 0.44%/73%/72% | 1.69%/75%/65% | 1.14%/75%/67% | is_fail |
| `lock+peakprox_lb21_e0.005_c0.65` | False | False | False | 0.43%/64%/73% | 1.88%/75%/67% | 1.25%/67%/72% | is_fail |
| `baseline_locked` | False | False | False | 0.42%/55%/74% | 2.33%/75%/70% | 1.52%/75%/81% | is_fail |
| `lock+peakprox_lb21_e0.02_c0.5` | False | False | False | 0.26%/64%/69% | 1.29%/75%/65% | 1.12%/83%/73% | is_fail |
| `lock+peakprox_lb21_e0.02_c0.35` | False | False | False | 0.21%/64%/69% | 0.97%/75%/64% | 1.00%/92%/68% | is_fail |
| `lock+peakprox_lb21_e0.01_c0.65` | False | False | False | 0.40%/64%/67% | 1.80%/75%/67% | 1.27%/75%/73% | is_fail |
| `lock+peakprox_lb21_e0.01_c0.5` | False | False | False | 0.39%/64%/64% | 1.57%/88%/66% | 1.16%/83%/69% | is_fail |
| `lock+peakprox_lb21_e0.01_c0.35` | False | False | False | 0.37%/73%/59% | 1.35%/88%/61% | 1.05%/83%/64% | soft_mean_fail |
| `lock+peakprox_lb21_e0.02_c0.65` | False | False | False | 0.31%/64%/72% | 1.60%/75%/67% | 1.24%/75%/76% | is_fail |

## Best soft (not auto-promote unless promote=True)

- **idea:** `wrebal_0.328_0.029_0.241_0.178_0.224+peakprox_lb63_e0.01_c0.35_frozen`
- 2024 **1.27%** / 82% / 49%
- 2025_IS **1.28%** / 88% / 63%
- HO **1.45%** / 75% / 60% note=year_clear_fail

**Promote: NO. Locked tag unchanged unless promote.**
