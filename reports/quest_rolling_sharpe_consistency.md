# Rolling Sharpe cool consistency wave

**Data:** `approximate_non_ftmo`  **RF:** 0.08  **VT:** 0.0025  **Locked:** `fx4plus_gbpcad_d1_voltarget_0025`
**Board:** n=20 soft=**3** hard=**0** promote=**0**

Sharpe = daily mean/std × √252 (annualized); lag-1 day; cool if >high or <low; hi≤1.

## Board (top by score)

| Idea | Soft | Hard | Promote | 2024 mo/pos/top3 | 2025_IS | HO | Note |
|------|:----:|:----:|:-------:|------------------|---------|----|------|
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen` | True | False | False | 1.18%/82%/45% | 1.74%/75%/64% | 1.69%/67%/71% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen+` | True | False | False | 1.02%/82%/39% | 1.51%/75%/65% | 1.06%/67%/67% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.65_frozen` | True | False | False | 1.03%/82%/48% | 1.95%/75%/66% | 1.70%/67%/73% | holdout_fail |
| `baseline_locked` | False | False | False | 0.42%/55%/74% | 2.33%/75%/70% | 1.52%/75%/81% | is_fail |
| `lock+rsharpe_lb42_hi1.0_lo-1.0_c0.5` | False | False | False | 0.44%/64%/70% | 1.46%/88%/65% | 0.89%/75%/68% | is_fail |
| `lock+rsharpe_lb42_hi1.0_lo-1.0_c0.35` | False | False | False | 0.44%/64%/69% | 1.20%/88%/60% | 0.70%/75%/62% | is_fail |
| `lock+rsharpe_lb63_hi1.5_lo0.0_c0.35` | False | False | False | 0.62%/64%/69% | 1.02%/75%/64% | 0.37%/67%/72% | is_fail |
| `lock+rsharpe_lb63_hi1.5_lo0.0_c0.5` | False | False | False | 0.58%/64%/70% | 1.32%/75%/66% | 0.63%/67%/77% | is_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224_frozen` | False | False | False | 0.68%/82%/59% | 2.44%/75%/70% | 1.72%/67%/77% | soft_mean_fail |
| `lock+rsharpe_lb63_hi1.0_lo0.0_c0.5` | False | False | False | 0.38%/64%/73% | 1.28%/75%/70% | 0.65%/67%/84% | is_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+rsharpe_lb42_hi1.0_lo-1` | False | False | False | 0.78%/82%/51% | 1.77%/75%/68% | 1.09%/67%/73% | soft_mean_fail |
| `lock+rsharpe_lb63_hi1.5_lo0.0_c0.65` | False | False | False | 0.53%/64%/71% | 1.62%/75%/68% | 0.90%/75%/80% | is_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+rsharpe_lb42_hi1.0_lo-1` | False | False | False | 0.82%/82%/53% | 1.49%/88%/65% | 0.82%/67%/71% | soft_mean_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.65_frozen` | False | False | False | 0.93%/82%/41% | 1.66%/75%/66% | 1.09%/67%/69% | soft_mean_fail |
| `blend_a0.40_lock_vs_wrebal_0.328_0.029_0.241_0.178_0.224+mtd` | False | False | False | 0.66%/73%/59% | 2.00%/75%/68% | 1.34%/75%/77% | soft_mean_fail |

## Best soft (not auto-promote unless promote=True)

- **idea:** `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen`
- 2024 **1.18%** / 82% / 45%
- 2025_IS **1.74%** / 75% / 64%
- HO **1.69%** / 67% / 71% note=holdout_fail

**Promote: NO. Locked tag unchanged unless promote.**
