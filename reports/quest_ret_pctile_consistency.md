# Rolling-return percentile cool consistency wave

**Data:** `approximate_non_ftmo`  **RF:** 0.08  **VT:** 0.0025  **Locked:** `fx4plus_gbpcad_d1_voltarget_0025`
**Board:** n=32 soft=**3** hard=**0** promote=**0**

Ret-pctile = lag-1 trailing return at/above causal rolling percentile → cool; hi≤1.
Distinct from runup (fixed thresh) and peak-proximity (near high).

## Board (top by score)

| Idea | Soft | Hard | Promote | 2024 mo/pos/top3 | 2025_IS | HO | Note |
|------|:----:|:----:|:-------:|------------------|---------|----|------|
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen` | True | False | False | 1.18%/82%/45% | 1.74%/75%/64% | 1.69%/67%/71% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen+` | True | False | False | 1.01%/82%/41% | 1.70%/75%/63% | 1.70%/75%/73% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.65_frozen` | True | False | False | 1.03%/82%/48% | 1.95%/75%/66% | 1.70%/67%/73% | holdout_fail |
| `baseline_locked` | False | False | False | 0.42%/55%/74% | 2.33%/75%/70% | 1.52%/75%/81% | is_fail |
| `lock+retpct_tb21_hb252_p0.85_c0.35` | False | False | False | 0.48%/55%/90% | 1.95%/75%/69% | 1.15%/67%/79% | is_fail |
| `lock+retpct_tb21_hb126_p0.85_c0.35` | False | False | False | 0.32%/55%/77% | 2.25%/75%/67% | 1.25%/67%/82% | is_fail |
| `lock+retpct_tb21_hb126_p0.85_c0.5` | False | False | False | 0.34%/55%/75% | 2.27%/75%/68% | 1.31%/67%/81% | is_fail |
| `lock+retpct_tb21_hb126_p0.85_c0.65` | False | False | False | 0.37%/55%/73% | 2.29%/75%/68% | 1.38%/67%/81% | is_fail |
| `lock+retpct_tb42_hb126_p0.85_c0.5` | False | False | False | 0.29%/55%/84% | 2.10%/75%/68% | 1.38%/67%/78% | is_fail |
| `lock+retpct_tb42_hb126_p0.85_c0.65` | False | False | False | 0.33%/55%/78% | 2.17%/75%/69% | 1.42%/67%/78% | is_fail |
| `lock+retpct_tb42_hb252_p0.85_c0.35` | False | False | False | 0.14%/45%/95% | 2.22%/75%/67% | 1.15%/75%/78% | is_fail |
| `lock+retpct_tb42_hb252_p0.85_c0.5` | False | False | False | 0.21%/55%/88% | 2.24%/75%/67% | 1.24%/75%/79% | is_fail |
| `lock+retpct_tb42_hb252_p0.85_c0.65` | False | False | False | 0.27%/55%/81% | 2.27%/75%/68% | 1.32%/75%/80% | is_fail |
| `lock+retpct_tb21_hb252_p0.85_c0.5` | False | False | False | 0.47%/55%/85% | 2.04%/75%/69% | 1.24%/67%/79% | is_fail |
| `lock+retpct_tb21_hb252_p0.85_c0.65` | False | False | False | 0.45%/55%/81% | 2.13%/75%/69% | 1.32%/67%/79% | is_fail |

## Best soft (not auto-promote unless promote=True)

- **idea:** `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen`
- 2024 **1.18%** / 82% / 45%
- 2025_IS **1.74%** / 75% / 64%
- HO **1.69%** / 67% / 71% note=holdout_fail

**Promote: NO. Locked tag unchanged unless promote.**
