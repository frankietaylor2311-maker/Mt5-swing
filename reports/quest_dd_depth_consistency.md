# Rolling DD-depth percentile cool consistency wave

**Data:** `approximate_non_ftmo`  **RF:** 0.08  **VT:** 0.0025  **Locked:** `fx4plus_gbpcad_d1_voltarget_0025`
**Board:** n=32 soft=**4** hard=**0** promote=**0**

DD-depth = lag-1 drawdown depth at/below causal rolling LOW percentile → cool; hi≤1.
Distinct from peak-proximity (fixed eps) and ret-pctile (rich trail return).

## Board (top by score)

| Idea | Soft | Hard | Promote | 2024 mo/pos/top3 | 2025_IS | HO | Note |
|------|:----:|:----:|:-------:|------------------|---------|----|------|
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen+` | True | False | False | 1.20%/82%/44% | 1.52%/75%/64% | 1.43%/67%/68% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen` | True | False | False | 1.18%/82%/45% | 1.74%/75%/64% | 1.69%/67%/71% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.65_frozen` | True | False | False | 1.03%/82%/48% | 1.95%/75%/66% | 1.70%/67%/73% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.65_frozen` | True | False | False | 0.96%/82%/41% | 1.67%/75%/65% | 1.48%/75%/68% | year_clear_fail |
| `baseline_locked` | False | False | False | 0.42%/55%/74% | 2.33%/75%/70% | 1.52%/75%/81% | is_fail |
| `lock+dddepth_pb42_hb126_p0.2_c0.35` | False | False | False | 0.54%/55%/81% | 1.91%/75%/63% | 1.17%/67%/77% | is_fail |
| `lock+dddepth_pb42_hb126_p0.2_c0.5` | False | False | False | 0.51%/55%/78% | 2.01%/75%/65% | 1.25%/67%/78% | is_fail |
| `lock+dddepth_pb42_hb126_p0.2_c0.65` | False | False | False | 0.49%/55%/75% | 2.11%/75%/67% | 1.33%/75%/79% | is_fail |
| `lock+dddepth_pb63_hb126_p0.2_c0.5` | False | False | False | 0.37%/55%/71% | 2.05%/75%/66% | 1.00%/67%/73% | is_fail |
| `lock+dddepth_pb63_hb126_p0.2_c0.65` | False | False | False | 0.39%/55%/71% | 2.13%/75%/67% | 1.16%/67%/76% | is_fail |
| `lock+dddepth_pb63_hb252_p0.2_c0.35` | False | False | False | 0.37%/55%/85% | 2.05%/75%/66% | 0.92%/67%/69% | is_fail |
| `lock+dddepth_pb63_hb252_p0.2_c0.5` | False | False | False | 0.38%/55%/81% | 2.11%/75%/67% | 1.06%/67%/73% | is_fail |
| `lock+dddepth_pb42_hb252_p0.2_c0.35` | False | False | False | 0.49%/55%/81% | 1.90%/75%/64% | 0.75%/67%/71% | is_fail |
| `lock+dddepth_pb42_hb252_p0.2_c0.5` | False | False | False | 0.47%/55%/78% | 2.00%/75%/65% | 0.93%/67%/74% | is_fail |
| `lock+dddepth_pb42_hb252_p0.2_c0.65` | False | False | False | 0.46%/55%/76% | 2.10%/75%/67% | 1.11%/75%/76% | is_fail |

## Best soft (not auto-promote unless promote=True)

- **idea:** `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen+dddepth_pb63_hb126_p0.15_c0.65`
- 2024 **1.20%** / 82% / 44%
- 2025_IS **1.52%** / 75% / 64%
- HO **1.43%** / 67% / 68% note=holdout_fail

**Promote: NO. Locked tag unchanged unless promote.**
