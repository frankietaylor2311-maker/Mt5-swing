# Quest wave: literature-backed macro / geopolitics factors

**Data:** `approximate_non_ftmo`. **RF:** 0.08 (no hike). **Holdout:** confirm only.
**NOT** an overlay-knob grid on locked equity — new research line.
**Board:** n=6 soft=0 hard=0 promote=0

## Idea vs tuning

| Label | Kind | Notes |
|-------|------|-------|
| fred_carry_d1 | **idea** | FRED rate-diff carry (Lustig–Verdelhan sign); fixed min_diff=0.25, month lag=1 |
| locked+vix_gate | **idea** | Menkhoff vol-risk gate; fixed z≥1.0 → scale 0.35 |
| locked+gpr_gate | **idea** | Caldara–Iacoviello / Liu–Zhang GPR risk-off; fixed z≥1.5 → scale 0.35 |
| locked+vix_gpr / fred_carry+vix_gpr | **idea** | Sequential literature gates (not a cool-knob search) |
| baseline_locked_vt | baseline | Official locked tag reference |

## Results

| Idea | soft | hard | promote | reason | IS min_mo | HO mo/%pos |
|------|:----:|:----:|:-------:|--------|----------:|-----------|
| `baseline_locked_vt` | False | False | False | soft_mean_fail | 0.36% | 1.54%/67% |
| `fred_carry_d1` | False | False | False | soft_mean_fail | -0.09% | 0.06%/50% |
| `locked+vix_gate` | False | False | False | soft_mean_fail | -0.19% | 1.55%/58% |
| `locked+gpr_gate` | False | False | False | soft_mean_fail | 0.41% | 1.75%/67% |
| `locked+vix_gpr` | False | False | False | soft_mean_fail | -0.19% | 1.75%/58% |
| `fred_carry+vix_gpr` | False | False | False | soft_mean_fail | 0.14% | -0.12%/50% |

## Datasets used vs required

See `reports/MACRO_FACTOR_LITERATURE.md`.

**Promote?** **NO** — locked tag unchanged.

