# Scholarly FX: EPU/TPU-conditioned Menkhoff FX momentum wave (§79)

**Path:** Baker–Bloom–Davis **EPU** (US USEPUINDXM / GEPU) + **TPU** (categorical Trade policy) stress × Menkhoff (2012 JFE) FX momentum gate/cool — **distinct** from combo (§8), raw fx_momentum, soft VIX/GPR (§72), carry VIX/GPR (§73), mom VIX/GPR (§74), value VIX/GPR (§75), EPU/TPU soft (§76), EPU/TPU×REER (§77), EPU/TPU×carry (§78), capital-sleeve (§53/§70), standalone EPU/TPU (§18). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED USEPUINDXM/GEPU + policyuncertainty.com TPU + Yahoo D1 G10. **PIT:** EPU/TPU pub_lag_months=1 + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU soft §76 / REER §77 / carry §78 mirror) + weight_lag_days=1; mom formation=63d skip=21d signal_lag=1d. Costs 1.5 bps/side.
**Primary:** `mom_low_epu` (trade mom only when lagged EPU z ≤ 0). Companions: mom_epu_cool / mom_low_tpu / mom_tpu_cool / mom_raw / mom_high_epu / mom_epu_tpu_stack / mom_epu_tpu_ew. Cool z_high=1.0 for both EPU and TPU (§71–§78 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| mom_low_epu | -0.027% | -0.50 | -0.57 | 28% | 18% | -0.12 |
| mom_epu_cool | -0.068% | -1.11 | -1.28 | 45% | 13% | -0.26 |
| mom_low_tpu | -0.061% | -1.15 | -1.34 | 29% | 18% | -0.28 |
| mom_tpu_cool | -0.070% | -1.18 | -1.37 | 46% | 16% | -0.28 |
| mom_raw | -0.076% | -1.08 | -1.10 | 45% | 12% | -0.25 |
| mom_high_epu | -0.024% | -0.80 | -0.73 | 13% | 35% | -0.18 |
| mom_epu_tpu_stack | -0.059% | -1.09 | -1.34 | 46% | 16% | -0.27 |
| mom_epu_tpu_ew | -0.056% | -1.07 | -1.29 | 46% | 14% | -0.26 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| mom_raw | year_2024 | -0.19% | 45% | PASS | no |
| mom_raw | year_2025 | +0.43% | 91% | PASS | no |
| mom_raw | year_2026 | +0.33% | 62% | PASS | no |
| mom_raw | holdout_365d | +0.37% | 75% | PASS | no |
| mom_epu_cool | year_2024 | -0.19% | 45% | PASS | no |
| mom_epu_cool | year_2025 | +0.20% | 91% | PASS | no |
| mom_epu_cool | year_2026 | +0.11% | 75% | PASS | no |
| mom_epu_cool | holdout_365d | +0.14% | 83% | PASS | no |
| mom_low_epu | year_2024 | -0.19% | 45% | PASS | no |
| mom_low_epu | year_2025 | +0.03% | 18% | PASS | no |
| mom_low_epu | year_2026 | +0.00% | 0% | PASS | no |
| mom_low_epu | holdout_365d | +0.00% | 0% | PASS | no |
| mom_high_epu | year_2024 | +0.00% | 0% | PASS | no |
| mom_high_epu | year_2025 | +0.28% | 64% | PASS | no |
| mom_high_epu | year_2026 | +0.29% | 38% | PASS | no |
| mom_high_epu | holdout_365d | +0.24% | 42% | PASS | no |
| mom_tpu_cool | year_2024 | -0.19% | 45% | PASS | no |
| mom_tpu_cool | year_2025 | +0.16% | 91% | PASS | no |
| mom_tpu_cool | year_2026 | +0.19% | 75% | PASS | no |
| mom_tpu_cool | holdout_365d | +0.18% | 83% | PASS | no |
| mom_low_tpu | year_2024 | -0.22% | 45% | PASS | no |
| mom_low_tpu | year_2025 | +0.00% | 0% | PASS | no |
| mom_low_tpu | year_2026 | +0.00% | 0% | PASS | no |
| mom_low_tpu | holdout_365d | +0.00% | 0% | PASS | no |
| mom_epu_tpu_stack | year_2024 | -0.19% | 45% | PASS | no |
| mom_epu_tpu_stack | year_2025 | +0.08% | 91% | PASS | no |
| mom_epu_tpu_stack | year_2026 | +0.07% | 75% | PASS | no |
| mom_epu_tpu_stack | holdout_365d | +0.07% | 83% | PASS | no |
| mom_epu_tpu_ew | year_2024 | -0.20% | 45% | PASS | no |
| mom_epu_tpu_ew | year_2025 | +0.10% | 91% | PASS | no |
| mom_epu_tpu_ew | year_2026 | +0.08% | 75% | PASS | no |
| mom_epu_tpu_ew | holdout_365d | +0.08% | 83% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** EPU [1985-02-01..2026-09-01]; TPU [1985-02-01..2026-09-01]. Gate z≤0 / cool z≥1 fixed for both (§71–§78 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily).

Artifacts: `reports/scholarly_fx_epu_tpu_conditioned_mom_*.csv`, `scholarly_fx_epu_tpu_conditioned_mom_meta.json`, `quest_locked_verify_epu_tpu_conditioned_mom.md`.
