# Scholarly FX: EPU/TPU-conditioned Lustig–Verdelhan carry wave (§78)

**Path:** Baker–Bloom–Davis **EPU** (US USEPUINDXM / GEPU) + **TPU** (categorical Trade policy) stress × Lustig–Verdelhan IR3M carry gate/cool — **distinct** from fwd_carry (§19), CIP×carry (§68), VIX/GPR×carry (§73), soft EW (§66), EPU/TPU soft (§76), EPU/TPU×REER (§77), capital-sleeve (§53/§70), combo (§8), standalone EPU/TPU (§18). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED USEPUINDXM/GEPU + policyuncertainty.com TPU + FRED IR3M rates (pub_lag=1m). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** EPU/TPU pub_lag_months=1 + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU soft §76 / REER §77 mirror) + weight_lag_days=1; carry signal_lag=1d. Costs 1.5 bps/side.
**Primary:** `carry_low_epu` (trade carry only when lagged EPU z ≤ 0). Companions: carry_epu_cool / carry_low_tpu / carry_tpu_cool / carry_raw / carry_high_epu / carry_epu_tpu_stack / carry_epu_tpu_ew. Cool z_high=1.0 for both EPU and TPU (§71–§77 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| carry_low_epu | -0.037% | -0.63 | -0.66 | 29% | 17% | -0.14 |
| carry_epu_cool | -0.040% | -0.61 | -0.69 | 51% | 12% | -0.15 |
| carry_low_tpu | -0.006% | -0.11 | -0.11 | 29% | 16% | -0.02 |
| carry_tpu_cool | -0.007% | -0.11 | -0.13 | 52% | 12% | -0.03 |
| carry_raw | -0.002% | -0.02 | -0.03 | 51% | 10% | -0.04 |
| carry_high_epu | +0.065% | +1.73 | +1.79 | 19% | 31% | +0.42 |
| carry_epu_tpu_stack | -0.032% | -0.55 | -0.61 | 52% | 15% | -0.12 |
| carry_epu_tpu_ew | -0.022% | -0.39 | -0.43 | 53% | 14% | -0.09 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| carry_raw | year_2024 | +0.15% | 64% | PASS | no |
| carry_raw | year_2025 | +0.12% | 64% | PASS | no |
| carry_raw | year_2026 | +0.13% | 62% | PASS | no |
| carry_raw | holdout_365d | +0.31% | 75% | PASS | no |
| carry_epu_cool | year_2024 | +0.15% | 64% | PASS | no |
| carry_epu_cool | year_2025 | -0.01% | 64% | PASS | no |
| carry_epu_cool | year_2026 | +0.02% | 62% | PASS | no |
| carry_epu_cool | holdout_365d | +0.10% | 75% | PASS | no |
| carry_low_epu | year_2024 | +0.15% | 64% | PASS | no |
| carry_low_epu | year_2025 | -0.00% | 9% | PASS | no |
| carry_low_epu | year_2026 | +0.00% | 0% | PASS | no |
| carry_low_epu | holdout_365d | +0.00% | 0% | PASS | no |
| carry_high_epu | year_2024 | +0.00% | 0% | PASS | no |
| carry_high_epu | year_2025 | +0.14% | 55% | PASS | no |
| carry_high_epu | year_2026 | +0.20% | 38% | PASS | no |
| carry_high_epu | holdout_365d | +0.20% | 42% | PASS | no |
| carry_tpu_cool | year_2024 | +0.16% | 64% | PASS | no |
| carry_tpu_cool | year_2025 | +0.06% | 73% | PASS | no |
| carry_tpu_cool | year_2026 | +0.02% | 62% | PASS | no |
| carry_tpu_cool | holdout_365d | +0.09% | 75% | PASS | no |
| carry_low_tpu | year_2024 | +0.20% | 73% | PASS | no |
| carry_low_tpu | year_2025 | +0.00% | 0% | PASS | no |
| carry_low_tpu | year_2026 | +0.00% | 0% | PASS | no |
| carry_low_tpu | holdout_365d | +0.00% | 0% | PASS | no |
| carry_epu_tpu_stack | year_2024 | +0.16% | 64% | PASS | no |
| carry_epu_tpu_stack | year_2025 | +0.01% | 73% | PASS | no |
| carry_epu_tpu_stack | year_2026 | -0.02% | 62% | PASS | no |
| carry_epu_tpu_stack | holdout_365d | +0.02% | 75% | PASS | no |
| carry_epu_tpu_ew | year_2024 | +0.16% | 73% | PASS | no |
| carry_epu_tpu_ew | year_2025 | +0.01% | 73% | PASS | no |
| carry_epu_tpu_ew | year_2026 | +0.01% | 62% | PASS | no |
| carry_epu_tpu_ew | holdout_365d | +0.05% | 75% | PASS | no |

**Board:** n=8 soft_nw_pos=1 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** EPU [1985-02-01..2026-09-01]; TPU [1985-02-01..2026-09-01]. Gate z≤0 / cool z≥1 fixed for both (§71–§77 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily).

Artifacts: `reports/scholarly_fx_epu_tpu_conditioned_carry_*.csv`, `scholarly_fx_epu_tpu_conditioned_carry_meta.json`, `quest_locked_verify_epu_tpu_conditioned_carry.md`.
