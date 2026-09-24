# Scholarly FX: EPU/TPU-conditioned BIS REER HML-FX value wave (§77)

**Path:** Baker–Bloom–Davis **EPU** (US USEPUINDXM / GEPU) + **TPU** (categorical Trade policy) stress × BIS multilateral REER undervaluation / HML-FX value (``reer_cheap_xs`` §31) gate/cool — **distinct** from raw bis_reer (§31), raw ppp, VIX/GPR×PPP value (§75), soft EW (§66), EPU/TPU soft (§76), carry/mom VIX/GPR (§73–§74), CIP waves, capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED USEPUINDXM/GEPU + policyuncertainty.com TPU + FRED BIS RB*BIS (pub_lag=2m). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** EPU/TPU pub_lag_months=1 + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU soft §76 mirror) + weight_lag_days=1; REER pub_lag=2 + signal_lag=1m + weight lag 1d. Costs 1.5 bps/side.
**Primary:** `reer_low_epu` (trade REER-value only when lagged EPU z ≤ 0). Companions: reer_epu_cool / reer_low_tpu / reer_tpu_cool / reer_raw / reer_high_epu / reer_epu_tpu_stack / reer_epu_tpu_ew. Cool z_high=1.0 for both EPU and TPU (§71–§76 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| reer_low_epu | +0.065% | +1.29 | +1.32 | 33% | 20% | +0.32 |
| reer_epu_cool | +0.042% | +0.73 | +0.79 | 51% | 16% | +0.19 |
| reer_low_tpu | +0.004% | +0.09 | +0.09 | 31% | 21% | +0.02 |
| reer_tpu_cool | +0.034% | +0.63 | +0.74 | 52% | 16% | +0.16 |
| reer_raw | +0.034% | +0.51 | +0.60 | 52% | 13% | +0.15 |
| reer_high_epu | -0.024% | -0.82 | -0.99 | 13% | 39% | -0.18 |
| reer_epu_tpu_stack | +0.034% | +0.69 | +0.75 | 51% | 18% | +0.17 |
| reer_epu_tpu_ew | +0.036% | +0.75 | +0.82 | 51% | 16% | +0.19 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| reer_raw | year_2024 | -0.11% | 36% | PASS | no |
| reer_raw | year_2025 | -0.27% | 27% | PASS | no |
| reer_raw | year_2026 | +0.16% | 62% | PASS | no |
| reer_raw | holdout_365d | +0.08% | 58% | PASS | no |
| reer_epu_cool | year_2024 | -0.11% | 36% | PASS | no |
| reer_epu_cool | year_2025 | -0.17% | 27% | PASS | no |
| reer_epu_cool | year_2026 | +0.09% | 62% | PASS | no |
| reer_epu_cool | holdout_365d | +0.05% | 58% | PASS | no |
| reer_low_epu | year_2024 | -0.11% | 36% | PASS | no |
| reer_low_epu | year_2025 | -0.09% | 9% | PASS | no |
| reer_low_epu | year_2026 | +0.00% | 0% | PASS | no |
| reer_low_epu | holdout_365d | +0.00% | 0% | PASS | no |
| reer_high_epu | year_2024 | +0.00% | 0% | PASS | no |
| reer_high_epu | year_2025 | -0.14% | 18% | PASS | no |
| reer_high_epu | year_2026 | +0.01% | 50% | PASS | no |
| reer_high_epu | holdout_365d | -0.04% | 42% | PASS | no |
| reer_tpu_cool | year_2024 | -0.13% | 36% | PASS | no |
| reer_tpu_cool | year_2025 | -0.10% | 18% | PASS | no |
| reer_tpu_cool | year_2026 | +0.11% | 62% | PASS | no |
| reer_tpu_cool | holdout_365d | +0.07% | 58% | PASS | no |
| reer_low_tpu | year_2024 | -0.19% | 36% | PASS | no |
| reer_low_tpu | year_2025 | +0.00% | 0% | PASS | no |
| reer_low_tpu | year_2026 | +0.00% | 0% | PASS | no |
| reer_low_tpu | holdout_365d | +0.00% | 0% | PASS | no |
| reer_epu_tpu_stack | year_2024 | -0.13% | 36% | PASS | no |
| reer_epu_tpu_stack | year_2025 | -0.07% | 18% | PASS | no |
| reer_epu_tpu_stack | year_2026 | +0.07% | 62% | PASS | no |
| reer_epu_tpu_stack | holdout_365d | +0.04% | 58% | PASS | no |
| reer_epu_tpu_ew | year_2024 | -0.13% | 36% | PASS | no |
| reer_epu_tpu_ew | year_2025 | -0.09% | 18% | PASS | no |
| reer_epu_tpu_ew | year_2026 | +0.05% | 62% | PASS | no |
| reer_epu_tpu_ew | holdout_365d | +0.03% | 58% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** EPU [1985-02-01..2026-09-01]; TPU [1985-02-01..2026-09-01]; REER [1994-03-01..2026-09-01]. Gate z≤0 / cool z≥1 fixed for both (§71–§76 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily).

Artifacts: `reports/scholarly_fx_epu_tpu_conditioned_reer_*.csv`, `scholarly_fx_epu_tpu_conditioned_reer_meta.json`, `quest_locked_verify_epu_tpu_conditioned_reer.md`.
