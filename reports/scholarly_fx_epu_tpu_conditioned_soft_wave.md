# Scholarly FX: EPU/TPU-conditioned soft-signal EW wave (§76)

**Path:** Baker–Bloom–Davis **EPU** (US USEPUINDXM / GEPU) + **TPU** (categorical Trade policy) stress × Dahlquist–Hasseltoft ``soft_ew_macro5`` gate/cool — **distinct** from standalone EPU/TPU wave, soft_ew (§66 ungated), soft-stack CIP enrichment (§69), CIP-conditioned carry (§68), CIP XS (§67), CIP-conditioned soft (§71), VIX/GPR soft (§72), carry/mom/value VIX/GPR (§73–§75), capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED USEPUINDXM/GEPU + policyuncertainty.com TPU + FRED/OECD soft legs (§66 PIT). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** EPU/TPU pub_lag_months=1 + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 mirror) + weight_lag_days=1; soft legs source-wave pub/signal lags. Costs 1.5 bps/side inside source factors.
**Primary:** `soft_low_epu` (trade soft only when lagged EPU z ≤ 0). Companions: soft_epu_cool / soft_low_tpu / soft_tpu_cool / soft_raw / soft_high_epu / soft_epu_tpu_stack / soft_epu_tpu_ew. Cool z_high=1.0 for both EPU and TPU (§71–§75 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| soft_low_epu | +0.061% | +2.81 | +2.63 | 37% | 18% | +0.68 |
| soft_epu_cool | +0.090% | +3.69 | +3.37 | 63% | 13% | +0.89 |
| soft_low_tpu | +0.079% | +3.38 | +2.83 | 39% | 16% | +0.85 |
| soft_tpu_cool | +0.103% | +4.08 | +3.48 | 61% | 13% | +1.02 |
| soft_raw | +0.118% | +4.26 | +3.63 | 61% | 11% | +1.02 |
| soft_high_epu | +0.041% | +3.19 | +2.16 | 21% | 29% | +0.85 |
| soft_epu_tpu_stack | +0.081% | +3.60 | +3.33 | 62% | 15% | +0.88 |
| soft_epu_tpu_ew | +0.083% | +3.78 | +3.38 | 62% | 13% | +0.92 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| soft_raw | year_2024 | +0.08% | 55% | PASS | no |
| soft_raw | year_2025 | +0.04% | 64% | PASS | no |
| soft_raw | year_2026 | +0.14% | 88% | PASS | no |
| soft_raw | holdout_365d | +0.06% | 67% | PASS | no |
| soft_epu_cool | year_2024 | +0.08% | 55% | PASS | no |
| soft_epu_cool | year_2025 | +0.02% | 73% | PASS | no |
| soft_epu_cool | year_2026 | +0.06% | 88% | PASS | no |
| soft_epu_cool | holdout_365d | +0.03% | 67% | PASS | no |
| soft_low_epu | year_2024 | +0.08% | 55% | PASS | no |
| soft_low_epu | year_2025 | +0.01% | 18% | PASS | no |
| soft_low_epu | year_2026 | +0.00% | 0% | PASS | no |
| soft_low_epu | holdout_365d | +0.00% | 0% | PASS | no |
| soft_high_epu | year_2024 | +0.00% | 0% | PASS | no |
| soft_high_epu | year_2025 | +0.04% | 45% | PASS | no |
| soft_high_epu | year_2026 | +0.08% | 75% | PASS | no |
| soft_high_epu | holdout_365d | +0.05% | 58% | PASS | no |
| soft_tpu_cool | year_2024 | +0.09% | 55% | PASS | no |
| soft_tpu_cool | year_2025 | +0.01% | 64% | PASS | no |
| soft_tpu_cool | year_2026 | +0.08% | 88% | PASS | no |
| soft_tpu_cool | holdout_365d | +0.04% | 67% | PASS | no |
| soft_low_tpu | year_2024 | +0.12% | 73% | PASS | no |
| soft_low_tpu | year_2025 | +0.00% | 0% | PASS | no |
| soft_low_tpu | year_2026 | +0.00% | 0% | PASS | no |
| soft_low_tpu | holdout_365d | +0.00% | 0% | PASS | no |
| soft_epu_tpu_stack | year_2024 | +0.09% | 55% | PASS | no |
| soft_epu_tpu_stack | year_2025 | +0.01% | 73% | PASS | no |
| soft_epu_tpu_stack | year_2026 | +0.03% | 88% | PASS | no |
| soft_epu_tpu_stack | holdout_365d | +0.02% | 67% | PASS | no |
| soft_epu_tpu_ew | year_2024 | +0.10% | 55% | PASS | no |
| soft_epu_tpu_ew | year_2025 | +0.01% | 73% | PASS | no |
| soft_epu_tpu_ew | year_2026 | +0.03% | 88% | PASS | no |
| soft_epu_tpu_ew | holdout_365d | +0.02% | 67% | PASS | no |

**Board:** n=8 soft_nw_pos=8 hard_nw_pos=8 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** EPU [1985-02-01..2026-09-01]; TPU [1985-02-01..2026-09-01]. Gate z≤0 / cool z≥1 fixed for both (§71–§75 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily).

Artifacts: `reports/scholarly_fx_epu_tpu_conditioned_soft_*.csv`, `scholarly_fx_epu_tpu_conditioned_soft_meta.json`, `quest_locked_verify_epu_tpu_conditioned_soft.md`.
