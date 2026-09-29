# Scholarly FX: EPU/TPU-conditioned Rogoff PPP / real-FX value wave (§80)

**Path:** Baker–Bloom–Davis **EPU** (US USEPUINDXM / GEPU) + **TPU** (categorical Trade policy) stress × Rogoff PPP / real-FX cross-sectional value gate/cool — **distinct** from raw PPP, BIS REER (§31), VIX/GPR×PPP (§75), EPU/TPU soft (§76), EPU/TPU×REER (§77), EPU/TPU×carry (§78), EPU/TPU×mom (§79), capital-sleeve (§53/§70), combo (§8), standalone EPU/TPU (§18). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED USEPUINDXM/GEPU + policyuncertainty.com TPU + FRED CPI levels (pub_lag=1m) + Yahoo D1 G10. **PIT:** EPU/TPU pub_lag_months=1 + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU soft §76 / REER §77 / carry §78 / mom §79 mirror) + weight_lag_days=1; PPP lookback=60m, CPI pub_lag + ppp_signal_lag=1m + weight lag 1d. Costs 1.5 bps/side.
**Primary:** `value_low_epu` (trade value only when lagged EPU z ≤ 0). Companions: value_epu_cool / value_low_tpu / value_tpu_cool / value_raw / value_high_epu / value_epu_tpu_stack / value_epu_tpu_ew. Cool z_high=1.0 for both EPU and TPU (§71–§79 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| value_low_epu | -0.071% | -1.64 | -1.62 | 24% | 21% | -0.35 |
| value_epu_cool | -0.039% | -0.77 | -0.78 | 43% | 13% | -0.16 |
| value_low_tpu | -0.016% | -0.38 | -0.36 | 23% | 21% | -0.08 |
| value_tpu_cool | -0.012% | -0.25 | -0.24 | 42% | 14% | -0.05 |
| value_raw | -0.014% | -0.22 | -0.22 | 42% | 12% | -0.05 |
| value_high_epu | +0.030% | +1.03 | +0.92 | 14% | 36% | +0.23 |
| value_epu_tpu_stack | -0.039% | -0.88 | -0.88 | 43% | 15% | -0.19 |
| value_epu_tpu_ew | -0.035% | -0.79 | -0.78 | 43% | 15% | -0.17 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| value_raw | year_2024 | -0.10% | 64% | PASS | no |
| value_raw | year_2025 | -0.30% | 27% | PASS | no |
| value_raw | year_2026 | +0.15% | 75% | PASS | no |
| value_raw | holdout_365d | +0.10% | 67% | PASS | no |
| value_epu_cool | year_2024 | -0.10% | 64% | PASS | no |
| value_epu_cool | year_2025 | -0.19% | 36% | PASS | no |
| value_epu_cool | year_2026 | +0.07% | 75% | PASS | no |
| value_epu_cool | holdout_365d | +0.04% | 67% | PASS | no |
| value_low_epu | year_2024 | -0.10% | 64% | PASS | no |
| value_low_epu | year_2025 | -0.02% | 18% | PASS | no |
| value_low_epu | year_2026 | +0.00% | 0% | PASS | no |
| value_low_epu | holdout_365d | +0.00% | 0% | PASS | no |
| value_high_epu | year_2024 | +0.00% | 0% | PASS | no |
| value_high_epu | year_2025 | -0.20% | 18% | PASS | no |
| value_high_epu | year_2026 | +0.06% | 62% | PASS | no |
| value_high_epu | holdout_365d | +0.04% | 50% | PASS | no |
| value_tpu_cool | year_2024 | -0.11% | 64% | PASS | no |
| value_tpu_cool | year_2025 | -0.10% | 27% | PASS | no |
| value_tpu_cool | year_2026 | +0.10% | 75% | PASS | no |
| value_tpu_cool | holdout_365d | +0.07% | 67% | PASS | no |
| value_low_tpu | year_2024 | -0.14% | 64% | PASS | no |
| value_low_tpu | year_2025 | +0.00% | 0% | PASS | no |
| value_low_tpu | year_2026 | +0.00% | 0% | PASS | no |
| value_low_tpu | holdout_365d | +0.00% | 0% | PASS | no |
| value_epu_tpu_stack | year_2024 | -0.11% | 64% | PASS | no |
| value_epu_tpu_stack | year_2025 | -0.06% | 36% | PASS | no |
| value_epu_tpu_stack | year_2026 | +0.05% | 75% | PASS | no |
| value_epu_tpu_stack | holdout_365d | +0.03% | 67% | PASS | no |
| value_epu_tpu_ew | year_2024 | -0.11% | 64% | PASS | no |
| value_epu_tpu_ew | year_2025 | -0.07% | 36% | PASS | no |
| value_epu_tpu_ew | year_2026 | +0.04% | 75% | PASS | no |
| value_epu_tpu_ew | holdout_365d | +0.03% | 67% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** EPU [1985-02-01..2026-09-01]; TPU [1985-02-01..2026-09-01]; CPI [1914-02-01..2026-09-01]. Gate z≤0 / cool z≥1 fixed for both (§71–§79 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). PPP lookback=60m (honesty 120m in raw ppp wave). AU/NZ CPI quarterly ffilled.

Artifacts: `reports/scholarly_fx_epu_tpu_conditioned_value_*.csv`, `scholarly_fx_epu_tpu_conditioned_value_meta.json`, `quest_locked_verify_epu_tpu_conditioned_value.md`.
