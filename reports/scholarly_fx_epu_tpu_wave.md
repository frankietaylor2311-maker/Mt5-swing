# Scholarly FX: EPU / TPU wave (Baker–Bloom–Davis)

**Path:** FX IV/RR **unavailable** free → EPU/TPU fallback.
**Data:** `approximate_non_ftmo` + FRED USEPUINDXM/GEPUCURRENT + policyuncertainty.com All_Country + Categorical Trade policy (TPU). **PIT:** pub_lag=1m + signal_lag_months=1 + signal_lag_days=1.
**Primary:** `country_epu_xs`. Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| country_epu_xs | +0.031% | +0.54 | +0.63 | 53% | 12% | +0.17 |
| epu_diff_xs | +0.031% | +0.54 | +0.63 | 53% | 12% | +0.17 |
| epu_us_usd | +0.011% | +1.06 | +0.87 | 17% | 29% | +0.28 |
| tpu_us_usd | +0.011% | +1.38 | +1.13 | 19% | 27% | +0.29 |
| gepu_usd | +0.001% | +0.14 | +0.11 | 19% | 33% | +0.03 |
| carry_epu_cool | +0.035% | +0.53 | +0.60 | 52% | 11% | +0.08 |
| carry_tpu_cool | -0.005% | -0.07 | -0.08 | 52% | 12% | -0.05 |
| epu_ew | +0.021% | +0.70 | +0.81 | 53% | 12% | +0.21 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| country_epu_xs | year_2024 | -0.07% | 36% | PASS | no |
| country_epu_xs | year_2025 | +0.05% | 55% | PASS | no |
| country_epu_xs | year_2026 | +0.19% | 62% | PASS | no |
| country_epu_xs | holdout_365d | +0.30% | 75% | PASS | no |
| epu_diff_xs | year_2024 | -0.07% | 36% | PASS | no |
| epu_diff_xs | year_2025 | +0.05% | 55% | PASS | no |
| epu_diff_xs | year_2026 | +0.19% | 62% | PASS | no |
| epu_diff_xs | holdout_365d | +0.30% | 75% | PASS | no |
| epu_us_usd | year_2024 | +0.04% | 9% | PASS | no |
| epu_us_usd | year_2025 | -0.12% | 0% | PASS | no |
| epu_us_usd | year_2026 | +0.00% | 0% | PASS | no |
| epu_us_usd | holdout_365d | +0.00% | 0% | PASS | no |
| tpu_us_usd | year_2024 | +0.05% | 64% | PASS | no |
| tpu_us_usd | year_2025 | -0.12% | 0% | PASS | no |
| tpu_us_usd | year_2026 | +0.00% | 0% | PASS | no |
| tpu_us_usd | holdout_365d | +0.00% | 0% | PASS | no |
| gepu_usd | year_2024 | +0.01% | 18% | PASS | no |
| gepu_usd | year_2025 | -0.12% | 0% | PASS | no |
| gepu_usd | year_2026 | +0.00% | 0% | PASS | no |
| gepu_usd | holdout_365d | +0.00% | 0% | PASS | no |
| carry_epu_cool | year_2024 | +0.20% | 64% | PASS | no |
| carry_epu_cool | year_2025 | +0.17% | 64% | PASS | no |
| carry_epu_cool | year_2026 | +0.13% | 62% | PASS | no |
| carry_epu_cool | holdout_365d | +0.31% | 75% | PASS | no |
| carry_tpu_cool | year_2024 | +0.01% | 55% | PASS | no |
| carry_tpu_cool | year_2025 | +0.16% | 64% | PASS | no |
| carry_tpu_cool | year_2026 | +0.13% | 62% | PASS | no |
| carry_tpu_cool | holdout_365d | +0.30% | 75% | PASS | no |
| epu_ew | year_2024 | -0.02% | 36% | PASS | no |
| epu_ew | year_2025 | -0.04% | 45% | PASS | no |
| epu_ew | year_2026 | +0.09% | 62% | PASS | no |
| epu_ew | holdout_365d | +0.15% | 75% | PASS | no |

**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_epu_tpu_*.csv`, `scholarly_fx_epu_tpu_meta.json`.
