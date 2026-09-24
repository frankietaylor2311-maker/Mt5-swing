# Scholarly FX: BIS/FRED household-credit differential wave (§63)

**Path:** FRED BIS **household-credit YoY** (`CRDQ*AHABIS` abs→YoY; EUR=XM euro-area; NZD unmapped; not %GDP) — HH-credit YoY XS — **distinct** from private credit/GDP (§32), money-growth (§33), DSR (§47), debt/GDP (§29), house-price (§35), retail (§46), cars (§60), import-value (§62).
**Data:** `approximate_non_ftmo` + free FRED CRDQ*AHABIS YoY after pub_lag. EUR=CRDQXMAHABIS euro-area (CRDQEAAHABIS 404); NZD unmapped. `n_long=n_short=2` on available foreign panel. **PIT:** pub_lag_months=5 (BIS credit ~4–5m lag a priori) + signal_lag_months=1 + 1d weight lag. Score basis = **household-credit YoY %** (abs→YoY, not %GDP).
**Primary:** `high_hhcred_xs` (long high relative household-credit YoY / short low — consumer-demand / credit-financed absorption → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_hhcred_xs | -0.055% | -0.81 | -0.83 | 48% | 12% | -0.22 |
| low_hhcred_xs | +0.051% | +0.75 | +0.77 | 52% | 9% | +0.21 |
| high_hhcred_z_xs | +0.067% | +0.98 | +1.00 | 52% | 17% | +0.27 |
| hhcred_chg_xs | +0.015% | +0.22 | +0.26 | 49% | 16% | +0.07 |
| us_hhcred_stress_fx | +0.002% | +0.12 | +0.21 | 4% | 54% | +0.03 |
| us_hhcred_haven_usd | -0.002% | -0.12 | -0.21 | 5% | 66% | -0.03 |
| hhcred_ew | -0.013% | -0.40 | -0.41 | 48% | 19% | -0.10 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_hhcred_xs | year_2024 | -0.09% | 55% | PASS | no |
| high_hhcred_xs | year_2025 | -0.22% | 36% | PASS | no |
| high_hhcred_xs | year_2026 | +0.28% | 75% | PASS | no |
| high_hhcred_xs | holdout_365d | +0.35% | 83% | PASS | no |
| low_hhcred_xs | year_2024 | +0.09% | 45% | PASS | no |
| low_hhcred_xs | year_2025 | +0.21% | 64% | PASS | no |
| low_hhcred_xs | year_2026 | -0.28% | 25% | PASS | no |
| low_hhcred_xs | holdout_365d | -0.36% | 17% | PASS | no |
| high_hhcred_z_xs | year_2024 | -0.05% | 64% | PASS | no |
| high_hhcred_z_xs | year_2025 | -0.25% | 36% | PASS | no |
| high_hhcred_z_xs | year_2026 | +0.19% | 62% | PASS | no |
| high_hhcred_z_xs | holdout_365d | +0.29% | 67% | PASS | no |
| hhcred_chg_xs | year_2024 | -0.05% | 45% | PASS | no |
| hhcred_chg_xs | year_2025 | -0.04% | 45% | PASS | no |
| hhcred_chg_xs | year_2026 | -0.30% | 12% | PASS | no |
| hhcred_chg_xs | holdout_365d | -0.11% | 33% | PASS | no |
| us_hhcred_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_hhcred_stress_fx | year_2025 | +0.04% | 36% | PASS | no |
| us_hhcred_stress_fx | year_2026 | -0.19% | 38% | PASS | no |
| us_hhcred_stress_fx | holdout_365d | -0.01% | 42% | PASS | no |
| us_hhcred_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_hhcred_haven_usd | year_2025 | -0.04% | 36% | PASS | no |
| us_hhcred_haven_usd | year_2026 | +0.19% | 62% | PASS | no |
| us_hhcred_haven_usd | holdout_365d | +0.01% | 58% | PASS | no |
| hhcred_ew | year_2024 | -0.05% | 45% | PASS | no |
| hhcred_ew | year_2025 | -0.07% | 45% | PASS | no |
| hhcred_ew | year_2026 | -0.07% | 38% | PASS | no |
| hhcred_ew | holdout_365d | +0.08% | 58% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_household_credit_*.csv`, `scholarly_fx_household_credit_meta.json`.
