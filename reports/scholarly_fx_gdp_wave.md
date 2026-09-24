# Scholarly FX: OECD/FRED real GDP growth differential wave (§58)

**Path:** FRED OECD MEI **real GDP YoY** (`NAEXKP01*Q657S` QoQ→YoY; full G10; EUR=Germany DEQ proxy) — GDP YoY XS — **distinct** from IP (§42), CLI (§37), BCI (§39), emp persons (§41), UR (§54), debt/GDP (§29), CA/GDP (§21).
**Data:** `approximate_non_ftmo` + free FRED NAEXKP01 YoY (from Q657S QoQ) after pub_lag. EUR EA EZQ657S stale ~2023-01 → DEQ657S Germany proxy; full G10 mapped. `n_long=n_short=2` on available foreign panel. **PIT:** pub_lag_months=3 (quarterly GDP lag a priori) + signal_lag_months=1 + 1d weight lag. Score basis = **real GDP YoY %**.
**Primary:** `high_gdp_xs` (long high relative GDP YoY / short low — growth / economic-momentum → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_gdp_xs | -0.109% | -1.50 | -1.65 | 46% | 11% | -0.38 |
| low_gdp_xs | +0.102% | +1.40 | +1.55 | 54% | 12% | +0.36 |
| high_gdp_z_xs | -0.057% | -0.99 | -1.15 | 48% | 10% | -0.20 |
| gdp_chg_xs | +0.016% | +0.25 | +0.27 | 54% | 12% | +0.06 |
| us_gdp_stress_fx | +0.013% | +0.59 | +0.59 | 4% | 62% | +0.17 |
| us_gdp_haven_usd | -0.013% | -0.60 | -0.60 | 4% | 70% | -0.17 |
| gdp_ew | -0.027% | -0.66 | -0.76 | 44% | 11% | -0.17 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_gdp_xs | year_2024 | -0.13% | 45% | PASS | no |
| high_gdp_xs | year_2025 | +0.05% | 45% | PASS | no |
| high_gdp_xs | year_2026 | +0.03% | 62% | PASS | no |
| high_gdp_xs | holdout_365d | +0.21% | 67% | PASS | no |
| low_gdp_xs | year_2024 | +0.13% | 55% | PASS | no |
| low_gdp_xs | year_2025 | -0.06% | 55% | PASS | no |
| low_gdp_xs | year_2026 | -0.04% | 38% | PASS | no |
| low_gdp_xs | holdout_365d | -0.22% | 33% | PASS | no |
| high_gdp_z_xs | year_2024 | -0.08% | 36% | PASS | no |
| high_gdp_z_xs | year_2025 | -0.13% | 36% | PASS | no |
| high_gdp_z_xs | year_2026 | +0.19% | 62% | PASS | no |
| high_gdp_z_xs | holdout_365d | +0.06% | 50% | PASS | no |
| gdp_chg_xs | year_2024 | -0.25% | 27% | PASS | no |
| gdp_chg_xs | year_2025 | -0.02% | 36% | PASS | no |
| gdp_chg_xs | year_2026 | +0.12% | 75% | PASS | no |
| gdp_chg_xs | holdout_365d | +0.09% | 58% | PASS | no |
| us_gdp_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_gdp_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_gdp_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_gdp_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_gdp_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_gdp_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_gdp_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_gdp_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| gdp_ew | year_2024 | -0.13% | 36% | PASS | no |
| gdp_ew | year_2025 | +0.01% | 45% | PASS | no |
| gdp_ew | year_2026 | +0.05% | 62% | PASS | no |
| gdp_ew | holdout_365d | +0.10% | 58% | PASS | no |

**Board:** n=7 soft_nw_pos=1 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_gdp_*.csv`, `scholarly_fx_gdp_meta.json`.
