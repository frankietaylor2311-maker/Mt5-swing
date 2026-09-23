# Scholarly FX: OECD CPI / inflation YoY differential wave (§52)

**Path:** FRED CPI/HICP index→YoY (`CPIAUCSL` / `CP0000EZ19M086NEST` / `*CPIALLMINMEI` / `*CPIALLQINMEI`) — CPI YoY XS — **distinct** from PPI (§51), IP (§42), retail (§46), employment (§41), ULC/LP/CU (§48–50), building-permits (§43), CLI/CCI/BCI, macro_diff EW blend, PPP real FX, WUI, EPU/TPU.
**Data:** `approximate_non_ftmo` + free FRED CPI/HICP index→YoY after pub_lag. **STALE** — JPY ends ~2021-06 (hard); GBP/CAD ~2025-03, CHF ~2025-04, AUD/NZD Q ~2025-01 (mild); USD/EUR live mid-2026. EUR = `CP0000EZ19M086NEST`. Full G10 mapped. `n_long=n_short=2`. **PIT:** pub_lag_months=1 (CPI a priori = macro_diff default) + signal_lag_months=1 + 1d weight lag. Score basis = **CPI YoY %**.
**Primary:** `high_cpi_xs` (long high relative CPI YoY / short low — Dahlquist economic-momentum → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_cpi_xs | +0.015% | +0.24 | +0.27 | 54% | 10% | +0.03 |
| low_cpi_xs | -0.025% | -0.41 | -0.45 | 46% | 13% | -0.06 |
| high_cpi_z_xs | +0.054% | +0.96 | +1.10 | 53% | 9% | +0.17 |
| cpi_chg_xs | +0.070% | +1.25 | +1.20 | 54% | 11% | +0.22 |
| us_cpi_stress_fx | -0.009% | -0.41 | -0.51 | 6% | 72% | -0.10 |
| us_cpi_haven_usd | +0.008% | +0.39 | +0.49 | 4% | 49% | +0.10 |
| cpi_ew | +0.026% | +0.85 | +0.79 | 51% | 12% | +0.13 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_cpi_xs | year_2024 | +0.02% | 55% | PASS | no |
| high_cpi_xs | year_2025 | -0.13% | 55% | PASS | no |
| high_cpi_xs | year_2026 | +0.15% | 75% | PASS | no |
| high_cpi_xs | holdout_365d | +0.10% | 67% | PASS | no |
| low_cpi_xs | year_2024 | -0.03% | 45% | PASS | no |
| low_cpi_xs | year_2025 | +0.12% | 45% | PASS | no |
| low_cpi_xs | year_2026 | -0.15% | 25% | PASS | no |
| low_cpi_xs | holdout_365d | -0.10% | 33% | PASS | no |
| high_cpi_z_xs | year_2024 | +0.04% | 55% | PASS | no |
| high_cpi_z_xs | year_2025 | -0.03% | 55% | PASS | no |
| high_cpi_z_xs | year_2026 | +0.15% | 75% | PASS | no |
| high_cpi_z_xs | holdout_365d | +0.10% | 67% | PASS | no |
| cpi_chg_xs | year_2024 | -0.07% | 45% | PASS | no |
| cpi_chg_xs | year_2025 | +0.18% | 64% | PASS | no |
| cpi_chg_xs | year_2026 | +0.14% | 62% | PASS | no |
| cpi_chg_xs | holdout_365d | +0.04% | 50% | PASS | no |
| us_cpi_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_cpi_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_cpi_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_cpi_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_cpi_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_cpi_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_cpi_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_cpi_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| cpi_ew | year_2024 | -0.02% | 55% | PASS | no |
| cpi_ew | year_2025 | +0.02% | 55% | PASS | no |
| cpi_ew | year_2026 | +0.09% | 75% | PASS | no |
| cpi_ew | holdout_365d | +0.05% | 67% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_cpi_*.csv`, `scholarly_fx_cpi_meta.json`.
