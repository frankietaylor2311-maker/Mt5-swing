# Scholarly FX: BIS debt-service (DSR / PNFS) differential wave (§47)

**Path:** BIS SDMX `WS_DSR` Q..P private non-financial debt-service ratio (FRED Q*PNFDSR / BISDSR* **404**) — DSR level XS — **distinct** from credit/GDP stock (§32), gov debt (§29), fiscal (§28), funding-liq (§20), IG OAS (§36), CA/TB, REER, money, reserves, retail/IP/emp.
**Data:** `approximate_non_ftmo` + free BIS SDMX WS_DSR (quarterly → monthly after pub_lag). EUR = `Q.DE.P` (Germany proxy; XM/EA missing). NZD unmapped. `n_long=n_short=2` a priori. **PIT:** pub_lag_months=5 (BIS quarterly a priori) + signal_lag_months=1 + 1d weight lag. Score basis = **PNFS DSR %**.
**Primary:** `high_dsr_xs` (long high relative DSR / short low — Drehmann–Juselius risk premia). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_dsr_xs | -0.067% | -0.98 | -1.05 | 47% | 11% | -0.29 |
| low_dsr_xs | +0.065% | +0.94 | +1.02 | 54% | 11% | +0.28 |
| high_dsr_z_xs | -0.057% | -0.89 | -0.96 | 48% | 14% | -0.23 |
| dsr_chg_xs | -0.059% | -0.84 | -0.89 | 47% | 14% | -0.21 |
| us_dsr_stress_fx | +0.002% | +0.08 | +0.09 | 4% | 68% | +0.02 |
| us_dsr_haven_usd | -0.002% | -0.09 | -0.10 | 6% | 57% | -0.02 |
| dsr_ew | -0.041% | -1.16 | -1.23 | 44% | 15% | -0.31 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_dsr_xs | year_2024 | -0.29% | 27% | PASS | no |
| high_dsr_xs | year_2025 | -0.17% | 27% | PASS | no |
| high_dsr_xs | year_2026 | +0.13% | 62% | PASS | no |
| high_dsr_xs | holdout_365d | +0.19% | 67% | PASS | no |
| low_dsr_xs | year_2024 | +0.29% | 73% | PASS | no |
| low_dsr_xs | year_2025 | +0.16% | 73% | PASS | no |
| low_dsr_xs | year_2026 | -0.13% | 38% | PASS | no |
| low_dsr_xs | holdout_365d | -0.19% | 33% | PASS | no |
| high_dsr_z_xs | year_2024 | -0.15% | 36% | PASS | no |
| high_dsr_z_xs | year_2025 | -0.16% | 27% | PASS | no |
| high_dsr_z_xs | year_2026 | +0.04% | 50% | PASS | no |
| high_dsr_z_xs | holdout_365d | +0.10% | 58% | PASS | no |
| dsr_chg_xs | year_2024 | -0.17% | 55% | PASS | no |
| dsr_chg_xs | year_2025 | -0.05% | 45% | PASS | no |
| dsr_chg_xs | year_2026 | -0.01% | 50% | PASS | no |
| dsr_chg_xs | holdout_365d | -0.02% | 50% | PASS | no |
| us_dsr_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_dsr_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_dsr_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_dsr_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_dsr_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_dsr_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_dsr_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_dsr_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| dsr_ew | year_2024 | -0.16% | 36% | PASS | no |
| dsr_ew | year_2025 | -0.07% | 36% | PASS | no |
| dsr_ew | year_2026 | +0.04% | 62% | PASS | no |
| dsr_ew | holdout_365d | +0.06% | 58% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_bis_dsr_*.csv`, `scholarly_fx_bis_dsr_meta.json`.
