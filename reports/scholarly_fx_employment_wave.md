# Scholarly FX: Employment-growth (LFEMTTTT) differential wave (§41)

**Path:** OECD MEI employment persons via FRED (`LFEMTTTT*M647S` / `*Q647S`) — emp YoY XS — **distinct** from macro-diff UR/CPI/IP, OECD CLI (§37) / CCI (§38) / BCI (§39), WUI (§40), EPU/TPU, IG OAS (§36), CA/TB, fiscal/debt, BIS REER/credit, reserves, house-price, money-growth, equity-diff. Retail-sales SARTMISMEI stale — skipped.
**Data:** `approximate_non_ftmo` + free FRED LFEMTTTT persons (monthly + quarterly → monthly after pub_lag). EUR = `LFEMTTTTDEQ647S` (Germany proxy; EZ stale). Full G10 mapped. `n_long=n_short=2` a priori. **PIT:** pub_lag_months=3 (labour a priori) + signal_lag_months=0 + 1d weight lag. Score basis = **YoY % of levels**.
**Primary:** `high_emp_xs` (long high relative emp YoY / short low — labour strength → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_emp_xs | -0.072% | -1.15 | -1.24 | 48% | 13% | -0.29 |
| low_emp_xs | +0.062% | +0.98 | +1.06 | 52% | 10% | +0.26 |
| high_emp_z_xs | -0.020% | -0.35 | -0.35 | 49% | 11% | -0.07 |
| emp_chg_xs | -0.034% | -0.55 | -0.55 | 46% | 16% | -0.12 |
| us_emp_stress_fx | +0.027% | +1.09 | +1.36 | 7% | 42% | +0.34 |
| us_emp_haven_usd | -0.027% | -1.10 | -1.37 | 6% | 64% | -0.34 |
| emp_ew | -0.026% | -0.78 | -0.79 | 43% | 16% | -0.20 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_emp_xs | year_2024 | +0.01% | 55% | PASS | no |
| high_emp_xs | year_2025 | -0.10% | 36% | PASS | no |
| high_emp_xs | year_2026 | +0.14% | 75% | PASS | no |
| high_emp_xs | holdout_365d | +0.13% | 67% | PASS | no |
| low_emp_xs | year_2024 | -0.02% | 45% | PASS | no |
| low_emp_xs | year_2025 | +0.10% | 55% | PASS | no |
| low_emp_xs | year_2026 | -0.16% | 25% | PASS | no |
| low_emp_xs | holdout_365d | -0.15% | 33% | PASS | no |
| high_emp_z_xs | year_2024 | +0.12% | 45% | PASS | no |
| high_emp_z_xs | year_2025 | -0.20% | 27% | PASS | no |
| high_emp_z_xs | year_2026 | -0.03% | 62% | PASS | no |
| high_emp_z_xs | holdout_365d | -0.26% | 42% | PASS | no |
| emp_chg_xs | year_2024 | -0.10% | 27% | PASS | no |
| emp_chg_xs | year_2025 | +0.01% | 55% | PASS | no |
| emp_chg_xs | year_2026 | -0.32% | 25% | PASS | no |
| emp_chg_xs | holdout_365d | -0.38% | 25% | PASS | no |
| us_emp_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_emp_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_emp_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_emp_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_emp_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_emp_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_emp_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_emp_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| emp_ew | year_2024 | -0.03% | 55% | PASS | no |
| emp_ew | year_2025 | -0.03% | 36% | PASS | no |
| emp_ew | year_2026 | -0.06% | 50% | PASS | no |
| emp_ew | holdout_365d | -0.08% | 42% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_employment_*.csv`, `scholarly_fx_employment_meta.json`.
