# Scholarly FX: Building-permits (ODCNPI03) housing-activity wave (§43)

**Path:** OECD MEI permits YoY via FRED (`{ISO3}ODCNPI03GYSAM`) + USD Census `PERMIT`→YoY — housing-*activity* XS — **distinct** from house-price §35 (BIS HPI), OECD CLI (§37) / CCI (§38) / BCI (§39), IP (§42), employment (§41), WUI (§40), EPU/TPU, IG OAS (§36), CA/TB, fiscal/debt, BIS, reserves, money-growth, equity, commodity.
**Data:** `approximate_non_ftmo` + free FRED ODCNPI03 YoY + US PERMIT. EUR = `DEUODCNPI03GYSAM` (Germany; EA19 stale). GBP/JPY/CHF unmapped. `n_long=n_short=2` a priori. **PIT:** pub_lag_months=2 (housing a priori) + signal_lag_months=0 + 1d weight lag. Score basis = **YoY growth**. ACM THREEFYTP10 fallback **not used**.
**Primary:** `high_housing_xs` (long high relative permits YoY / short low — construction-activity → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_housing_xs | +0.006% | +0.11 | +0.11 | 52% | 10% | +0.05 |
| low_housing_xs | -0.025% | -0.48 | -0.50 | 44% | 13% | -0.13 |
| high_housing_z_xs | +0.015% | +0.26 | +0.27 | 55% | 9% | +0.09 |
| housing_chg_xs | -0.083% | -1.65 | -1.62 | 52% | 10% | -0.32 |
| us_housing_stress_fx | +0.027% | +1.13 | +1.19 | 11% | 44% | +0.26 |
| us_housing_haven_usd | -0.030% | -1.22 | -1.29 | 9% | 52% | -0.28 |
| housing_ew | -0.016% | -0.58 | -0.57 | 47% | 13% | -0.08 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_housing_xs | year_2024 | -0.01% | 45% | PASS | no |
| high_housing_xs | year_2025 | -0.26% | 27% | PASS | no |
| high_housing_xs | year_2026 | -0.07% | 50% | PASS | no |
| high_housing_xs | holdout_365d | -0.04% | 50% | PASS | no |
| low_housing_xs | year_2024 | +0.01% | 55% | PASS | no |
| low_housing_xs | year_2025 | +0.24% | 64% | PASS | no |
| low_housing_xs | year_2026 | +0.06% | 50% | PASS | no |
| low_housing_xs | holdout_365d | +0.02% | 42% | PASS | no |
| high_housing_z_xs | year_2024 | -0.22% | 36% | PASS | no |
| high_housing_z_xs | year_2025 | +0.14% | 55% | PASS | no |
| high_housing_z_xs | year_2026 | -0.05% | 50% | PASS | no |
| high_housing_z_xs | holdout_365d | -0.08% | 50% | PASS | no |
| housing_chg_xs | year_2024 | -0.16% | 55% | PASS | no |
| housing_chg_xs | year_2025 | -0.02% | 45% | PASS | no |
| housing_chg_xs | year_2026 | -0.02% | 38% | PASS | no |
| housing_chg_xs | holdout_365d | -0.07% | 42% | PASS | no |
| us_housing_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_housing_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_housing_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_housing_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_housing_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_housing_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_housing_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_housing_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| housing_ew | year_2024 | -0.06% | 45% | PASS | no |
| housing_ew | year_2025 | -0.09% | 18% | PASS | no |
| housing_ew | year_2026 | -0.03% | 38% | PASS | no |
| housing_ew | holdout_365d | -0.04% | 42% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_building_permits_*.csv`, `scholarly_fx_building_permits_meta.json`.
