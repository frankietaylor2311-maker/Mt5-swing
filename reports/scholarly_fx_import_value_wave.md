# Scholarly FX: OECD/FRED merchandise import-value differential wave (§62)

**Path:** FRED OECD MEI **merchandise import-value YoY** (`XTIMVA01` M657S; EUR=Germany DEM proxy; full G10; unit=VALUE not volume) — import-value YoY XS — **distinct** from trade-balance (§30), CA (§21), ToT (UV 404), retail (§46), IP (§42), passenger-cars (§60), construction (§59), GDP (§58).
**Data:** `approximate_non_ftmo` + free FRED XTIMVA01 YoY after pub_lag. EUR=XTIMVA01DEM657S Germany proxy (EZM657S 404); full G10 live. `n_long=n_short=2` on available foreign panel. **PIT:** pub_lag_months=2 (trade/import ~1–2m lag a priori) + signal_lag_months=1 + 1d weight lag. Score basis = **merchandise import-value YoY %** (value not volume).
**Primary:** `high_ival_xs` (long high relative import-value YoY / short low — growth-channel / domestic absorption / import demand → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_ival_xs | -0.125% | -2.03 | -1.77 | 44% | 13% | -0.48 |
| low_ival_xs | +0.073% | +1.19 | +1.06 | 53% | 15% | +0.28 |
| high_ival_z_xs | -0.084% | -1.55 | -1.43 | 43% | 11% | -0.33 |
| ival_chg_xs | -0.095% | -1.62 | -1.51 | 43% | 13% | -0.38 |
| us_ival_stress_fx | +0.015% | +0.62 | +0.57 | 9% | 50% | +0.15 |
| us_ival_haven_usd | -0.018% | -0.74 | -0.68 | 11% | 50% | -0.17 |
| ival_ew | -0.068% | -1.82 | -1.62 | 41% | 15% | -0.43 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_ival_xs | year_2024 | +0.03% | 64% | PASS | no |
| high_ival_xs | year_2025 | +0.10% | 64% | PASS | no |
| high_ival_xs | year_2026 | -0.02% | 62% | PASS | no |
| high_ival_xs | holdout_365d | +0.03% | 67% | PASS | no |
| low_ival_xs | year_2024 | -0.08% | 27% | PASS | no |
| low_ival_xs | year_2025 | -0.15% | 36% | PASS | no |
| low_ival_xs | year_2026 | -0.03% | 38% | PASS | no |
| low_ival_xs | holdout_365d | -0.09% | 33% | PASS | no |
| high_ival_z_xs | year_2024 | +0.12% | 45% | PASS | no |
| high_ival_z_xs | year_2025 | +0.10% | 64% | PASS | no |
| high_ival_z_xs | year_2026 | +0.04% | 50% | PASS | no |
| high_ival_z_xs | holdout_365d | +0.10% | 67% | PASS | no |
| ival_chg_xs | year_2024 | -0.13% | 27% | PASS | no |
| ival_chg_xs | year_2025 | +0.18% | 64% | PASS | no |
| ival_chg_xs | year_2026 | -0.06% | 50% | PASS | no |
| ival_chg_xs | holdout_365d | -0.02% | 50% | PASS | no |
| us_ival_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_ival_stress_fx | year_2025 | +0.08% | 36% | PASS | no |
| us_ival_stress_fx | year_2026 | -0.03% | 0% | PASS | no |
| us_ival_stress_fx | holdout_365d | -0.05% | 8% | PASS | no |
| us_ival_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_ival_haven_usd | year_2025 | -0.09% | 27% | PASS | no |
| us_ival_haven_usd | year_2026 | +0.02% | 25% | PASS | no |
| us_ival_haven_usd | holdout_365d | +0.04% | 42% | PASS | no |
| ival_ew | year_2024 | -0.03% | 45% | PASS | no |
| ival_ew | year_2025 | +0.12% | 64% | PASS | no |
| ival_ew | year_2026 | -0.03% | 38% | PASS | no |
| ival_ew | holdout_365d | -0.01% | 42% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_import_value_*.csv`, `scholarly_fx_import_value_meta.json`.
