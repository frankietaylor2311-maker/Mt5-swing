# Scholarly FX: OECD/FRED merchandise export-value differential wave (§61)

**Path:** FRED OECD MEI **merchandise export-value YoY** (`XTEXVA01` M657S; EUR=Germany DEM proxy; full G10; unit=VALUE not volume) — export-value YoY XS — **distinct** from trade-balance (§30), CA (§21), ToT (UV 404), retail (§46), IP (§42), passenger-cars (§60), construction (§59), GDP (§58).
**Data:** `approximate_non_ftmo` + free FRED XTEXVA01 YoY after pub_lag. EUR=XTEXVA01DEM657S Germany proxy (EZM657S 404); full G10 live. `n_long=n_short=2` on available foreign panel. **PIT:** pub_lag_months=2 (trade/export ~1–2m lag a priori) + signal_lag_months=1 + 1d weight lag. Score basis = **merchandise export-value YoY %** (value not volume).
**Primary:** `high_xval_xs` (long high relative export-value YoY / short low — growth-channel / external demand → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_xval_xs | -0.058% | -0.87 | -0.79 | 46% | 12% | -0.17 |
| low_xval_xs | +0.008% | +0.12 | +0.11 | 52% | 13% | +0.00 |
| high_xval_z_xs | -0.064% | -0.98 | -0.92 | 48% | 12% | -0.19 |
| xval_chg_xs | -0.034% | -0.57 | -0.51 | 46% | 17% | -0.08 |
| us_xval_stress_fx | -0.028% | -1.27 | -1.32 | 8% | 61% | -0.32 |
| us_xval_haven_usd | +0.025% | +1.16 | +1.22 | 11% | 42% | +0.29 |
| xval_ew | -0.039% | -1.05 | -0.94 | 46% | 13% | -0.20 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_xval_xs | year_2024 | +0.55% | 64% | PASS | no |
| high_xval_xs | year_2025 | -0.06% | 45% | PASS | no |
| high_xval_xs | year_2026 | +0.12% | 50% | PASS | no |
| high_xval_xs | holdout_365d | +0.08% | 58% | PASS | no |
| low_xval_xs | year_2024 | -0.59% | 36% | PASS | no |
| low_xval_xs | year_2025 | +0.00% | 55% | PASS | no |
| low_xval_xs | year_2026 | -0.17% | 38% | PASS | no |
| low_xval_xs | holdout_365d | -0.13% | 33% | PASS | no |
| high_xval_z_xs | year_2024 | +0.54% | 64% | PASS | no |
| high_xval_z_xs | year_2025 | +0.00% | 45% | PASS | no |
| high_xval_z_xs | year_2026 | +0.22% | 62% | PASS | no |
| high_xval_z_xs | holdout_365d | +0.11% | 58% | PASS | no |
| xval_chg_xs | year_2024 | +0.52% | 73% | PASS | no |
| xval_chg_xs | year_2025 | -0.08% | 45% | PASS | no |
| xval_chg_xs | year_2026 | -0.04% | 25% | PASS | no |
| xval_chg_xs | holdout_365d | -0.08% | 33% | PASS | no |
| us_xval_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_xval_stress_fx | year_2025 | -0.01% | 9% | PASS | no |
| us_xval_stress_fx | year_2026 | -0.14% | 12% | PASS | no |
| us_xval_stress_fx | holdout_365d | -0.09% | 17% | PASS | no |
| us_xval_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_xval_haven_usd | year_2025 | +0.01% | 9% | PASS | no |
| us_xval_haven_usd | year_2026 | +0.13% | 25% | PASS | no |
| us_xval_haven_usd | holdout_365d | +0.08% | 17% | PASS | no |
| xval_ew | year_2024 | +0.36% | 64% | PASS | no |
| xval_ew | year_2025 | -0.05% | 36% | PASS | no |
| xval_ew | year_2026 | -0.02% | 38% | PASS | no |
| xval_ew | holdout_365d | -0.03% | 42% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_export_value_*.csv`, `scholarly_fx_export_value_meta.json`.
