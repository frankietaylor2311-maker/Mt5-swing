# Scholarly FX: OECD/FRED real GFCF / investment differential wave (§64)

**Path:** FRED OECD MEI **real GFCF YoY** (`NAEXKP04*Q657S` QoQ→YoY; full G10; EUR=Germany DEQ proxy) — GFCF YoY XS — **distinct** from GDP (§58), construction (§59), IP (§42), BCI (§39), building permits (§43), HPI (§35), CLI (§37), retail (§46), cars (§60), export/import (§61/§62), household credit (§63), private credit/GDP (§32).
**Data:** `approximate_non_ftmo` + free FRED NAEXKP04 YoY (from Q657S QoQ) after pub_lag. EUR EA EZQ657S stale ~2023-01 → DEQ657S Germany proxy; full G10 mapped. `n_long=n_short=2` on available foreign panel. **PIT:** pub_lag_months=3 (quarterly GFCF/NA lag a priori) + signal_lag_months=1 + 1d weight lag. Score basis = **real GFCF YoY %**.
**Primary:** `high_gfcf_xs` (long high relative GFCF YoY / short low — investment boom / economic-momentum → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_gfcf_xs | +0.018% | +0.27 | +0.35 | 53% | 13% | +0.04 |
| low_gfcf_xs | -0.025% | -0.38 | -0.48 | 46% | 12% | -0.06 |
| high_gfcf_z_xs | +0.040% | +0.75 | +0.87 | 54% | 11% | +0.17 |
| gfcf_chg_xs | +0.047% | +0.81 | +0.85 | 53% | 9% | +0.18 |
| us_gfcf_stress_fx | +0.014% | +0.52 | +0.58 | 7% | 46% | +0.14 |
| us_gfcf_haven_usd | -0.015% | -0.55 | -0.60 | 9% | 38% | -0.15 |
| gfcf_ew | +0.027% | +0.83 | +1.02 | 47% | 13% | +0.17 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_gfcf_xs | year_2024 | -0.23% | 45% | PASS | no |
| high_gfcf_xs | year_2025 | +0.10% | 55% | PASS | no |
| high_gfcf_xs | year_2026 | +0.03% | 62% | PASS | no |
| high_gfcf_xs | holdout_365d | +0.10% | 67% | PASS | no |
| low_gfcf_xs | year_2024 | +0.22% | 45% | PASS | no |
| low_gfcf_xs | year_2025 | -0.11% | 45% | PASS | no |
| low_gfcf_xs | year_2026 | -0.04% | 38% | PASS | no |
| low_gfcf_xs | holdout_365d | -0.11% | 33% | PASS | no |
| high_gfcf_z_xs | year_2024 | -0.15% | 45% | PASS | no |
| high_gfcf_z_xs | year_2025 | +0.02% | 45% | PASS | no |
| high_gfcf_z_xs | year_2026 | -0.05% | 50% | PASS | no |
| high_gfcf_z_xs | holdout_365d | -0.14% | 42% | PASS | no |
| gfcf_chg_xs | year_2024 | -0.13% | 45% | PASS | no |
| gfcf_chg_xs | year_2025 | +0.20% | 64% | PASS | no |
| gfcf_chg_xs | year_2026 | +0.05% | 62% | PASS | no |
| gfcf_chg_xs | holdout_365d | +0.12% | 67% | PASS | no |
| us_gfcf_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_gfcf_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_gfcf_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_gfcf_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_gfcf_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_gfcf_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_gfcf_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_gfcf_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| gfcf_ew | year_2024 | -0.12% | 55% | PASS | no |
| gfcf_ew | year_2025 | +0.10% | 55% | PASS | no |
| gfcf_ew | year_2026 | +0.03% | 75% | PASS | no |
| gfcf_ew | holdout_365d | +0.07% | 75% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_gfcf_*.csv`, `scholarly_fx_gfcf_meta.json`.
