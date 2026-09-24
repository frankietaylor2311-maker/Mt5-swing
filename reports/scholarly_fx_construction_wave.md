# Scholarly FX: OECD/FRED construction-production differential wave (§59)

**Path:** FRED OECD MEI **construction production YoY** (`PRCNTO01` GYSAM/GYSAQ/Q657S; full G10; EUR=Germany DEU proxy) — construction YoY XS — **distinct** from building-permits (§43), IP (§42), GDP (§58), CLI (§37), BCI (§39).
**Data:** `approximate_non_ftmo` + free FRED PRCNTO01 YoY after pub_lag. EUR=DEUPRCNTO01GYSAM; JPY=GYSAQ; AUD/NZD/CHF QoQ→YoY; full G10 mapped. `n_long=n_short=2` on available foreign panel. **PIT:** pub_lag_months=2 (construction/IP-like lag a priori) + signal_lag_months=1 + 1d weight lag. Score basis = **construction YoY %**.
**Primary:** `high_cons_xs` (long high relative construction YoY / short low — growth-channel / coincident construction → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_cons_xs | -0.072% | -1.24 | -1.24 | 43% | 13% | -0.27 |
| low_cons_xs | +0.057% | +0.98 | +0.99 | 55% | 12% | +0.22 |
| high_cons_z_xs | -0.081% | -1.31 | -1.37 | 44% | 13% | -0.27 |
| cons_chg_xs | -0.002% | -0.04 | -0.04 | 52% | 11% | +0.05 |
| us_cons_stress_fx | +0.019% | +0.63 | +0.72 | 11% | 33% | +0.17 |
| us_cons_haven_usd | -0.020% | -0.65 | -0.75 | 11% | 38% | -0.18 |
| cons_ew | -0.018% | -0.51 | -0.56 | 48% | 12% | -0.09 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_cons_xs | year_2024 | -0.01% | 55% | PASS | no |
| high_cons_xs | year_2025 | +0.04% | 36% | PASS | no |
| high_cons_xs | year_2026 | -0.14% | 25% | PASS | no |
| high_cons_xs | holdout_365d | -0.03% | 33% | PASS | no |
| low_cons_xs | year_2024 | -0.01% | 45% | PASS | no |
| low_cons_xs | year_2025 | -0.05% | 64% | PASS | no |
| low_cons_xs | year_2026 | +0.12% | 75% | PASS | no |
| low_cons_xs | holdout_365d | +0.02% | 67% | PASS | no |
| high_cons_z_xs | year_2024 | -0.04% | 55% | PASS | no |
| high_cons_z_xs | year_2025 | +0.09% | 55% | PASS | no |
| high_cons_z_xs | year_2026 | -0.11% | 25% | PASS | no |
| high_cons_z_xs | holdout_365d | -0.11% | 33% | PASS | no |
| cons_chg_xs | year_2024 | +0.06% | 45% | PASS | no |
| cons_chg_xs | year_2025 | +0.10% | 55% | PASS | no |
| cons_chg_xs | year_2026 | +0.05% | 62% | PASS | no |
| cons_chg_xs | holdout_365d | +0.08% | 67% | PASS | no |
| us_cons_stress_fx | year_2024 | -0.14% | 0% | PASS | no |
| us_cons_stress_fx | year_2025 | +0.37% | 64% | PASS | no |
| us_cons_stress_fx | year_2026 | -0.19% | 38% | PASS | no |
| us_cons_stress_fx | holdout_365d | -0.01% | 42% | PASS | no |
| us_cons_haven_usd | year_2024 | +0.14% | 9% | PASS | no |
| us_cons_haven_usd | year_2025 | -0.37% | 36% | PASS | no |
| us_cons_haven_usd | year_2026 | +0.19% | 62% | PASS | no |
| us_cons_haven_usd | holdout_365d | +0.01% | 58% | PASS | no |
| cons_ew | year_2024 | -0.03% | 45% | PASS | no |
| cons_ew | year_2025 | +0.17% | 73% | PASS | no |
| cons_ew | year_2026 | -0.09% | 38% | PASS | no |
| cons_ew | holdout_365d | +0.01% | 50% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_construction_*.csv`, `scholarly_fx_construction_meta.json`.
