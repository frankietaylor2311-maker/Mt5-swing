# Scholarly FX: country-GPR bilateral (rel-to-US) wave (§45)

**Path:** Caldara–Iacoviello country GPRC_* *relative-to-US* (home − US) cross-section + US geopolitics tilts — **distinct** from absolute country-GPR sort/LP (§9), aggregate GPR regime (§8), AI-GPR threats/acts/oil (§25).
**Data:** `approximate_non_ftmo` + free Caldara–Iacoviello `gpr_country_monthly.csv`. **PIT:** pub_lag=1m, signal_lag=0m (a priori 0 — pub_lag + 1d weight lag), z_window=60, z_high=1.0, usd_tilt=0.5, n_long=n_short=2. **Gap:** NZD (no GPRC_NZL). EUR=EW of ['DEU', 'FRA', 'ITA', 'ESP', 'NLD', 'BEL'].
**Primary:** `low_rel_gpr_xs` (long low (home−US) GPR z / short high). Locked sleeve untouched (no cooler overlay).

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| low_rel_gpr_xs | +0.076% | +1.26 | +1.25 | 51% | 13% | +0.29 |
| high_rel_gpr_xs | -0.108% | -1.79 | -1.72 | 46% | 11% | -0.39 |
| low_rel_gpr_lvl_xs | +0.024% | +0.40 | +0.41 | 51% | 12% | +0.07 |
| rel_gpr_chg_xs | +0.054% | +0.91 | +0.89 | 50% | 11% | +0.20 |
| us_gpr_stress_fx | -0.079% | -2.20 | -2.41 | 15% | 44% | -0.57 |
| us_gpr_haven_usd | +0.076% | +2.08 | +2.31 | 21% | 25% | +0.54 |
| gpr_bilat_ew | +0.017% | +0.49 | +0.52 | 50% | 13% | +0.11 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| low_rel_gpr_xs | year_2024 | -0.02% | 45% | PASS | no |
| low_rel_gpr_xs | year_2025 | -0.06% | 45% | PASS | no |
| low_rel_gpr_xs | year_2026 | -0.03% | 50% | PASS | no |
| low_rel_gpr_xs | holdout_365d | -0.01% | 50% | PASS | no |
| high_rel_gpr_xs | year_2024 | -0.00% | 55% | PASS | no |
| high_rel_gpr_xs | year_2025 | +0.02% | 45% | PASS | no |
| high_rel_gpr_xs | year_2026 | +0.01% | 50% | PASS | no |
| high_rel_gpr_xs | holdout_365d | -0.01% | 50% | PASS | no |
| low_rel_gpr_lvl_xs | year_2024 | -0.17% | 27% | PASS | no |
| low_rel_gpr_lvl_xs | year_2025 | +0.15% | 55% | PASS | no |
| low_rel_gpr_lvl_xs | year_2026 | +0.02% | 50% | PASS | no |
| low_rel_gpr_lvl_xs | holdout_365d | +0.09% | 58% | PASS | no |
| rel_gpr_chg_xs | year_2024 | +0.15% | 45% | PASS | no |
| rel_gpr_chg_xs | year_2025 | -0.01% | 45% | PASS | no |
| rel_gpr_chg_xs | year_2026 | -0.33% | 38% | PASS | no |
| rel_gpr_chg_xs | holdout_365d | -0.22% | 42% | PASS | no |
| us_gpr_stress_fx | year_2024 | -0.15% | 0% | PASS | no |
| us_gpr_stress_fx | year_2025 | -0.03% | 27% | PASS | no |
| us_gpr_stress_fx | year_2026 | -0.23% | 38% | PASS | no |
| us_gpr_stress_fx | holdout_365d | -0.15% | 25% | PASS | no |
| us_gpr_haven_usd | year_2024 | +0.14% | 45% | PASS | no |
| us_gpr_haven_usd | year_2025 | +0.02% | 9% | PASS | no |
| us_gpr_haven_usd | year_2026 | +0.22% | 38% | PASS | no |
| us_gpr_haven_usd | holdout_365d | +0.15% | 25% | PASS | no |
| gpr_bilat_ew | year_2024 | -0.01% | 45% | PASS | no |
| gpr_bilat_ew | year_2025 | -0.03% | 45% | PASS | no |
| gpr_bilat_ew | year_2026 | -0.20% | 38% | PASS | no |
| gpr_bilat_ew | holdout_365d | -0.13% | 42% | PASS | no |

**Board:** n=7 soft_nw_pos=1 hard_nw_pos=1 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_cgpr_bilat_*.csv`, `scholarly_fx_cgpr_bilat_meta.json`.
