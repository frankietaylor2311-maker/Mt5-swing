# Scholarly FX: OECD/FRED employment-rate differential wave (§57)

**Path:** FRED OECD MEI **employment-rate levels** (`LREM64TT*`; full G10; EUR=Germany DEQ proxy) — ER level XS — **distinct** from employment persons (§41), UR levels (§54), LFP (§56), wages (§55), ULC (§48), LP (§49), CU (§50).
**Data:** `approximate_non_ftmo` + free FRED LREM64TT levels after pub_lag. EUR EA EZQ156S stale ~2022-10 → DEQ156S Germany proxy; full G10 mapped. `n_long=n_short=2` on available foreign panel. **PIT:** pub_lag_months=2 (employment-rate lag a priori; mixed M+Q) + signal_lag_months=1 + 1d weight lag. Score basis = **employment-rate level %**.
**Primary:** `high_er_xs` (long high relative ER / short low — labour-strength/engagement stock → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_er_xs | -0.013% | -0.21 | -0.23 | 47% | 16% | -0.08 |
| low_er_xs | +0.007% | +0.11 | +0.13 | 52% | 11% | +0.06 |
| high_er_z_xs | -0.076% | -1.19 | -1.13 | 41% | 13% | -0.26 |
| er_chg_xs | -0.089% | -1.43 | -1.40 | 43% | 13% | -0.32 |
| us_er_stress_fx | +0.011% | +0.50 | +0.78 | 3% | 83% | +0.02 |
| us_er_haven_usd | -0.011% | -0.50 | -0.79 | 3% | 92% | -0.02 |
| er_ew | -0.030% | -0.87 | -0.91 | 46% | 17% | -0.23 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_er_xs | year_2024 | -0.04% | 27% | PASS | no |
| high_er_xs | year_2025 | -0.06% | 45% | PASS | no |
| high_er_xs | year_2026 | -0.09% | 38% | PASS | no |
| high_er_xs | holdout_365d | -0.17% | 33% | PASS | no |
| low_er_xs | year_2024 | +0.03% | 73% | PASS | no |
| low_er_xs | year_2025 | +0.05% | 55% | PASS | no |
| low_er_xs | year_2026 | +0.09% | 62% | PASS | no |
| low_er_xs | holdout_365d | +0.17% | 67% | PASS | no |
| high_er_z_xs | year_2024 | -0.16% | 27% | PASS | no |
| high_er_z_xs | year_2025 | -0.20% | 27% | PASS | no |
| high_er_z_xs | year_2026 | +0.22% | 50% | PASS | no |
| high_er_z_xs | holdout_365d | -0.09% | 33% | PASS | no |
| er_chg_xs | year_2024 | +0.02% | 36% | PASS | no |
| er_chg_xs | year_2025 | -0.09% | 36% | PASS | no |
| er_chg_xs | year_2026 | +0.14% | 38% | PASS | no |
| er_chg_xs | holdout_365d | -0.14% | 25% | PASS | no |
| us_er_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_er_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_er_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_er_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_er_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_er_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_er_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_er_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| er_ew | year_2024 | -0.00% | 27% | PASS | no |
| er_ew | year_2025 | -0.05% | 45% | PASS | no |
| er_ew | year_2026 | +0.02% | 50% | PASS | no |
| er_ew | holdout_365d | -0.10% | 33% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_employment_rate_*.csv`, `scholarly_fx_employment_rate_meta.json`.
