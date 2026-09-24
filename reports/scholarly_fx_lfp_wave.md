# Scholarly FX: OECD/FRED labour-force participation / activity-rate differential wave (§56)

**Path:** FRED OECD MEI **activity / LFP rate levels** (`LRAC64TT*`; full G10; EUR=Germany DEQ proxy) — LFP level XS — **distinct** from employment persons (§41), UR levels (§54), wages (§55), ULC (§48), LP (§49), CU (§50).
**Data:** `approximate_non_ftmo` + free FRED LRAC64TT levels after pub_lag. EUR EA EZQ156S stale ~2022-10 → DEQ156S Germany proxy; full G10 mapped. `n_long=n_short=2` on available foreign panel. **PIT:** pub_lag_months=2 (participation lag a priori; mixed M+Q) + signal_lag_months=1 + 1d weight lag. Score basis = **LFP / activity rate level %**.
**Primary:** `high_lfp_xs` (long high relative LFP / short low — growth/engagement → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_lfp_xs | -0.050% | -1.02 | -1.09 | 49% | 12% | -0.26 |
| low_lfp_xs | +0.047% | +0.95 | +1.01 | 51% | 10% | +0.25 |
| high_lfp_z_xs | -0.052% | -0.89 | -0.93 | 44% | 12% | -0.18 |
| lfp_chg_xs | -0.070% | -1.11 | -1.14 | 44% | 13% | -0.23 |
| us_lfp_stress_fx | -0.016% | -0.42 | -0.52 | 18% | 31% | -0.18 |
| us_lfp_haven_usd | +0.015% | +0.39 | +0.49 | 13% | 35% | +0.17 |
| lfp_ew | -0.045% | -1.51 | -1.64 | 47% | 11% | -0.34 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_lfp_xs | year_2024 | -0.18% | 36% | PASS | no |
| high_lfp_xs | year_2025 | -0.07% | 36% | PASS | no |
| high_lfp_xs | year_2026 | -0.09% | 38% | PASS | no |
| high_lfp_xs | holdout_365d | -0.17% | 33% | PASS | no |
| low_lfp_xs | year_2024 | +0.17% | 64% | PASS | no |
| low_lfp_xs | year_2025 | +0.06% | 64% | PASS | no |
| low_lfp_xs | year_2026 | +0.09% | 62% | PASS | no |
| low_lfp_xs | holdout_365d | +0.17% | 67% | PASS | no |
| high_lfp_z_xs | year_2024 | -0.14% | 36% | PASS | no |
| high_lfp_z_xs | year_2025 | -0.13% | 36% | PASS | no |
| high_lfp_z_xs | year_2026 | +0.21% | 50% | PASS | no |
| high_lfp_z_xs | holdout_365d | -0.04% | 33% | PASS | no |
| lfp_chg_xs | year_2024 | -0.09% | 36% | PASS | no |
| lfp_chg_xs | year_2025 | -0.08% | 36% | PASS | no |
| lfp_chg_xs | year_2026 | +0.03% | 38% | PASS | no |
| lfp_chg_xs | holdout_365d | -0.20% | 25% | PASS | no |
| us_lfp_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_lfp_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_lfp_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_lfp_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_lfp_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_lfp_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_lfp_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_lfp_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| lfp_ew | year_2024 | -0.09% | 27% | PASS | no |
| lfp_ew | year_2025 | -0.05% | 45% | PASS | no |
| lfp_ew | year_2026 | -0.02% | 25% | PASS | no |
| lfp_ew | holdout_365d | -0.12% | 17% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_lfp_*.csv`, `scholarly_fx_lfp_meta.json`.

Locked `fx4plus_gbpcad_d1_voltarget_0025` **untouched PASS** (see `quest_locked_verify_lfp.md`).
