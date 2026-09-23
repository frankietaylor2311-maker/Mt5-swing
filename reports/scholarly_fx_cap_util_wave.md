# Scholarly FX: OECD capacity-utilization (ULQECU01) differential wave (§50)

**Path:** OECD MEI early-estimate capacity-utilization YoY via FRED (`ULQECU01{ISO2}Q657S`) — CU YoY XS cyclical strength — **companion** completing ULC (§48) / LP (§49) trilogy; **distinct** from employment (§41), IP (§42), ULC (§48), LP (§49), CLI/BCI, building-permits (§43), retail (§46).
**Data:** `approximate_non_ftmo` + free FRED ULQECU01 Q657S YoY (quarterly → monthly after pub_lag). **STALE** — all G10 end ~2023-04..07 on free FRED. EUR = `ULQECU01DEQ657S` (Germany proxy). Full G10 mapped. `n_long=n_short=2`. **PIT:** pub_lag_months=3 (CU labour/activity a priori) + signal_lag_months=1 + 1d weight lag. Score basis = **YoY growth as reported**.
**Primary:** `high_cu_xs` (long high relative CU YoY / short low — cyclical strength → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_cu_xs | -0.058% | -0.91 | -0.99 | 48% | 13% | -0.24 |
| low_cu_xs | +0.045% | +0.71 | +0.78 | 52% | 13% | +0.20 |
| high_cu_z_xs | -0.051% | -1.00 | -1.05 | 47% | 13% | -0.17 |
| cu_chg_xs | -0.022% | -0.38 | -0.42 | 46% | 11% | -0.03 |
| us_cu_stress_fx | -0.019% | -0.91 | -0.95 | 5% | 71% | -0.23 |
| us_cu_haven_usd | +0.018% | +0.85 | +0.90 | 8% | 55% | +0.22 |
| cu_ew | -0.033% | -0.99 | -1.08 | 48% | 14% | -0.21 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_cu_xs | year_2024 | -0.00% | 64% | PASS | no |
| high_cu_xs | year_2025 | -0.33% | 27% | PASS | no |
| high_cu_xs | year_2026 | +0.09% | 50% | PASS | no |
| high_cu_xs | holdout_365d | +0.09% | 50% | PASS | no |
| low_cu_xs | year_2024 | +0.00% | 36% | PASS | no |
| low_cu_xs | year_2025 | +0.33% | 73% | PASS | no |
| low_cu_xs | year_2026 | -0.09% | 50% | PASS | no |
| low_cu_xs | holdout_365d | -0.10% | 50% | PASS | no |
| high_cu_z_xs | year_2024 | +0.01% | 64% | PASS | no |
| high_cu_z_xs | year_2025 | -0.20% | 36% | PASS | no |
| high_cu_z_xs | year_2026 | -0.17% | 25% | PASS | no |
| high_cu_z_xs | holdout_365d | -0.13% | 33% | PASS | no |
| cu_chg_xs | year_2024 | +0.03% | 55% | PASS | no |
| cu_chg_xs | year_2025 | -0.23% | 27% | PASS | no |
| cu_chg_xs | year_2026 | -0.06% | 50% | PASS | no |
| cu_chg_xs | holdout_365d | -0.10% | 42% | PASS | no |
| us_cu_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_cu_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_cu_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_cu_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_cu_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_cu_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_cu_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_cu_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| cu_ew | year_2024 | +0.01% | 64% | PASS | no |
| cu_ew | year_2025 | -0.19% | 27% | PASS | no |
| cu_ew | year_2026 | +0.01% | 50% | PASS | no |
| cu_ew | holdout_365d | -0.00% | 50% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_cap_util_*.csv`, `scholarly_fx_cap_util_meta.json`.

Locked `fx4plus_gbpcad_d1_voltarget_0025` **untouched PASS** (see `quest_locked_verify_cap_util.md`).
