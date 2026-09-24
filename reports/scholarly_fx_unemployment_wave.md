# Scholarly FX: OECD/FRED unemployment-rate differential wave (§54)

**Path:** FRED OECD MEI / national unemployment **rate levels** (`LRHUTTTT*M156S`; CHF=`LMUNRRTTCHM156S`; NZD unmapped) — UR level XS — **distinct** from employment persons (§41), macro_diff EW CPI+IP+UR blend (§12), CPI (§52), PPI (§51), CLI/CCI/BCI, WUI, ULC/LP/CU.
**Data:** `approximate_non_ftmo` + free FRED UR levels after pub_lag. EUR EA UR sparse tail ~2023-01; NZD unmapped; CHF registered UR. `n_long=n_short=2` on available foreign panel. **PIT:** pub_lag_months=1 (matches macro_diff ur default) + signal_lag_months=1 + 1d weight lag. Score basis = **UR level %**.
**Primary:** `low_ur_xs` (long low relative UR / short high — Dahlquist labour strength → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| low_ur_xs | -0.009% | -0.15 | -0.16 | 46% | 16% | -0.03 |
| high_ur_xs | +0.007% | +0.12 | +0.13 | 54% | 9% | +0.03 |
| low_ur_z_xs | +0.038% | +0.65 | +0.69 | 57% | 12% | +0.13 |
| ur_chg_xs | -0.026% | -0.42 | -0.41 | 49% | 13% | -0.09 |
| us_ur_stress_fx | +0.023% | +1.43 | +1.36 | 3% | 89% | +0.50 |
| us_ur_haven_usd | -0.023% | -1.44 | -1.36 | 1% | 100% | -0.50 |
| ur_ew | -0.004% | -0.14 | -0.14 | 49% | 13% | -0.03 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| low_ur_xs | year_2024 | -0.10% | 45% | PASS | no |
| low_ur_xs | year_2025 | -0.18% | 36% | PASS | no |
| low_ur_xs | year_2026 | +0.18% | 62% | PASS | no |
| low_ur_xs | holdout_365d | +0.06% | 50% | PASS | no |
| high_ur_xs | year_2024 | +0.08% | 55% | PASS | no |
| high_ur_xs | year_2025 | +0.18% | 64% | PASS | no |
| high_ur_xs | year_2026 | -0.18% | 38% | PASS | no |
| high_ur_xs | holdout_365d | -0.06% | 50% | PASS | no |
| low_ur_z_xs | year_2024 | -0.18% | 64% | PASS | no |
| low_ur_z_xs | year_2025 | -0.18% | 36% | PASS | no |
| low_ur_z_xs | year_2026 | +0.08% | 50% | PASS | no |
| low_ur_z_xs | holdout_365d | -0.01% | 42% | PASS | no |
| ur_chg_xs | year_2024 | +0.01% | 45% | PASS | no |
| ur_chg_xs | year_2025 | -0.16% | 36% | PASS | no |
| ur_chg_xs | year_2026 | +0.04% | 50% | PASS | no |
| ur_chg_xs | holdout_365d | -0.04% | 42% | PASS | no |
| us_ur_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_ur_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_ur_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_ur_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_ur_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_ur_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_ur_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_ur_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| ur_ew | year_2024 | -0.03% | 36% | PASS | no |
| ur_ew | year_2025 | -0.11% | 36% | PASS | no |
| ur_ew | year_2026 | +0.07% | 50% | PASS | no |
| ur_ew | holdout_365d | +0.01% | 42% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_unemployment_*.csv`, `scholarly_fx_unemployment_meta.json`.

Locked `fx4plus_gbpcad_d1_voltarget_0025` **untouched PASS** (see `quest_locked_verify_unemployment.md`).
