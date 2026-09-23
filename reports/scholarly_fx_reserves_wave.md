# Scholarly FX: IMF/WEO reserves / external-buffer wave (§34)

**Path:** IMF IFS `TRESEG*M052N` via FRED — Aizenman–Jeanne–Rancière / reserve-adequacy XS — **distinct** from CA (§21), CB-BS (§22), fiscal (§28), debt (§29), TB (§30), BIS REER (§31), BIS credit (§32), money-growth (§33), funding-liq, macro-diff, equity-diff, AI-GPR, ToT, dollar-beta, crash-skew.
**Data:** `approximate_non_ftmo` + free FRED `TRESEG*M052N` (IMF IFS excl. gold, log USD mn). EUR = Germany `TRESEGDEM052N` (`TRESEGEZM052N` stale 2018). NZD/CHF **unmapped** (404). Reserves/GDP **not formed** (GDP units heterogeneous / many 404). **PIT:** pub_lag_months=3 (conservative IMF IFS) + signal_lag=1m + 1d weight lag.
**Primary:** `high_reserves_xs` (long high relative external buffer / short low). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_reserves_xs | -0.038% | -0.59 | -0.58 | 48% | 12% | -0.09 |
| low_reserves_xs | +0.036% | +0.57 | +0.56 | 52% | 13% | +0.09 |
| high_reserves_z_xs | -0.027% | -0.43 | -0.47 | 52% | 10% | -0.11 |
| reserves_chg_xs | +0.048% | +0.74 | +0.81 | 52% | 11% | +0.18 |
| us_reserves_stress_fx | -0.002% | -0.08 | -0.09 | 9% | 49% | -0.02 |
| us_reserves_haven_usd | +0.002% | +0.06 | +0.07 | 8% | 36% | +0.02 |
| reserves_ew | +0.003% | +0.10 | +0.10 | 46% | 16% | +0.06 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_reserves_xs | year_2024 | +0.04% | 36% | PASS | no |
| high_reserves_xs | year_2025 | -0.32% | 27% | PASS | no |
| high_reserves_xs | year_2026 | -0.06% | 50% | PASS | no |
| high_reserves_xs | holdout_365d | -0.24% | 33% | PASS | no |
| low_reserves_xs | year_2024 | -0.04% | 64% | PASS | no |
| low_reserves_xs | year_2025 | +0.32% | 73% | PASS | no |
| low_reserves_xs | year_2026 | +0.06% | 50% | PASS | no |
| low_reserves_xs | holdout_365d | +0.23% | 67% | PASS | no |
| high_reserves_z_xs | year_2024 | -0.06% | 64% | PASS | no |
| high_reserves_z_xs | year_2025 | +0.18% | 55% | PASS | no |
| high_reserves_z_xs | year_2026 | +0.14% | 62% | PASS | no |
| high_reserves_z_xs | holdout_365d | +0.29% | 75% | PASS | no |
| reserves_chg_xs | year_2024 | +0.10% | 45% | PASS | no |
| reserves_chg_xs | year_2025 | -0.08% | 45% | PASS | no |
| reserves_chg_xs | year_2026 | +0.10% | 62% | PASS | no |
| reserves_chg_xs | holdout_365d | -0.01% | 58% | PASS | no |
| us_reserves_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_reserves_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_reserves_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_reserves_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_reserves_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_reserves_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_reserves_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_reserves_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| reserves_ew | year_2024 | +0.05% | 45% | PASS | no |
| reserves_ew | year_2025 | -0.13% | 18% | PASS | no |
| reserves_ew | year_2026 | +0.01% | 62% | PASS | no |
| reserves_ew | holdout_365d | -0.08% | 42% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_reserves_*.csv`, `scholarly_fx_reserves_meta.json`.
