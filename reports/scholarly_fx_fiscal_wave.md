# Scholarly FX: fiscal-balance / government-budget wave

**Path:** Twin deficits / fiscal sustainability — IMF WEO general-government net lending/borrowing (% GDP) via FRED — **distinct** from CA (§21), CB-BS, macro-diff, PPP/BS.
**Data:** `approximate_non_ftmo` + free FRED `GGNLBA*188N` (+ US MTS `MTSDS133FMS`). **PIT:** pub_lag_months=15 (~April Y+1 for year-Y) + signal_lag=1m + 1d weight lag.
**Primary:** `fiscal_surplus_xs` (long surplus / short deficit). NZD/CHF unmapped on free FRED. Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| fiscal_surplus_xs | -0.037% | -0.70 | -0.75 | 48% | 12% | -0.19 |
| fiscal_deficit_xs | +0.035% | +0.65 | +0.70 | 52% | 12% | +0.19 |
| fiscal_chg_xs | +0.021% | +0.34 | +0.36 | 48% | 14% | +0.09 |
| us_fiscal_twin_fx | -0.029% | -0.71 | -0.79 | 13% | 33% | -0.26 |
| us_fiscal_haven_usd | +0.030% | +0.72 | +0.79 | 13% | 29% | +0.25 |
| us_mts_chg_usd | -0.003% | -0.11 | -0.11 | 16% | 45% | -0.03 |
| fiscal_ca_blend | -0.032% | -0.73 | -0.88 | 47% | 12% | -0.14 |
| fiscal_ew | -0.015% | -0.48 | -0.58 | 48% | 14% | -0.15 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| fiscal_surplus_xs | year_2024 | +0.05% | 64% | PASS | no |
| fiscal_surplus_xs | year_2025 | -0.18% | 36% | PASS | no |
| fiscal_surplus_xs | year_2026 | +0.18% | 62% | PASS | no |
| fiscal_surplus_xs | holdout_365d | +0.06% | 50% | PASS | no |
| fiscal_deficit_xs | year_2024 | -0.05% | 36% | PASS | no |
| fiscal_deficit_xs | year_2025 | +0.18% | 64% | PASS | no |
| fiscal_deficit_xs | year_2026 | -0.18% | 38% | PASS | no |
| fiscal_deficit_xs | holdout_365d | -0.06% | 50% | PASS | no |
| fiscal_chg_xs | year_2024 | +0.04% | 64% | PASS | no |
| fiscal_chg_xs | year_2025 | -0.34% | 18% | PASS | no |
| fiscal_chg_xs | year_2026 | +0.19% | 75% | PASS | no |
| fiscal_chg_xs | holdout_365d | +0.09% | 58% | PASS | no |
| us_fiscal_twin_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_fiscal_twin_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_fiscal_twin_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_fiscal_twin_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_fiscal_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_fiscal_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_fiscal_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_fiscal_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| us_mts_chg_usd | year_2024 | +0.09% | 18% | PASS | no |
| us_mts_chg_usd | year_2025 | +0.05% | 18% | PASS | no |
| us_mts_chg_usd | year_2026 | -0.13% | 0% | PASS | no |
| us_mts_chg_usd | holdout_365d | -0.14% | 8% | PASS | no |
| fiscal_ca_blend | year_2024 | +0.03% | 36% | PASS | no |
| fiscal_ca_blend | year_2025 | -0.13% | 45% | PASS | no |
| fiscal_ca_blend | year_2026 | +0.07% | 50% | PASS | no |
| fiscal_ca_blend | holdout_365d | -0.12% | 33% | PASS | no |
| fiscal_ew | year_2024 | +0.03% | 64% | PASS | no |
| fiscal_ew | year_2025 | -0.17% | 27% | PASS | no |
| fiscal_ew | year_2026 | +0.12% | 75% | PASS | no |
| fiscal_ew | holdout_365d | +0.05% | 58% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_fiscal_*.csv`, `scholarly_fx_fiscal_meta.json`.
