# Scholarly FX: OECD/BIS residential house-price wave (§35)

**Path:** BIS `Q*R628BIS` via FRED — Aoki–Proudman–Vlieghe / housing wealth-collateral XS — **distinct** from CA (§21), CB-BS (§22), fiscal (§28), debt (§29), TB (§30), BIS REER (§31), BIS credit (§32), money-growth (§33), reserves (§34), funding-liq, macro-diff, equity-diff, AI-GPR, ToT, dollar-beta, crash-skew, IG OAS (deferred).
**Data:** `approximate_non_ftmo` + free FRED BIS `Q*R628BIS` (real residential HPI index). EUR = Germany `QDER628BIS` (`QEUR628BIS` 404). NZD/CHF **mapped** (`QNZR628BIS` / `QCHR628BIS`). Scores = YoY log-diff (not levels). **PIT:** pub_lag_months=4 (conservative quarterly property prices) + signal_lag=1m + 1d weight lag.
**Primary:** `high_hpi_xs` (long high relative HPI momentum / short low). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_hpi_xs | +0.036% | +0.55 | +0.60 | 52% | 12% | +0.16 |
| low_hpi_xs | -0.040% | -0.61 | -0.67 | 47% | 13% | -0.18 |
| high_hpi_z_xs | +0.028% | +0.45 | +0.46 | 52% | 12% | +0.16 |
| hpi_chg_xs | +0.108% | +1.60 | +1.37 | 54% | 12% | +0.40 |
| us_hpi_stress_fx | -0.015% | -0.45 | -0.65 | 13% | 28% | -0.11 |
| us_hpi_haven_usd | +0.013% | +0.41 | +0.59 | 16% | 25% | +0.10 |
| hpi_ew | +0.043% | +1.10 | +1.05 | 56% | 11% | +0.30 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_hpi_xs | year_2024 | -0.16% | 36% | PASS | no |
| high_hpi_xs | year_2025 | +0.29% | 82% | PASS | no |
| high_hpi_xs | year_2026 | +0.09% | 50% | PASS | no |
| high_hpi_xs | holdout_365d | +0.11% | 50% | PASS | no |
| low_hpi_xs | year_2024 | +0.15% | 55% | PASS | no |
| low_hpi_xs | year_2025 | -0.29% | 18% | PASS | no |
| low_hpi_xs | year_2026 | -0.09% | 50% | PASS | no |
| low_hpi_xs | holdout_365d | -0.11% | 50% | PASS | no |
| high_hpi_z_xs | year_2024 | -0.14% | 45% | PASS | no |
| high_hpi_z_xs | year_2025 | +0.39% | 82% | PASS | no |
| high_hpi_z_xs | year_2026 | -0.13% | 38% | PASS | no |
| high_hpi_z_xs | holdout_365d | +0.04% | 58% | PASS | no |
| hpi_chg_xs | year_2024 | -0.22% | 36% | PASS | no |
| hpi_chg_xs | year_2025 | +0.18% | 64% | PASS | no |
| hpi_chg_xs | year_2026 | -0.06% | 38% | PASS | no |
| hpi_chg_xs | holdout_365d | -0.14% | 33% | PASS | no |
| us_hpi_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_hpi_stress_fx | year_2025 | -0.18% | 18% | PASS | no |
| us_hpi_stress_fx | year_2026 | -0.19% | 38% | PASS | no |
| us_hpi_stress_fx | holdout_365d | -0.07% | 42% | PASS | no |
| us_hpi_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_hpi_haven_usd | year_2025 | +0.18% | 27% | PASS | no |
| us_hpi_haven_usd | year_2026 | +0.19% | 62% | PASS | no |
| us_hpi_haven_usd | holdout_365d | +0.07% | 58% | PASS | no |
| hpi_ew | year_2024 | -0.13% | 55% | PASS | no |
| hpi_ew | year_2025 | +0.10% | 73% | PASS | no |
| hpi_ew | year_2026 | -0.05% | 25% | PASS | no |
| hpi_ew | holdout_365d | -0.03% | 33% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_house_price_*.csv`, `scholarly_fx_house_price_meta.json`.
