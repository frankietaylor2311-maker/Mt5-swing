# Scholarly FX: OECD CCI consumer-confidence differential wave (§38)

**Path:** OECD CCI via FRED `CSCICP02*M460S` (+ US `USACSCICP02STSAM`) — Ludvigson-style sentiment XS — **distinct** from OECD CLI (§37), coincident macro-diff CPI/IP/UR, house-price (§35), money-growth (§33), IG OAS (§36), CA/TB, BIS REER/credit, reserves, equity-diff, GPR/EPU.
**Data:** `approximate_non_ftmo` + free FRED `CSCICP02*M460S` / `USACSCICP02STSAM`. EUR = `CSCICP02EZM460S` (EZ aggregate). CAD/NZD/CHF **unmapped** (CSCICP02 404; CSCICP03 amplitude stale). **PIT:** pub_lag_months=2 (conservative monthly OECD CCI) + signal_lag_months=0 + 1d weight lag.
**Primary:** `high_cci_xs` (long high relative CCI YoY change / short low — consumer-confidence strength → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_cci_xs | +0.016% | +0.24 | +0.26 | 53% | 11% | +0.03 |
| low_cci_xs | -0.029% | -0.44 | -0.47 | 47% | 12% | -0.07 |
| high_cci_z_xs | -0.034% | -0.53 | -0.54 | 46% | 13% | -0.15 |
| cci_chg_xs | -0.027% | -0.43 | -0.50 | 52% | 11% | -0.13 |
| us_cci_weak_fx | +0.002% | +0.06 | +0.06 | 14% | 33% | +0.02 |
| us_cci_haven_usd | -0.004% | -0.12 | -0.12 | 11% | 34% | -0.03 |
| cci_ew | -0.003% | -0.07 | -0.08 | 52% | 12% | -0.05 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_cci_xs | year_2024 | +0.55% | 64% | PASS | no |
| high_cci_xs | year_2025 | +0.21% | 64% | PASS | no |
| high_cci_xs | year_2026 | +0.09% | 50% | PASS | no |
| high_cci_xs | holdout_365d | +0.22% | 58% | PASS | no |
| low_cci_xs | year_2024 | -0.49% | 45% | PASS | no |
| low_cci_xs | year_2025 | -0.31% | 27% | PASS | no |
| low_cci_xs | year_2026 | -0.09% | 50% | PASS | no |
| low_cci_xs | holdout_365d | -0.30% | 33% | PASS | no |
| high_cci_z_xs | year_2024 | +0.18% | 36% | PASS | no |
| high_cci_z_xs | year_2025 | +0.25% | 73% | PASS | no |
| high_cci_z_xs | year_2026 | +0.09% | 50% | PASS | no |
| high_cci_z_xs | holdout_365d | +0.30% | 67% | PASS | no |
| cci_chg_xs | year_2024 | +0.02% | 73% | PASS | no |
| cci_chg_xs | year_2025 | +0.11% | 55% | PASS | no |
| cci_chg_xs | year_2026 | -0.03% | 50% | PASS | no |
| cci_chg_xs | holdout_365d | -0.09% | 42% | PASS | no |
| us_cci_weak_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_cci_weak_fx | year_2025 | -0.00% | 18% | PASS | no |
| us_cci_weak_fx | year_2026 | -0.15% | 12% | PASS | no |
| us_cci_weak_fx | holdout_365d | -0.10% | 8% | PASS | no |
| us_cci_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_cci_haven_usd | year_2025 | +0.00% | 9% | PASS | no |
| us_cci_haven_usd | year_2026 | +0.14% | 25% | PASS | no |
| us_cci_haven_usd | holdout_365d | +0.09% | 17% | PASS | no |
| cci_ew | year_2024 | +0.19% | 82% | PASS | no |
| cci_ew | year_2025 | +0.11% | 82% | PASS | no |
| cci_ew | year_2026 | -0.03% | 50% | PASS | no |
| cci_ew | holdout_365d | +0.01% | 58% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_oecd_cci_*.csv`, `scholarly_fx_oecd_cci_meta.json`.
