# Scholarly FX: OECD ULC / wage-growth (ULQEUL01) differential wave (§48)

**Path:** OECD MEI early-estimate ULC YoY via FRED (`ULQEUL01{ISO2}Q657S`) — ULC YoY XS competitiveness — **distinct** from employment (§41), IP (§42), retail (§46), house-price (§35), REER (§31), money (§33), credit (§32), DSR (§47), OECD CLI/CCI/BCI, WUI, EPU/TPU, macro-diff EW CPI/IP/UR.
**Data:** `approximate_non_ftmo` + free FRED ULQEUL01 Q657S YoY (quarterly → monthly after pub_lag). EUR = `ULQEUL01DEQ657S` (Germany proxy; EZ stale 2024-01). Full G10 mapped incl NZD/CHF. `n_long=n_short=2` a priori. **PIT:** pub_lag_months=3 (ULC labour a priori) + signal_lag_months=1 + 1d weight lag. Score basis = **YoY growth as reported**.
**Primary:** `low_ulc_xs` (long low relative ULC YoY / short high — competitiveness → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| low_ulc_xs | -0.044% | -0.66 | -0.78 | 46% | 14% | -0.18 |
| high_ulc_xs | +0.029% | +0.44 | +0.52 | 54% | 10% | +0.13 |
| low_ulc_z_xs | -0.100% | -1.70 | -1.77 | 44% | 13% | -0.42 |
| ulc_chg_xs | -0.108% | -1.69 | -1.98 | 42% | 15% | -0.43 |
| us_ulc_stress_fx | +0.019% | +0.81 | +0.97 | 8% | 47% | +0.23 |
| us_ulc_haven_usd | -0.020% | -0.85 | -1.01 | 7% | 56% | -0.24 |
| ulc_ew | -0.044% | -1.20 | -1.39 | 45% | 17% | -0.31 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| low_ulc_xs | year_2024 | -0.44% | 18% | PASS | no |
| low_ulc_xs | year_2025 | +0.17% | 64% | PASS | no |
| low_ulc_xs | year_2026 | -0.12% | 38% | PASS | no |
| low_ulc_xs | holdout_365d | -0.03% | 42% | PASS | no |
| high_ulc_xs | year_2024 | +0.42% | 82% | PASS | no |
| high_ulc_xs | year_2025 | -0.19% | 36% | PASS | no |
| high_ulc_xs | year_2026 | +0.11% | 62% | PASS | no |
| high_ulc_xs | holdout_365d | +0.02% | 58% | PASS | no |
| low_ulc_z_xs | year_2024 | -0.55% | 45% | PASS | no |
| low_ulc_z_xs | year_2025 | +0.17% | 55% | PASS | no |
| low_ulc_z_xs | year_2026 | +0.09% | 50% | PASS | no |
| low_ulc_z_xs | holdout_365d | +0.21% | 67% | PASS | no |
| ulc_chg_xs | year_2024 | -0.33% | 36% | PASS | no |
| ulc_chg_xs | year_2025 | -0.13% | 36% | PASS | no |
| ulc_chg_xs | year_2026 | +0.18% | 62% | PASS | no |
| ulc_chg_xs | holdout_365d | +0.10% | 58% | PASS | no |
| us_ulc_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_ulc_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_ulc_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_ulc_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_ulc_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_ulc_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_ulc_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_ulc_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| ulc_ew | year_2024 | -0.26% | 18% | PASS | no |
| ulc_ew | year_2025 | +0.01% | 55% | PASS | no |
| ulc_ew | year_2026 | +0.02% | 50% | PASS | no |
| ulc_ew | holdout_365d | +0.02% | 50% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_ulc_*.csv`, `scholarly_fx_ulc_meta.json`.
