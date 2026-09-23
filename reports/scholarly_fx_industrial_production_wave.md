# Scholarly FX: Industrial-production (PRINTO01) differential wave (§42)

**Path:** OECD MEI industry YoY via FRED (`{ISO3}PRINTO01GYSAM`) + AUD/NZD/CHF manufacturing YoY (`{ISO3}PRMNTO01GYSAQ`) — IP YoY XS — **distinct** from macro-diff EW CPI/IP/UR, OECD CLI (§37) / CCI (§38) / BCI (§39), employment (§41), WUI (§40), EPU/TPU, IG OAS (§36), CA/TB, fiscal/debt, BIS, reserves, house-price, money-growth, equity, commodity.
**Data:** `approximate_non_ftmo` + free FRED PRINTO01/PRMNTO01 YoY (monthly + quarterly → monthly after pub_lag). EUR = `FRAPRINTO01GYSAM` (France proxy; EA19/DEU stale). Full G10 mapped. `n_long=n_short=2` a priori. **PIT:** pub_lag_months=2 (IP a priori) + signal_lag_months=0 + 1d weight lag. Score basis = **YoY growth as reported**.
**Primary:** `high_ip_xs` (long high relative IP YoY / short low — growth-channel → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_ip_xs | +0.101% | +1.81 | +1.88 | 54% | 9% | +0.37 |
| low_ip_xs | -0.122% | -2.19 | -2.23 | 44% | 13% | -0.44 |
| high_ip_z_xs | +0.078% | +1.28 | +1.35 | 54% | 11% | +0.26 |
| ip_chg_xs | +0.038% | +0.60 | +0.60 | 53% | 8% | +0.15 |
| us_ip_stress_fx | +0.005% | +0.17 | +0.18 | 11% | 45% | +0.04 |
| us_ip_haven_usd | -0.006% | -0.20 | -0.20 | 8% | 33% | -0.05 |
| ip_ew | +0.048% | +1.28 | +1.32 | 57% | 9% | +0.28 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_ip_xs | year_2024 | +0.05% | 64% | PASS | no |
| high_ip_xs | year_2025 | +0.16% | 64% | PASS | no |
| high_ip_xs | year_2026 | +0.11% | 50% | PASS | no |
| high_ip_xs | holdout_365d | +0.16% | 50% | PASS | no |
| low_ip_xs | year_2024 | -0.08% | 36% | PASS | no |
| low_ip_xs | year_2025 | -0.18% | 27% | PASS | no |
| low_ip_xs | year_2026 | -0.14% | 50% | PASS | no |
| low_ip_xs | holdout_365d | -0.18% | 50% | PASS | no |
| high_ip_z_xs | year_2024 | -0.11% | 45% | PASS | no |
| high_ip_z_xs | year_2025 | +0.01% | 45% | PASS | no |
| high_ip_z_xs | year_2026 | -0.14% | 38% | PASS | no |
| high_ip_z_xs | holdout_365d | -0.10% | 42% | PASS | no |
| ip_chg_xs | year_2024 | -0.13% | 36% | PASS | no |
| ip_chg_xs | year_2025 | +0.12% | 45% | PASS | no |
| ip_chg_xs | year_2026 | +0.29% | 62% | PASS | no |
| ip_chg_xs | holdout_365d | +0.27% | 58% | PASS | no |
| us_ip_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_ip_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_ip_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_ip_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_ip_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_ip_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_ip_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_ip_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| ip_ew | year_2024 | -0.03% | 45% | PASS | no |
| ip_ew | year_2025 | +0.09% | 64% | PASS | no |
| ip_ew | year_2026 | +0.13% | 62% | PASS | no |
| ip_ew | holdout_365d | +0.14% | 58% | PASS | no |

**Board:** n=7 soft_nw_pos=1 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_industrial_production_*.csv`, `scholarly_fx_industrial_production_meta.json`.
