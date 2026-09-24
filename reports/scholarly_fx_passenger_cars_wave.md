# Scholarly FX: OECD/FRED passenger-car registrations differential wave (§60)

**Path:** FRED OECD MEI **passenger-car registrations YoY** (`SLRTCR03` GYSAM; EUR=Spain ESP proxy; NZD/CHF unmapped) — car-regs YoY XS — **distinct** from retail-sales (§46), building-permits (§43), construction (§59), IP (§42), CLI (§37), BCI (§39).
**Data:** `approximate_non_ftmo` + free FRED SLRTCR03 YoY after pub_lag. EUR=ESPSLRTCR03GYSAM Spain proxy; NZD/CHF unmapped stale; USD/GBP/JPY/CAD/AUD live. `n_long=n_short=2` on available foreign panel. **PIT:** pub_lag_months=2 (auto/retail ~1–2m lag a priori) + signal_lag_months=1 + 1d weight lag. Score basis = **passenger-car registrations YoY %**.
**Primary:** `high_cars_xs` (long high relative car-regs YoY / short low — growth-channel / durable auto demand → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_cars_xs | -0.138% | -2.15 | -2.18 | 45% | 12% | -0.53 |
| low_cars_xs | +0.120% | +1.87 | +1.93 | 54% | 11% | +0.47 |
| high_cars_z_xs | -0.091% | -1.55 | -1.43 | 47% | 12% | -0.36 |
| cars_chg_xs | -0.090% | -1.44 | -1.54 | 45% | 13% | -0.36 |
| us_cars_stress_fx | +0.012% | +0.38 | +0.38 | 14% | 32% | +0.09 |
| us_cars_haven_usd | -0.013% | -0.43 | -0.44 | 14% | 26% | -0.10 |
| cars_ew | -0.072% | -2.00 | -2.19 | 44% | 13% | -0.50 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_cars_xs | year_2024 | +0.01% | 55% | PASS | no |
| high_cars_xs | year_2025 | +0.18% | 64% | PASS | no |
| high_cars_xs | year_2026 | -0.11% | 50% | PASS | no |
| high_cars_xs | holdout_365d | +0.02% | 58% | PASS | no |
| low_cars_xs | year_2024 | -0.04% | 36% | PASS | no |
| low_cars_xs | year_2025 | -0.21% | 27% | PASS | no |
| low_cars_xs | year_2026 | +0.10% | 50% | PASS | no |
| low_cars_xs | holdout_365d | -0.03% | 42% | PASS | no |
| high_cars_z_xs | year_2024 | -0.09% | 64% | PASS | no |
| high_cars_z_xs | year_2025 | +0.02% | 73% | PASS | no |
| high_cars_z_xs | year_2026 | -0.27% | 38% | PASS | no |
| high_cars_z_xs | holdout_365d | -0.14% | 50% | PASS | no |
| cars_chg_xs | year_2024 | -0.16% | 45% | PASS | no |
| cars_chg_xs | year_2025 | +0.09% | 73% | PASS | no |
| cars_chg_xs | year_2026 | -0.11% | 38% | PASS | no |
| cars_chg_xs | holdout_365d | +0.06% | 50% | PASS | no |
| us_cars_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_cars_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_cars_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_cars_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_cars_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_cars_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_cars_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_cars_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| cars_ew | year_2024 | -0.05% | 36% | PASS | no |
| cars_ew | year_2025 | +0.09% | 73% | PASS | no |
| cars_ew | year_2026 | -0.07% | 38% | PASS | no |
| cars_ew | holdout_365d | +0.03% | 58% | PASS | no |

**Board:** n=7 soft_nw_pos=1 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_passenger_cars_*.csv`, `scholarly_fx_passenger_cars_meta.json`.
