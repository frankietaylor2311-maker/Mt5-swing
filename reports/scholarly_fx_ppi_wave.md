# Scholarly FX: OECD producer-price (PIEAMP01) differential wave (§51)

**Path:** OECD MEI manufacturing PPI YoY via FRED (`{ISO3}PIEAMP01GYM` / `JPNPPDMMINMEI` / `PIEAMP01*Q661N`) — PPI YoY XS — **distinct** from IP (§42), retail (§46), employment (§41), ULC/LP/CU (§48–50), building-permits (§43), CLI/CCI/BCI, WUI, EPU/TPU.
**Data:** `approximate_non_ftmo` + free FRED PIEAMP01 GYM YoY (monthly + JPY/AUD/NZD level→YoY after pub_lag). **STALE** — G10 end ~2022-12 on free FRED (AUD Q ~2023-01; NZD Q ~2022-07). EUR = `DEUPIEAMP01GYM` (Germany proxy). Full G10 mapped. `n_long=n_short=2`. **PIT:** pub_lag_months=2 (PPI a priori) + signal_lag_months=1 + 1d weight lag. Score basis = **PPI YoY %**.
**Primary:** `high_ppi_xs` (long high relative PPI YoY / short low — Dahlquist economic-momentum → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_ppi_xs | +0.116% | +1.85 | +2.00 | 59% | 8% | +0.36 |
| low_ppi_xs | -0.126% | -2.00 | -2.14 | 41% | 14% | -0.38 |
| high_ppi_z_xs | +0.059% | +1.03 | +1.03 | 49% | 12% | +0.24 |
| ppi_chg_xs | +0.005% | +0.07 | +0.08 | 52% | 11% | -0.01 |
| us_ppi_stress_fx | -0.005% | -0.25 | -0.29 | 6% | 68% | -0.05 |
| us_ppi_haven_usd | +0.005% | +0.22 | +0.25 | 6% | 48% | +0.05 |
| ppi_ew | +0.039% | +1.19 | +1.38 | 50% | 10% | +0.20 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_ppi_xs | year_2024 | +0.06% | 73% | PASS | no |
| high_ppi_xs | year_2025 | -0.01% | 64% | PASS | no |
| high_ppi_xs | year_2026 | +0.03% | 50% | PASS | no |
| high_ppi_xs | holdout_365d | +0.17% | 67% | PASS | no |
| low_ppi_xs | year_2024 | -0.06% | 27% | PASS | no |
| low_ppi_xs | year_2025 | -0.00% | 36% | PASS | no |
| low_ppi_xs | year_2026 | -0.03% | 50% | PASS | no |
| low_ppi_xs | holdout_365d | -0.17% | 33% | PASS | no |
| high_ppi_z_xs | year_2024 | +0.07% | 55% | PASS | no |
| high_ppi_z_xs | year_2025 | -0.17% | 27% | PASS | no |
| high_ppi_z_xs | year_2026 | +0.09% | 50% | PASS | no |
| high_ppi_z_xs | holdout_365d | +0.07% | 50% | PASS | no |
| ppi_chg_xs | year_2024 | +0.17% | 64% | PASS | no |
| ppi_chg_xs | year_2025 | +0.16% | 73% | PASS | no |
| ppi_chg_xs | year_2026 | -0.13% | 38% | PASS | no |
| ppi_chg_xs | holdout_365d | -0.19% | 33% | PASS | no |
| us_ppi_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_ppi_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_ppi_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_ppi_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_ppi_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_ppi_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_ppi_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_ppi_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| ppi_ew | year_2024 | +0.08% | 55% | PASS | no |
| ppi_ew | year_2025 | +0.06% | 45% | PASS | no |
| ppi_ew | year_2026 | -0.03% | 50% | PASS | no |
| ppi_ew | holdout_365d | -0.01% | 42% | PASS | no |

**Board:** n=7 soft_nw_pos=1 hard_nw_pos=1 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_ppi_*.csv`, `scholarly_fx_ppi_meta.json`.
