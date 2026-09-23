# Scholarly FX: OECD CLI leading-indicator differential wave (§37)

**Path:** OECD amplitude-adjusted CLI via FRED `*LOLITOAASTSAM` — Estrella–Mishkin / leading-activity XS — **distinct** from coincident macro-diff CPI/IP/UR, house-price (§35), money-growth (§33), IG OAS (§36), CA/TB, BIS REER/credit, reserves, equity-diff, GPR/EPU.
**Data:** `approximate_non_ftmo` + free FRED `*LOLITOAASTSAM`. EUR = `DEULOLITOAASTSAM` (Germany proxy; EA19 ends 2022-11). NZD/CHF **unmapped** on primary (NZL 2019 / CHE 2022 stale). **PIT:** pub_lag_months=2 (conservative monthly OECD CLI) + signal_lag_months=0 + 1d weight lag.
**Primary:** `high_cli_xs` (long high relative CLI YoY growth / short low — leading-activity strength → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_cli_xs | +0.045% | +0.71 | +0.72 | 54% | 11% | +0.19 |
| low_cli_xs | -0.051% | -0.82 | -0.83 | 44% | 14% | -0.22 |
| high_cli_z_xs | -0.007% | -0.10 | -0.11 | 51% | 10% | +0.02 |
| cli_chg_xs | -0.033% | -0.50 | -0.49 | 47% | 12% | -0.13 |
| us_cli_weak_fx | +0.005% | +0.18 | +0.23 | 9% | 50% | +0.05 |
| us_cli_haven_usd | -0.005% | -0.20 | -0.26 | 9% | 35% | -0.05 |
| cli_ew | +0.005% | +0.14 | +0.15 | 47% | 12% | +0.04 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_cli_xs | year_2024 | +0.20% | 55% | PASS | no |
| high_cli_xs | year_2025 | +0.08% | 55% | PASS | no |
| high_cli_xs | year_2026 | +0.05% | 50% | PASS | no |
| high_cli_xs | holdout_365d | +0.16% | 58% | PASS | no |
| low_cli_xs | year_2024 | -0.20% | 45% | PASS | no |
| low_cli_xs | year_2025 | -0.09% | 45% | PASS | no |
| low_cli_xs | year_2026 | -0.06% | 25% | PASS | no |
| low_cli_xs | holdout_365d | -0.16% | 25% | PASS | no |
| high_cli_z_xs | year_2024 | -0.21% | 36% | PASS | no |
| high_cli_z_xs | year_2025 | +0.14% | 55% | PASS | no |
| high_cli_z_xs | year_2026 | +0.19% | 50% | PASS | no |
| high_cli_z_xs | holdout_365d | +0.33% | 67% | PASS | no |
| cli_chg_xs | year_2024 | -0.04% | 55% | PASS | no |
| cli_chg_xs | year_2025 | -0.11% | 36% | PASS | no |
| cli_chg_xs | year_2026 | -0.13% | 38% | PASS | no |
| cli_chg_xs | holdout_365d | -0.22% | 33% | PASS | no |
| us_cli_weak_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_cli_weak_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_cli_weak_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_cli_weak_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_cli_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_cli_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_cli_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_cli_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| cli_ew | year_2024 | +0.05% | 45% | PASS | no |
| cli_ew | year_2025 | -0.01% | 45% | PASS | no |
| cli_ew | year_2026 | -0.03% | 38% | PASS | no |
| cli_ew | holdout_365d | -0.02% | 42% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_oecd_cli_*.csv`, `scholarly_fx_oecd_cli_meta.json`.
