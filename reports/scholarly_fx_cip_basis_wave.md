# Scholarly FX: Du–Schreger CIP / U.S. Treasury premium wave (§67)

**Path:** Du–Keerati–Schreger **government-bond CIP** (`cip_govt` bps, tenor=**5y** a priori) + U.S. Treasury Premium (−mean CIP) — **distinct** from funding_liq (§20), fwd_carry (§19 rate-implied FD), IG OAS (§36), CB-BS (§22).
**Data:** `approximate_non_ftmo` + public `cip_dataset_v4.csv` after pub_lag_days=1. G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. `n_long=n_short=2`. **PIT:** pub_lag_days=1 + signal_lag_months=1 + 1d weight lag. Score basis = **cip_govt bps** (month-end of daily).
**Primary:** `low_cip_xs` (long low / more-negative CIP / short high — Du–Schreger convenience + DTV dollar-funding channel). Honesty `high_cip_xs`. Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| low_cip_xs | +0.047% | +0.54 | +0.63 | 53% | 10% | +0.10 |
| high_cip_xs | -0.051% | -0.58 | -0.67 | 46% | 14% | -0.11 |
| low_cip_z_xs | +0.041% | +0.57 | +0.68 | 49% | 13% | +0.14 |
| cip_chg_xs | +0.059% | +0.82 | +0.88 | 56% | 11% | +0.21 |
| ust_premium_haven_usd | +0.076% | +1.86 | +1.94 | 22% | 24% | +0.50 |
| ust_premium_stress_fx | -0.077% | -1.91 | -1.99 | 16% | 27% | -0.51 |
| cip_ew | +0.061% | +1.64 | +1.68 | 58% | 14% | +0.38 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| low_cip_xs | year_2024 | +0.12% | 64% | PASS | no |
| low_cip_xs | year_2025 | +0.44% | 82% | PASS | no |
| low_cip_xs | year_2026 | +0.14% | 62% | PASS | no |
| low_cip_xs | holdout_365d | +0.24% | 75% | PASS | no |
| high_cip_xs | year_2024 | -0.12% | 36% | PASS | no |
| high_cip_xs | year_2025 | -0.44% | 18% | PASS | no |
| high_cip_xs | year_2026 | -0.14% | 38% | PASS | no |
| high_cip_xs | holdout_365d | -0.25% | 25% | PASS | no |
| low_cip_z_xs | year_2024 | +0.07% | 27% | PASS | no |
| low_cip_z_xs | year_2025 | +0.06% | 55% | PASS | no |
| low_cip_z_xs | year_2026 | -0.01% | 38% | PASS | no |
| low_cip_z_xs | holdout_365d | -0.16% | 25% | PASS | no |
| cip_chg_xs | year_2024 | -0.37% | 18% | PASS | no |
| cip_chg_xs | year_2025 | -0.06% | 55% | PASS | no |
| cip_chg_xs | year_2026 | +0.09% | 38% | PASS | no |
| cip_chg_xs | holdout_365d | -0.07% | 33% | PASS | no |
| ust_premium_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| ust_premium_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| ust_premium_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| ust_premium_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| ust_premium_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| ust_premium_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| ust_premium_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| ust_premium_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| cip_ew | year_2024 | -0.08% | 45% | PASS | no |
| cip_ew | year_2025 | +0.13% | 73% | PASS | no |
| cip_ew | year_2026 | +0.08% | 75% | PASS | no |
| cip_ew | holdout_365d | +0.06% | 75% | PASS | no |

**Board:** n=7 soft_nw_pos=2 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** panel ends **2025-07-31** (cip_dataset_v4 vintage); 2026 FX window uses stale/ffilled CIP. Tenor=5y a priori (not HO-tuned).

Artifacts: `reports/scholarly_fx_cip_basis_*.csv`, `scholarly_fx_cip_basis_meta.json`, `quest_locked_verify_cip_basis.md`.
