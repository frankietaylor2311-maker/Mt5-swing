# Scholarly FX: CIP-conditioned Lustig–Verdelhan carry wave (§68)

**Path:** Du–Schreger **government-bond CIP** stress (−mean `cip_govt` bps, tenor=**5y**) × Lustig–Verdelhan IR3M carry gate/cool — **distinct** from fwd_carry (§19), funding_liq (§20), IG OAS (§36), ACM TP (§44), CIP XS (§67), combo (§8), capital-sleeve (§53) / soft EW (§66).
**Data:** `approximate_non_ftmo` + public `cip_dataset_v4.csv` after pub_lag_days=1 + FRED IR3M rates (pub_lag_months=1). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. `n_long=n_short=2`. **PIT:** CIP pub_lag_days=1 + signal_lag_months=1 + 1d weight lag; carry signal_lag=1d.
**Primary:** `carry_low_cip_stress` (trade carry only when lagged CIP-stress z ≤ 0). Honesty `carry_raw` + `carry_high_cip_stress`. Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| cip_stress_haven_usd | +0.073% | +1.78 | +1.91 | 21% | 24% | +0.47 |
| carry_raw | -0.002% | -0.02 | -0.03 | 51% | 10% | -0.04 |
| carry_cip_cool | +0.039% | +0.75 | +0.78 | 51% | 13% | +0.14 |
| carry_low_cip_stress | +0.057% | +1.67 | +1.49 | 23% | 21% | +0.34 |
| carry_high_cip_stress | -0.031% | -0.81 | -0.88 | 16% | 27% | -0.18 |
| cip_carry_ew | +0.056% | +1.98 | +1.89 | 57% | 11% | +0.40 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| cip_stress_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| cip_stress_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| cip_stress_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| cip_stress_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| carry_raw | year_2024 | +0.15% | 64% | PASS | no |
| carry_raw | year_2025 | +0.12% | 64% | PASS | no |
| carry_raw | year_2026 | +0.13% | 62% | PASS | no |
| carry_raw | holdout_365d | +0.31% | 75% | PASS | no |
| carry_cip_cool | year_2024 | +0.10% | 64% | PASS | no |
| carry_cip_cool | year_2025 | +0.12% | 64% | PASS | no |
| carry_cip_cool | year_2026 | +0.13% | 62% | PASS | no |
| carry_cip_cool | holdout_365d | +0.31% | 75% | PASS | no |
| carry_low_cip_stress | year_2024 | -0.15% | 27% | PASS | no |
| carry_low_cip_stress | year_2025 | +0.12% | 64% | PASS | no |
| carry_low_cip_stress | year_2026 | +0.13% | 62% | PASS | no |
| carry_low_cip_stress | holdout_365d | +0.31% | 75% | PASS | no |
| carry_high_cip_stress | year_2024 | +0.00% | 0% | PASS | no |
| carry_high_cip_stress | year_2025 | +0.00% | 0% | PASS | no |
| carry_high_cip_stress | year_2026 | +0.00% | 0% | PASS | no |
| carry_high_cip_stress | holdout_365d | +0.00% | 0% | PASS | no |
| cip_carry_ew | year_2024 | -0.02% | 73% | PASS | no |
| cip_carry_ew | year_2025 | +0.08% | 64% | PASS | no |
| cip_carry_ew | year_2026 | +0.09% | 62% | PASS | no |
| cip_carry_ew | holdout_365d | +0.21% | 75% | PASS | no |

**Board:** n=6 soft_nw_pos=2 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** CIP panel ends **2025-07-31** (cip_dataset_v4 vintage); 2026 FX window uses stale/ffilled CIP. Tenor=5y a priori (not HO-tuned). Gate z≤0 / cool z≥1 fixed (not HO-tuned).

Artifacts: `reports/scholarly_fx_cip_conditioned_carry_*.csv`, `scholarly_fx_cip_conditioned_carry_meta.json`, `quest_locked_verify_cip_conditioned_carry.md`.
