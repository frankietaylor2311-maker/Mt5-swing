# Scholarly FX: CIP-conditioned soft-signal EW wave (§71)

**Path:** Du–Schreger **government-bond CIP** stress (−mean `cip_govt` bps, tenor=**5y**) × Dahlquist–Hasseltoft ``soft_ew_macro5`` gate/cool — **distinct** from soft_ew (§66 ungated), soft-stack CIP enrichment (§69 EW membership), CIP-conditioned carry (§68 gates Lustig–Verdelhan *carry*), CIP XS (§67), capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + public `cip_dataset_v4.csv` after pub_lag_days=1 + FRED/OECD soft legs (§66 PIT). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** CIP pub_lag_days=1 + signal_lag_months=1 + 1d weight lag; soft legs source-wave pub/signal lags. Costs 1.5 bps/side inside source factors.
**Primary:** `soft_low_cip_stress` (trade soft only when lagged CIP-stress z ≤ 0). Honesty `soft_raw` + `soft_high_cip_stress`. Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| soft_low_cip_stress | +0.040% | +2.66 | +1.98 | 21% | 30% | +0.67 |
| soft_cip_cool | +0.083% | +4.28 | +3.36 | 61% | 14% | +1.03 |
| soft_raw | +0.118% | +4.26 | +3.63 | 61% | 11% | +1.02 |
| soft_high_cip_stress | +0.019% | +1.18 | +1.11 | 20% | 30% | +0.30 |
| cip_stress_haven_usd | +0.073% | +1.78 | +1.91 | 21% | 24% | +0.47 |
| soft_cip_ew | +0.065% | +3.85 | +3.33 | 61% | 12% | +0.96 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| soft_raw | year_2024 | +0.08% | 55% | PASS | no |
| soft_raw | year_2025 | +0.04% | 64% | PASS | no |
| soft_raw | year_2026 | +0.14% | 88% | PASS | no |
| soft_raw | holdout_365d | +0.06% | 67% | PASS | no |
| soft_cip_cool | year_2024 | +0.06% | 55% | PASS | no |
| soft_cip_cool | year_2025 | +0.04% | 64% | PASS | no |
| soft_cip_cool | year_2026 | +0.14% | 88% | PASS | no |
| soft_cip_cool | holdout_365d | +0.06% | 67% | PASS | no |
| soft_low_cip_stress | year_2024 | -0.04% | 18% | PASS | no |
| soft_low_cip_stress | year_2025 | +0.04% | 64% | PASS | no |
| soft_low_cip_stress | year_2026 | +0.14% | 88% | PASS | no |
| soft_low_cip_stress | holdout_365d | +0.06% | 67% | PASS | no |
| soft_high_cip_stress | year_2024 | +0.00% | 0% | PASS | no |
| soft_high_cip_stress | year_2025 | +0.00% | 0% | PASS | no |
| soft_high_cip_stress | year_2026 | +0.00% | 0% | PASS | no |
| soft_high_cip_stress | holdout_365d | +0.00% | 0% | PASS | no |
| cip_stress_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| cip_stress_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| cip_stress_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| cip_stress_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| soft_cip_ew | year_2024 | +0.01% | 55% | PASS | no |
| soft_cip_ew | year_2025 | +0.03% | 64% | PASS | no |
| soft_cip_ew | year_2026 | +0.09% | 88% | PASS | no |
| soft_cip_ew | holdout_365d | +0.04% | 67% | PASS | no |

**Board:** n=6 soft_nw_pos=5 hard_nw_pos=3 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** CIP panel ends **2025-07-31** (cip_dataset_v4 vintage); 2026 FX window uses stale/ffilled CIP. Tenor=5y a priori (not HO-tuned). Gate z≤0 / cool z≥1 fixed (not HO-tuned).

Artifacts: `reports/scholarly_fx_cip_conditioned_soft_*.csv`, `scholarly_fx_cip_conditioned_soft_meta.json`, `quest_locked_verify_cip_conditioned_soft.md`.
