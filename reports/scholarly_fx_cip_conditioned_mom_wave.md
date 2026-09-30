# Scholarly FX: CIP-conditioned Menkhoff FX momentum wave (§82)

**Path:** Du–Schreger **government-bond CIP** stress (−mean `cip_govt` bps, tenor=**5y**) × Menkhoff (2012 JFE) FX momentum gate/cool — **distinct** from combo (§8), raw fx_momentum, CIP×carry (§68), CIP soft (§71), VIX/GPR mom (§74), EPU/TPU mom (§79), soft/carry/value/REER stress (§72–§81), capital-sleeve (§53/§70). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + public `cip_dataset_v4.csv` after pub_lag_days=1 + Yahoo D1 G10. **PIT:** CIP pub_lag_days=1 + signal_lag_months=1 on trailing **monthly** z (z_window=60m, §68/§71 mirror) + weight_lag_days=1; mom formation=63d skip=21d signal_lag=1d. Costs 1.5 bps/side.
**Primary:** `mom_low_cip` (trade mom only when lagged CIP-stress z ≤ 0). Companions: mom_cip_cool / mom_raw / mom_high_cip / cip_stress_haven_usd / mom_cip_stack / mom_cip_ew / mom_cip_regime. Cool z_high=1.0 (§68/§71 symmetry). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| mom_low_cip | -0.005% | -0.12 | -0.10 | 19% | 31% | -0.03 |
| mom_cip_cool | -0.055% | -1.08 | -1.05 | 43% | 19% | -0.25 |
| mom_raw | -0.076% | -1.08 | -1.10 | 45% | 12% | -0.25 |
| mom_high_cip | +0.007% | +0.17 | +0.18 | 18% | 21% | +0.04 |
| cip_stress_haven_usd | +0.073% | +1.78 | +1.91 | 21% | 24% | +0.47 |
| mom_cip_stack | +0.009% | +0.28 | +0.30 | 46% | 14% | +0.07 |
| mom_cip_ew | +0.004% | +0.13 | +0.13 | 47% | 17% | +0.03 |
| mom_cip_regime | +0.068% | +1.13 | +1.10 | 39% | 16% | +0.29 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| mom_raw | year_2024 | -0.19% | 45% | PASS | no |
| mom_raw | year_2025 | +0.43% | 91% | PASS | no |
| mom_raw | year_2026 | +0.33% | 62% | PASS | no |
| mom_raw | holdout_365d | +0.37% | 75% | PASS | no |
| mom_cip_cool | year_2024 | -0.17% | 45% | PASS | no |
| mom_cip_cool | year_2025 | +0.43% | 91% | PASS | no |
| mom_cip_cool | year_2026 | +0.33% | 62% | PASS | no |
| mom_cip_cool | holdout_365d | +0.37% | 75% | PASS | no |
| mom_low_cip | year_2024 | -0.26% | 27% | PASS | no |
| mom_low_cip | year_2025 | +0.43% | 91% | PASS | no |
| mom_low_cip | year_2026 | +0.33% | 62% | PASS | no |
| mom_low_cip | holdout_365d | +0.37% | 75% | PASS | no |
| mom_high_cip | year_2024 | +0.00% | 0% | PASS | no |
| mom_high_cip | year_2025 | +0.00% | 0% | PASS | no |
| mom_high_cip | year_2026 | +0.00% | 0% | PASS | no |
| mom_high_cip | holdout_365d | +0.00% | 0% | PASS | no |
| cip_stress_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| cip_stress_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| cip_stress_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| cip_stress_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| mom_cip_stack | year_2024 | -0.09% | 45% | PASS | no |
| mom_cip_stack | year_2025 | +0.21% | 91% | PASS | no |
| mom_cip_stack | year_2026 | +0.17% | 62% | PASS | no |
| mom_cip_stack | holdout_365d | +0.19% | 75% | PASS | no |
| mom_cip_ew | year_2024 | -0.14% | 45% | PASS | no |
| mom_cip_ew | year_2025 | +0.29% | 91% | PASS | no |
| mom_cip_ew | year_2026 | +0.22% | 62% | PASS | no |
| mom_cip_ew | holdout_365d | +0.25% | 75% | PASS | no |
| mom_cip_regime | year_2024 | -0.26% | 27% | PASS | no |
| mom_cip_regime | year_2025 | +0.43% | 91% | PASS | no |
| mom_cip_regime | year_2026 | +0.33% | 62% | PASS | no |
| mom_cip_regime | holdout_365d | +0.37% | 75% | PASS | no |

**Board:** n=8 soft_nw_pos=1 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** CIP panel ends **2025-07-31** (cip_dataset_v4 vintage); 2026 FX window uses stale/ffilled CIP. Tenor=5y a priori (not HO-tuned). Gate z≤0 / cool z≥1 fixed (not HO-tuned). Monthly z_window=60m (not 252d daily).

Artifacts: `reports/scholarly_fx_cip_conditioned_mom_*.csv`, `scholarly_fx_cip_conditioned_mom_meta.json`, `quest_locked_verify_cip_conditioned_mom.md`.
