# Scholarly FX: CIP-conditioned BIS REER HML-FX value wave (§84)

**Path:** Du–Schreger **government-bond CIP** stress (−mean `cip_govt` bps, tenor=**5y**) × BIS multilateral REER undervaluation / HML-FX value (`reer_cheap_xs` §31) gate/cool — **distinct** from raw bis_reer (§31), raw PPP, VIX/GPR×PPP (§75), EPU×PPP (§80), CIP×PPP (§83), EPU×REER (§77), VIX×REER (§81), CIP soft (§71), CIP×carry (§68), CIP×mom (§82), soft/carry/value/REER stress (§72–§81), capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + public `cip_dataset_v4.csv` after pub_lag_days=1 + FRED BIS REER (pub_lag=2m) + Yahoo D1 G10. **PIT:** CIP pub_lag_days=1 + signal_lag_months=1 on trailing **monthly** z (z_window=60m, §68/§71/§82/§83 mirror) + weight_lag_days=1; REER pub_lag=2m signal_lag=1m. Costs 1.5 bps/side.
**Primary:** `reer_low_cip` (trade REER-value only when lagged CIP-stress z ≤ 0). Companions: reer_cip_cool / reer_raw / reer_high_cip / cip_stress_haven_usd / reer_cip_stack / reer_cip_ew / reer_cip_regime. Cool z_high=1.0 (§68/§71/§82/§83 symmetry). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| reer_low_cip | +0.078% | +2.19 | +1.93 | 21% | 30% | +0.52 |
| reer_cip_cool | +0.046% | +1.00 | +1.11 | 53% | 17% | +0.26 |
| reer_raw | +0.034% | +0.51 | +0.60 | 52% | 13% | +0.15 |
| reer_high_cip | -0.012% | -0.32 | -0.34 | 19% | 22% | -0.08 |
| cip_stress_haven_usd | +0.076% | +1.86 | +1.94 | 22% | 24% | +0.50 |
| reer_cip_stack | +0.061% | +2.03 | +2.21 | 56% | 12% | +0.50 |
| reer_cip_ew | +0.066% | +2.33 | +2.32 | 56% | 17% | +0.57 |
| reer_cip_regime | +0.154% | +2.87 | +2.70 | 42% | 15% | +0.72 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| reer_raw | year_2024 | -0.11% | 36% | PASS | no |
| reer_raw | year_2025 | -0.27% | 27% | PASS | no |
| reer_raw | year_2026 | +0.16% | 62% | PASS | no |
| reer_raw | holdout_365d | +0.08% | 58% | PASS | no |
| reer_cip_cool | year_2024 | -0.09% | 36% | PASS | no |
| reer_cip_cool | year_2025 | -0.27% | 27% | PASS | no |
| reer_cip_cool | year_2026 | +0.16% | 62% | PASS | no |
| reer_cip_cool | holdout_365d | +0.08% | 58% | PASS | no |
| reer_low_cip | year_2024 | +0.08% | 36% | PASS | no |
| reer_low_cip | year_2025 | -0.27% | 27% | PASS | no |
| reer_low_cip | year_2026 | +0.16% | 62% | PASS | no |
| reer_low_cip | holdout_365d | +0.08% | 58% | PASS | no |
| reer_high_cip | year_2024 | +0.00% | 0% | PASS | no |
| reer_high_cip | year_2025 | +0.00% | 0% | PASS | no |
| reer_high_cip | year_2026 | +0.00% | 0% | PASS | no |
| reer_high_cip | holdout_365d | +0.00% | 0% | PASS | no |
| cip_stress_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| cip_stress_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| cip_stress_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| cip_stress_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| reer_cip_stack | year_2024 | -0.05% | 36% | PASS | no |
| reer_cip_stack | year_2025 | -0.13% | 27% | PASS | no |
| reer_cip_stack | year_2026 | +0.08% | 62% | PASS | no |
| reer_cip_stack | holdout_365d | +0.04% | 58% | PASS | no |
| reer_cip_ew | year_2024 | -0.01% | 45% | PASS | no |
| reer_cip_ew | year_2025 | -0.18% | 27% | PASS | no |
| reer_cip_ew | year_2026 | +0.11% | 62% | PASS | no |
| reer_cip_ew | holdout_365d | +0.06% | 58% | PASS | no |
| reer_cip_regime | year_2024 | +0.08% | 36% | PASS | no |
| reer_cip_regime | year_2025 | -0.27% | 27% | PASS | no |
| reer_cip_regime | year_2026 | +0.16% | 62% | PASS | no |
| reer_cip_regime | holdout_365d | +0.08% | 58% | PASS | no |

**Board:** n=8 soft_nw_pos=5 hard_nw_pos=3 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** CIP panel ends **2025-07-31** (cip_dataset_v4 vintage); 2026 FX window uses stale/ffilled CIP. Tenor=5y a priori (not HO-tuned). Gate z≤0 / cool z≥1 fixed (not HO-tuned). Monthly z_window=60m (not 252d daily).

Artifacts: `reports/scholarly_fx_cip_conditioned_reer_*.csv`, `scholarly_fx_cip_conditioned_reer_meta.json`, `quest_locked_verify_cip_conditioned_reer.md`.
