# Scholarly FX: CIP-conditioned Rogoff PPP / real-FX value wave (§83)

**Path:** Du–Schreger **government-bond CIP** stress (−mean `cip_govt` bps, tenor=**5y**) × Rogoff (1996) PPP / real-FX value gate/cool — **distinct** from raw PPP, BIS REER (§31), VIX/GPR×PPP (§75), EPU×PPP (§80), EPU×REER (§77), VIX×REER (§81), CIP soft (§71), CIP×carry (§68), CIP×mom (§82), soft/carry/value/REER stress (§72–§81), capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + public `cip_dataset_v4.csv` after pub_lag_days=1 + FRED CPI levels (pub_lag=1m) + Yahoo D1 G10. **PIT:** CIP pub_lag_days=1 + signal_lag_months=1 on trailing **monthly** z (z_window=60m, §68/§71/§82 mirror) + weight_lag_days=1; PPP lookback=60m signal_lag=1m. Costs 1.5 bps/side.
**Primary:** `value_low_cip` (trade value only when lagged CIP-stress z ≤ 0). Companions: value_cip_cool / value_raw / value_high_cip / cip_stress_haven_usd / value_cip_stack / value_cip_ew / value_cip_regime. Cool z_high=1.0 (§68/§71/§82 symmetry). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| value_low_cip | +0.024% | +0.67 | +0.54 | 18% | 29% | +0.15 |
| value_cip_cool | +0.022% | +0.52 | +0.49 | 41% | 17% | +0.11 |
| value_raw | -0.014% | -0.22 | -0.22 | 42% | 12% | -0.05 |
| value_high_cip | -0.033% | -0.95 | -0.97 | 13% | 29% | -0.24 |
| cip_stress_haven_usd | +0.073% | +1.78 | +1.91 | 21% | 24% | +0.47 |
| value_cip_stack | +0.048% | +1.74 | +1.71 | 47% | 13% | +0.39 |
| value_cip_ew | +0.040% | +1.45 | +1.29 | 46% | 16% | +0.32 |
| value_cip_regime | +0.096% | +1.80 | +1.65 | 39% | 15% | +0.43 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| value_raw | year_2024 | -0.10% | 64% | PASS | no |
| value_raw | year_2025 | -0.30% | 27% | PASS | no |
| value_raw | year_2026 | +0.15% | 75% | PASS | no |
| value_raw | holdout_365d | +0.10% | 67% | PASS | no |
| value_cip_cool | year_2024 | -0.12% | 64% | PASS | no |
| value_cip_cool | year_2025 | -0.30% | 27% | PASS | no |
| value_cip_cool | year_2026 | +0.15% | 75% | PASS | no |
| value_cip_cool | holdout_365d | +0.10% | 67% | PASS | no |
| value_low_cip | year_2024 | -0.21% | 27% | PASS | no |
| value_low_cip | year_2025 | -0.30% | 27% | PASS | no |
| value_low_cip | year_2026 | +0.15% | 75% | PASS | no |
| value_low_cip | holdout_365d | +0.10% | 67% | PASS | no |
| value_high_cip | year_2024 | +0.00% | 0% | PASS | no |
| value_high_cip | year_2025 | +0.00% | 0% | PASS | no |
| value_high_cip | year_2026 | +0.00% | 0% | PASS | no |
| value_high_cip | holdout_365d | +0.00% | 0% | PASS | no |
| cip_stress_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| cip_stress_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| cip_stress_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| cip_stress_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| value_cip_stack | year_2024 | -0.06% | 64% | PASS | no |
| value_cip_stack | year_2025 | -0.15% | 27% | PASS | no |
| value_cip_stack | year_2026 | +0.07% | 75% | PASS | no |
| value_cip_stack | holdout_365d | +0.05% | 67% | PASS | no |
| value_cip_ew | year_2024 | -0.11% | 64% | PASS | no |
| value_cip_ew | year_2025 | -0.20% | 27% | PASS | no |
| value_cip_ew | year_2026 | +0.10% | 75% | PASS | no |
| value_cip_ew | holdout_365d | +0.06% | 67% | PASS | no |
| value_cip_regime | year_2024 | -0.21% | 27% | PASS | no |
| value_cip_regime | year_2025 | -0.30% | 27% | PASS | no |
| value_cip_regime | year_2026 | +0.15% | 75% | PASS | no |
| value_cip_regime | holdout_365d | +0.10% | 67% | PASS | no |

**Board:** n=8 soft_nw_pos=3 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** CIP panel ends **2025-07-31** (cip_dataset_v4 vintage); 2026 FX window uses stale/ffilled CIP. Tenor=5y a priori (not HO-tuned). Gate z≤0 / cool z≥1 fixed (not HO-tuned). Monthly z_window=60m (not 252d daily).

Artifacts: `reports/scholarly_fx_cip_conditioned_value_*.csv`, `scholarly_fx_cip_conditioned_value_meta.json`, `quest_locked_verify_cip_conditioned_value.md`.
