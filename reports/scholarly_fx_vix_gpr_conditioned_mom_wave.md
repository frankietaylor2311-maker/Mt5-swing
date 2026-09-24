# Scholarly FX: VIX/GPR-conditioned Menkhoff FX momentum wave (§74)

**Path:** Menkhoff **VIX** (global FX-vol proxy) + Caldara–Iacoviello **aggregate GPR** stress × Menkhoff–Sarno–Schmeling–Schrimpf (2012 JFE) cross-sectional FX momentum gate/cool — **distinct** from combo (§8), raw fx_momentum, soft EW (§66), soft CIP (§69), CIP XS (§67), CIP×carry (§68), CIP-conditioned soft (§71), VIX/GPR-conditioned soft (§72), VIX/GPR-conditioned carry (§73), capital-sleeve (§53/§70), funding_liq (§20), gpr_regime / AI-GPR (§25) / country-GPR (§45).
**Data:** `approximate_non_ftmo` + `vix_yahoo.csv` + `gpr_daily.csv` (bar_lag=1). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. `formation=63d` `skip=21d` `n_long=n_short=2`. **PIT:** VIX/GPR bar_lag=1 + signal_lag=1d on trailing z (252d); mom signal_lag=1d.
**Primary:** `mom_low_vix` (trade mom only when lagged VIX z ≤ 0). Honesty `mom_raw` + `mom_high_vix`. Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| mom_raw | -0.076% | -1.08 | -1.10 | 45% | 12% | -0.25 |
| mom_vix_cool | -0.035% | -0.58 | -0.60 | 47% | 14% | -0.14 |
| mom_gpr_cool | -0.120% | -1.99 | -1.85 | 43% | 14% | -0.46 |
| mom_low_vix | -0.017% | -0.33 | -0.30 | 42% | 14% | -0.08 |
| mom_low_gpr | -0.170% | -3.25 | -2.71 | 42% | 16% | -0.76 |
| mom_high_vix | -0.043% | -1.44 | -1.59 | 17% | 26% | -0.30 |
| mom_vix_gpr_stack | -0.068% | -1.29 | -1.25 | 46% | 14% | -0.31 |
| mom_vix_gpr_ew | -0.085% | -1.67 | -1.59 | 45% | 14% | -0.40 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| mom_raw | year_2024 | -0.19% | 45% | PASS | no |
| mom_raw | year_2025 | +0.43% | 91% | PASS | no |
| mom_raw | year_2026 | +0.33% | 62% | PASS | no |
| mom_raw | holdout_365d | +0.37% | 75% | PASS | no |
| mom_vix_cool | year_2024 | -0.11% | 45% | PASS | no |
| mom_vix_cool | year_2025 | +0.38% | 91% | PASS | no |
| mom_vix_cool | year_2026 | +0.29% | 75% | PASS | no |
| mom_vix_cool | holdout_365d | +0.33% | 75% | PASS | no |
| mom_gpr_cool | year_2024 | -0.35% | 27% | PASS | no |
| mom_gpr_cool | year_2025 | +0.32% | 91% | PASS | no |
| mom_gpr_cool | year_2026 | +0.27% | 75% | PASS | no |
| mom_gpr_cool | holdout_365d | +0.30% | 83% | PASS | no |
| mom_low_vix | year_2024 | -0.07% | 36% | PASS | no |
| mom_low_vix | year_2025 | +0.27% | 64% | PASS | no |
| mom_low_vix | year_2026 | +0.30% | 88% | PASS | no |
| mom_low_vix | holdout_365d | +0.31% | 83% | PASS | no |
| mom_low_gpr | year_2024 | -0.52% | 27% | PASS | no |
| mom_low_gpr | year_2025 | +0.18% | 73% | PASS | no |
| mom_low_gpr | year_2026 | +0.13% | 62% | PASS | no |
| mom_low_gpr | holdout_365d | +0.18% | 67% | PASS | no |
| mom_high_vix | year_2024 | -0.13% | 27% | PASS | no |
| mom_high_vix | year_2025 | -0.00% | 27% | PASS | no |
| mom_high_vix | year_2026 | +0.01% | 25% | PASS | no |
| mom_high_vix | holdout_365d | +0.00% | 25% | PASS | no |
| mom_vix_gpr_stack | year_2024 | -0.20% | 36% | PASS | no |
| mom_vix_gpr_stack | year_2025 | +0.30% | 91% | PASS | no |
| mom_vix_gpr_stack | year_2026 | +0.25% | 88% | PASS | no |
| mom_vix_gpr_stack | holdout_365d | +0.28% | 92% | PASS | no |
| mom_vix_gpr_ew | year_2024 | -0.26% | 36% | PASS | no |
| mom_vix_gpr_ew | year_2025 | +0.29% | 91% | PASS | no |
| mom_vix_gpr_ew | year_2026 | +0.25% | 88% | PASS | no |
| mom_vix_gpr_ew | holdout_365d | +0.28% | 92% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** VIX ends **2026-09-16**; GPR ends **2026-09-14**. Gate z≤0 / cool z≥1 fixed (§71/§72/§73 symmetry; lit GPR often 1.5 — documented, not HO-tuned). z_window=252 daily (NOT monthly CIP 60). Menkhoff formation=63d skip=21d (fx_momentum defaults).

Artifacts: `reports/scholarly_fx_vix_gpr_conditioned_mom_*.csv`, `scholarly_fx_vix_gpr_conditioned_mom_meta.json`, `quest_locked_verify_vix_gpr_conditioned_mom.md`.
