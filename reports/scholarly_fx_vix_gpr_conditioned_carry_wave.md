# Scholarly FX: VIX/GPR-conditioned Lustig–Verdelhan carry wave (§73)

**Path:** Menkhoff **VIX** (global FX-vol proxy) + Caldara–Iacoviello **aggregate GPR** stress × Lustig–Verdelhan IR3M carry gate/cool — **distinct** from fwd_carry (§19), funding_liq (§20), IG OAS (§36), ACM TP (§44), CIP XS (§67), CIP×carry (§68), soft EW (§66), soft CIP (§69), capital-sleeve (§53/§70), CIP-conditioned soft (§71), VIX/GPR-conditioned soft (§72), combo (§8), gpr_regime / AI-GPR (§25) / country-GPR (§45).
**Data:** `approximate_non_ftmo` + `vix_yahoo.csv` + `gpr_daily.csv` (bar_lag=1) + FRED IR3M rates (pub_lag_months=1). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. `n_long=n_short=2`. **PIT:** VIX/GPR bar_lag=1 + signal_lag=1d on trailing z (252d); carry signal_lag=1d.
**Primary:** `carry_low_vix` (trade carry only when lagged VIX z ≤ 0). Honesty `carry_raw` + `carry_high_vix`. Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| carry_raw | -0.002% | -0.02 | -0.03 | 51% | 10% | -0.04 |
| carry_vix_cool | -0.035% | -0.53 | -0.57 | 48% | 13% | -0.16 |
| carry_gpr_cool | -0.038% | -0.57 | -0.67 | 53% | 13% | -0.17 |
| carry_low_vix | -0.091% | -1.63 | -1.59 | 43% | 15% | -0.37 |
| carry_low_gpr | -0.086% | -1.52 | -1.70 | 49% | 12% | -0.35 |
| carry_high_vix | +0.003% | +0.09 | +0.10 | 19% | 22% | +0.02 |
| carry_vix_gpr_stack | -0.055% | -0.96 | -1.06 | 49% | 13% | -0.27 |
| carry_vix_gpr_ew | -0.062% | -1.13 | -1.22 | 49% | 11% | -0.28 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| carry_raw | year_2024 | +0.15% | 64% | PASS | no |
| carry_raw | year_2025 | +0.12% | 64% | PASS | no |
| carry_raw | year_2026 | +0.13% | 62% | PASS | no |
| carry_raw | holdout_365d | +0.31% | 75% | PASS | no |
| carry_vix_cool | year_2024 | +0.11% | 64% | PASS | no |
| carry_vix_cool | year_2025 | +0.14% | 55% | PASS | no |
| carry_vix_cool | year_2026 | +0.08% | 75% | PASS | no |
| carry_vix_cool | holdout_365d | +0.25% | 83% | PASS | no |
| carry_gpr_cool | year_2024 | +0.08% | 73% | PASS | no |
| carry_gpr_cool | year_2025 | +0.05% | 64% | PASS | no |
| carry_gpr_cool | year_2026 | -0.01% | 75% | PASS | no |
| carry_gpr_cool | holdout_365d | +0.14% | 83% | PASS | no |
| carry_low_vix | year_2024 | +0.12% | 55% | PASS | no |
| carry_low_vix | year_2025 | +0.12% | 55% | PASS | no |
| carry_low_vix | year_2026 | +0.06% | 62% | PASS | no |
| carry_low_vix | holdout_365d | +0.19% | 75% | PASS | no |
| carry_low_gpr | year_2024 | -0.19% | 55% | PASS | no |
| carry_low_gpr | year_2025 | -0.06% | 55% | PASS | no |
| carry_low_gpr | year_2026 | -0.12% | 25% | PASS | no |
| carry_low_gpr | holdout_365d | -0.07% | 25% | PASS | no |
| carry_high_vix | year_2024 | -0.05% | 27% | PASS | no |
| carry_high_vix | year_2025 | -0.09% | 18% | PASS | no |
| carry_high_vix | year_2026 | +0.04% | 12% | PASS | no |
| carry_high_vix | holdout_365d | +0.03% | 17% | PASS | no |
| carry_vix_gpr_stack | year_2024 | +0.06% | 73% | PASS | no |
| carry_vix_gpr_stack | year_2025 | +0.10% | 55% | PASS | no |
| carry_vix_gpr_stack | year_2026 | -0.01% | 62% | PASS | no |
| carry_vix_gpr_stack | holdout_365d | +0.12% | 75% | PASS | no |
| carry_vix_gpr_ew | year_2024 | +0.03% | 64% | PASS | no |
| carry_vix_gpr_ew | year_2025 | +0.06% | 64% | PASS | no |
| carry_vix_gpr_ew | year_2026 | +0.00% | 62% | PASS | no |
| carry_vix_gpr_ew | holdout_365d | +0.13% | 75% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** VIX ends **2026-09-16**; GPR ends **2026-09-14**. Gate z≤0 / cool z≥1 fixed (§71/§72 symmetry; lit GPR often 1.5 — documented, not HO-tuned). z_window=252 daily (NOT monthly CIP 60).

Artifacts: `reports/scholarly_fx_vix_gpr_conditioned_carry_*.csv`, `scholarly_fx_vix_gpr_conditioned_carry_meta.json`, `quest_locked_verify_vix_gpr_conditioned_carry.md`.
