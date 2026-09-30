# Scholarly FX: STLFSI4-conditioned Menkhoff FX momentum wave (§102)

**Path:** St. Louis Fed **US STLFSI4** (FRED STLFSI4; redesigned Financial Stress Index) stress × Menkhoff (2012 JFE) FX momentum gate/cool — **distinct** from funding_liquidity (§20), NFCI mom (§92), ANFCI mom (§97), STLFSI soft (§100), STLFSI carry (§101), WUI mom (§87), CIP×mom (§82), VIX/GPR×mom (§74), EPU/TPU×mom (§79), soft–carry–value–REER stacks, capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED STLFSI4. G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** STLFSI4 pub_lag_days=7 → month-end + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 / NFCI soft §90 / NFCI mom §92 / ANFCI mom §97 / STLFSI soft §100 / STLFSI carry §101 mirror) + weight_lag_days=1; mom formation=63d skip=21d signal_lag=1d. Costs 1.5 bps/side.
**Primary:** `mom_low_stlfsi` (trade mom only when lagged US STLFSI4 z ≤ 0). Companions: mom_stlfsi_cool / mom_raw / mom_high_stlfsi / us_stlfsi_haven_usd / mom_stlfsi_stack / mom_stlfsi_ew / mom_stlfsi_regime. Cool z_high=1.0 (§71–§101 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| mom_low_stlfsi | -0.067% | -1.18 | -1.29 | 36% | 15% | -0.28 |
| mom_stlfsi_cool | -0.056% | -0.88 | -0.93 | 45% | 12% | -0.21 |
| mom_raw | -0.076% | -1.08 | -1.10 | 45% | 12% | -0.25 |
| mom_high_stlfsi | -0.037% | -1.90 | -1.70 | 3% | 90% | -0.43 |
| us_stlfsi_haven_usd | -0.021% | -1.19 | -1.09 | 4% | 79% | -0.31 |
| mom_stlfsi_stack | -0.067% | -1.18 | -1.29 | 36% | 15% | -0.28 |
| mom_stlfsi_ew | -0.048% | -1.22 | -1.28 | 44% | 13% | -0.28 |
| mom_stlfsi_regime | -0.089% | -1.49 | -1.56 | 38% | 14% | -0.35 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| mom_raw | year_2024 | -0.19% | 45% | PASS | no |
| mom_raw | year_2025 | +0.43% | 91% | PASS | no |
| mom_raw | year_2026 | +0.33% | 62% | PASS | no |
| mom_raw | holdout_365d | +0.37% | 75% | PASS | no |
| mom_stlfsi_cool | year_2024 | -0.19% | 45% | PASS | no |
| mom_stlfsi_cool | year_2025 | +0.38% | 91% | PASS | no |
| mom_stlfsi_cool | year_2026 | +0.25% | 62% | PASS | no |
| mom_stlfsi_cool | holdout_365d | +0.30% | 75% | PASS | no |
| mom_low_stlfsi | year_2024 | -0.19% | 45% | PASS | no |
| mom_low_stlfsi | year_2025 | +0.33% | 82% | PASS | no |
| mom_low_stlfsi | year_2026 | -0.02% | 25% | PASS | no |
| mom_low_stlfsi | holdout_365d | +0.11% | 50% | PASS | no |
| mom_high_stlfsi | year_2024 | +0.00% | 0% | PASS | no |
| mom_high_stlfsi | year_2025 | +0.00% | 0% | PASS | no |
| mom_high_stlfsi | year_2026 | -0.01% | 0% | PASS | no |
| mom_high_stlfsi | holdout_365d | +0.02% | 8% | PASS | no |
| us_stlfsi_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_stlfsi_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_stlfsi_haven_usd | year_2026 | +0.08% | 12% | PASS | no |
| us_stlfsi_haven_usd | holdout_365d | -0.09% | 8% | PASS | no |
| mom_stlfsi_stack | year_2024 | -0.19% | 45% | PASS | no |
| mom_stlfsi_stack | year_2025 | +0.33% | 82% | PASS | no |
| mom_stlfsi_stack | year_2026 | -0.02% | 25% | PASS | no |
| mom_stlfsi_stack | holdout_365d | +0.11% | 50% | PASS | no |
| mom_stlfsi_ew | year_2024 | -0.13% | 45% | PASS | no |
| mom_stlfsi_ew | year_2025 | +0.24% | 91% | PASS | no |
| mom_stlfsi_ew | year_2026 | +0.10% | 62% | PASS | no |
| mom_stlfsi_ew | holdout_365d | +0.11% | 67% | PASS | no |
| mom_stlfsi_regime | year_2024 | -0.19% | 45% | PASS | no |
| mom_stlfsi_regime | year_2025 | +0.33% | 82% | PASS | no |
| mom_stlfsi_regime | year_2026 | +0.06% | 38% | PASS | no |
| mom_stlfsi_regime | holdout_365d | +0.03% | 50% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US STLFSI4 [1994-01-07..2026-09-25]. Gate z≤0 / cool z≥1 fixed (§71–§101 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). Single-stress design (like CIP×mom §82 / WUI×mom §87 / NFCI×mom §92 / ANFCI×mom §97). Distinct from funding_liq §20 / NFCI mom §92 / ANFCI mom §97 / STLFSI soft §100 / STLFSI carry §101.

Artifacts: `reports/scholarly_fx_stlfsi_conditioned_mom_*.csv`, `scholarly_fx_stlfsi_conditioned_mom_meta.json`, `quest_locked_verify_stlfsi_conditioned_mom.md`.
