# Scholarly FX: KCFSI-conditioned Menkhoff FX momentum wave (§107)

**Path:** Kansas City Fed **US KCFSI** (FRED KCFSI; Financial Stress Index) stress × Menkhoff (2012 JFE) FX momentum gate/cool — **distinct** from funding_liquidity (§20), NFCI mom (§92), ANFCI mom (§97), STLFSI mom (§102), KCFSI soft (§105), KCFSI carry (§106), WUI mom (§87), CIP×mom (§82), VIX/GPR×mom (§74), EPU/TPU×mom (§79), soft–carry–value–REER stacks, capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED KCFSI. G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** KCFSI pub_lag_months=1 + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 / NFCI soft §90 / NFCI mom §92 / ANFCI mom §97 / STLFSI mom §102 / KCFSI soft §105 / KCFSI carry §106 mirror) + weight_lag_days=1; mom formation=63d skip=21d signal_lag=1d. Costs 1.5 bps/side.
**Primary:** `mom_low_kcfsi` (trade mom only when lagged US KCFSI z ≤ 0). Companions: mom_kcfsi_cool / mom_raw / mom_high_kcfsi / us_kcfsi_haven_usd / mom_kcfsi_stack / mom_kcfsi_ew / mom_kcfsi_regime. Cool z_high=1.0 (§71–§106 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| mom_low_kcfsi | -0.059% | -1.03 | -1.08 | 34% | 13% | -0.24 |
| mom_kcfsi_cool | -0.093% | -1.45 | -1.43 | 45% | 10% | -0.34 |
| mom_raw | -0.076% | -1.08 | -1.10 | 45% | 12% | -0.25 |
| mom_high_kcfsi | -0.005% | -0.33 | -0.29 | 4% | 70% | -0.07 |
| us_kcfsi_haven_usd | -0.012% | -0.75 | -0.80 | 3% | 88% | -0.19 |
| mom_kcfsi_stack | -0.059% | -1.03 | -1.08 | 34% | 13% | -0.24 |
| mom_kcfsi_ew | -0.055% | -1.38 | -1.37 | 43% | 11% | -0.33 |
| mom_kcfsi_regime | -0.071% | -1.20 | -1.25 | 37% | 12% | -0.29 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| mom_raw | year_2024 | -0.19% | 45% | PASS | no |
| mom_raw | year_2025 | +0.43% | 91% | PASS | no |
| mom_raw | year_2026 | +0.33% | 62% | PASS | no |
| mom_raw | holdout_365d | +0.37% | 75% | PASS | no |
| mom_kcfsi_cool | year_2024 | -0.18% | 45% | PASS | no |
| mom_kcfsi_cool | year_2025 | +0.42% | 91% | PASS | no |
| mom_kcfsi_cool | year_2026 | +0.33% | 62% | PASS | no |
| mom_kcfsi_cool | holdout_365d | +0.37% | 75% | PASS | no |
| mom_low_kcfsi | year_2024 | -0.16% | 45% | PASS | no |
| mom_low_kcfsi | year_2025 | +0.42% | 91% | PASS | no |
| mom_low_kcfsi | year_2026 | +0.33% | 62% | PASS | no |
| mom_low_kcfsi | holdout_365d | +0.37% | 75% | PASS | no |
| mom_high_kcfsi | year_2024 | +0.00% | 0% | PASS | no |
| mom_high_kcfsi | year_2025 | +0.00% | 0% | PASS | no |
| mom_high_kcfsi | year_2026 | +0.00% | 0% | PASS | no |
| mom_high_kcfsi | holdout_365d | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| mom_kcfsi_stack | year_2024 | -0.16% | 45% | PASS | no |
| mom_kcfsi_stack | year_2025 | +0.42% | 91% | PASS | no |
| mom_kcfsi_stack | year_2026 | +0.33% | 62% | PASS | no |
| mom_kcfsi_stack | holdout_365d | +0.37% | 75% | PASS | no |
| mom_kcfsi_ew | year_2024 | -0.12% | 45% | PASS | no |
| mom_kcfsi_ew | year_2025 | +0.28% | 91% | PASS | no |
| mom_kcfsi_ew | year_2026 | +0.22% | 62% | PASS | no |
| mom_kcfsi_ew | holdout_365d | +0.25% | 75% | PASS | no |
| mom_kcfsi_regime | year_2024 | -0.16% | 45% | PASS | no |
| mom_kcfsi_regime | year_2025 | +0.42% | 91% | PASS | no |
| mom_kcfsi_regime | year_2026 | +0.33% | 62% | PASS | no |
| mom_kcfsi_regime | holdout_365d | +0.37% | 75% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US KCFSI [1990-03-01..2026-09-01]. Gate z≤0 / cool z≥1 fixed (§71–§106 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). Single-stress design (like CIP×mom §82 / WUI×mom §87 / NFCI×mom §92 / ANFCI×mom §97). Distinct from funding_liq §20 / NFCI mom §92 / ANFCI mom §97 / STLFSI mom §102 / KCFSI soft §105 / KCFSI carry §106.

Artifacts: `reports/scholarly_fx_kcfsi_conditioned_mom_*.csv`, `scholarly_fx_kcfsi_conditioned_mom_meta.json`, `quest_locked_verify_kcfsi_conditioned_mom.md`.
