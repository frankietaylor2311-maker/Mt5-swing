# Scholarly FX: NFCI-conditioned Menkhoff FX momentum wave (§92)

**Path:** Chicago Fed **US NFCI** (FRED NFCI) stress × Menkhoff (2012 JFE) FX momentum gate/cool — **distinct** from funding_liquidity (§20), NFCI soft (§90), NFCI carry (§91), WUI mom (§87), CIP×mom (§82), VIX/GPR×mom (§74), EPU/TPU×mom (§79), soft–carry–value–REER stacks, capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED NFCI. G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** NFCI pub_lag_days=7 → month-end + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 / NFCI soft §90 / NFCI carry §91 mirror) + weight_lag_days=1; mom formation=63d skip=21d signal_lag=1d. Costs 1.5 bps/side.
**Primary:** `mom_low_nfci` (trade mom only when lagged US NFCI z ≤ 0). Companions: mom_nfci_cool / mom_raw / mom_high_nfci / us_nfci_haven_usd / mom_nfci_stack / mom_nfci_ew / mom_nfci_regime. Cool z_high=1.0 (§71–§91 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| mom_low_nfci | -0.042% | -0.75 | -0.79 | 33% | 16% | -0.18 |
| mom_nfci_cool | -0.058% | -0.96 | -0.99 | 44% | 12% | -0.23 |
| mom_raw | -0.076% | -1.08 | -1.10 | 45% | 12% | -0.25 |
| mom_high_nfci | -0.016% | -0.64 | -0.60 | 7% | 47% | -0.14 |
| us_nfci_haven_usd | -0.008% | -0.26 | -0.25 | 6% | 47% | -0.07 |
| mom_nfci_stack | -0.042% | -0.75 | -0.79 | 33% | 16% | -0.18 |
| mom_nfci_ew | -0.036% | -0.91 | -0.94 | 43% | 12% | -0.22 |
| mom_nfci_regime | -0.050% | -0.78 | -0.80 | 38% | 12% | -0.19 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| mom_raw | year_2024 | -0.19% | 45% | PASS | no |
| mom_raw | year_2025 | +0.43% | 91% | PASS | no |
| mom_raw | year_2026 | +0.33% | 62% | PASS | no |
| mom_raw | holdout_365d | +0.37% | 75% | PASS | no |
| mom_nfci_cool | year_2024 | -0.17% | 45% | PASS | no |
| mom_nfci_cool | year_2025 | +0.42% | 91% | PASS | no |
| mom_nfci_cool | year_2026 | +0.33% | 62% | PASS | no |
| mom_nfci_cool | holdout_365d | +0.37% | 75% | PASS | no |
| mom_low_nfci | year_2024 | -0.22% | 36% | PASS | no |
| mom_low_nfci | year_2025 | +0.33% | 82% | PASS | no |
| mom_low_nfci | year_2026 | +0.33% | 62% | PASS | no |
| mom_low_nfci | holdout_365d | +0.37% | 75% | PASS | no |
| mom_high_nfci | year_2024 | +0.00% | 0% | PASS | no |
| mom_high_nfci | year_2025 | +0.00% | 0% | PASS | no |
| mom_high_nfci | year_2026 | +0.00% | 0% | PASS | no |
| mom_high_nfci | holdout_365d | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| mom_nfci_stack | year_2024 | -0.22% | 36% | PASS | no |
| mom_nfci_stack | year_2025 | +0.33% | 82% | PASS | no |
| mom_nfci_stack | year_2026 | +0.33% | 62% | PASS | no |
| mom_nfci_stack | holdout_365d | +0.37% | 75% | PASS | no |
| mom_nfci_ew | year_2024 | -0.13% | 36% | PASS | no |
| mom_nfci_ew | year_2025 | +0.25% | 91% | PASS | no |
| mom_nfci_ew | year_2026 | +0.22% | 62% | PASS | no |
| mom_nfci_ew | holdout_365d | +0.25% | 75% | PASS | no |
| mom_nfci_regime | year_2024 | -0.22% | 36% | PASS | no |
| mom_nfci_regime | year_2025 | +0.33% | 82% | PASS | no |
| mom_nfci_regime | year_2026 | +0.33% | 62% | PASS | no |
| mom_nfci_regime | holdout_365d | +0.37% | 75% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US NFCI [1971-01-15..2026-09-18]. Gate z≤0 / cool z≥1 fixed (§71–§91 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). Single-stress design (like CIP×mom §82 / WUI×mom §87 / NFCI×carry §91). Distinct from funding_liq §20 / NFCI soft §90 / NFCI carry §91.

Artifacts: `reports/scholarly_fx_nfci_conditioned_mom_*.csv`, `scholarly_fx_nfci_conditioned_mom_meta.json`, `quest_locked_verify_nfci_conditioned_mom.md`.
