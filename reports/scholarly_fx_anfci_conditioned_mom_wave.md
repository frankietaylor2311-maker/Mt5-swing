# Scholarly FX: ANFCI-conditioned Menkhoff FX momentum wave (§97)

**Path:** Chicago Fed **US ANFCI** (FRED ANFCI) stress × Menkhoff (2012 JFE) FX momentum gate/cool — **distinct** from funding_liquidity (§20), NFCI mom (§92), ANFCI soft (§95), ANFCI carry (§96), NFCI soft (§90), WUI mom (§87), CIP×mom (§82), VIX/GPR×mom (§74), EPU/TPU×mom (§79), soft–carry–value–REER stacks, capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED ANFCI. G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** ANFCI pub_lag_days=7 → month-end + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 / NFCI soft §90 / NFCI mom §92 / ANFCI soft §95 / ANFCI carry §96 mirror) + weight_lag_days=1; mom formation=63d skip=21d signal_lag=1d. Costs 1.5 bps/side.
**Primary:** `mom_low_anfci` (trade mom only when lagged US ANFCI z ≤ 0). Companions: mom_anfci_cool / mom_raw / mom_high_anfci / us_anfci_haven_usd / mom_anfci_stack / mom_anfci_ew / mom_anfci_regime. Cool z_high=1.0 (§71–§96 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| mom_low_anfci | -0.028% | -0.50 | -0.52 | 34% | 15% | -0.12 |
| mom_anfci_cool | -0.061% | -0.98 | -1.02 | 45% | 12% | -0.23 |
| mom_raw | -0.076% | -1.08 | -1.10 | 45% | 12% | -0.25 |
| mom_high_anfci | -0.011% | -0.44 | -0.54 | 7% | 46% | -0.09 |
| us_anfci_haven_usd | -0.003% | -0.10 | -0.09 | 6% | 43% | -0.02 |
| mom_anfci_stack | -0.028% | -0.50 | -0.52 | 34% | 15% | -0.12 |
| mom_anfci_ew | -0.031% | -0.76 | -0.78 | 46% | 12% | -0.18 |
| mom_anfci_regime | -0.031% | -0.48 | -0.48 | 39% | 12% | -0.12 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| mom_raw | year_2024 | -0.19% | 45% | PASS | no |
| mom_raw | year_2025 | +0.43% | 91% | PASS | no |
| mom_raw | year_2026 | +0.33% | 62% | PASS | no |
| mom_raw | holdout_365d | +0.37% | 75% | PASS | no |
| mom_anfci_cool | year_2024 | -0.18% | 45% | PASS | no |
| mom_anfci_cool | year_2025 | +0.42% | 91% | PASS | no |
| mom_anfci_cool | year_2026 | +0.33% | 62% | PASS | no |
| mom_anfci_cool | holdout_365d | +0.37% | 75% | PASS | no |
| mom_low_anfci | year_2024 | -0.06% | 64% | PASS | no |
| mom_low_anfci | year_2025 | +0.33% | 82% | PASS | no |
| mom_low_anfci | year_2026 | +0.33% | 62% | PASS | no |
| mom_low_anfci | holdout_365d | +0.37% | 75% | PASS | no |
| mom_high_anfci | year_2024 | +0.00% | 0% | PASS | no |
| mom_high_anfci | year_2025 | +0.00% | 0% | PASS | no |
| mom_high_anfci | year_2026 | +0.00% | 0% | PASS | no |
| mom_high_anfci | holdout_365d | +0.00% | 0% | PASS | no |
| us_anfci_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_anfci_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_anfci_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_anfci_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| mom_anfci_stack | year_2024 | -0.06% | 64% | PASS | no |
| mom_anfci_stack | year_2025 | +0.33% | 82% | PASS | no |
| mom_anfci_stack | year_2026 | +0.33% | 62% | PASS | no |
| mom_anfci_stack | holdout_365d | +0.37% | 75% | PASS | no |
| mom_anfci_ew | year_2024 | -0.08% | 55% | PASS | no |
| mom_anfci_ew | year_2025 | +0.25% | 91% | PASS | no |
| mom_anfci_ew | year_2026 | +0.22% | 62% | PASS | no |
| mom_anfci_ew | holdout_365d | +0.25% | 75% | PASS | no |
| mom_anfci_regime | year_2024 | -0.06% | 64% | PASS | no |
| mom_anfci_regime | year_2025 | +0.33% | 82% | PASS | no |
| mom_anfci_regime | year_2026 | +0.33% | 62% | PASS | no |
| mom_anfci_regime | holdout_365d | +0.37% | 75% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US ANFCI [1971-01-15..2026-09-18]. Gate z≤0 / cool z≥1 fixed (§71–§96 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). Single-stress design (like CIP×mom §82 / WUI×mom §87 / NFCI×mom §92). Distinct from funding_liq §20 / NFCI mom §92 / ANFCI soft §95 / ANFCI carry §96.

Artifacts: `reports/scholarly_fx_anfci_conditioned_mom_*.csv`, `scholarly_fx_anfci_conditioned_mom_meta.json`, `quest_locked_verify_anfci_conditioned_mom.md`.
