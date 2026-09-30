# Scholarly FX: ANFCI-conditioned Lustig–Verdelhan carry wave (§96)

**Path:** Chicago Fed **US ANFCI** (FRED ANFCI; Adjusted NFCI) stress × Lustig–Verdelhan IR3M carry gate/cool — **distinct** from funding_liquidity (§20), NFCI×carry (§91), ANFCI soft (§95), NFCI soft (§90), WUI carry (§86), soft_ew (§66), CIP×carry (§68), VIX/GPR×carry (§73), EPU/TPU×carry (§78), CIP/VIX/EPU/WUI soft/mom/value/REER (§71–§77/§79–§89), capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED ANFCI + FRED IR3M rates (pub_lag=1m). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** ANFCI pub_lag_days=7 → month-end + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 / NFCI soft §90 / NFCI carry §91 / ANFCI soft §95 mirror) + weight_lag_days=1; carry signal_lag=1d. Costs 1.5 bps/side.
**Primary:** `carry_low_anfci` (trade carry only when lagged US ANFCI z ≤ 0). Companions: carry_anfci_cool / carry_raw / carry_high_anfci / us_anfci_haven_usd / carry_anfci_stack / carry_anfci_ew / carry_anfci_regime. Cool z_high=1.0 (§71–§95 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| carry_low_anfci | -0.012% | -0.19 | -0.20 | 34% | 16% | -0.09 |
| carry_anfci_cool | -0.003% | -0.04 | -0.05 | 51% | 12% | -0.05 |
| carry_raw | -0.002% | -0.02 | -0.03 | 51% | 10% | -0.04 |
| carry_high_anfci | -0.002% | -0.10 | -0.10 | 8% | 37% | -0.01 |
| us_anfci_haven_usd | -0.003% | -0.10 | -0.09 | 6% | 43% | -0.02 |
| carry_anfci_stack | -0.012% | -0.19 | -0.20 | 34% | 16% | -0.09 |
| carry_anfci_ew | -0.006% | -0.13 | -0.14 | 51% | 13% | -0.08 |
| carry_anfci_regime | -0.015% | -0.21 | -0.22 | 40% | 13% | -0.10 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| carry_raw | year_2024 | +0.15% | 64% | PASS | no |
| carry_raw | year_2025 | +0.12% | 64% | PASS | no |
| carry_raw | year_2026 | +0.13% | 62% | PASS | no |
| carry_raw | holdout_365d | +0.31% | 75% | PASS | no |
| carry_anfci_cool | year_2024 | +0.14% | 64% | PASS | no |
| carry_anfci_cool | year_2025 | +0.12% | 64% | PASS | no |
| carry_anfci_cool | year_2026 | +0.13% | 62% | PASS | no |
| carry_anfci_cool | holdout_365d | +0.31% | 75% | PASS | no |
| carry_low_anfci | year_2024 | +0.02% | 55% | PASS | no |
| carry_low_anfci | year_2025 | +0.13% | 55% | PASS | no |
| carry_low_anfci | year_2026 | +0.13% | 62% | PASS | no |
| carry_low_anfci | holdout_365d | +0.31% | 75% | PASS | no |
| carry_high_anfci | year_2024 | +0.00% | 0% | PASS | no |
| carry_high_anfci | year_2025 | +0.00% | 0% | PASS | no |
| carry_high_anfci | year_2026 | +0.00% | 0% | PASS | no |
| carry_high_anfci | holdout_365d | +0.00% | 0% | PASS | no |
| us_anfci_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_anfci_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_anfci_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_anfci_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| carry_anfci_stack | year_2024 | +0.02% | 55% | PASS | no |
| carry_anfci_stack | year_2025 | +0.13% | 55% | PASS | no |
| carry_anfci_stack | year_2026 | +0.13% | 62% | PASS | no |
| carry_anfci_stack | holdout_365d | +0.31% | 75% | PASS | no |
| carry_anfci_ew | year_2024 | +0.05% | 64% | PASS | no |
| carry_anfci_ew | year_2025 | +0.08% | 55% | PASS | no |
| carry_anfci_ew | year_2026 | +0.09% | 62% | PASS | no |
| carry_anfci_ew | holdout_365d | +0.21% | 75% | PASS | no |
| carry_anfci_regime | year_2024 | +0.02% | 55% | PASS | no |
| carry_anfci_regime | year_2025 | +0.13% | 55% | PASS | no |
| carry_anfci_regime | year_2026 | +0.13% | 62% | PASS | no |
| carry_anfci_regime | holdout_365d | +0.31% | 75% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US ANFCI [1971-01-15..2026-09-18]. Gate z≤0 / cool z≥1 fixed (§71–§95 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). Single-stress design (like CIP×carry §68 / WUI×carry §86 / NFCI×carry §91). Distinct from funding_liq §20 / NFCI×carry §91 / ANFCI soft §95.

Artifacts: `reports/scholarly_fx_anfci_conditioned_carry_*.csv`, `scholarly_fx_anfci_conditioned_carry_meta.json`, `quest_locked_verify_anfci_conditioned_carry.md`.
