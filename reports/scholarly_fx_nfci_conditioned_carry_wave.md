# Scholarly FX: NFCI-conditioned Lustig–Verdelhan carry wave (§91)

**Path:** Chicago Fed **US NFCI** (FRED NFCI) stress × Lustig–Verdelhan IR3M carry gate/cool — **distinct** from funding_liquidity (§20), NFCI soft (§90), WUI carry (§86), soft_ew (§66), CIP×carry (§68), VIX/GPR×carry (§73), EPU/TPU×carry (§78), CIP/VIX/EPU/WUI soft/mom/value/REER (§71–§77/§79–§89), capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED NFCI + FRED IR3M rates (pub_lag=1m). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** NFCI pub_lag_days=7 → month-end + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 / NFCI soft §90 mirror) + weight_lag_days=1; carry signal_lag=1d. Costs 1.5 bps/side.
**Primary:** `carry_low_nfci` (trade carry only when lagged US NFCI z ≤ 0). Companions: carry_nfci_cool / carry_raw / carry_high_nfci / us_nfci_haven_usd / carry_nfci_stack / carry_nfci_ew / carry_nfci_regime. Cool z_high=1.0 (§71–§90 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| carry_low_nfci | -0.031% | -0.47 | -0.49 | 34% | 16% | -0.16 |
| carry_nfci_cool | -0.013% | -0.19 | -0.22 | 51% | 12% | -0.08 |
| carry_raw | -0.002% | -0.02 | -0.03 | 51% | 10% | -0.04 |
| carry_high_nfci | -0.003% | -0.12 | -0.13 | 6% | 48% | -0.02 |
| us_nfci_haven_usd | -0.008% | -0.26 | -0.25 | 6% | 47% | -0.07 |
| carry_nfci_stack | -0.031% | -0.47 | -0.49 | 34% | 16% | -0.16 |
| carry_nfci_ew | -0.017% | -0.38 | -0.41 | 51% | 13% | -0.14 |
| carry_nfci_regime | -0.039% | -0.54 | -0.54 | 40% | 13% | -0.17 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| carry_raw | year_2024 | +0.15% | 64% | PASS | no |
| carry_raw | year_2025 | +0.12% | 64% | PASS | no |
| carry_raw | year_2026 | +0.13% | 62% | PASS | no |
| carry_raw | holdout_365d | +0.31% | 75% | PASS | no |
| carry_nfci_cool | year_2024 | +0.12% | 64% | PASS | no |
| carry_nfci_cool | year_2025 | +0.12% | 64% | PASS | no |
| carry_nfci_cool | year_2026 | +0.13% | 62% | PASS | no |
| carry_nfci_cool | holdout_365d | +0.31% | 75% | PASS | no |
| carry_low_nfci | year_2024 | -0.12% | 27% | PASS | no |
| carry_low_nfci | year_2025 | +0.13% | 55% | PASS | no |
| carry_low_nfci | year_2026 | +0.13% | 62% | PASS | no |
| carry_low_nfci | holdout_365d | +0.31% | 75% | PASS | no |
| carry_high_nfci | year_2024 | +0.00% | 0% | PASS | no |
| carry_high_nfci | year_2025 | +0.00% | 0% | PASS | no |
| carry_high_nfci | year_2026 | +0.00% | 0% | PASS | no |
| carry_high_nfci | holdout_365d | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| carry_nfci_stack | year_2024 | -0.12% | 27% | PASS | no |
| carry_nfci_stack | year_2025 | +0.13% | 55% | PASS | no |
| carry_nfci_stack | year_2026 | +0.13% | 62% | PASS | no |
| carry_nfci_stack | holdout_365d | +0.31% | 75% | PASS | no |
| carry_nfci_ew | year_2024 | -0.00% | 55% | PASS | no |
| carry_nfci_ew | year_2025 | +0.08% | 55% | PASS | no |
| carry_nfci_ew | year_2026 | +0.09% | 62% | PASS | no |
| carry_nfci_ew | holdout_365d | +0.21% | 75% | PASS | no |
| carry_nfci_regime | year_2024 | -0.12% | 27% | PASS | no |
| carry_nfci_regime | year_2025 | +0.13% | 55% | PASS | no |
| carry_nfci_regime | year_2026 | +0.13% | 62% | PASS | no |
| carry_nfci_regime | holdout_365d | +0.31% | 75% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US NFCI [1971-01-15..2026-09-18]. Gate z≤0 / cool z≥1 fixed (§71–§90 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). Single-stress design (like CIP×carry §68 / WUI×carry §86). Distinct from funding_liq §20 / NFCI soft §90.

Artifacts: `reports/scholarly_fx_nfci_conditioned_carry_*.csv`, `scholarly_fx_nfci_conditioned_carry_meta.json`, `quest_locked_verify_nfci_conditioned_carry.md`.
