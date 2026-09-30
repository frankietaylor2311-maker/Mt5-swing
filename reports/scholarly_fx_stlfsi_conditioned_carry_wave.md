# Scholarly FX: STLFSI4-conditioned Lustig–Verdelhan carry wave (§101)

**Path:** St. Louis Fed **US STLFSI4** (FRED STLFSI4; redesigned Financial Stress Index) stress × Lustig–Verdelhan IR3M carry gate/cool — **distinct** from funding_liquidity (§20), NFCI×carry (§91), ANFCI×carry (§96), STLFSI soft (§100), ANFCI soft (§95), NFCI soft (§90), WUI carry (§86), soft_ew (§66), CIP×carry (§68), VIX/GPR×carry (§73), EPU/TPU×carry (§78), CIP/VIX/EPU/WUI soft/mom/value/REER (§71–§77/§79–§89), capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED STLFSI4 + FRED IR3M rates (pub_lag=1m). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** STLFSI4 pub_lag_days=7 → month-end + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 / NFCI soft §90 / NFCI carry §91 / ANFCI soft §95 / ANFCI carry §96 / STLFSI soft §100 mirror) + weight_lag_days=1; carry signal_lag=1d. Costs 1.5 bps/side.
**Primary:** `carry_low_stlfsi` (trade carry only when lagged US STLFSI4 z ≤ 0). Companions: carry_stlfsi_cool / carry_raw / carry_high_stlfsi / us_stlfsi_haven_usd / carry_stlfsi_stack / carry_stlfsi_ew / carry_stlfsi_regime. Cool z_high=1.0 (§71–§100 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| carry_low_stlfsi | -0.072% | -1.10 | -1.33 | 39% | 15% | -0.31 |
| carry_stlfsi_cool | -0.043% | -0.60 | -0.68 | 51% | 12% | -0.18 |
| carry_raw | -0.002% | -0.02 | -0.03 | 51% | 10% | -0.04 |
| carry_high_stlfsi | +0.030% | +2.21 | +2.01 | 7% | 49% | +0.33 |
| us_stlfsi_haven_usd | -0.021% | -1.19 | -1.09 | 4% | 79% | -0.31 |
| carry_stlfsi_stack | -0.072% | -1.10 | -1.33 | 39% | 15% | -0.31 |
| carry_stlfsi_ew | -0.045% | -1.00 | -1.18 | 48% | 13% | -0.28 |
| carry_stlfsi_regime | -0.093% | -1.37 | -1.62 | 42% | 14% | -0.38 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| carry_raw | year_2024 | +0.15% | 64% | PASS | no |
| carry_raw | year_2025 | +0.12% | 64% | PASS | no |
| carry_raw | year_2026 | +0.13% | 62% | PASS | no |
| carry_raw | holdout_365d | +0.31% | 75% | PASS | no |
| carry_stlfsi_cool | year_2024 | +0.15% | 64% | PASS | no |
| carry_stlfsi_cool | year_2025 | +0.13% | 55% | PASS | no |
| carry_stlfsi_cool | year_2026 | +0.05% | 62% | PASS | no |
| carry_stlfsi_cool | holdout_365d | +0.21% | 75% | PASS | no |
| carry_low_stlfsi | year_2024 | +0.15% | 64% | PASS | no |
| carry_low_stlfsi | year_2025 | +0.13% | 55% | PASS | no |
| carry_low_stlfsi | year_2026 | -0.14% | 38% | PASS | no |
| carry_low_stlfsi | holdout_365d | +0.06% | 58% | PASS | no |
| carry_high_stlfsi | year_2024 | +0.00% | 0% | PASS | no |
| carry_high_stlfsi | year_2025 | +0.00% | 0% | PASS | no |
| carry_high_stlfsi | year_2026 | +0.02% | 12% | PASS | no |
| carry_high_stlfsi | holdout_365d | +0.08% | 17% | PASS | no |
| us_stlfsi_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_stlfsi_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_stlfsi_haven_usd | year_2026 | +0.08% | 12% | PASS | no |
| us_stlfsi_haven_usd | holdout_365d | -0.09% | 8% | PASS | no |
| carry_stlfsi_stack | year_2024 | +0.15% | 64% | PASS | no |
| carry_stlfsi_stack | year_2025 | +0.13% | 55% | PASS | no |
| carry_stlfsi_stack | year_2026 | -0.14% | 38% | PASS | no |
| carry_stlfsi_stack | holdout_365d | +0.06% | 58% | PASS | no |
| carry_stlfsi_ew | year_2024 | +0.10% | 64% | PASS | no |
| carry_stlfsi_ew | year_2025 | +0.08% | 55% | PASS | no |
| carry_stlfsi_ew | year_2026 | +0.00% | 62% | PASS | no |
| carry_stlfsi_ew | holdout_365d | +0.06% | 67% | PASS | no |
| carry_stlfsi_regime | year_2024 | +0.15% | 64% | PASS | no |
| carry_stlfsi_regime | year_2025 | +0.13% | 55% | PASS | no |
| carry_stlfsi_regime | year_2026 | -0.05% | 50% | PASS | no |
| carry_stlfsi_regime | holdout_365d | -0.02% | 58% | PASS | no |

**Board:** n=8 soft_nw_pos=1 hard_nw_pos=1 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US STLFSI4 [1994-01-07..2026-09-25]. Gate z≤0 / cool z≥1 fixed (§71–§100 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). Single-stress design (like CIP×carry §68 / WUI×carry §86 / NFCI×carry §91). Distinct from funding_liq §20 / NFCI×carry §91 / ANFCI×carry §96 / STLFSI soft §100.

Artifacts: `reports/scholarly_fx_stlfsi_conditioned_carry_*.csv`, `scholarly_fx_stlfsi_conditioned_carry_meta.json`, `quest_locked_verify_stlfsi_conditioned_carry.md`.
