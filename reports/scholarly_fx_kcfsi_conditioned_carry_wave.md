# Scholarly FX: KCFSI-conditioned Lustig–Verdelhan carry wave (§106)

**Path:** Kansas City Fed **US KCFSI** (FRED KCFSI; Financial Stress Index) stress × Lustig–Verdelhan IR3M carry gate/cool — **distinct** from funding_liquidity (§20), NFCI×carry (§91), ANFCI×carry (§96), STLFSI×carry (§101), KCFSI soft (§105), ANFCI soft (§95), NFCI soft (§90), WUI carry (§86), soft_ew (§66), CIP×carry (§68), VIX/GPR×carry (§73), EPU/TPU×carry (§78), CIP/VIX/EPU/WUI soft/mom/value/REER (§71–§77/§79–§89), capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED KCFSI + FRED IR3M rates (pub_lag=1m). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** KCFSI pub_lag_months=1 + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 / NFCI soft §90 / ANFCI soft §95 / STLFSI soft §100 / KCFSI soft §105 mirror) + weight_lag_days=1; carry signal_lag=1d. Costs 1.5 bps/side.
**Primary:** `carry_low_kcfsi` (trade carry only when lagged US KCFSI z ≤ 0). Companions: carry_kcfsi_cool / carry_raw / carry_high_kcfsi / us_kcfsi_haven_usd / carry_kcfsi_stack / carry_kcfsi_ew / carry_kcfsi_regime. Cool z_high=1.0 (§71–§105 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| carry_low_kcfsi | -0.037% | -0.54 | -0.63 | 38% | 15% | -0.18 |
| carry_kcfsi_cool | -0.015% | -0.21 | -0.24 | 51% | 12% | -0.08 |
| carry_raw | -0.002% | -0.02 | -0.03 | 51% | 10% | -0.04 |
| carry_high_kcfsi | +0.005% | +0.44 | +0.68 | 4% | 66% | +0.07 |
| us_kcfsi_haven_usd | -0.012% | -0.75 | -0.80 | 3% | 88% | -0.19 |
| carry_kcfsi_stack | -0.037% | -0.54 | -0.63 | 38% | 15% | -0.18 |
| carry_kcfsi_ew | -0.021% | -0.46 | -0.53 | 51% | 13% | -0.15 |
| carry_kcfsi_regime | -0.049% | -0.70 | -0.82 | 41% | 14% | -0.21 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| carry_raw | year_2024 | +0.15% | 64% | PASS | no |
| carry_raw | year_2025 | +0.12% | 64% | PASS | no |
| carry_raw | year_2026 | +0.13% | 62% | PASS | no |
| carry_raw | holdout_365d | +0.31% | 75% | PASS | no |
| carry_kcfsi_cool | year_2024 | +0.15% | 64% | PASS | no |
| carry_kcfsi_cool | year_2025 | +0.12% | 73% | PASS | no |
| carry_kcfsi_cool | year_2026 | +0.13% | 62% | PASS | no |
| carry_kcfsi_cool | holdout_365d | +0.31% | 75% | PASS | no |
| carry_low_kcfsi | year_2024 | +0.16% | 64% | PASS | no |
| carry_low_kcfsi | year_2025 | +0.12% | 64% | PASS | no |
| carry_low_kcfsi | year_2026 | +0.13% | 62% | PASS | no |
| carry_low_kcfsi | holdout_365d | +0.31% | 75% | PASS | no |
| carry_high_kcfsi | year_2024 | +0.00% | 0% | PASS | no |
| carry_high_kcfsi | year_2025 | +0.00% | 0% | PASS | no |
| carry_high_kcfsi | year_2026 | +0.00% | 0% | PASS | no |
| carry_high_kcfsi | holdout_365d | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| carry_kcfsi_stack | year_2024 | +0.16% | 64% | PASS | no |
| carry_kcfsi_stack | year_2025 | +0.12% | 64% | PASS | no |
| carry_kcfsi_stack | year_2026 | +0.13% | 62% | PASS | no |
| carry_kcfsi_stack | holdout_365d | +0.31% | 75% | PASS | no |
| carry_kcfsi_ew | year_2024 | +0.10% | 64% | PASS | no |
| carry_kcfsi_ew | year_2025 | +0.08% | 64% | PASS | no |
| carry_kcfsi_ew | year_2026 | +0.09% | 62% | PASS | no |
| carry_kcfsi_ew | holdout_365d | +0.21% | 75% | PASS | no |
| carry_kcfsi_regime | year_2024 | +0.16% | 64% | PASS | no |
| carry_kcfsi_regime | year_2025 | +0.12% | 64% | PASS | no |
| carry_kcfsi_regime | year_2026 | +0.13% | 62% | PASS | no |
| carry_kcfsi_regime | holdout_365d | +0.31% | 75% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US KCFSI [1990-03-01..2026-09-01]. Gate z≤0 / cool z≥1 fixed (§71–§105 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). Single-stress design (like CIP×carry §68 / WUI×carry §86 / NFCI×carry §91 / STLFSI×carry §101). Distinct from funding_liq §20 / NFCI×carry §91 / ANFCI×carry §96 / STLFSI×carry §101 / KCFSI soft §105.

Artifacts: `reports/scholarly_fx_kcfsi_conditioned_carry_*.csv`, `scholarly_fx_kcfsi_conditioned_carry_meta.json`, `quest_locked_verify_kcfsi_conditioned_carry.md`.
