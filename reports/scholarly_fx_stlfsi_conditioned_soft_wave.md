# Scholarly FX: STLFSI4-conditioned soft-signal EW wave (§100)

**Path:** St. Louis Fed **US STLFSI4** (FRED STLFSI4) stress × Dahlquist–Hasseltoft ``soft_ew_macro5`` gate/cool — **distinct** from funding_liquidity (§20), soft_ew (§66 ungated), NFCI soft (§90), ANFCI soft (§95), soft-stack CIP enrichment (§69), CIP-conditioned soft (§71), VIX/GPR soft (§72), EPU/TPU soft (§76), WUI soft (§85), CIP/VIX/EPU/WUI × carry/mom/value/REER (§68/§73–§89), capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED STLFSI4 + FRED/OECD soft legs (§66 PIT). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** STLFSI4 pub_lag_days=7 → month-end + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 / NFCI soft §90 mirror) + weight_lag_days=1; soft legs source-wave pub/signal lags. Costs 1.5 bps/side inside source factors (haven applies own costs).
**Primary:** `soft_low_stlfsi` (trade soft only when lagged US STLFSI4 z ≤ 0). Companions: soft_stlfsi_cool / soft_raw / soft_high_stlfsi / us_stlfsi_haven_usd / soft_stlfsi_stack / soft_stlfsi_ew / soft_stlfsi_regime. Cool z_high=1.0 (§71–§99 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| soft_low_stlfsi | +0.084% | +3.74 | +2.98 | 52% | 15% | +0.87 |
| soft_stlfsi_cool | +0.110% | +4.42 | +3.52 | 61% | 12% | +1.02 |
| soft_raw | +0.118% | +4.26 | +3.63 | 61% | 11% | +1.02 |
| soft_high_stlfsi | +0.014% | +1.60 | +1.71 | 6% | 64% | +0.57 |
| us_stlfsi_haven_usd | -0.021% | -1.19 | -1.09 | 4% | 79% | -0.31 |
| soft_stlfsi_stack | +0.084% | +3.74 | +2.98 | 52% | 15% | +0.87 |
| soft_stlfsi_ew | +0.057% | +3.46 | +2.85 | 59% | 13% | +0.82 |
| soft_stlfsi_regime | +0.062% | +2.16 | +1.84 | 54% | 14% | +0.52 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| soft_raw | year_2024 | +0.08% | 55% | PASS | no |
| soft_raw | year_2025 | +0.04% | 64% | PASS | no |
| soft_raw | year_2026 | +0.14% | 88% | PASS | no |
| soft_raw | holdout_365d | +0.06% | 67% | PASS | no |
| soft_stlfsi_cool | year_2024 | +0.08% | 55% | PASS | no |
| soft_stlfsi_cool | year_2025 | +0.03% | 64% | PASS | no |
| soft_stlfsi_cool | year_2026 | +0.14% | 88% | PASS | no |
| soft_stlfsi_cool | holdout_365d | +0.07% | 67% | PASS | no |
| soft_low_stlfsi | year_2024 | +0.08% | 55% | PASS | no |
| soft_low_stlfsi | year_2025 | +0.02% | 64% | PASS | no |
| soft_low_stlfsi | year_2026 | +0.12% | 50% | PASS | no |
| soft_low_stlfsi | holdout_365d | +0.06% | 50% | PASS | no |
| soft_high_stlfsi | year_2024 | +0.00% | 0% | PASS | no |
| soft_high_stlfsi | year_2025 | +0.00% | 0% | PASS | no |
| soft_high_stlfsi | year_2026 | +0.00% | 12% | PASS | no |
| soft_high_stlfsi | holdout_365d | -0.01% | 8% | PASS | no |
| us_stlfsi_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_stlfsi_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_stlfsi_haven_usd | year_2026 | +0.08% | 12% | PASS | no |
| us_stlfsi_haven_usd | holdout_365d | -0.09% | 8% | PASS | no |
| soft_stlfsi_stack | year_2024 | +0.08% | 55% | PASS | no |
| soft_stlfsi_stack | year_2025 | +0.02% | 64% | PASS | no |
| soft_stlfsi_stack | year_2026 | +0.12% | 50% | PASS | no |
| soft_stlfsi_stack | holdout_365d | +0.06% | 50% | PASS | no |
| soft_stlfsi_ew | year_2024 | +0.06% | 55% | PASS | no |
| soft_stlfsi_ew | year_2025 | +0.02% | 64% | PASS | no |
| soft_stlfsi_ew | year_2026 | +0.11% | 88% | PASS | no |
| soft_stlfsi_ew | holdout_365d | +0.01% | 67% | PASS | no |
| soft_stlfsi_regime | year_2024 | +0.08% | 55% | PASS | no |
| soft_stlfsi_regime | year_2025 | +0.02% | 64% | PASS | no |
| soft_stlfsi_regime | year_2026 | +0.20% | 62% | PASS | no |
| soft_stlfsi_regime | holdout_365d | -0.03% | 50% | PASS | no |

**Board:** n=8 soft_nw_pos=7 hard_nw_pos=5 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US STLFSI4 [1994-01-07..2026-09-25]. Weekly→month-end. Gate z≤0 / cool z≥1 fixed (§71–§99 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). Single-stress design (like CIP soft §71 / WUI soft §85). Distinct from funding_liq §20.

Artifacts: `reports/scholarly_fx_stlfsi_conditioned_soft_*.csv`, `scholarly_fx_stlfsi_conditioned_soft_meta.json`, `quest_locked_verify_stlfsi_conditioned_soft.md`.
