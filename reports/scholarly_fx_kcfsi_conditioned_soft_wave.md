# Scholarly FX: KCFSI-conditioned soft-signal EW wave (§105)

**Path:** Kansas City Fed **US KCFSI** (FRED KCFSI) stress × Dahlquist–Hasseltoft ``soft_ew_macro5`` gate/cool — **distinct** from funding_liquidity (§20), soft_ew (§66 ungated), NFCI soft (§90), ANFCI soft (§95), STLFSI soft (§100), soft-stack CIP enrichment (§69), CIP-conditioned soft (§71), VIX/GPR soft (§72), EPU/TPU soft (§76), WUI soft (§85), CIP/VIX/EPU/WUI × carry/mom/value/REER (§68/§73–§89), capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED KCFSI + FRED/OECD soft legs (§66 PIT). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** KCFSI pub_lag_months=1 + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 / NFCI soft §90 mirror) + weight_lag_days=1; soft legs source-wave pub/signal lags. Costs 1.5 bps/side inside source factors (haven applies own costs).
**Primary:** `soft_low_kcfsi` (trade soft only when lagged US KCFSI z ≤ 0). Companions: soft_kcfsi_cool / soft_raw / soft_high_kcfsi / us_kcfsi_haven_usd / soft_kcfsi_stack / soft_kcfsi_ew / soft_kcfsi_regime. Cool z_high=1.0 (§71–§104 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| soft_low_kcfsi | +0.081% | +3.43 | +3.06 | 48% | 15% | +0.86 |
| soft_kcfsi_cool | +0.109% | +4.28 | +3.63 | 61% | 12% | +1.03 |
| soft_raw | +0.118% | +4.26 | +3.63 | 61% | 11% | +1.02 |
| soft_high_kcfsi | -0.001% | -0.19 | -0.22 | 3% | 84% | -0.05 |
| us_kcfsi_haven_usd | -0.012% | -0.75 | -0.80 | 3% | 88% | -0.19 |
| soft_kcfsi_stack | +0.081% | +3.43 | +3.06 | 48% | 15% | +0.86 |
| soft_kcfsi_ew | +0.059% | +3.53 | +3.16 | 62% | 13% | +0.87 |
| soft_kcfsi_regime | +0.069% | +2.41 | +2.31 | 51% | 14% | +0.60 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| soft_raw | year_2024 | +0.08% | 55% | PASS | no |
| soft_raw | year_2025 | +0.04% | 64% | PASS | no |
| soft_raw | year_2026 | +0.14% | 88% | PASS | no |
| soft_raw | holdout_365d | +0.06% | 67% | PASS | no |
| soft_kcfsi_cool | year_2024 | +0.09% | 55% | PASS | no |
| soft_kcfsi_cool | year_2025 | +0.04% | 64% | PASS | no |
| soft_kcfsi_cool | year_2026 | +0.14% | 88% | PASS | no |
| soft_kcfsi_cool | holdout_365d | +0.06% | 67% | PASS | no |
| soft_low_kcfsi | year_2024 | +0.09% | 55% | PASS | no |
| soft_low_kcfsi | year_2025 | +0.03% | 55% | PASS | no |
| soft_low_kcfsi | year_2026 | +0.14% | 88% | PASS | no |
| soft_low_kcfsi | holdout_365d | +0.06% | 67% | PASS | no |
| soft_high_kcfsi | year_2024 | +0.00% | 0% | PASS | no |
| soft_high_kcfsi | year_2025 | +0.00% | 0% | PASS | no |
| soft_high_kcfsi | year_2026 | +0.00% | 0% | PASS | no |
| soft_high_kcfsi | holdout_365d | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| soft_kcfsi_stack | year_2024 | +0.09% | 55% | PASS | no |
| soft_kcfsi_stack | year_2025 | +0.03% | 55% | PASS | no |
| soft_kcfsi_stack | year_2026 | +0.14% | 88% | PASS | no |
| soft_kcfsi_stack | holdout_365d | +0.06% | 67% | PASS | no |
| soft_kcfsi_ew | year_2024 | +0.06% | 55% | PASS | no |
| soft_kcfsi_ew | year_2025 | +0.02% | 64% | PASS | no |
| soft_kcfsi_ew | year_2026 | +0.09% | 88% | PASS | no |
| soft_kcfsi_ew | holdout_365d | +0.04% | 67% | PASS | no |
| soft_kcfsi_regime | year_2024 | +0.09% | 55% | PASS | no |
| soft_kcfsi_regime | year_2025 | +0.03% | 55% | PASS | no |
| soft_kcfsi_regime | year_2026 | +0.14% | 88% | PASS | no |
| soft_kcfsi_regime | holdout_365d | +0.06% | 67% | PASS | no |

**Board:** n=8 soft_nw_pos=6 hard_nw_pos=6 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US KCFSI [1990-03-01..2026-09-01]. Monthly. Gate z≤0 / cool z≥1 fixed (§71–§104 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). Single-stress design (like CIP soft §71 / WUI soft §85). Distinct from funding_liq §20.

Artifacts: `reports/scholarly_fx_kcfsi_conditioned_soft_*.csv`, `scholarly_fx_kcfsi_conditioned_soft_meta.json`, `quest_locked_verify_kcfsi_conditioned_soft.md`.
