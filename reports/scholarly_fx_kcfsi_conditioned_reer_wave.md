# Scholarly FX: KCFSI-conditioned BIS REER HML-FX value wave (§109)

**Path:** Kansas City Fed **US KCFSI** (FRED KCFSI) stress × BIS multilateral REER undervaluation / HML-FX value (`reer_cheap_xs` §31) gate/cool — **distinct** from funding_liquidity (§20), raw bis_reer (§31), NFCI soft–carry–mom–value–REER (§90–§94), ANFCI soft–carry–mom–value–REER (§95–§99), STLFSI soft–carry–mom–value–REER (§100–§104), KCFSI soft–carry–mom–PPP (§105–§108), CIP×REER (§84), VIX/GPR×REER (§81), EPU×REER (§77), WUI×REER (§89), CIP/VIX/EPU/WUI × soft/carry/mom/PPP, soft–carry–mom–value stacks, capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED KCFSI + FRED BIS REER (pub_lag=2m). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** KCFSI pub_lag_months=1 + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 / NFCI REER §94 / ANFCI REER §99 / STLFSI REER §104 / KCFSI soft–carry–mom–PPP §105–§108 mirror) + weight_lag_days=1; REER pub_lag=2m signal_lag=1m + weight lag 1d. Costs 1.5 bps/side.
**Primary:** `reer_low_kcfsi` (trade REER-value only when lagged US KCFSI z ≤ 0). Companions: reer_kcfsi_cool / reer_raw / reer_high_kcfsi / us_kcfsi_haven_usd / reer_kcfsi_stack / reer_kcfsi_ew / reer_kcfsi_regime. Cool z_high=1.0 (§71–§108 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| reer_low_kcfsi | -0.013% | -0.24 | -0.28 | 37% | 18% | -0.03 |
| reer_kcfsi_cool | +0.030% | +0.50 | +0.57 | 53% | 15% | +0.15 |
| reer_raw | +0.034% | +0.51 | +0.60 | 52% | 13% | +0.15 |
| reer_high_kcfsi | -0.018% | -1.22 | -1.54 | 2% | 99% | -0.22 |
| us_kcfsi_haven_usd | -0.008% | -0.49 | -0.51 | 4% | 84% | -0.12 |
| reer_kcfsi_stack | -0.013% | -0.24 | -0.28 | 37% | 18% | -0.03 |
| reer_kcfsi_ew | +0.003% | +0.09 | +0.10 | 51% | 15% | +0.05 |
| reer_kcfsi_regime | -0.021% | -0.37 | -0.42 | 41% | 17% | -0.06 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| reer_raw | year_2024 | -0.11% | 36% | PASS | no |
| reer_raw | year_2025 | -0.27% | 27% | PASS | no |
| reer_raw | year_2026 | +0.16% | 62% | PASS | no |
| reer_raw | holdout_365d | +0.08% | 58% | PASS | no |
| reer_kcfsi_cool | year_2024 | -0.11% | 36% | PASS | no |
| reer_kcfsi_cool | year_2025 | -0.28% | 27% | PASS | no |
| reer_kcfsi_cool | year_2026 | +0.16% | 62% | PASS | no |
| reer_kcfsi_cool | holdout_365d | +0.08% | 58% | PASS | no |
| reer_low_kcfsi | year_2024 | -0.11% | 36% | PASS | no |
| reer_low_kcfsi | year_2025 | -0.29% | 18% | PASS | no |
| reer_low_kcfsi | year_2026 | +0.16% | 62% | PASS | no |
| reer_low_kcfsi | holdout_365d | +0.08% | 58% | PASS | no |
| reer_high_kcfsi | year_2024 | +0.00% | 0% | PASS | no |
| reer_high_kcfsi | year_2025 | +0.00% | 0% | PASS | no |
| reer_high_kcfsi | year_2026 | +0.00% | 0% | PASS | no |
| reer_high_kcfsi | holdout_365d | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| reer_kcfsi_stack | year_2024 | -0.11% | 36% | PASS | no |
| reer_kcfsi_stack | year_2025 | -0.29% | 18% | PASS | no |
| reer_kcfsi_stack | year_2026 | +0.16% | 62% | PASS | no |
| reer_kcfsi_stack | holdout_365d | +0.08% | 58% | PASS | no |
| reer_kcfsi_ew | year_2024 | -0.07% | 36% | PASS | no |
| reer_kcfsi_ew | year_2025 | -0.19% | 18% | PASS | no |
| reer_kcfsi_ew | year_2026 | +0.11% | 62% | PASS | no |
| reer_kcfsi_ew | holdout_365d | +0.06% | 58% | PASS | no |
| reer_kcfsi_regime | year_2024 | -0.11% | 36% | PASS | no |
| reer_kcfsi_regime | year_2025 | -0.29% | 18% | PASS | no |
| reer_kcfsi_regime | year_2026 | +0.16% | 62% | PASS | no |
| reer_kcfsi_regime | holdout_365d | +0.08% | 58% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US KCFSI [1990-03-01..2026-09-01]. REER [1994-03-01..2026-09-01]. Gate z≤0 / cool z≥1 fixed (§71–§108 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). REER pub_lag=2m signal_lag=1m. Single-stress design (like CIP×REER §84 / WUI×REER §89 / NFCI×REER §94 / ANFCI×REER §99 / STLFSI×REER §104), not dual EPU+TPU. Closes KCFSI soft–carry–mom–value–REER stack after §105–§108.

Artifacts: `reports/scholarly_fx_kcfsi_conditioned_reer_*.csv`, `scholarly_fx_kcfsi_conditioned_reer_meta.json`, `quest_locked_verify_kcfsi_conditioned_reer.md`.
