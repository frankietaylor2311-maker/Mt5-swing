# Scholarly FX: STLFSI4-conditioned BIS REER HML-FX value wave (§104)

**Path:** St. Louis Fed **US STLFSI4** (FRED STLFSI4) stress × BIS multilateral REER undervaluation / HML-FX value (`reer_cheap_xs` §31) gate/cool — **distinct** from funding_liquidity (§20), raw bis_reer (§31), NFCI soft–carry–mom–value–REER (§90–§94), ANFCI soft–carry–mom–value–REER (§95–§99), STLFSI soft–carry–mom–PPP (§100–§103), CIP×REER (§84), VIX/GPR×REER (§81), EPU×REER (§77), WUI×REER (§89), CIP/VIX/EPU/WUI × soft/carry/mom/PPP, soft–carry–mom–value stacks, capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED STLFSI4 + FRED BIS REER (pub_lag=2m). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** STLFSI4 pub_lag_days=7 → month-end + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 / NFCI REER §94 / ANFCI REER §99 / STLFSI soft–carry–mom–PPP §100–§103 mirror) + weight_lag_days=1; REER pub_lag=2m signal_lag=1m + weight lag 1d. Costs 1.5 bps/side.
**Primary:** `reer_low_stlfsi` (trade REER-value only when lagged US STLFSI4 z ≤ 0). Companions: reer_stlfsi_cool / reer_raw / reer_high_stlfsi / us_stlfsi_haven_usd / reer_stlfsi_stack / reer_stlfsi_ew / reer_stlfsi_regime. Cool z_high=1.0 (§71–§103 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| reer_low_stlfsi | -0.056% | -0.99 | -1.15 | 37% | 18% | -0.21 |
| reer_stlfsi_cool | -0.000% | -0.00 | -0.00 | 53% | 14% | +0.03 |
| reer_raw | +0.034% | +0.51 | +0.60 | 52% | 13% | +0.15 |
| reer_high_stlfsi | +0.041% | +2.15 | +1.76 | 7% | 60% | +0.48 |
| us_stlfsi_haven_usd | -0.022% | -1.31 | -1.12 | 4% | 71% | -0.32 |
| reer_stlfsi_stack | -0.056% | -0.99 | -1.15 | 37% | 18% | -0.21 |
| reer_stlfsi_ew | -0.026% | -0.67 | -0.78 | 51% | 16% | -0.13 |
| reer_stlfsi_regime | -0.078% | -1.32 | -1.44 | 41% | 17% | -0.30 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| reer_raw | year_2024 | -0.11% | 36% | PASS | no |
| reer_raw | year_2025 | -0.27% | 27% | PASS | no |
| reer_raw | year_2026 | +0.16% | 62% | PASS | no |
| reer_raw | holdout_365d | +0.08% | 58% | PASS | no |
| reer_stlfsi_cool | year_2024 | -0.11% | 36% | PASS | no |
| reer_stlfsi_cool | year_2025 | -0.25% | 36% | PASS | no |
| reer_stlfsi_cool | year_2026 | +0.17% | 62% | PASS | no |
| reer_stlfsi_cool | holdout_365d | +0.06% | 50% | PASS | no |
| reer_low_stlfsi | year_2024 | -0.11% | 36% | PASS | no |
| reer_low_stlfsi | year_2025 | -0.24% | 36% | PASS | no |
| reer_low_stlfsi | year_2026 | +0.12% | 50% | PASS | no |
| reer_low_stlfsi | holdout_365d | +0.02% | 42% | PASS | no |
| reer_high_stlfsi | year_2024 | +0.00% | 0% | PASS | no |
| reer_high_stlfsi | year_2025 | +0.00% | 0% | PASS | no |
| reer_high_stlfsi | year_2026 | +0.00% | 12% | PASS | no |
| reer_high_stlfsi | holdout_365d | +0.03% | 17% | PASS | no |
| us_stlfsi_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_stlfsi_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_stlfsi_haven_usd | year_2026 | +0.07% | 12% | PASS | no |
| us_stlfsi_haven_usd | holdout_365d | -0.09% | 8% | PASS | no |
| reer_stlfsi_stack | year_2024 | -0.11% | 36% | PASS | no |
| reer_stlfsi_stack | year_2025 | -0.24% | 36% | PASS | no |
| reer_stlfsi_stack | year_2026 | +0.12% | 50% | PASS | no |
| reer_stlfsi_stack | holdout_365d | +0.02% | 42% | PASS | no |
| reer_stlfsi_ew | year_2024 | -0.07% | 36% | PASS | no |
| reer_stlfsi_ew | year_2025 | -0.16% | 36% | PASS | no |
| reer_stlfsi_ew | year_2026 | +0.12% | 75% | PASS | no |
| reer_stlfsi_ew | holdout_365d | -0.00% | 58% | PASS | no |
| reer_stlfsi_regime | year_2024 | -0.11% | 36% | PASS | no |
| reer_stlfsi_regime | year_2025 | -0.24% | 36% | PASS | no |
| reer_stlfsi_regime | year_2026 | +0.19% | 62% | PASS | no |
| reer_stlfsi_regime | holdout_365d | -0.07% | 50% | PASS | no |

**Board:** n=8 soft_nw_pos=1 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US STLFSI4 [1994-01-07..2026-09-25]. REER [1994-03-01..2026-09-01]. Gate z≤0 / cool z≥1 fixed (§71–§103 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). REER pub_lag=2m signal_lag=1m. Single-stress design (like CIP×REER §84 / WUI×REER §89 / NFCI×REER §94 / ANFCI×REER §99), not dual EPU+TPU. Closes STLFSI soft–carry–mom–value–REER stack after §100–§103.

Artifacts: `reports/scholarly_fx_stlfsi_conditioned_reer_*.csv`, `scholarly_fx_stlfsi_conditioned_reer_meta.json`, `quest_locked_verify_stlfsi_conditioned_reer.md`.
