# Scholarly FX: NFCI-conditioned BIS REER HML-FX value wave (§94)

**Path:** Chicago Fed **US NFCI** (FRED NFCI) stress × BIS multilateral REER undervaluation / HML-FX value (`reer_cheap_xs` §31) gate/cool — **distinct** from funding_liquidity (§20), raw bis_reer (§31), NFCI soft (§90), NFCI carry (§91), NFCI mom (§92), NFCI PPP (§93), CIP×REER (§84), VIX/GPR×REER (§81), EPU×REER (§77), WUI×REER (§89), CIP/VIX/EPU/WUI × soft/carry/mom/PPP, soft–carry–mom–value stacks, capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED NFCI + FRED BIS REER (pub_lag=2m). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** NFCI pub_lag_days=7 → month-end + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 / NFCI soft–carry–mom–PPP §90–§93 mirror) + weight_lag_days=1; REER pub_lag=2m signal_lag=1m + weight lag 1d. Costs 1.5 bps/side.
**Primary:** `reer_low_nfci` (trade REER-value only when lagged US NFCI z ≤ 0). Companions: reer_nfci_cool / reer_raw / reer_high_nfci / us_nfci_haven_usd / reer_nfci_stack / reer_nfci_ew / reer_nfci_regime. Cool z_high=1.0 (§71–§93 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| reer_low_nfci | -0.016% | -0.30 | -0.33 | 35% | 19% | -0.04 |
| reer_nfci_cool | +0.001% | +0.01 | +0.01 | 52% | 14% | +0.03 |
| reer_raw | +0.034% | +0.51 | +0.60 | 52% | 13% | +0.15 |
| reer_high_nfci | +0.028% | +1.16 | +1.18 | 8% | 54% | +0.27 |
| us_nfci_haven_usd | -0.011% | -0.37 | -0.35 | 6% | 49% | -0.09 |
| reer_nfci_stack | -0.016% | -0.30 | -0.33 | 35% | 19% | -0.04 |
| reer_nfci_ew | -0.009% | -0.23 | -0.25 | 51% | 15% | -0.03 |
| reer_nfci_regime | -0.027% | -0.44 | -0.45 | 41% | 15% | -0.08 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| reer_raw | year_2024 | -0.11% | 36% | PASS | no |
| reer_raw | year_2025 | -0.27% | 27% | PASS | no |
| reer_raw | year_2026 | +0.16% | 62% | PASS | no |
| reer_raw | holdout_365d | +0.08% | 58% | PASS | no |
| reer_nfci_cool | year_2024 | -0.11% | 36% | PASS | no |
| reer_nfci_cool | year_2025 | -0.26% | 27% | PASS | no |
| reer_nfci_cool | year_2026 | +0.16% | 62% | PASS | no |
| reer_nfci_cool | holdout_365d | +0.08% | 58% | PASS | no |
| reer_low_nfci | year_2024 | -0.08% | 27% | PASS | no |
| reer_low_nfci | year_2025 | -0.24% | 36% | PASS | no |
| reer_low_nfci | year_2026 | +0.16% | 62% | PASS | no |
| reer_low_nfci | holdout_365d | +0.08% | 58% | PASS | no |
| reer_high_nfci | year_2024 | +0.00% | 0% | PASS | no |
| reer_high_nfci | year_2025 | +0.00% | 0% | PASS | no |
| reer_high_nfci | year_2026 | +0.00% | 0% | PASS | no |
| reer_high_nfci | holdout_365d | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| reer_nfci_stack | year_2024 | -0.08% | 27% | PASS | no |
| reer_nfci_stack | year_2025 | -0.24% | 36% | PASS | no |
| reer_nfci_stack | year_2026 | +0.16% | 62% | PASS | no |
| reer_nfci_stack | holdout_365d | +0.08% | 58% | PASS | no |
| reer_nfci_ew | year_2024 | -0.06% | 36% | PASS | no |
| reer_nfci_ew | year_2025 | -0.17% | 36% | PASS | no |
| reer_nfci_ew | year_2026 | +0.11% | 62% | PASS | no |
| reer_nfci_ew | holdout_365d | +0.06% | 58% | PASS | no |
| reer_nfci_regime | year_2024 | -0.08% | 27% | PASS | no |
| reer_nfci_regime | year_2025 | -0.24% | 36% | PASS | no |
| reer_nfci_regime | year_2026 | +0.16% | 62% | PASS | no |
| reer_nfci_regime | holdout_365d | +0.08% | 58% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US NFCI [1971-01-15..2026-09-18]. REER [1994-03-01..2026-09-01]. Gate z≤0 / cool z≥1 fixed (§71–§93 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). REER pub_lag=2m signal_lag=1m. Single-stress design (like CIP×REER §84 / WUI×REER §89 / NFCI×PPP §93), not dual EPU+TPU. Closes NFCI triad after §90–§93.

Artifacts: `reports/scholarly_fx_nfci_conditioned_reer_*.csv`, `scholarly_fx_nfci_conditioned_reer_meta.json`, `quest_locked_verify_nfci_conditioned_reer.md`.
