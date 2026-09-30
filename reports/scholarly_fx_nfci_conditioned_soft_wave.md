# Scholarly FX: NFCI-conditioned soft-signal EW wave (§90)

**Path:** Chicago Fed **US NFCI** (FRED NFCI) stress × Dahlquist–Hasseltoft ``soft_ew_macro5`` gate/cool — **distinct** from funding_liquidity (§20 USD tilts / carry×NFCI), soft_ew (§66 ungated), soft-stack CIP enrichment (§69), CIP-conditioned soft (§71), VIX/GPR soft (§72), EPU/TPU soft (§76), WUI soft (§85), CIP/VIX/EPU/WUI × carry/mom/value/REER (§68/§73–§89), capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED NFCI + FRED/OECD soft legs (§66 PIT). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** NFCI pub_lag_days=7 → month-end + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 mirror) + weight_lag_days=1; soft legs source-wave pub/signal lags. Costs 1.5 bps/side inside source factors (haven applies own costs).
**Primary:** `soft_low_nfci` (trade soft only when lagged US NFCI z ≤ 0). Companions: soft_nfci_cool / soft_raw / soft_high_nfci / us_nfci_haven_usd / soft_nfci_stack / soft_nfci_ew / soft_nfci_regime. Cool z_high=1.0 (§71–§89 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| soft_low_nfci | +0.085% | +3.81 | +2.95 | 46% | 16% | +0.91 |
| soft_nfci_cool | +0.103% | +4.26 | +3.37 | 61% | 12% | +1.01 |
| soft_raw | +0.118% | +4.26 | +3.63 | 61% | 11% | +1.02 |
| soft_high_nfci | +0.015% | +1.68 | +1.64 | 7% | 41% | +0.42 |
| us_nfci_haven_usd | -0.008% | -0.26 | -0.25 | 6% | 47% | -0.07 |
| soft_nfci_stack | +0.085% | +3.81 | +2.95 | 46% | 16% | +0.91 |
| soft_nfci_ew | +0.060% | +3.29 | +2.70 | 59% | 12% | +0.79 |
| soft_nfci_regime | +0.078% | +2.06 | +1.78 | 52% | 14% | +0.51 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| soft_raw | year_2024 | +0.08% | 55% | PASS | no |
| soft_raw | year_2025 | +0.04% | 64% | PASS | no |
| soft_raw | year_2026 | +0.14% | 88% | PASS | no |
| soft_raw | holdout_365d | +0.06% | 67% | PASS | no |
| soft_nfci_cool | year_2024 | +0.08% | 55% | PASS | no |
| soft_nfci_cool | year_2025 | +0.04% | 64% | PASS | no |
| soft_nfci_cool | year_2026 | +0.14% | 88% | PASS | no |
| soft_nfci_cool | holdout_365d | +0.06% | 67% | PASS | no |
| soft_low_nfci | year_2024 | +0.04% | 36% | PASS | no |
| soft_low_nfci | year_2025 | +0.02% | 64% | PASS | no |
| soft_low_nfci | year_2026 | +0.14% | 88% | PASS | no |
| soft_low_nfci | holdout_365d | +0.06% | 67% | PASS | no |
| soft_high_nfci | year_2024 | +0.00% | 0% | PASS | no |
| soft_high_nfci | year_2025 | +0.00% | 0% | PASS | no |
| soft_high_nfci | year_2026 | +0.00% | 0% | PASS | no |
| soft_high_nfci | holdout_365d | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| soft_nfci_stack | year_2024 | +0.04% | 36% | PASS | no |
| soft_nfci_stack | year_2025 | +0.02% | 64% | PASS | no |
| soft_nfci_stack | year_2026 | +0.14% | 88% | PASS | no |
| soft_nfci_stack | holdout_365d | +0.06% | 67% | PASS | no |
| soft_nfci_ew | year_2024 | +0.04% | 55% | PASS | no |
| soft_nfci_ew | year_2025 | +0.02% | 64% | PASS | no |
| soft_nfci_ew | year_2026 | +0.09% | 88% | PASS | no |
| soft_nfci_ew | holdout_365d | +0.04% | 67% | PASS | no |
| soft_nfci_regime | year_2024 | +0.04% | 36% | PASS | no |
| soft_nfci_regime | year_2025 | +0.02% | 64% | PASS | no |
| soft_nfci_regime | year_2026 | +0.14% | 88% | PASS | no |
| soft_nfci_regime | holdout_365d | +0.06% | 67% | PASS | no |

**Board:** n=8 soft_nw_pos=7 hard_nw_pos=5 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US NFCI [1971-01-15..2026-09-18]. Weekly→month-end. Gate z≤0 / cool z≥1 fixed (§71–§89 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). Single-stress design (like CIP soft §71 / WUI soft §85). Distinct from funding_liq §20.

Artifacts: `reports/scholarly_fx_nfci_conditioned_soft_*.csv`, `scholarly_fx_nfci_conditioned_soft_meta.json`, `quest_locked_verify_nfci_conditioned_soft.md`.
