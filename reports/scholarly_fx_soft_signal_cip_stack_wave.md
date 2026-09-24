# Scholarly FX: soft-signal CIP-enriched EW recombination (§69)

**Path:** Equal-weight of pre-registered soft-boarded scholarly XS daily returns (macro5 + CIP soft2) — **distinct** from soft_ew (§66), CIP XS (§67), CIP-conditioned carry (§68), combo (§8), capital-sleeve mix (§53).
**Data:** `approximate_non_ftmo` + free FRED/OECD panels + Du–Schreger CIP slim rebuilt with source-wave PIT lags. **SOFT_LEGS_CIP (a priori):** `['bci_chg_xs', 'high_ip_xs', 'high_ppi_xs', 'low_gdp_xs', 'low_cars_xs', 'ust_premium_haven_usd', 'cip_ew']`. Row-wise nanmean; require ≥2 legs/day. Costs 1.5 bps/side (inside source factors).
**Primary:** `soft_ew7_cip` (EW of all 7 soft CIP legs). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| soft_ew7_cip | +0.104% | +5.09 | +3.94 | 64% | 11% | +1.18 |
| soft_ew_macro5 | +0.118% | +4.26 | +3.63 | 61% | 11% | +1.02 |
| soft_ew_cip2 | +0.069% | +2.29 | +2.17 | 60% | 14% | +0.59 |
| soft_ew6_no_haven | +0.109% | +4.71 | +3.78 | 63% | 11% | +1.11 |
| soft_ew6_no_cip_ew | +0.111% | +4.71 | +3.87 | 63% | 10% | +1.10 |
| soft_ew_growth_cip | +0.101% | +4.03 | +3.34 | 64% | 10% | +0.92 |
| soft_ew_honesty_cip | +0.090% | +3.80 | +3.36 | 62% | 13% | +0.90 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| soft_ew7_cip | year_2024 | +0.05% | 45% | PASS | no |
| soft_ew7_cip | year_2025 | +0.05% | 73% | PASS | no |
| soft_ew7_cip | year_2026 | +0.11% | 88% | PASS | no |
| soft_ew7_cip | holdout_365d | +0.05% | 67% | PASS | no |
| soft_ew_macro5 | year_2024 | +0.08% | 55% | PASS | no |
| soft_ew_macro5 | year_2025 | +0.04% | 64% | PASS | no |
| soft_ew_macro5 | year_2026 | +0.14% | 88% | PASS | no |
| soft_ew_macro5 | holdout_365d | +0.06% | 67% | PASS | no |
| soft_ew_cip2 | year_2024 | -0.04% | 45% | PASS | no |
| soft_ew_cip2 | year_2025 | +0.06% | 73% | PASS | no |
| soft_ew_cip2 | year_2026 | +0.04% | 75% | PASS | no |
| soft_ew_cip2 | holdout_365d | +0.03% | 75% | PASS | no |
| soft_ew6_no_haven | year_2024 | +0.06% | 45% | PASS | no |
| soft_ew6_no_haven | year_2025 | +0.06% | 73% | PASS | no |
| soft_ew6_no_haven | year_2026 | +0.13% | 88% | PASS | no |
| soft_ew6_no_haven | holdout_365d | +0.06% | 67% | PASS | no |
| soft_ew6_no_cip_ew | year_2024 | +0.07% | 55% | PASS | no |
| soft_ew6_no_cip_ew | year_2025 | +0.04% | 64% | PASS | no |
| soft_ew6_no_cip_ew | year_2026 | +0.11% | 88% | PASS | no |
| soft_ew6_no_cip_ew | holdout_365d | +0.05% | 67% | PASS | no |
| soft_ew_growth_cip | year_2024 | +0.05% | 64% | PASS | no |
| soft_ew_growth_cip | year_2025 | +0.12% | 91% | PASS | no |
| soft_ew_growth_cip | year_2026 | +0.14% | 75% | PASS | no |
| soft_ew_growth_cip | holdout_365d | +0.12% | 83% | PASS | no |
| soft_ew_honesty_cip | year_2024 | +0.00% | 45% | PASS | no |
| soft_ew_honesty_cip | year_2025 | -0.03% | 45% | PASS | no |
| soft_ew_honesty_cip | year_2026 | +0.03% | 75% | PASS | no |
| soft_ew_honesty_cip | holdout_365d | -0.05% | 50% | PASS | no |

**Board:** n=7 soft_nw_pos=7 hard_nw_pos=7 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_soft_signal_cip_stack_*.csv`, `scholarly_fx_soft_signal_cip_stack_meta.json`, `quest_locked_verify_soft_signal_cip_stack.md`.
