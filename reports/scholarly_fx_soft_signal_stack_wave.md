# Scholarly FX: Dahlquist-style soft-signal EW recombination (§66)

**Path:** Equal-weight of pre-registered soft-boarded scholarly XS daily returns — **distinct** from combo (§8), capital-sleeve mix (§53), macro_diff blend, and single-series BCI/IP/PPI/GDP/cars waves.
**Data:** `approximate_non_ftmo` + free FRED/OECD panels rebuilt with source-wave PIT lags. **SOFT_LEGS (a priori):** `['bci_chg_xs', 'high_ip_xs', 'high_ppi_xs', 'low_gdp_xs', 'low_cars_xs']`. Row-wise nanmean; require ≥2 legs/day. Costs 1.5 bps/side (inside source factors).
**Primary:** `soft_ew5` (EW of all 5 soft legs). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| soft_ew5 | +0.118% | +4.26 | +3.63 | 61% | 11% | +1.02 |
| soft_ew_growth | +0.123% | +3.28 | +2.93 | 61% | 10% | +0.75 |
| soft_ew_honesty | +0.111% | +2.60 | +2.64 | 57% | 14% | +0.64 |
| soft_ew4_no_bci | +0.111% | +3.90 | +3.44 | 62% | 12% | +0.91 |
| soft_ew4_no_ip | +0.122% | +3.68 | +3.41 | 61% | 10% | +0.88 |
| soft_ew4_no_ppi | +0.118% | +3.84 | +3.57 | 59% | 11% | +0.90 |
| soft_ew3_core | +0.123% | +3.28 | +2.93 | 61% | 10% | +0.75 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| soft_ew5 | year_2024 | +0.08% | 55% | PASS | no |
| soft_ew5 | year_2025 | +0.04% | 64% | PASS | no |
| soft_ew5 | year_2026 | +0.14% | 88% | PASS | no |
| soft_ew5 | holdout_365d | +0.06% | 67% | PASS | no |
| soft_ew_growth | year_2024 | +0.11% | 64% | PASS | no |
| soft_ew_growth | year_2025 | +0.16% | 91% | PASS | no |
| soft_ew_growth | year_2026 | +0.21% | 62% | PASS | no |
| soft_ew_growth | holdout_365d | +0.19% | 75% | PASS | no |
| soft_ew_honesty | year_2024 | +0.05% | 45% | PASS | no |
| soft_ew_honesty | year_2025 | -0.13% | 45% | PASS | no |
| soft_ew_honesty | year_2026 | +0.03% | 75% | PASS | no |
| soft_ew_honesty | holdout_365d | -0.13% | 50% | PASS | no |
| soft_ew4_no_bci | year_2024 | +0.05% | 64% | PASS | no |
| soft_ew4_no_bci | year_2025 | -0.03% | 45% | PASS | no |
| soft_ew4_no_bci | year_2026 | +0.05% | 50% | PASS | no |
| soft_ew4_no_bci | holdout_365d | +0.02% | 50% | PASS | no |
| soft_ew4_no_ip | year_2024 | +0.09% | 55% | PASS | no |
| soft_ew4_no_ip | year_2025 | +0.01% | 55% | PASS | no |
| soft_ew4_no_ip | year_2026 | +0.14% | 75% | PASS | no |
| soft_ew4_no_ip | holdout_365d | +0.04% | 58% | PASS | no |
| soft_ew4_no_ppi | year_2024 | +0.09% | 36% | PASS | no |
| soft_ew4_no_ppi | year_2025 | +0.05% | 55% | PASS | no |
| soft_ew4_no_ppi | year_2026 | +0.16% | 88% | PASS | no |
| soft_ew4_no_ppi | holdout_365d | +0.04% | 58% | PASS | no |
| soft_ew3_core | year_2024 | +0.11% | 64% | PASS | no |
| soft_ew3_core | year_2025 | +0.16% | 91% | PASS | no |
| soft_ew3_core | year_2026 | +0.21% | 62% | PASS | no |
| soft_ew3_core | holdout_365d | +0.19% | 75% | PASS | no |

**Board:** n=7 soft_nw_pos=7 hard_nw_pos=7 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_soft_signal_stack_*.csv`, `scholarly_fx_soft_signal_stack_meta.json`.
