# Scholarly FX: global imbalances / current-account wave

**Path:** Gourinchas–Rey / Della Corte–Riddiough–Sarno external-adjustment / NFA–CA channel — **distinct** from PPP, BS, macro-diff, Hau–Rey equity.
**Data:** `approximate_non_ftmo` + free FRED IMF BOP `{ISO3}B6BLTT02STSAQ` (CA/GDP %). **PIT:** pub_lag_quarters=2 (=6m) + signal_lag=1m + 1d weight lag.
**Primary:** `ca_debtor_xs` (long deficit / short surplus). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| ca_debtor_xs | +0.020% | +0.26 | +0.33 | 52% | 11% | +0.02 |
| ca_surplus_xs | -0.027% | -0.35 | -0.45 | 47% | 10% | -0.04 |
| ca_chg_xs | -0.021% | -0.36 | -0.35 | 48% | 14% | -0.05 |
| us_ca_gr_fx | -0.021% | -0.50 | -0.58 | 14% | 28% | -0.14 |
| us_ca_haven_usd | +0.021% | +0.50 | +0.58 | 16% | 24% | +0.13 |
| ca_ew | -0.007% | -0.20 | -0.24 | 55% | 12% | -0.06 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| ca_debtor_xs | year_2024 | -0.02% | 64% | PASS | no |
| ca_debtor_xs | year_2025 | +0.07% | 64% | PASS | no |
| ca_debtor_xs | year_2026 | +0.05% | 62% | PASS | no |
| ca_debtor_xs | holdout_365d | +0.30% | 75% | PASS | no |
| ca_surplus_xs | year_2024 | +0.02% | 36% | PASS | no |
| ca_surplus_xs | year_2025 | -0.08% | 36% | PASS | no |
| ca_surplus_xs | year_2026 | -0.05% | 38% | PASS | no |
| ca_surplus_xs | holdout_365d | -0.30% | 25% | PASS | no |
| ca_chg_xs | year_2024 | -0.23% | 45% | PASS | no |
| ca_chg_xs | year_2025 | -0.28% | 36% | PASS | no |
| ca_chg_xs | year_2026 | +0.09% | 50% | PASS | no |
| ca_chg_xs | holdout_365d | +0.09% | 50% | PASS | no |
| us_ca_gr_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_ca_gr_fx | year_2025 | +0.29% | 55% | PASS | no |
| us_ca_gr_fx | year_2026 | -0.19% | 38% | PASS | no |
| us_ca_gr_fx | holdout_365d | -0.01% | 42% | PASS | no |
| us_ca_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_ca_haven_usd | year_2025 | -0.29% | 36% | PASS | no |
| us_ca_haven_usd | year_2026 | +0.19% | 62% | PASS | no |
| us_ca_haven_usd | holdout_365d | +0.01% | 58% | PASS | no |
| ca_ew | year_2024 | -0.08% | 45% | PASS | no |
| ca_ew | year_2025 | +0.03% | 64% | PASS | no |
| ca_ew | year_2026 | -0.02% | 50% | PASS | no |
| ca_ew | holdout_365d | +0.13% | 58% | PASS | no |

**Board:** n=6 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_ca_*.csv`, `scholarly_fx_ca_meta.json`.
