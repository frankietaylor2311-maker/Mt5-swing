# Scholarly FX: OECD/FRED manufacturing wage/earnings differential wave (§55)

**Path:** FRED OECD MEI **manufacturing earnings YoY** (`LCEAMN01*`; CHF unmapped 404) — wage YoY XS — **distinct** from ULC (§48), employment persons (§41), UR levels (§54), LP (§49), CU (§50), CPI (§52), PPI (§51).
**Data:** `approximate_non_ftmo` + free FRED LCEAMN YoY after pub_lag. EUR EA EZQ657S ends ~2025-07; CHF unmapped; GBP/JPY/AUD index→YoY. `n_long=n_short=2` on available foreign panel. **PIT:** pub_lag_months=2 (earnings lag UR/CPI a priori) + signal_lag_months=1 + 1d weight lag. Score basis = **wage YoY %**.
**Primary:** `low_wage_growth_xs` (long low relative wage growth / short high — competitiveness → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| low_wage_growth_xs | -0.070% | -1.22 | -1.19 | 46% | 13% | -0.23 |
| high_wage_growth_xs | +0.046% | +0.80 | +0.79 | 53% | 9% | +0.14 |
| low_wage_z_xs | -0.065% | -1.08 | -1.02 | 47% | 14% | -0.21 |
| wage_chg_xs | -0.021% | -0.32 | -0.36 | 49% | 10% | -0.04 |
| us_wage_stress_fx | -0.047% | -1.46 | -1.59 | 16% | 36% | -0.37 |
| us_wage_haven_usd | +0.042% | +1.30 | +1.43 | 20% | 28% | +0.33 |
| wage_ew | -0.046% | -1.26 | -1.29 | 48% | 13% | -0.26 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| low_wage_growth_xs | year_2024 | +0.11% | 36% | PASS | no |
| low_wage_growth_xs | year_2025 | +0.11% | 45% | PASS | no |
| low_wage_growth_xs | year_2026 | -0.13% | 62% | PASS | no |
| low_wage_growth_xs | holdout_365d | -0.03% | 50% | PASS | no |
| high_wage_growth_xs | year_2024 | -0.13% | 64% | PASS | no |
| high_wage_growth_xs | year_2025 | -0.13% | 55% | PASS | no |
| high_wage_growth_xs | year_2026 | +0.13% | 38% | PASS | no |
| high_wage_growth_xs | holdout_365d | +0.03% | 50% | PASS | no |
| low_wage_z_xs | year_2024 | -0.10% | 45% | PASS | no |
| low_wage_z_xs | year_2025 | +0.17% | 73% | PASS | no |
| low_wage_z_xs | year_2026 | -0.02% | 50% | PASS | no |
| low_wage_z_xs | holdout_365d | +0.05% | 58% | PASS | no |
| wage_chg_xs | year_2024 | +0.01% | 45% | PASS | no |
| wage_chg_xs | year_2025 | +0.18% | 73% | PASS | no |
| wage_chg_xs | year_2026 | +0.14% | 62% | PASS | no |
| wage_chg_xs | holdout_365d | +0.19% | 67% | PASS | no |
| us_wage_stress_fx | year_2024 | -0.08% | 9% | PASS | no |
| us_wage_stress_fx | year_2025 | -0.06% | 18% | PASS | no |
| us_wage_stress_fx | year_2026 | -0.17% | 0% | PASS | no |
| us_wage_stress_fx | holdout_365d | -0.12% | 0% | PASS | no |
| us_wage_haven_usd | year_2024 | +0.07% | 18% | PASS | no |
| us_wage_haven_usd | year_2025 | +0.06% | 9% | PASS | no |
| us_wage_haven_usd | year_2026 | +0.17% | 25% | PASS | no |
| us_wage_haven_usd | holdout_365d | +0.11% | 17% | PASS | no |
| wage_ew | year_2024 | +0.02% | 45% | PASS | no |
| wage_ew | year_2025 | +0.08% | 73% | PASS | no |
| wage_ew | year_2026 | -0.06% | 50% | PASS | no |
| wage_ew | holdout_365d | +0.01% | 58% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_wage_earnings_*.csv`, `scholarly_fx_wage_earnings_meta.json`.

Locked `fx4plus_gbpcad_d1_voltarget_0025` **untouched PASS** (see `quest_locked_verify_wage_earnings.md`).
Soft-best-ish: `high_wage_growth_xs` (~+4.6 bp/mo NW t +0.79); `us_wage_haven_usd` (~+4.2 bp/mo NW t +1.43) — below soft |t|≥1.5.
Risk sweep primary scale≈0.801 bind=static; scaled HO clear **NO**.
EUR `LCEAMN01EZQ657S` ends ~2025-07 (PIT known ~2025-09 after pub_lag=2); CHF unmapped 404; GBP/JPY/AUD index→YoY.
