# Scholarly FX: monetary / money-growth differential wave (§33)

**Path:** OECD broad-money growth via FRED `MABMM301*M657S` — Frenkel–Bilson / sticky-price monetary XS — **distinct** from CB-BS/QE (§22), real-rate (§23), debt (§29), fiscal (§28), TB (§30), BIS REER (§31), BIS credit (§32), CA, funding-liq, macro-diff, equity-diff, AI-GPR, ToT, dollar-beta, crash-skew.
**Data:** `approximate_non_ftmo` + free FRED `MABMM301*M657S` (OECD MEI measure 657). EUR = `MABMM301EZM657S`. NZD/CHF **unmapped** on primary (657S ends 2018). US `M2SL` level documented alt. **PIT:** pub_lag_months=2 (conservative monthly money) + signal_lag=1m + 1d weight lag.
**Primary:** `low_money_growth_xs` (long low relative money growth / short high — Frenkel–Bilson tightness → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| low_money_growth_xs | +0.034% | +0.55 | +0.59 | 52% | 11% | +0.12 |
| high_money_growth_xs | -0.058% | -0.92 | -0.99 | 46% | 12% | -0.21 |
| low_money_growth_z_xs | -0.003% | -0.04 | -0.04 | 46% | 13% | -0.06 |
| money_growth_chg_xs | +0.041% | +0.63 | +0.65 | 48% | 12% | +0.12 |
| us_money_stress_fx | -0.021% | -0.96 | -0.98 | 6% | 65% | -0.24 |
| us_money_haven_usd | +0.020% | +0.88 | +0.91 | 8% | 51% | +0.22 |
| money_ew | +0.018% | +0.48 | +0.54 | 48% | 12% | +0.10 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| low_money_growth_xs | year_2024 | +0.17% | 64% | PASS | no |
| low_money_growth_xs | year_2025 | -0.33% | 27% | PASS | no |
| low_money_growth_xs | year_2026 | +0.05% | 50% | PASS | no |
| low_money_growth_xs | holdout_365d | -0.11% | 42% | PASS | no |
| high_money_growth_xs | year_2024 | -0.19% | 36% | PASS | no |
| high_money_growth_xs | year_2025 | +0.30% | 73% | PASS | no |
| high_money_growth_xs | year_2026 | -0.07% | 50% | PASS | no |
| high_money_growth_xs | holdout_365d | +0.09% | 58% | PASS | no |
| low_money_growth_z_xs | year_2024 | -0.02% | 55% | PASS | no |
| low_money_growth_z_xs | year_2025 | -0.27% | 27% | PASS | no |
| low_money_growth_z_xs | year_2026 | +0.03% | 38% | PASS | no |
| low_money_growth_z_xs | holdout_365d | -0.05% | 33% | PASS | no |
| money_growth_chg_xs | year_2024 | +0.05% | 45% | PASS | no |
| money_growth_chg_xs | year_2025 | -0.40% | 27% | PASS | no |
| money_growth_chg_xs | year_2026 | -0.07% | 38% | PASS | no |
| money_growth_chg_xs | holdout_365d | -0.03% | 33% | PASS | no |
| us_money_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_money_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_money_stress_fx | year_2026 | -0.19% | 0% | PASS | no |
| us_money_stress_fx | holdout_365d | -0.13% | 0% | PASS | no |
| us_money_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_money_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_money_haven_usd | year_2026 | +0.19% | 38% | PASS | no |
| us_money_haven_usd | holdout_365d | +0.12% | 25% | PASS | no |
| money_ew | year_2024 | +0.07% | 55% | PASS | no |
| money_ew | year_2025 | -0.24% | 27% | PASS | no |
| money_ew | year_2026 | -0.07% | 25% | PASS | no |
| money_ew | holdout_365d | -0.09% | 25% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_money_growth_*.csv`, `scholarly_fx_money_growth_meta.json`.
