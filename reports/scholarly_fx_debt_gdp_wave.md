# Scholarly FX: government debt/GDP wave (§29)

**Path:** Debt overhang / fiscal sustainability — IMF WEO general-government gross debt (% GDP) via FRED — **distinct** from fiscal balance GGNLBA (§28), CA (§21), CB-BS, macro-diff, equity-diff, AI-GPR, ToT, dollar-beta, crash-skew.
**Data:** `approximate_non_ftmo` + free FRED `GGGDTA*188N` (brief `GGXWDG*188N` **404 on FRED** — documenting honest alias failure) + US quarterly `GFDEGDQ188S` for tilts. **PIT:** pub_lag_months=15 (~April Y+1 for year-Y) + signal_lag=1m + 1d weight lag; US-Q pub_lag=4m.
**Primary:** `low_debt_xs` (long low debt / short high debt). NZD/CHF unmapped on free FRED. Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| low_debt_xs | +0.019% | +0.27 | +0.27 | 53% | 12% | +0.03 |
| high_debt_xs | -0.020% | -0.28 | -0.27 | 47% | 12% | -0.03 |
| debt_chg_xs | +0.014% | +0.26 | +0.27 | 52% | 11% | +0.02 |
| us_debt_twin_fx | -0.024% | -0.48 | -0.58 | 27% | 18% | -0.17 |
| us_debt_haven_usd | +0.023% | +0.46 | +0.56 | 28% | 19% | +0.17 |
| debt_fiscal_blend | -0.008% | -0.15 | -0.15 | 54% | 13% | -0.09 |
| debt_ew | +0.004% | +0.09 | +0.10 | 50% | 14% | -0.05 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| low_debt_xs | year_2024 | -0.04% | 64% | PASS | no |
| low_debt_xs | year_2025 | +0.32% | 73% | PASS | no |
| low_debt_xs | year_2026 | +0.06% | 50% | PASS | no |
| low_debt_xs | holdout_365d | +0.23% | 67% | PASS | no |
| high_debt_xs | year_2024 | +0.04% | 36% | PASS | no |
| high_debt_xs | year_2025 | -0.32% | 27% | PASS | no |
| high_debt_xs | year_2026 | -0.06% | 50% | PASS | no |
| high_debt_xs | holdout_365d | -0.24% | 33% | PASS | no |
| debt_chg_xs | year_2024 | -0.05% | 36% | PASS | no |
| debt_chg_xs | year_2025 | -0.08% | 55% | PASS | no |
| debt_chg_xs | year_2026 | +0.02% | 50% | PASS | no |
| debt_chg_xs | holdout_365d | -0.14% | 33% | PASS | no |
| us_debt_twin_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_debt_twin_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_debt_twin_fx | year_2026 | +0.02% | 38% | PASS | no |
| us_debt_twin_fx | holdout_365d | +0.01% | 25% | PASS | no |
| us_debt_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_debt_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_debt_haven_usd | year_2026 | -0.02% | 38% | PASS | no |
| us_debt_haven_usd | holdout_365d | -0.01% | 25% | PASS | no |
| debt_fiscal_blend | year_2024 | +0.01% | 73% | PASS | no |
| debt_fiscal_blend | year_2025 | +0.07% | 55% | PASS | no |
| debt_fiscal_blend | year_2026 | +0.12% | 62% | PASS | no |
| debt_fiscal_blend | holdout_365d | +0.15% | 67% | PASS | no |
| debt_ew | year_2024 | -0.03% | 36% | PASS | no |
| debt_ew | year_2025 | +0.08% | 64% | PASS | no |
| debt_ew | year_2026 | +0.04% | 62% | PASS | no |
| debt_ew | holdout_365d | +0.04% | 58% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_debt_gdp_*.csv`, `scholarly_fx_debt_gdp_meta.json`.

## Primary `low_debt_xs` consistency (detail)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| year_2024 | −0.04% | 64% | 68% | PASS | no |
| year_2025 | +0.32% | 73% | 60% | PASS | no |
| year_2026 | +0.06% | 50% | 81% | PASS | no |
| holdout_365d | +0.23% | 67% | 54% | PASS | no |

## Risk sweep

Sweep run (positive IS mean on some legs). Primary scale≈0.668 bind=static; scaled HO mean≈0.16%/mo %pos≈67% — **does not** clear 1% bar. Soft/hard NW boards empty.

## FRED mnemonic note

Brief suggested `GGXWDG*188N` (IMF `GGXWDG_NGDP`); those IDs **404 on FRED**. Live actuals are `GGGDTA*188N` (same WEO release / country codes as fiscal `GGNLBA*`). NZD/CHF still unmapped. US tilt uses quarterly `GFDEGDQ188S`.

## Locked sleeve

`fx4plus_gbpcad_d1_voltarget_0025` **untouched**; re-verify PASS (see `quest_locked_verify_debt_gdp.md`).

**Suggested next:** monthly trade-balance (higher-freq vs quarterly CA) or BIS/FRED REER misalignment — not a locked-sleeve cooler.
