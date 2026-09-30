# Scholarly FX: KCFSI-conditioned Rogoff PPP / real-FX value wave (§108)

**Path:** Kansas City Fed **US KCFSI** (FRED KCFSI; Financial Stress Index) stress × Rogoff (1996) PPP / real-FX value gate/cool — **distinct** from funding_liquidity (§20), NFCI value (§93), ANFCI value (§98), STLFSI value (§103), KCFSI soft (§105), KCFSI carry (§106), KCFSI mom (§107), WUI×PPP (§88), CIP×PPP (§83), VIX/GPR×PPP (§75), EPU×PPP (§80), raw PPP, BIS REER (§31), soft–carry–mom–value–REER stacks, capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED KCFSI + FRED CPI levels (pub_lag=1m). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** KCFSI pub_lag_months=1 + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 / NFCI value §93 / ANFCI value §98 / STLFSI value §103 / KCFSI soft–carry–mom §105–§107 mirror) + weight_lag_days=1; PPP lookback=60m ppp_signal_lag=1m + weight lag 1d. Costs 1.5 bps/side.
**Primary:** `value_low_kcfsi` (trade value only when lagged US KCFSI z ≤ 0). Companions: value_kcfsi_cool / value_raw / value_high_kcfsi / us_kcfsi_haven_usd / value_kcfsi_stack / value_kcfsi_ew / value_kcfsi_regime. Cool z_high=1.0 (§71–§107 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| value_low_kcfsi | -0.056% | -1.15 | -1.18 | 29% | 22% | -0.26 |
| value_kcfsi_cool | -0.028% | -0.50 | -0.51 | 41% | 14% | -0.11 |
| value_raw | -0.014% | -0.22 | -0.22 | 42% | 12% | -0.05 |
| value_high_kcfsi | +0.010% | +0.73 | +0.95 | 4% | 72% | +0.12 |
| us_kcfsi_haven_usd | -0.012% | -0.75 | -0.80 | 3% | 88% | -0.19 |
| value_kcfsi_stack | -0.056% | -1.15 | -1.18 | 29% | 22% | -0.26 |
| value_kcfsi_ew | -0.032% | -0.94 | -1.00 | 41% | 17% | -0.21 |
| value_kcfsi_regime | -0.069% | -1.33 | -1.37 | 32% | 20% | -0.30 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| value_raw | year_2024 | -0.10% | 64% | PASS | no |
| value_raw | year_2025 | -0.30% | 27% | PASS | no |
| value_raw | year_2026 | +0.15% | 75% | PASS | no |
| value_raw | holdout_365d | +0.10% | 67% | PASS | no |
| value_kcfsi_cool | year_2024 | -0.09% | 64% | PASS | no |
| value_kcfsi_cool | year_2025 | -0.30% | 18% | PASS | no |
| value_kcfsi_cool | year_2026 | +0.15% | 75% | PASS | no |
| value_kcfsi_cool | holdout_365d | +0.10% | 67% | PASS | no |
| value_low_kcfsi | year_2024 | -0.08% | 64% | PASS | no |
| value_low_kcfsi | year_2025 | -0.30% | 18% | PASS | no |
| value_low_kcfsi | year_2026 | +0.15% | 75% | PASS | no |
| value_low_kcfsi | holdout_365d | +0.10% | 67% | PASS | no |
| value_high_kcfsi | year_2024 | +0.00% | 0% | PASS | no |
| value_high_kcfsi | year_2025 | +0.00% | 0% | PASS | no |
| value_high_kcfsi | year_2026 | +0.00% | 0% | PASS | no |
| value_high_kcfsi | holdout_365d | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_kcfsi_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| value_kcfsi_stack | year_2024 | -0.08% | 64% | PASS | no |
| value_kcfsi_stack | year_2025 | -0.30% | 18% | PASS | no |
| value_kcfsi_stack | year_2026 | +0.15% | 75% | PASS | no |
| value_kcfsi_stack | holdout_365d | +0.10% | 67% | PASS | no |
| value_kcfsi_ew | year_2024 | -0.06% | 64% | PASS | no |
| value_kcfsi_ew | year_2025 | -0.20% | 18% | PASS | no |
| value_kcfsi_ew | year_2026 | +0.10% | 75% | PASS | no |
| value_kcfsi_ew | holdout_365d | +0.06% | 67% | PASS | no |
| value_kcfsi_regime | year_2024 | -0.08% | 64% | PASS | no |
| value_kcfsi_regime | year_2025 | -0.30% | 18% | PASS | no |
| value_kcfsi_regime | year_2026 | +0.15% | 75% | PASS | no |
| value_kcfsi_regime | holdout_365d | +0.10% | 67% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US KCFSI [1990-03-01..2026-09-01]. CPI [1914-02-01..2026-09-01]. Gate z≤0 / cool z≥1 fixed (§71–§107 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). PPP lookback=60m (honesty 120m in raw ppp wave). AU/NZ CPI quarterly ffilled. Single-stress design (like CIP×PPP §83 / WUI×PPP §88 / STLFSI×PPP §103 / ANFCI×PPP §98). Distinct from funding_liq §20 / STLFSI value §103 / ANFCI value §98 / KCFSI soft §105 / KCFSI carry §106 / KCFSI mom §107.

Artifacts: `reports/scholarly_fx_kcfsi_conditioned_value_*.csv`, `scholarly_fx_kcfsi_conditioned_value_meta.json`, `quest_locked_verify_kcfsi_conditioned_value.md`.
