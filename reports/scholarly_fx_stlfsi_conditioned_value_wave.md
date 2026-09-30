# Scholarly FX: ANFCI-conditioned Rogoff PPP / real-FX value wave (§103)

**Path:** Chicago Fed **US ANFCI** (FRED ANFCI) stress × Rogoff (1996) PPP / real-FX value gate/cool — **distinct** from funding_liquidity (§20), NFCI value (§93), ANFCI soft (§95), ANFCI carry (§96), ANFCI mom (§97), WUI×PPP (§88), CIP×PPP (§83), VIX/GPR×PPP (§75), EPU×PPP (§80), raw PPP, BIS REER (§31), soft–carry–mom–value–REER stacks, capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED ANFCI + FRED CPI levels (pub_lag=1m). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** ANFCI pub_lag_days=7 → month-end + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 / NFCI value §93 / ANFCI soft–carry–mom §95–§97 mirror) + weight_lag_days=1; PPP lookback=60m ppp_signal_lag=1m + weight lag 1d. Costs 1.5 bps/side.
**Primary:** `value_low_stlfsi` (trade value only when lagged US ANFCI z ≤ 0). Companions: value_stlfsi_cool / value_raw / value_high_stlfsi / us_stlfsi_haven_usd / value_stlfsi_stack / value_stlfsi_ew / value_stlfsi_regime. Cool z_high=1.0 (§71–§97 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| value_low_stlfsi | -0.033% | -0.64 | -0.76 | 34% | 20% | -0.15 |
| value_stlfsi_cool | -0.032% | -0.58 | -0.59 | 42% | 14% | -0.13 |
| value_raw | -0.014% | -0.22 | -0.22 | 42% | 12% | -0.05 |
| value_high_stlfsi | +0.033% | +1.66 | +1.45 | 6% | 57% | +0.35 |
| us_stlfsi_haven_usd | -0.021% | -1.19 | -1.09 | 4% | 79% | -0.31 |
| value_stlfsi_stack | -0.033% | -0.64 | -0.76 | 34% | 20% | -0.15 |
| value_stlfsi_ew | -0.029% | -0.81 | -0.88 | 42% | 17% | -0.18 |
| value_stlfsi_regime | -0.054% | -0.99 | -1.11 | 37% | 18% | -0.23 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| value_raw | year_2024 | -0.10% | 64% | PASS | no |
| value_raw | year_2025 | -0.30% | 27% | PASS | no |
| value_raw | year_2026 | +0.15% | 75% | PASS | no |
| value_raw | holdout_365d | +0.10% | 67% | PASS | no |
| value_stlfsi_cool | year_2024 | -0.10% | 64% | PASS | no |
| value_stlfsi_cool | year_2025 | -0.25% | 27% | PASS | no |
| value_stlfsi_cool | year_2026 | +0.15% | 75% | PASS | no |
| value_stlfsi_cool | holdout_365d | +0.11% | 67% | PASS | no |
| value_low_stlfsi | year_2024 | -0.10% | 64% | PASS | no |
| value_low_stlfsi | year_2025 | -0.20% | 36% | PASS | no |
| value_low_stlfsi | year_2026 | +0.14% | 75% | PASS | no |
| value_low_stlfsi | holdout_365d | +0.12% | 67% | PASS | no |
| value_high_stlfsi | year_2024 | +0.00% | 0% | PASS | no |
| value_high_stlfsi | year_2025 | +0.00% | 0% | PASS | no |
| value_high_stlfsi | year_2026 | +0.03% | 12% | PASS | no |
| value_high_stlfsi | holdout_365d | -0.01% | 8% | PASS | no |
| us_stlfsi_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_stlfsi_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_stlfsi_haven_usd | year_2026 | +0.08% | 12% | PASS | no |
| us_stlfsi_haven_usd | holdout_365d | -0.09% | 8% | PASS | no |
| value_stlfsi_stack | year_2024 | -0.10% | 64% | PASS | no |
| value_stlfsi_stack | year_2025 | -0.20% | 36% | PASS | no |
| value_stlfsi_stack | year_2026 | +0.14% | 75% | PASS | no |
| value_stlfsi_stack | holdout_365d | +0.12% | 67% | PASS | no |
| value_stlfsi_ew | year_2024 | -0.07% | 64% | PASS | no |
| value_stlfsi_ew | year_2025 | -0.15% | 27% | PASS | no |
| value_stlfsi_ew | year_2026 | +0.12% | 88% | PASS | no |
| value_stlfsi_ew | holdout_365d | +0.05% | 75% | PASS | no |
| value_stlfsi_regime | year_2024 | -0.10% | 64% | PASS | no |
| value_stlfsi_regime | year_2025 | -0.20% | 36% | PASS | no |
| value_stlfsi_regime | year_2026 | +0.22% | 88% | PASS | no |
| value_stlfsi_regime | holdout_365d | +0.03% | 75% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US STLFSI4 [1994-01-07..2026-09-25]. CPI [1914-02-01..2026-09-01]. Gate z≤0 / cool z≥1 fixed (§71–§102 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). PPP lookback=60m (honesty 120m in raw ppp wave). AU/NZ CPI quarterly ffilled. Single-stress design (like CIP×PPP §83 / WUI×PPP §88 / ANFCI×PPP §98). Distinct from funding_liq §20 / ANFCI value §98 / STLFSI soft §100 / STLFSI carry §101 / STLFSI mom §102.

Artifacts: `reports/scholarly_fx_stlfsi_conditioned_value_*.csv`, `scholarly_fx_stlfsi_conditioned_value_meta.json`, `quest_locked_verify_stlfsi_conditioned_value.md`.
