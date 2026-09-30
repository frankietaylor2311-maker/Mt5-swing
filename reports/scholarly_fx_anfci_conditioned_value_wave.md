# Scholarly FX: ANFCI-conditioned Rogoff PPP / real-FX value wave (§98)

**Path:** Chicago Fed **US ANFCI** (FRED ANFCI) stress × Rogoff (1996) PPP / real-FX value gate/cool — **distinct** from funding_liquidity (§20), NFCI value (§93), ANFCI soft (§95), ANFCI carry (§96), ANFCI mom (§97), WUI×PPP (§88), CIP×PPP (§83), VIX/GPR×PPP (§75), EPU×PPP (§80), raw PPP, BIS REER (§31), soft–carry–mom–value–REER stacks, capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED ANFCI + FRED CPI levels (pub_lag=1m). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** ANFCI pub_lag_days=7 → month-end + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 / NFCI value §93 / ANFCI soft–carry–mom §95–§97 mirror) + weight_lag_days=1; PPP lookback=60m ppp_signal_lag=1m + weight lag 1d. Costs 1.5 bps/side.
**Primary:** `value_low_anfci` (trade value only when lagged US ANFCI z ≤ 0). Companions: value_anfci_cool / value_raw / value_high_anfci / us_anfci_haven_usd / value_anfci_stack / value_anfci_ew / value_anfci_regime. Cool z_high=1.0 (§71–§97 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| value_low_anfci | -0.038% | -0.78 | -0.79 | 27% | 23% | -0.18 |
| value_anfci_cool | -0.022% | -0.41 | -0.43 | 41% | 15% | -0.09 |
| value_raw | -0.014% | -0.22 | -0.22 | 42% | 12% | -0.05 |
| value_high_anfci | +0.025% | +0.99 | +0.87 | 8% | 44% | +0.21 |
| us_anfci_haven_usd | -0.003% | -0.10 | -0.09 | 6% | 43% | -0.02 |
| value_anfci_stack | -0.038% | -0.78 | -0.79 | 27% | 23% | -0.18 |
| value_anfci_ew | -0.021% | -0.60 | -0.61 | 41% | 16% | -0.14 |
| value_anfci_regime | -0.041% | -0.71 | -0.69 | 32% | 16% | -0.16 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| value_raw | year_2024 | -0.10% | 64% | PASS | no |
| value_raw | year_2025 | -0.30% | 27% | PASS | no |
| value_raw | year_2026 | +0.15% | 75% | PASS | no |
| value_raw | holdout_365d | +0.10% | 67% | PASS | no |
| value_anfci_cool | year_2024 | -0.10% | 64% | PASS | no |
| value_anfci_cool | year_2025 | -0.29% | 27% | PASS | no |
| value_anfci_cool | year_2026 | +0.15% | 75% | PASS | no |
| value_anfci_cool | holdout_365d | +0.10% | 67% | PASS | no |
| value_low_anfci | year_2024 | -0.13% | 64% | PASS | no |
| value_low_anfci | year_2025 | -0.20% | 36% | PASS | no |
| value_low_anfci | year_2026 | +0.15% | 75% | PASS | no |
| value_low_anfci | holdout_365d | +0.10% | 67% | PASS | no |
| value_high_anfci | year_2024 | +0.00% | 0% | PASS | no |
| value_high_anfci | year_2025 | +0.00% | 0% | PASS | no |
| value_high_anfci | year_2026 | +0.00% | 0% | PASS | no |
| value_high_anfci | holdout_365d | +0.00% | 0% | PASS | no |
| us_anfci_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_anfci_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_anfci_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_anfci_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| value_anfci_stack | year_2024 | -0.13% | 64% | PASS | no |
| value_anfci_stack | year_2025 | -0.20% | 36% | PASS | no |
| value_anfci_stack | year_2026 | +0.15% | 75% | PASS | no |
| value_anfci_stack | holdout_365d | +0.10% | 67% | PASS | no |
| value_anfci_ew | year_2024 | -0.08% | 64% | PASS | no |
| value_anfci_ew | year_2025 | -0.16% | 27% | PASS | no |
| value_anfci_ew | year_2026 | +0.10% | 75% | PASS | no |
| value_anfci_ew | holdout_365d | +0.06% | 67% | PASS | no |
| value_anfci_regime | year_2024 | -0.13% | 64% | PASS | no |
| value_anfci_regime | year_2025 | -0.20% | 36% | PASS | no |
| value_anfci_regime | year_2026 | +0.15% | 75% | PASS | no |
| value_anfci_regime | holdout_365d | +0.10% | 67% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US ANFCI [1971-01-15..2026-09-18]. CPI [1914-02-01..2026-09-01]. Gate z≤0 / cool z≥1 fixed (§71–§97 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). PPP lookback=60m (honesty 120m in raw ppp wave). AU/NZ CPI quarterly ffilled. Single-stress design (like CIP×PPP §83 / WUI×PPP §88 / NFCI×mom §92). Distinct from funding_liq §20 / ANFCI soft §95 / ANFCI carry §96 / ANFCI mom §97.

Artifacts: `reports/scholarly_fx_anfci_conditioned_value_*.csv`, `scholarly_fx_anfci_conditioned_value_meta.json`, `quest_locked_verify_anfci_conditioned_value.md`.
