# Scholarly FX: NFCI-conditioned Rogoff PPP / real-FX value wave (§93)

**Path:** Chicago Fed **US NFCI** (FRED NFCI) stress × Rogoff (1996) PPP / real-FX value gate/cool — **distinct** from funding_liquidity (§20), NFCI soft (§90), NFCI carry (§91), NFCI mom (§92), WUI×PPP (§88), CIP×PPP (§83), VIX/GPR×PPP (§75), EPU×PPP (§80), raw PPP, BIS REER (§31), soft–carry–mom–value–REER stacks, capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED NFCI + FRED CPI levels (pub_lag=1m). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** NFCI pub_lag_days=7 → month-end + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 / NFCI soft §90 / NFCI carry §91 / NFCI mom §92 mirror) + weight_lag_days=1; PPP lookback=60m ppp_signal_lag=1m + weight lag 1d. Costs 1.5 bps/side.
**Primary:** `value_low_nfci` (trade value only when lagged US NFCI z ≤ 0). Companions: value_nfci_cool / value_raw / value_high_nfci / us_nfci_haven_usd / value_nfci_stack / value_nfci_ew / value_nfci_regime. Cool z_high=1.0 (§71–§92 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| value_low_nfci | -0.040% | -0.85 | -0.77 | 28% | 19% | -0.19 |
| value_nfci_cool | -0.033% | -0.62 | -0.61 | 42% | 13% | -0.14 |
| value_raw | -0.014% | -0.22 | -0.22 | 42% | 12% | -0.05 |
| value_high_nfci | +0.031% | +1.23 | +1.00 | 7% | 43% | +0.27 |
| us_nfci_haven_usd | -0.008% | -0.26 | -0.25 | 6% | 47% | -0.07 |
| value_nfci_stack | -0.040% | -0.85 | -0.77 | 28% | 19% | -0.19 |
| value_nfci_ew | -0.027% | -0.79 | -0.74 | 41% | 14% | -0.18 |
| value_nfci_regime | -0.048% | -0.85 | -0.75 | 33% | 15% | -0.20 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| value_raw | year_2024 | -0.10% | 64% | PASS | no |
| value_raw | year_2025 | -0.30% | 27% | PASS | no |
| value_raw | year_2026 | +0.15% | 75% | PASS | no |
| value_raw | holdout_365d | +0.10% | 67% | PASS | no |
| value_nfci_cool | year_2024 | -0.10% | 64% | PASS | no |
| value_nfci_cool | year_2025 | -0.29% | 27% | PASS | no |
| value_nfci_cool | year_2026 | +0.15% | 75% | PASS | no |
| value_nfci_cool | holdout_365d | +0.10% | 67% | PASS | no |
| value_low_nfci | year_2024 | -0.17% | 36% | PASS | no |
| value_low_nfci | year_2025 | -0.20% | 36% | PASS | no |
| value_low_nfci | year_2026 | +0.15% | 75% | PASS | no |
| value_low_nfci | holdout_365d | +0.10% | 67% | PASS | no |
| value_high_nfci | year_2024 | +0.00% | 0% | PASS | no |
| value_high_nfci | year_2025 | +0.00% | 0% | PASS | no |
| value_high_nfci | year_2026 | +0.00% | 0% | PASS | no |
| value_high_nfci | holdout_365d | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_nfci_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| value_nfci_stack | year_2024 | -0.17% | 36% | PASS | no |
| value_nfci_stack | year_2025 | -0.20% | 36% | PASS | no |
| value_nfci_stack | year_2026 | +0.15% | 75% | PASS | no |
| value_nfci_stack | holdout_365d | +0.10% | 67% | PASS | no |
| value_nfci_ew | year_2024 | -0.09% | 55% | PASS | no |
| value_nfci_ew | year_2025 | -0.16% | 27% | PASS | no |
| value_nfci_ew | year_2026 | +0.10% | 75% | PASS | no |
| value_nfci_ew | holdout_365d | +0.06% | 67% | PASS | no |
| value_nfci_regime | year_2024 | -0.17% | 36% | PASS | no |
| value_nfci_regime | year_2025 | -0.20% | 36% | PASS | no |
| value_nfci_regime | year_2026 | +0.15% | 75% | PASS | no |
| value_nfci_regime | holdout_365d | +0.10% | 67% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US NFCI [1971-01-15..2026-09-18]. CPI [1914-02-01..2026-09-01]. Gate z≤0 / cool z≥1 fixed (§71–§92 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). PPP lookback=60m (honesty 120m in raw ppp wave). AU/NZ CPI quarterly ffilled. Single-stress design (like CIP×PPP §83 / WUI×PPP §88 / NFCI×mom §92). Distinct from funding_liq §20 / NFCI soft §90 / NFCI carry §91 / NFCI mom §92.

Artifacts: `reports/scholarly_fx_nfci_conditioned_value_*.csv`, `scholarly_fx_nfci_conditioned_value_meta.json`, `quest_locked_verify_nfci_conditioned_value.md`.
