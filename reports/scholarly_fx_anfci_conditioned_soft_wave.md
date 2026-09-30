# Scholarly FX: ANFCI-conditioned soft-signal EW wave (§95)

**Path:** Chicago Fed **US ANFCI** (FRED ANFCI) stress × Dahlquist–Hasseltoft ``soft_ew_macro5`` gate/cool — **distinct** from funding_liquidity (§20 USD tilts / carry×NFCI), soft_ew (§66 ungated), soft-stack CIP enrichment (§69), CIP-conditioned soft (§71), VIX/GPR soft (§72), EPU/TPU soft (§76), WUI soft (§85), CIP/VIX/EPU/WUI × carry/mom/value/REER (§68/§73–§89), capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED ANFCI + FRED/OECD soft legs (§66 PIT). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** ANFCI pub_lag_days=7 → month-end + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 / NFCI soft §90 mirror) + weight_lag_days=1; soft legs source-wave pub/signal lags. Costs 1.5 bps/side inside source factors (haven applies own costs).
**Primary:** `soft_low_anfci` (trade soft only when lagged US ANFCI z ≤ 0). Companions: soft_anfci_cool / soft_raw / soft_high_anfci / us_anfci_haven_usd / soft_anfci_stack / soft_anfci_ew / soft_anfci_regime. Cool z_high=1.0 (§71–§94 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| soft_low_anfci | +0.082% | +3.66 | +2.84 | 44% | 16% | +0.89 |
| soft_anfci_cool | +0.102% | +4.23 | +3.34 | 62% | 12% | +1.01 |
| soft_raw | +0.118% | +4.26 | +3.63 | 61% | 11% | +1.02 |
| soft_high_anfci | +0.013% | +1.43 | +1.35 | 9% | 42% | +0.31 |
| us_anfci_haven_usd | -0.003% | -0.10 | -0.09 | 6% | 43% | -0.02 |
| soft_anfci_stack | +0.082% | +3.66 | +2.84 | 44% | 16% | +0.89 |
| soft_anfci_ew | +0.061% | +3.30 | +2.69 | 59% | 12% | +0.79 |
| soft_anfci_regime | +0.080% | +2.12 | +1.79 | 49% | 14% | +0.51 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| soft_raw | year_2024 | +0.08% | 55% | PASS | no |
| soft_raw | year_2025 | +0.04% | 64% | PASS | no |
| soft_raw | year_2026 | +0.14% | 88% | PASS | no |
| soft_raw | holdout_365d | +0.06% | 67% | PASS | no |
| soft_anfci_cool | year_2024 | +0.08% | 55% | PASS | no |
| soft_anfci_cool | year_2025 | +0.04% | 64% | PASS | no |
| soft_anfci_cool | year_2026 | +0.14% | 88% | PASS | no |
| soft_anfci_cool | holdout_365d | +0.06% | 67% | PASS | no |
| soft_low_anfci | year_2024 | +0.04% | 36% | PASS | no |
| soft_low_anfci | year_2025 | +0.02% | 64% | PASS | no |
| soft_low_anfci | year_2026 | +0.14% | 88% | PASS | no |
| soft_low_anfci | holdout_365d | +0.06% | 67% | PASS | no |
| soft_high_anfci | year_2024 | +0.00% | 0% | PASS | no |
| soft_high_anfci | year_2025 | +0.00% | 0% | PASS | no |
| soft_high_anfci | year_2026 | +0.00% | 0% | PASS | no |
| soft_high_anfci | holdout_365d | +0.00% | 0% | PASS | no |
| us_anfci_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_anfci_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_anfci_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_anfci_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| soft_anfci_stack | year_2024 | +0.04% | 36% | PASS | no |
| soft_anfci_stack | year_2025 | +0.02% | 64% | PASS | no |
| soft_anfci_stack | year_2026 | +0.14% | 88% | PASS | no |
| soft_anfci_stack | holdout_365d | +0.06% | 67% | PASS | no |
| soft_anfci_ew | year_2024 | +0.04% | 45% | PASS | no |
| soft_anfci_ew | year_2025 | +0.02% | 64% | PASS | no |
| soft_anfci_ew | year_2026 | +0.09% | 88% | PASS | no |
| soft_anfci_ew | holdout_365d | +0.04% | 67% | PASS | no |
| soft_anfci_regime | year_2024 | +0.04% | 36% | PASS | no |
| soft_anfci_regime | year_2025 | +0.02% | 64% | PASS | no |
| soft_anfci_regime | year_2026 | +0.14% | 88% | PASS | no |
| soft_anfci_regime | holdout_365d | +0.06% | 67% | PASS | no |

**Board:** n=8 soft_nw_pos=6 hard_nw_pos=5 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US ANFCI [1971-01-15..2026-09-18]. Weekly→month-end. Gate z≤0 / cool z≥1 fixed (§71–§94 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). Single-stress design (like CIP soft §71 / WUI soft §85). Distinct from funding_liq §20.

Artifacts: `reports/scholarly_fx_anfci_conditioned_soft_*.csv`, `scholarly_fx_anfci_conditioned_soft_meta.json`, `quest_locked_verify_anfci_conditioned_soft.md`.
