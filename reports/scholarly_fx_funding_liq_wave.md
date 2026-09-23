# Scholarly FX: funding-liquidity / financial-conditions wave

**Path:** Brunnermeier–Nagel–Pedersen / Menkhoff funding-liquidity channel (NFCI/ANFCI + TED/CPFF/BAA) — **distinct** from VIX, GPR, FX-RV, EPU.
**Data:** `approximate_non_ftmo` + free FRED. **PIT:** weekly_pub_lag=7d, daily_pub_lag=1d, signal_lag=1d.
**Primary:** `nfci_usd`. Locked sleeve untouched (no cooler overlay).

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| nfci_usd | +0.062% | +1.80 | +1.77 | 13% | 30% | +0.42 |
| nfci_lvl_usd | -0.014% | -1.11 | -1.00 | 1% | 100% | -0.33 |
| anfci_usd | +0.065% | +1.72 | +1.82 | 18% | 28% | +0.41 |
| nfci_chg_usd | +0.073% | +2.02 | +2.03 | 20% | 27% | +0.53 |
| cpff_usd | +0.015% | +0.41 | +0.50 | 23% | 28% | +0.11 |
| ted_usd | +0.048% | +1.70 | +2.05 | 13% | 30% | +0.40 |
| baa_usd | -0.022% | -0.72 | -0.81 | 11% | 36% | -0.15 |
| spread_usd | +0.055% | +1.40 | +1.78 | 20% | 25% | +0.39 |
| carry_nfci_cool | +0.016% | +0.25 | +0.29 | 52% | 12% | +0.01 |
| carry_nfci_loose | -0.012% | -0.16 | -0.18 | 51% | 10% | -0.08 |
| carry_cpff_cool | -0.021% | -0.31 | -0.34 | 54% | 12% | -0.12 |
| funding_ew | +0.050% | +1.85 | +1.88 | 30% | 25% | +0.44 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| nfci_usd | year_2024 | +0.00% | 0% | PASS | no |
| nfci_usd | year_2025 | -0.01% | 0% | PASS | no |
| nfci_usd | year_2026 | +0.00% | 0% | PASS | no |
| nfci_usd | holdout_365d | +0.00% | 0% | PASS | no |
| nfci_lvl_usd | year_2024 | +0.00% | 0% | PASS | no |
| nfci_lvl_usd | year_2025 | +0.00% | 0% | PASS | no |
| nfci_lvl_usd | year_2026 | +0.00% | 0% | PASS | no |
| nfci_lvl_usd | holdout_365d | +0.00% | 0% | PASS | no |
| anfci_usd | year_2024 | +0.00% | 0% | PASS | no |
| anfci_usd | year_2025 | +0.06% | 9% | PASS | no |
| anfci_usd | year_2026 | -0.10% | 38% | PASS | no |
| anfci_usd | holdout_365d | -0.07% | 25% | PASS | no |
| nfci_chg_usd | year_2024 | +0.01% | 18% | PASS | no |
| nfci_chg_usd | year_2025 | -0.24% | 9% | PASS | no |
| nfci_chg_usd | year_2026 | +0.21% | 38% | PASS | no |
| nfci_chg_usd | holdout_365d | +0.14% | 25% | PASS | no |
| cpff_usd | year_2024 | +0.02% | 9% | PASS | no |
| cpff_usd | year_2025 | -0.17% | 18% | PASS | no |
| cpff_usd | year_2026 | -0.00% | 62% | PASS | no |
| cpff_usd | holdout_365d | -0.07% | 42% | PASS | no |
| ted_usd | year_2024 | +0.00% | 0% | PASS | no |
| ted_usd | year_2025 | +0.00% | 0% | PASS | no |
| ted_usd | year_2026 | +0.00% | 0% | PASS | no |
| ted_usd | holdout_365d | +0.00% | 0% | PASS | no |
| baa_usd | year_2024 | -0.04% | 0% | PASS | no |
| baa_usd | year_2025 | -0.27% | 0% | PASS | no |
| baa_usd | year_2026 | -0.01% | 0% | PASS | no |
| baa_usd | holdout_365d | -0.01% | 0% | PASS | no |
| spread_usd | year_2024 | +0.02% | 9% | PASS | no |
| spread_usd | year_2025 | -0.17% | 18% | PASS | no |
| spread_usd | year_2026 | -0.00% | 62% | PASS | no |
| spread_usd | holdout_365d | -0.07% | 42% | PASS | no |
| carry_nfci_cool | year_2024 | +0.14% | 64% | PASS | no |
| carry_nfci_cool | year_2025 | +0.05% | 64% | PASS | no |
| carry_nfci_cool | year_2026 | +0.08% | 75% | PASS | no |
| carry_nfci_cool | holdout_365d | +0.27% | 83% | PASS | no |
| carry_nfci_loose | year_2024 | +0.15% | 64% | PASS | no |
| carry_nfci_loose | year_2025 | +0.12% | 64% | PASS | no |
| carry_nfci_loose | year_2026 | +0.13% | 62% | PASS | no |
| carry_nfci_loose | holdout_365d | +0.31% | 75% | PASS | no |
| carry_cpff_cool | year_2024 | +0.17% | 64% | PASS | no |
| carry_cpff_cool | year_2025 | +0.05% | 73% | PASS | no |
| carry_cpff_cool | year_2026 | +0.08% | 88% | PASS | no |
| carry_cpff_cool | holdout_365d | +0.18% | 92% | PASS | no |
| funding_ew | year_2024 | +0.01% | 27% | PASS | no |
| funding_ew | year_2025 | -0.14% | 9% | PASS | no |
| funding_ew | year_2026 | +0.07% | 62% | PASS | no |
| funding_ew | holdout_365d | +0.02% | 42% | PASS | no |

**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_funding_liq_*.csv`, `scholarly_fx_funding_liq_meta.json`.
