# Scholarly FX: Lustig–Verdelhan dollar-factor beta wave

**Path:** Rolling β of currency returns on the dollar factor RX (EW foreign excess returns vs USD) — long high-$β / short low-$β. **Distinct** from Lustig carry (rate sort), Menkhoff FX-RV (§15), Hau–Rey equity-diff, BNP crash-skew (§26), AI-GPR (§25).
**Data:** `approximate_non_ftmo` Yahoo D1 USD majors + free FRED short rates for AFD. **PIT:** skip=1m + signal_lag=1m; β window 60/36m (monthly OLS); RX TSMOM 12m; costs 1.5 bps/side.
**Primary:** `dollar_beta_xs` (unconditional high−low $β HML). Locked sleeve untouched.

**Board:** n=5 soft_nw_pos=0 hard_nw_pos=0 promote=0

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| dollar_beta_xs | +0.014% | +0.18 | +0.20 | 39% | 15% | +0.05 |
| dollar_beta_xs_36m | -0.033% | -0.46 | -0.51 | 37% | 14% | -0.10 |
| dollar_beta_afd | +0.020% | +0.25 | +0.27 | 43% | 16% | +0.06 |
| dollar_rx_tsmom | -0.232% | -1.68 | -1.67 | 38% | 12% | -0.43 |
| dollar_ew | -0.057% | -1.33 | -1.20 | 38% | 17% | -0.34 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | top3 | gates | 1% bar |
|----------|--------|--------:|-----:|-----:|:-----:|:------:|
| dollar_beta_xs | year_2024 | -0.04% | 64% | 62% | PASS | no |
| dollar_beta_xs | year_2025 | -0.18% | 27% | 100% | PASS | no |
| dollar_beta_xs | year_2026 | +0.10% | 62% | 79% | PASS | no |
| dollar_beta_xs | holdout_365d | +0.18% | 58% | 67% | PASS | no |
| dollar_beta_xs_36m | year_2024 | -0.04% | 73% | 72% | PASS | no |
| dollar_beta_xs_36m | year_2025 | -0.37% | 18% | 100% | PASS | no |
| dollar_beta_xs_36m | year_2026 | -0.17% | 38% | 100% | PASS | no |
| dollar_beta_xs_36m | holdout_365d | -0.21% | 33% | 96% | PASS | no |
| dollar_beta_afd | year_2024 | +0.05% | 36% | 95% | PASS | no |
| dollar_beta_afd | year_2025 | +0.16% | 73% | 56% | PASS | no |
| dollar_beta_afd | year_2026 | -0.10% | 38% | 100% | PASS | no |
| dollar_beta_afd | holdout_365d | -0.18% | 42% | 86% | PASS | no |
| dollar_rx_tsmom | year_2024 | -0.57% | 27% | 100% | PASS | no |
| dollar_rx_tsmom | year_2025 | -1.26% | 9% | 100% | FAIL | no |
| dollar_rx_tsmom | year_2026 | -0.24% | 50% | 89% | PASS | no |
| dollar_rx_tsmom | holdout_365d | -0.15% | 42% | 84% | PASS | no |
| dollar_ew | year_2024 | -0.15% | 27% | 100% | PASS | no |
| dollar_ew | year_2025 | -0.41% | 9% | 100% | PASS | no |
| dollar_ew | year_2026 | -0.10% | 50% | 99% | PASS | no |
| dollar_ew | holdout_365d | -0.09% | 42% | 93% | PASS | no |

**Risk sweep run:** YES.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_dollar_beta_*.csv`, `scholarly_fx_dollar_beta_meta.json`.
