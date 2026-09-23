# Scholarly FX: CFTC COT positioning wave

**Data:** `approximate_non_ftmo` + free CFTC TFF/Legacy SODA. **PIT:** release_lag_days=3 (Tue→Fri) + signal_lag=1d.
**Primary:** `cot_lev_net_xs`. Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| cot_lev_net_xs | -0.139% | -2.34 | -2.26 | 46% | 10% | -0.55 |
| cot_lev_chg_xs | -0.213% | -3.11 | -2.71 | 41% | 16% | -0.72 |
| cot_lev_z_xs | -0.129% | -1.86 | -1.87 | 44% | 15% | -0.42 |
| cot_lev_z_mr_xs | +0.078% | +1.12 | +1.16 | 54% | 13% | +0.25 |
| cot_noncomm_net_xs | -0.187% | -2.80 | -2.68 | 41% | 14% | -0.70 |
| cot_dx_usd | +0.148% | +0.94 | +0.86 | 53% | 14% | +0.23 |
| cot_ew | -0.176% | -3.70 | -3.18 | 41% | 12% | -0.87 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| cot_lev_net_xs | year_2024 | +0.09% | 55% | PASS | no |
| cot_lev_net_xs | year_2025 | +0.03% | 73% | PASS | no |
| cot_lev_net_xs | year_2026 | +0.17% | 75% | PASS | no |
| cot_lev_net_xs | holdout_365d | +0.27% | 83% | PASS | no |
| cot_lev_chg_xs | year_2024 | +0.08% | 36% | PASS | no |
| cot_lev_chg_xs | year_2025 | -0.25% | 36% | PASS | no |
| cot_lev_chg_xs | year_2026 | -0.39% | 25% | PASS | no |
| cot_lev_chg_xs | holdout_365d | -0.25% | 42% | PASS | no |
| cot_lev_z_xs | year_2024 | -0.25% | 27% | PASS | no |
| cot_lev_z_xs | year_2025 | +0.07% | 55% | PASS | no |
| cot_lev_z_xs | year_2026 | -0.23% | 38% | PASS | no |
| cot_lev_z_xs | holdout_365d | -0.02% | 42% | PASS | no |
| cot_lev_z_mr_xs | year_2024 | +0.19% | 55% | PASS | no |
| cot_lev_z_mr_xs | year_2025 | -0.13% | 45% | PASS | no |
| cot_lev_z_mr_xs | year_2026 | +0.17% | 62% | PASS | no |
| cot_lev_z_mr_xs | holdout_365d | -0.03% | 58% | PASS | no |
| cot_noncomm_net_xs | year_2024 | -0.21% | 36% | PASS | no |
| cot_noncomm_net_xs | year_2025 | -0.22% | 45% | PASS | no |
| cot_noncomm_net_xs | year_2026 | -0.08% | 50% | PASS | no |
| cot_noncomm_net_xs | holdout_365d | -0.28% | 33% | PASS | no |
| cot_dx_usd | year_2024 | -0.86% | 18% | PASS | no |
| cot_dx_usd | year_2025 | -0.60% | 27% | PASS | no |
| cot_dx_usd | year_2026 | +0.22% | 50% | PASS | no |
| cot_dx_usd | holdout_365d | +0.44% | 50% | PASS | no |
| cot_ew | year_2024 | +0.08% | 64% | PASS | no |
| cot_ew | year_2025 | -0.11% | 55% | PASS | no |
| cot_ew | year_2026 | -0.11% | 25% | PASS | no |
| cot_ew | holdout_365d | +0.01% | 42% | PASS | no |

**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_cot_*.csv`, `scholarly_fx_cot_meta.json`.

## Risk sweep

See `scholarly_fx_cot_risk_sweep.csv`. Best scaled IS (`cot_lev_z_mr_xs` @ ~2.07×) ≈ +0.18%/mo ≪ 1%. Primary continuation scaled IS negative. Scaled clears: NO.

## Honest read

Continuation speculative-pressure sorts are **significantly negative** full-sample (NW t ≈ −2 to −3). Pre-specified mean-reversion z and DX USD tilt mildly positive (~0.08–0.15%/mo). Far from 1%/mo + 70% hit-rate. Locked sleeve untouched. promote=0.

## PIT

Tuesday report_date → Friday release → release_lag_days=3 + signal_lag=1 trading day. Free CFTC SODA; no paid NLP.
