# Scholarly FX factor stats (v1)
**data_source:** `approximate_non_ftmo` (Yahoo D1) + FRED rates + VIX + Caldara–Iacoviello GPR
**signal_lag:** 1 | **pub_lag rates:** 1 month | **VIX lag:** 1 day | **GPR lag:** 1 month
**No holdout tuning** — fixed literature priors (n_long=n_short=2, mom 63d/skip21, regime z_high=1 cool=0.35).

## Data landed
- FRED immediate-rate CSVs: 11 series under `data/macro/fred_*.csv`
- Rates panel shape `(866, 8)` currencies ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
- VIX: `vix_yahoo.csv` n=9246
- GPR: OK monthly n=500 lag=1m [1985-02-01 → 2026-09-01]
- FX D1 pairs: ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDJPY', 'USDCAD', 'USDCHF'] bars≈3905

## Full-sample monthly factor moments
| Factor | n_mo | mean_mo | t-stat | boot 95% CI | %pos | ann Sharpe (d) |
|--------|-----:|--------:|-------:|------------|-----:|---------------:|
| carry_rank | 180 | -0.001% | -0.02 | [-0.0015, 0.0014] | 51.1% | -0.04 |
| fx_momentum | 180 | -0.064% | -0.90 | [-0.0020, 0.0007] | 45.6% | -0.21 |
| dollar_factor_avg | 180 | -0.123% | -0.80 | [-0.0043, 0.0017] | 47.8% | -0.25 |
| dollar_tsmom | 180 | 0.111% | 0.80 | [-0.0015, 0.0039] | 50.6% | 0.19 |
| carry_gpr_vix | 180 | -0.001% | -0.02 | [-0.0011, 0.0011] | 49.4% | -0.05 |
| mom_gpr_vix | 180 | -0.010% | -0.19 | [-0.0012, 0.0010] | 46.1% | -0.05 |

## Calendar / walk-forward windows (mean monthly)
| Factor | Window | mean_mo | %pos | Sharpe | gates | role |
|--------|--------|--------:|-----:|-------:|:-----:|------|
| carry_rank | IS_2018_2023 | -0.061% | 46.5% | -0.21 | PASS | eval |
| carry_rank | year_2024 | 0.148% | 63.6% | 0.76 | PASS | eval |
| carry_rank | year_2025 | 0.125% | 63.6% | 0.21 | PASS | eval |
| carry_rank | year_2026 | 0.128% | 62.5% | 1.01 | PASS | eval |
| carry_rank | holdout_365d_confirm_only | 0.310% | 75.0% | 1.35 | PASS | confirm_only |
| fx_momentum | IS_2018_2023 | -0.094% | 45.1% | -0.28 | PASS | eval |
| fx_momentum | year_2024 | -0.171% | 45.5% | -0.64 | PASS | eval |
| fx_momentum | year_2025 | 0.437% | 90.9% | 1.28 | PASS | eval |
| fx_momentum | year_2026 | 0.344% | 62.5% | 1.65 | PASS | eval |
| fx_momentum | holdout_365d_confirm_only | 0.384% | 75.0% | 2.05 | PASS | confirm_only |
| dollar_factor_avg | IS_2018_2023 | -0.144% | 45.1% | -0.16 | FAIL | eval |
| dollar_factor_avg | year_2024 | -0.476% | 36.4% | -1.25 | PASS | eval |
| dollar_factor_avg | year_2025 | 0.635% | 63.6% | 0.98 | PASS | eval |
| dollar_factor_avg | year_2026 | -0.386% | 37.5% | -0.01 | PASS | eval |
| dollar_factor_avg | holdout_365d_confirm_only | -0.023% | 41.7% | -0.31 | PASS | confirm_only |
| dollar_tsmom | IS_2018_2023 | 0.024% | 52.1% | 0.14 | FAIL | eval |
| dollar_tsmom | year_2024 | 0.775% | 54.5% | 1.32 | PASS | eval |
| dollar_tsmom | year_2025 | 0.049% | 63.6% | -0.10 | PASS | eval |
| dollar_tsmom | year_2026 | 0.296% | 37.5% | 0.84 | PASS | eval |
| dollar_tsmom | holdout_365d_confirm_only | 0.527% | 58.3% | 0.58 | PASS | confirm_only |
| carry_gpr_vix | IS_2018_2023 | -0.058% | 45.1% | -0.27 | PASS | eval |
| carry_gpr_vix | year_2024 | -0.021% | 54.5% | 0.07 | PASS | eval |
| carry_gpr_vix | year_2025 | 0.164% | 63.6% | 0.74 | PASS | eval |
| carry_gpr_vix | year_2026 | 0.043% | 62.5% | 0.93 | PASS | eval |
| carry_gpr_vix | holdout_365d_confirm_only | 0.248% | 75.0% | 1.42 | PASS | confirm_only |
| mom_gpr_vix | IS_2018_2023 | -0.052% | 47.9% | -0.20 | PASS | eval |
| mom_gpr_vix | year_2024 | -0.079% | 45.5% | -0.42 | PASS | eval |
| mom_gpr_vix | year_2025 | 0.338% | 90.9% | 1.98 | PASS | eval |
| mom_gpr_vix | year_2026 | 0.195% | 75.0% | 1.56 | PASS | eval |
| mom_gpr_vix | holdout_365d_confirm_only | 0.274% | 75.0% | 2.14 | PASS | confirm_only |

## Honest read vs 1%/mo FTMO goal
- Literature carry/momentum/dollar premia are **real but small** after costs; full-sample mean_mo above is **not** a claim of stable ≥1%/month.
- Year windows will often miss 1% mean and/or 70% positive months — that is expected.
- Overlay hunt on technical sleeves is **deprecated**; this line is the active research path.
- Still **approximate_non_ftmo** — never golive without FTMO CSVs.

Artifacts: `scholarly_fx_factor_summary.csv`, `scholarly_fx_window_stats.csv`, `scholarly_fx_monthly_returns.csv`, `scholarly_fx_gates.csv`.
