# Scholarly FX: macro differentials (Dahlquist-style FRED)

**Prior (fixed):** lagged CPI/IP/UR differentials vs USD sort G10 FX. pub_lags={'cpi': 1, 'ip': 2, 'ur': 1} + signal_lag=1m; n_long=2, n_short=2.

**Data:** `approximate_non_ftmo` D1 FX + free FRED macro panels under `data/macro/`.

## Factor board (full sample)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| macro_cpi | -0.022% | -0.36 | -0.39 | 45.6% | 13% | -0.05 |
| macro_ip | -0.032% | -0.55 | -0.60 | 48.9% | 11% | -0.15 |
| macro_ur | -0.008% | -0.14 | -0.15 | 46.1% | 16% | -0.03 |
| macro_diff_ew | -0.020% | -0.57 | -0.59 | 46.1% | 16% | -0.12 |

## Calendar / holdout windows

| Strategy | Window | mean_mo | %pos | top3 | t NW | Sharpe | gates | 1% bar | role |
|----------|--------|--------:|-----:|-----:|-----:|-------:|:-----:|:------:|------|
| macro_cpi | full_sample | -0.022% | 45.6% | 13% | -0.39 | -0.05 | PASS | no | eval |
| macro_cpi | year_2024 | -0.026% | 45.5% | 94% | -0.18 | -0.16 | PASS | no | eval |
| macro_cpi | year_2025 | 0.121% | 45.5% | 89% | 0.75 | 0.53 | PASS | no | eval |
| macro_cpi | year_2026 | -0.149% | 25.0% | 100% | -1.28 | -0.73 | PASS | no | eval |
| macro_cpi | holdout_365d | -0.097% | 33.3% | 91% | -1.22 | -0.45 | PASS | no | confirm_only |
| macro_ip | full_sample | -0.032% | 48.9% | 11% | -0.60 | -0.15 | FAIL | no | eval |
| macro_ip | year_2024 | 0.064% | 72.7% | 66% | 0.35 | 0.66 | PASS | no | eval |
| macro_ip | year_2025 | 0.072% | 45.5% | 73% | 0.48 | -0.04 | PASS | no | eval |
| macro_ip | year_2026 | -0.024% | 50.0% | 95% | -0.21 | -0.12 | PASS | no | eval |
| macro_ip | holdout_365d | 0.138% | 66.7% | 73% | 0.94 | 0.64 | PASS | no | confirm_only |
| macro_ur | full_sample | -0.008% | 46.1% | 16% | -0.15 | -0.03 | PASS | no | eval |
| macro_ur | year_2024 | -0.091% | 45.5% | 86% | -0.42 | -0.79 | PASS | no | eval |
| macro_ur | year_2025 | -0.179% | 36.4% | 96% | -1.32 | -0.41 | PASS | no | eval |
| macro_ur | year_2026 | 0.179% | 62.5% | 92% | 1.23 | 1.33 | PASS | no | eval |
| macro_ur | holdout_365d | 0.059% | 50.0% | 86% | 0.38 | 0.38 | PASS | no | confirm_only |
| macro_diff_ew | full_sample | -0.020% | 46.1% | 16% | -0.59 | -0.12 | PASS | no | eval |
| macro_diff_ew | year_2024 | -0.017% | 45.5% | 83% | -0.31 | -0.32 | PASS | no | eval |
| macro_diff_ew | year_2025 | 0.006% | 54.5% | 79% | 0.10 | 0.15 | PASS | no | eval |
| macro_diff_ew | year_2026 | 0.003% | 37.5% | 100% | 0.07 | 0.86 | PASS | no | eval |
| macro_diff_ew | holdout_365d | 0.034% | 41.7% | 83% | 0.73 | 0.77 | PASS | no | confirm_only |

## Promote / 1%/mo bar

**Joint promote:** **NO**. No overlay on locked sleeve; Yahoo/FRED ≠ FTMO MT5; no go-live.
