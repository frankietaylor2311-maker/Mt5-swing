# Scholarly FX: country-level GPR → FX depreciation

**Prior (fixed):** high home-country Caldara–Iacoviello GPR → depreciate that currency vs USD. Sort = long low-GPR / short high-GPR. pub_lag=1m + signal_lag=1m; z_window=60m.

**Data:** `approximate_non_ftmo` Yahoo D1 + GPRC_* from monthly export. **NZD gap:** no `GPRC_NZL` in 44-country file.

## Lagged sort portfolio

| mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------:|------:|-----:|-----:|-----:|-------:|
| 0.034% | 0.55 | 0.64 | 48.3% | 16% | 0.13 |

## Calendar / holdout windows

| Window | mean_mo | %pos | top3 | t NW | gates | 1% bar | role |
|--------|--------:|-----:|-----:|-----:|:-----:|:------:|------|
| IS_2018_2023 | -0.017% | 47.9% | 24% | -0.25 | PASS | no | eval |
| year_2024 | 0.013% | 36.4% | 98% | 0.04 | PASS | no | eval |
| year_2025 | 0.044% | 45.5% | 82% | 0.25 | PASS | no | eval |
| year_2026 | -0.127% | 37.5% | 100% | -1.03 | PASS | no | eval |
| holdout_365d | -0.151% | 33.3% | 92% | -1.05 | PASS | no | confirm_only |

## Local projections (FX cumret on lagged home GPR z)

Hypothesis support if **β < 0** (high GPR → subsequent depreciation).

| Scope | h | beta | t | n | sign_ok |
|-------|--:|-----:|--:|--:|:------:|
| AUD | 1 | -0.108%/σ | -0.58 | 180 | yes |
| AUD | 3 | -0.140%/σ | -0.47 | 178 | yes |
| AUD | 6 | -0.685%/σ | -1.81 | 175 | yes |
| CAD | 1 | -0.048%/σ | -0.38 | 180 | yes |
| CAD | 3 | -0.213%/σ | -1.03 | 178 | yes |
| CAD | 6 | -0.538%/σ | -2.16 | 175 | yes |
| CHF | 1 | -0.015%/σ | -0.11 | 180 | yes |
| CHF | 3 | -0.272%/σ | -1.22 | 178 | yes |
| CHF | 6 | -0.701%/σ | -2.59 | 175 | yes |
| EUR | 1 | -0.131%/σ | -0.92 | 180 | yes |
| EUR | 3 | -0.606%/σ | -2.46 | 178 | yes |
| EUR | 6 | -0.980%/σ | -2.74 | 175 | yes |
| GBP | 1 | -0.054%/σ | -0.35 | 180 | yes |
| GBP | 3 | -0.304%/σ | -1.16 | 178 | yes |
| GBP | 6 | -0.401%/σ | -1.06 | 175 | yes |
| JPY | 1 | -0.055%/σ | -0.39 | 180 | yes |
| JPY | 3 | 0.053%/σ | 0.21 | 178 | no |
| JPY | 6 | -0.044%/σ | -0.13 | 175 | yes |
| POOLED | 1 | -0.067%/σ | -1.12 | 1080 | yes |
| POOLED | 3 | -0.250%/σ | -2.47 | 1068 | yes |
| POOLED | 6 | -0.579%/σ | -4.28 | 1050 | yes |

## FRED OECD rate coverage (sparse / missing)

Extended panel: 26 series OK; 13 sparse/stale/missing flagged.

| CCY | series | end | months_lag | note |
|-----|--------|-----|----------:|------|
| CHF | IRSTCI01CHM156N | 2024-03-01 | 30.0 | tail lags OECD release / FRED update |
| CNY | IRSTCI01CNM156N | 2025-06-01 | 15.0 | ok |
| DKK | IRSTCI01DKM156N | 2025-12-01 | 9.0 | ok |
| EUR | IRSTCI01EZM156N | 2026-01-01 | 8.0 | ok |
| HKD | IRSTCI01HKM156N | nan | nan | IRSTCI01* not on FRED (404) |
| MYR | IRSTCI01MYM156N | nan | nan | IRSTCI01* not on FRED (404) |
| NZD | IRSTCI01NZM156N | 2024-12-01 | 21.0 | tail lags OECD release / FRED update |
| PHP | IRSTCI01PHM156N | nan | nan | IRSTCI01* not on FRED (404) |
| RUB | IRSTCI01RUM156N | 2025-10-01 | 11.0 | ok |
| SEK | IRSTCI01SEM156N | 2020-10-01 | 71.0 | OECD series ends ~2020 on FRED — treat as discontinued for live carry |
| SGD | IRSTCI01SGM156N | nan | nan | IRSTCI01* not on FRED (404) |
| THB | IRSTCI01THM156N | nan | nan | IRSTCI01* not on FRED (404) |
| TWD | IRSTCI01TWM156N | nan | nan | IRSTCI01* not on FRED (404) |

## Consistency vs ~1%/mo

- Any window clears? **NO**
- Joint years+holdout? **NO**
- **Do not claim 1%/mo.** Still `approximate_non_ftmo`.
- **Gaps:** news NLP not wired; FTMO MT5 CSVs not present; NZD country GPR missing; SEK FRED discontinued ~2020.
