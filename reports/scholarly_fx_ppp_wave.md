# Scholarly FX: PPP / real exchange-rate value (Rogoff-style)

**Prior (fixed):** real FX q = S·(CPI_US/CPI_f); trailing z; long undervalued / short overvalued. pub_lag=1m + signal_lag=1m; lookbacks=[60, 120]m; n_long=2, n_short=2. No HO tuning.

**Data:** `approximate_non_ftmo` D1 FX + FRED CPI index levels (USD, EUR, GBP, JPY, CAD, CHF, AUD, NZD). AU/NZ quarterly CPI ffilled to monthly.

## Factor board (full sample, unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| ppp_xs_60m | -0.010% | -0.15 | -0.15 | 42.2% | 12% | -0.03 |
| ppp_ts_60m | -0.014% | -0.25 | -0.30 | 39.4% | 13% | -0.05 |
| ppp_xs_120m | 0.007% | 0.11 | 0.11 | 42.8% | 12% | 0.03 |
| ppp_ts_120m | -0.016% | -0.28 | -0.32 | 41.1% | 13% | -0.06 |
| ppp_xs_ew | -0.001% | -0.02 | -0.02 | 42.2% | 12% | -0.00 |

## Calendar / holdout windows (unscaled)

| Strategy | Window | mean_mo | %pos | top3 | t NW | Sharpe | gates | 1% bar | role |
|----------|--------|--------:|-----:|-----:|-----:|-------:|:-----:|:------:|------|
| ppp_xs_60m | full_sample | -0.010% | 42.2% | 12% | -0.15 | -0.03 | PASS | no | eval |
| ppp_xs_60m | year_2024 | -0.104% | 63.6% | 65% | -0.49 | -0.66 | PASS | no | eval |
| ppp_xs_60m | year_2025 | -0.353% | 27.3% | 100% | -1.56 | -0.97 | PASS | no | eval |
| ppp_xs_60m | year_2026 | 0.148% | 75.0% | 63% | 1.28 | 0.73 | PASS | no | eval |
| ppp_xs_60m | holdout_365d | 0.096% | 66.7% | 52% | 1.21 | 0.45 | PASS | no | confirm_only |
| ppp_ts_60m | full_sample | -0.014% | 39.4% | 13% | -0.30 | -0.05 | PASS | no | eval |
| ppp_ts_60m | year_2024 | -0.039% | 54.5% | 72% | -0.18 | -0.12 | PASS | no | eval |
| ppp_ts_60m | year_2025 | -0.263% | 27.3% | 100% | -1.36 | -0.68 | PASS | no | eval |
| ppp_ts_60m | year_2026 | 0.221% | 62.5% | 79% | 1.24 | 0.39 | PASS | no | eval |
| ppp_ts_60m | holdout_365d | 0.064% | 50.0% | 66% | 0.48 | 0.26 | PASS | no | confirm_only |
| ppp_xs_120m | full_sample | 0.007% | 42.8% | 12% | 0.11 | 0.03 | PASS | no | eval |
| ppp_xs_120m | year_2024 | -0.098% | 72.7% | 65% | -0.52 | -0.54 | PASS | no | eval |
| ppp_xs_120m | year_2025 | -0.317% | 36.4% | 97% | -1.55 | -0.76 | PASS | no | eval |
| ppp_xs_120m | year_2026 | 0.148% | 75.0% | 63% | 1.28 | 0.73 | PASS | no | eval |
| ppp_xs_120m | holdout_365d | 0.096% | 66.7% | 52% | 1.21 | 0.45 | PASS | no | confirm_only |
| ppp_ts_120m | full_sample | -0.016% | 41.1% | 13% | -0.32 | -0.06 | PASS | no | eval |
| ppp_ts_120m | year_2024 | 0.034% | 54.5% | 73% | 0.15 | 0.15 | PASS | no | eval |
| ppp_ts_120m | year_2025 | -0.263% | 27.3% | 100% | -1.36 | -0.68 | PASS | no | eval |
| ppp_ts_120m | year_2026 | 0.221% | 62.5% | 79% | 1.24 | 0.39 | PASS | no | eval |
| ppp_ts_120m | holdout_365d | 0.064% | 50.0% | 66% | 0.48 | 0.26 | PASS | no | confirm_only |
| ppp_xs_ew | full_sample | -0.001% | 42.2% | 12% | -0.02 | -0.00 | PASS | no | eval |
| ppp_xs_ew | year_2024 | -0.101% | 72.7% | 66% | -0.52 | -0.62 | PASS | no | eval |
| ppp_xs_ew | year_2025 | -0.335% | 27.3% | 100% | -1.56 | -0.87 | PASS | no | eval |
| ppp_xs_ew | year_2026 | 0.148% | 75.0% | 63% | 1.28 | 0.73 | PASS | no | eval |
| ppp_xs_ew | holdout_365d | 0.096% | 66.7% | 52% | 1.21 | 0.45 | PASS | no | confirm_only |

## FTMO risk sweep (IS → OOS confirm)

IS = [2011-09-18 … 2025-09-14]; OOS holdout = [2025-09-15 … 2026-09-15]. Scale chosen on IS only to sit just under 10% static / 5% daily; OOS not retuned.

| Strategy | scale | bind | IS mean_mo | IS static | IS daily | IS gate | OOS mean_mo | OOS static | OOS daily | OOS gate |
|----------|------:|:----:|-----------:|----------:|---------:|:-------:|-----------:|-----------:|----------:|:--------:|
| ppp_xs_60m | 1.15 | static | -0.017% | 10.00% | 4.27% | PASS | 0.110% | 1.15% | 0.44% | PASS |
| ppp_ts_60m | 1.82 | static | -0.036% | 10.00% | 2.96% | PASS | 0.116% | 2.30% | 1.16% | PASS |
| ppp_xs_120m | 1.15 | static | 0.003% | 10.00% | 4.27% | PASS | 0.110% | 1.15% | 0.44% | PASS |
| ppp_ts_120m | 1.82 | static | -0.040% | 10.00% | 2.94% | PASS | 0.116% | 2.30% | 1.16% | PASS |
| ppp_xs_ew | 1.15 | static | -0.007% | 10.00% | 4.27% | PASS | 0.110% | 1.15% | 0.44% | PASS |

## Promote / 1%/mo bar

**Unscaled joint promote:** **NO**. **Scaled primary (`ppp_xs_60m`) promote:** **NO**. Do not claim 1%/mo unless earned. No overlay on locked sleeve; Yahoo/FRED ≠ FTMO MT5; no go-live.

### Honest gaps

- Real FX mean reversion is **slow** (Rogoff half-life years) — monthly prop-firm hit-rate is a different objective.
- AU/NZ CPI quarterly on FRED (ffill).
- Absolute PPP levels depend on index base; we use **relative** z vs own history.

Artifacts: `scholarly_fx_ppp_*.csv`, `scholarly_fx_ppp_meta.json`.
