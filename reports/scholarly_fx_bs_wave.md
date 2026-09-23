# Scholarly FX: Balassa–Samuelson / productivity-adjusted real FX

**Prior (fixed):** HBS — relative productivity co-moves with real FX; residuals / gaps are productivity-adjusted value. CPI pub_lag=1m, IP pub_lag=2m + signal_lag=1m; lookbacks=[60, 120]m. No HO tuning.

**Data:** `approximate_non_ftmo` D1 FX + FRED CPI levels (USD, EUR, GBP, JPY, CAD, CHF, AUD, NZD) + IP levels (USD, EUR, GBP, JPY, CAD). IP missing: ['AUD', 'NZD', 'CHF'].

## Factor board (full sample, unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| bs_gap_60m | -0.045% | -0.71 | -0.76 | 41.1% | 14% | -0.15 |
| bs_resid_60m | 0.109% | 1.72 | 1.62 | 43.9% | 18% | 0.38 |
| bs_prod_60m | -0.011% | -0.17 | -0.19 | 47.2% | 12% | -0.07 |
| bs_gap_120m | -0.044% | -0.68 | -0.74 | 40.0% | 14% | -0.15 |
| bs_resid_120m | 0.117% | 1.81 | 1.73 | 45.0% | 17% | 0.39 |
| bs_prod_120m | -0.003% | -0.05 | -0.05 | 47.2% | 10% | -0.04 |
| bs_ew | 0.018% | 0.50 | 0.60 | 52.8% | 15% | 0.09 |

## Calendar / holdout windows (unscaled)

| Strategy | Window | mean_mo | %pos | top3 | t NW | Sharpe | gates | 1% bar | role |
|----------|--------|--------:|-----:|-----:|-----:|-------:|:-----:|:------:|------|
| bs_gap_60m | full_sample | -0.045% | 41.1% | 14% | -0.76 | -0.15 | FAIL | no | eval |
| bs_gap_60m | year_2024 | -0.273% | 36.4% | 88% | -1.53 | -1.32 | PASS | no | eval |
| bs_gap_60m | year_2025 | -0.129% | 36.4% | 88% | -0.94 | -0.44 | PASS | no | eval |
| bs_gap_60m | year_2026 | -0.054% | 50.0% | 93% | -0.39 | -0.49 | PASS | no | eval |
| bs_gap_60m | holdout_365d | -0.020% | 50.0% | 82% | -0.18 | -0.04 | PASS | no | confirm_only |
| bs_resid_60m | full_sample | 0.109% | 43.9% | 18% | 1.62 | 0.38 | PASS | no | eval |
| bs_resid_60m | year_2024 | -0.122% | 36.4% | 91% | -0.75 | -0.85 | PASS | no | eval |
| bs_resid_60m | year_2025 | -0.129% | 36.4% | 88% | -0.94 | -0.44 | PASS | no | eval |
| bs_resid_60m | year_2026 | -0.054% | 50.0% | 93% | -0.39 | -0.49 | PASS | no | eval |
| bs_resid_60m | holdout_365d | -0.020% | 50.0% | 82% | -0.18 | -0.04 | PASS | no | confirm_only |
| bs_prod_60m | full_sample | -0.011% | 47.2% | 12% | -0.19 | -0.07 | FAIL | no | eval |
| bs_prod_60m | year_2024 | 0.017% | 63.6% | 77% | 0.07 | 0.10 | PASS | no | eval |
| bs_prod_60m | year_2025 | 0.463% | 72.7% | 50% | 1.62 | 0.88 | PASS | no | eval |
| bs_prod_60m | year_2026 | -0.063% | 62.5% | 84% | -0.25 | -0.10 | PASS | no | eval |
| bs_prod_60m | holdout_365d | 0.219% | 75.0% | 62% | 0.82 | 0.66 | PASS | no | confirm_only |
| bs_gap_120m | full_sample | -0.044% | 40.0% | 14% | -0.74 | -0.15 | FAIL | no | eval |
| bs_gap_120m | year_2024 | -0.273% | 36.4% | 88% | -1.53 | -1.32 | PASS | no | eval |
| bs_gap_120m | year_2025 | -0.129% | 36.4% | 88% | -0.94 | -0.44 | PASS | no | eval |
| bs_gap_120m | year_2026 | -0.054% | 50.0% | 93% | -0.39 | -0.49 | PASS | no | eval |
| bs_gap_120m | holdout_365d | -0.020% | 50.0% | 82% | -0.18 | -0.04 | PASS | no | confirm_only |
| bs_resid_120m | full_sample | 0.117% | 45.0% | 17% | 1.73 | 0.39 | PASS | no | eval |
| bs_resid_120m | year_2024 | 0.271% | 63.6% | 65% | 1.52 | 1.32 | PASS | no | eval |
| bs_resid_120m | year_2025 | 0.125% | 63.6% | 61% | 0.92 | 0.44 | PASS | no | eval |
| bs_resid_120m | year_2026 | 0.049% | 50.0% | 91% | 0.36 | 0.49 | PASS | no | eval |
| bs_resid_120m | holdout_365d | 0.016% | 50.0% | 76% | 0.15 | 0.04 | PASS | no | confirm_only |
| bs_prod_120m | full_sample | -0.003% | 47.2% | 10% | -0.05 | -0.04 | FAIL | no | eval |
| bs_prod_120m | year_2024 | 0.157% | 72.7% | 64% | 0.50 | 0.42 | PASS | no | eval |
| bs_prod_120m | year_2025 | 0.463% | 72.7% | 50% | 1.62 | 0.88 | PASS | no | eval |
| bs_prod_120m | year_2026 | -0.063% | 62.5% | 84% | -0.25 | -0.10 | PASS | no | eval |
| bs_prod_120m | holdout_365d | 0.219% | 75.0% | 62% | 0.82 | 0.66 | PASS | no | confirm_only |
| bs_ew | full_sample | 0.018% | 52.8% | 15% | 0.60 | 0.09 | PASS | no | eval |
| bs_ew | year_2024 | -0.125% | 54.5% | 86% | -0.75 | -0.78 | PASS | no | eval |
| bs_ew | year_2025 | 0.070% | 54.5% | 86% | 0.54 | 0.13 | PASS | no | eval |
| bs_ew | year_2026 | -0.055% | 50.0% | 94% | -0.66 | -0.55 | PASS | no | eval |
| bs_ew | holdout_365d | 0.061% | 58.3% | 78% | 0.56 | 0.38 | PASS | no | confirm_only |

## FTMO risk sweep (IS → OOS confirm)

IS = [2011-09-18 … 2025-09-14]; OOS = [2025-09-15 … 2026-09-15]. Run because ≥1 factor had positive IS mean.

| Strategy | scale | bind | IS mean_mo | IS static | IS daily | IS gate | OOS mean_mo | OOS gate |
|----------|------:|:----:|-----------:|----------:|---------:|:-------:|-----------:|:--------:|
| bs_gap_60m | 0.80 | static | -0.038% | 10.00% | 1.94% | PASS | -0.016% | PASS |
| bs_resid_60m | 1.98 | static | 0.229% | 10.00% | 2.68% | PASS | -0.043% | PASS |
| bs_prod_60m | 0.79 | static | -0.020% | 10.00% | 1.25% | PASS | 0.174% | PASS |
| bs_gap_120m | 0.80 | static | -0.037% | 10.00% | 1.94% | PASS | -0.016% | PASS |
| bs_resid_120m | 1.98 | static | 0.243% | 10.00% | 3.87% | PASS | 0.028% | PASS |
| bs_prod_120m | 0.62 | static | -0.011% | 10.00% | 0.61% | PASS | 0.137% | PASS |
| bs_ew | 2.13 | static | 0.031% | 10.00% | 1.90% | PASS | 0.129% | PASS |

## Promote / 1%/mo bar

**Unscaled joint promote:** **NO**. **Scaled primary (`bs_gap_60m`) promote:** **NO**. Do not claim 1%/mo unless earned. Locked sleeve untouched; Yahoo/FRED ≠ FTMO MT5; no go-live.

### Honest gaps

- IP is a **manufacturing/output proxy**, not TFP or tradable labour productivity.
- AUD/NZD/CHF lack clean FRED IP → thin cross-section (EUR/GBP/JPY/CAD).
- EUR IP series sparse after ~2023 on FRED.
- HBS is a **long-run** relative-price relation; monthly prop hit-rate is different.

Artifacts: `scholarly_fx_bs_*.csv`, `scholarly_fx_bs_meta.json`.
