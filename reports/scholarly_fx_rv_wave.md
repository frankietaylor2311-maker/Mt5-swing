# Scholarly FX: Menkhoff global FX realized-vol risk factor

**Prior (fixed):** Menkhoff et al. (2012 JF) — global FX volatility is a priced state variable; high FX vol predicts carry underperformance / risk-off USD. RV windows=[21, 63]d; z_window=252; z_high=1.0; cool=0.35; signal_lag=1. **Not VIX** (already tried). No HO tuning.

**Data:** `approximate_non_ftmo` D1 USD majors (EURUSD, GBPUSD, AUDUSD, NZDUSD, USDJPY, USDCAD, USDCHF). Rates: USD, EUR, GBP, JPY, AUD, CAD, CHF, NZD. corr(fx_rv_21d, VIX)=0.44236153857735566.

## Factor board (full sample, unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| carry_raw | -0.002% | -0.02 | -0.03 | 51.1% | 10% | -0.04 |
| fxrv_usd_tilt_21d | 0.021% | 0.56 | 0.52 | 25.6% | 28% | 0.13 |
| fxrv_innov_usd_21d | 0.032% | 0.79 | 0.76 | 36.7% | 18% | 0.19 |
| fxrv_highvol_usd_21d | 0.024% | 0.72 | 0.65 | 12.8% | 32% | 0.17 |
| carry_fxrv_cool_21d | 0.018% | 0.27 | 0.34 | 49.4% | 12% | 0.02 |
| carry_fxrv_lowvol_only_21d | 0.012% | 0.22 | 0.25 | 40.0% | 13% | 0.05 |
| fxrv_usd_tilt_63d | 0.026% | 0.63 | 0.57 | 20.6% | 28% | 0.16 |
| fxrv_innov_usd_63d | 0.012% | 0.29 | 0.27 | 29.4% | 28% | 0.07 |
| fxrv_highvol_usd_63d | 0.033% | 0.79 | 0.73 | 12.2% | 31% | 0.20 |
| carry_fxrv_cool_63d | 0.023% | 0.34 | 0.41 | 51.1% | 12% | 0.04 |
| carry_fxrv_lowvol_only_63d | 0.012% | 0.20 | 0.24 | 39.4% | 13% | 0.05 |
| fxrv_ew_21d | 0.026% | 0.71 | 0.67 | 38.3% | 22% | 0.17 |

## Calendar / holdout windows (unscaled)

| Strategy | Window | mean_mo | %pos | top3 | t NW | Sharpe | gates | 1% bar | role |
|----------|--------|--------:|-----:|-----:|-----:|-------:|:-----:|:------:|------|
| carry_raw | full_sample | -0.002% | 51.1% | 10% | -0.03 | -0.04 | FAIL | no | eval |
| carry_raw | year_2024 | 0.147% | 63.6% | 59% | 0.63 | 0.76 | PASS | no | eval |
| carry_raw | year_2025 | 0.124% | 63.6% | 63% | 0.71 | 0.21 | PASS | no | eval |
| carry_raw | year_2026 | 0.127% | 62.5% | 80% | 0.74 | 1.00 | PASS | no | eval |
| carry_raw | holdout_365d | 0.309% | 75.0% | 53% | 1.43 | 1.26 | PASS | no | confirm_only |
| fxrv_usd_tilt_21d | full_sample | 0.021% | 25.6% | 28% | 0.52 | 0.13 | PASS | no | eval |
| fxrv_usd_tilt_21d | year_2024 | 0.104% | 27.3% | 100% | 0.52 | 0.74 | PASS | no | eval |
| fxrv_usd_tilt_21d | year_2025 | -0.334% | 9.1% | 100% | -1.19 | -1.64 | PASS | no | eval |
| fxrv_usd_tilt_21d | year_2026 | -0.037% | 37.5% | 100% | -0.63 | -0.39 | PASS | no | eval |
| fxrv_usd_tilt_21d | holdout_365d | -0.025% | 25.0% | 100% | -0.63 | -0.33 | PASS | no | confirm_only |
| fxrv_innov_usd_21d | full_sample | 0.032% | 36.7% | 18% | 0.76 | 0.19 | PASS | no | eval |
| fxrv_innov_usd_21d | year_2024 | -0.006% | 36.4% | 100% | -0.04 | 0.03 | PASS | no | eval |
| fxrv_innov_usd_21d | year_2025 | -0.251% | 18.2% | 100% | -1.06 | -1.34 | PASS | no | eval |
| fxrv_innov_usd_21d | year_2026 | 0.132% | 50.0% | 97% | 0.84 | 0.02 | PASS | no | eval |
| fxrv_innov_usd_21d | holdout_365d | 0.001% | 33.3% | 97% | 0.01 | -0.07 | PASS | no | confirm_only |
| fxrv_highvol_usd_21d | full_sample | 0.024% | 12.8% | 32% | 0.65 | 0.17 | PASS | no | eval |
| fxrv_highvol_usd_21d | year_2024 | 0.118% | 18.2% | 100% | 0.75 | 0.96 | PASS | no | eval |
| fxrv_highvol_usd_21d | year_2025 | -0.272% | 18.2% | 100% | -1.06 | -1.44 | PASS | no | eval |
| fxrv_highvol_usd_21d | year_2026 | 0.000% | 0.0% | nan% | 0.00 | nan | PASS | no | eval |
| fxrv_highvol_usd_21d | holdout_365d | 0.000% | 0.0% | nan% | 0.00 | nan | PASS | no | confirm_only |
| carry_fxrv_cool_21d | full_sample | 0.018% | 49.4% | 12% | 0.34 | 0.02 | PASS | no | eval |
| carry_fxrv_cool_21d | year_2024 | 0.121% | 54.5% | 60% | 0.50 | 0.78 | PASS | no | eval |
| carry_fxrv_cool_21d | year_2025 | 0.087% | 63.6% | 68% | 0.48 | 0.16 | PASS | no | eval |
| carry_fxrv_cool_21d | year_2026 | 0.105% | 75.0% | 69% | 0.70 | 1.03 | PASS | no | eval |
| carry_fxrv_cool_21d | holdout_365d | 0.294% | 83.3% | 51% | 1.40 | 1.31 | PASS | no | confirm_only |
| carry_fxrv_lowvol_only_21d | full_sample | 0.012% | 40.0% | 13% | 0.25 | 0.05 | PASS | no | eval |
| carry_fxrv_lowvol_only_21d | year_2024 | 0.008% | 54.5% | 65% | 0.03 | 0.34 | PASS | no | eval |
| carry_fxrv_lowvol_only_21d | year_2025 | 0.215% | 54.5% | 65% | 1.49 | 1.29 | PASS | no | eval |
| carry_fxrv_lowvol_only_21d | year_2026 | 0.049% | 62.5% | 83% | 0.41 | 0.89 | PASS | no | eval |
| carry_fxrv_lowvol_only_21d | holdout_365d | 0.256% | 75.0% | 56% | 1.30 | 1.24 | PASS | no | confirm_only |
| fxrv_usd_tilt_63d | full_sample | 0.026% | 20.6% | 28% | 0.57 | 0.16 | PASS | no | eval |
| fxrv_usd_tilt_63d | year_2024 | 0.346% | 27.3% | 100% | 1.17 | 2.29 | PASS | no | eval |
| fxrv_usd_tilt_63d | year_2025 | -0.370% | 9.1% | 100% | -1.18 | -1.39 | PASS | no | eval |
| fxrv_usd_tilt_63d | year_2026 | -0.046% | 37.5% | 100% | -0.74 | -0.69 | PASS | no | eval |
| fxrv_usd_tilt_63d | holdout_365d | -0.030% | 25.0% | 100% | -0.74 | -0.58 | PASS | no | confirm_only |
| fxrv_innov_usd_63d | full_sample | 0.012% | 29.4% | 28% | 0.27 | 0.07 | PASS | no | eval |
| fxrv_innov_usd_63d | year_2024 | 0.188% | 27.3% | 100% | 0.88 | 1.67 | PASS | no | eval |
| fxrv_innov_usd_63d | year_2025 | -0.355% | 9.1% | 100% | -1.31 | -1.63 | PASS | no | eval |
| fxrv_innov_usd_63d | year_2026 | 0.048% | 37.5% | 100% | 0.36 | 0.26 | PASS | no | eval |
| fxrv_innov_usd_63d | holdout_365d | 0.031% | 25.0% | 100% | 0.35 | 0.22 | PASS | no | confirm_only |
| fxrv_highvol_usd_63d | full_sample | 0.033% | 12.2% | 31% | 0.73 | 0.20 | PASS | no | eval |
| fxrv_highvol_usd_63d | year_2024 | 0.370% | 27.3% | 100% | 1.20 | 2.80 | PASS | no | eval |
| fxrv_highvol_usd_63d | year_2025 | -0.367% | 9.1% | 100% | -1.17 | -1.59 | PASS | no | eval |
| fxrv_highvol_usd_63d | year_2026 | 0.000% | 0.0% | nan% | 0.00 | nan | PASS | no | eval |
| fxrv_highvol_usd_63d | holdout_365d | 0.000% | 0.0% | nan% | 0.00 | nan | PASS | no | confirm_only |
| carry_fxrv_cool_63d | full_sample | 0.023% | 51.1% | 12% | 0.41 | 0.04 | PASS | no | eval |
| carry_fxrv_cool_63d | year_2024 | 0.164% | 63.6% | 62% | 0.71 | 0.94 | PASS | no | eval |
| carry_fxrv_cool_63d | year_2025 | 0.150% | 63.6% | 74% | 1.04 | 0.69 | PASS | no | eval |
| carry_fxrv_cool_63d | year_2026 | 0.087% | 62.5% | 76% | 0.62 | 0.90 | PASS | no | eval |
| carry_fxrv_cool_63d | holdout_365d | 0.282% | 75.0% | 52% | 1.37 | 1.20 | PASS | no | confirm_only |
| carry_fxrv_lowvol_only_63d | full_sample | 0.012% | 39.4% | 13% | 0.24 | 0.05 | PASS | no | eval |
| carry_fxrv_lowvol_only_63d | year_2024 | 0.000% | 54.5% | 72% | 0.00 | 0.27 | PASS | no | eval |
| carry_fxrv_lowvol_only_63d | year_2025 | 0.173% | 45.5% | 84% | 1.25 | 1.16 | PASS | no | eval |
| carry_fxrv_lowvol_only_63d | year_2026 | 0.068% | 50.0% | 94% | 0.47 | 0.92 | PASS | no | eval |
| carry_fxrv_lowvol_only_63d | holdout_365d | 0.270% | 66.7% | 55% | 1.26 | 1.24 | PASS | no | confirm_only |
| fxrv_ew_21d | full_sample | 0.026% | 38.3% | 22% | 0.67 | 0.17 | PASS | no | eval |
| fxrv_ew_21d | year_2024 | 0.049% | 36.4% | 100% | 0.29 | 0.39 | PASS | no | eval |
| fxrv_ew_21d | year_2025 | -0.292% | 9.1% | 100% | -1.14 | -1.53 | PASS | no | eval |
| fxrv_ew_21d | year_2026 | 0.048% | 62.5% | 96% | 0.50 | -0.12 | PASS | no | eval |
| fxrv_ew_21d | holdout_365d | -0.012% | 41.7% | 96% | -0.23 | -0.17 | PASS | no | confirm_only |

## FTMO risk sweep (IS → OOS confirm)

IS = [2011-09-18 … 2025-09-14]; OOS = [2025-09-15 … 2026-09-15]. Run because ≥1 factor had positive IS mean.

| Strategy | scale | bind | IS mean_mo | IS static | IS daily | IS gate | OOS mean_mo | OOS gate |
|----------|------:|:----:|-----------:|----------:|---------:|:-------:|-----------:|:--------:|
| carry_raw | 0.96 | static | -0.021% | 10.00% | 2.96% | PASS | 0.298% | PASS |
| fxrv_usd_tilt_21d | 3.29 | daily | 0.077% | 0.91% | 5.00% | PASS | -0.082% | PASS |
| fxrv_innov_usd_21d | 3.06 | daily | 0.106% | 2.52% | 5.00% | PASS | 0.003% | PASS |
| fxrv_highvol_usd_21d | 3.35 | daily | 0.084% | 3.31% | 5.00% | PASS | 0.000% | PASS |
| carry_fxrv_cool_21d | 1.41 | static | 0.000% | 10.00% | 2.42% | PASS | 0.415% | PASS |
| carry_fxrv_lowvol_only_21d | 1.18 | static | -0.004% | 10.00% | 2.07% | PASS | 0.302% | PASS |
| fxrv_usd_tilt_63d | 3.32 | daily | 0.101% | 4.56% | 5.00% | PASS | -0.101% | PASS |
| fxrv_innov_usd_63d | 3.53 | daily | 0.037% | 0.71% | 5.00% | PASS | 0.107% | PASS |
| fxrv_highvol_usd_63d | 3.24 | daily | 0.115% | 4.01% | 5.00% | PASS | 0.000% | PASS |
| carry_fxrv_cool_63d | 1.34 | static | 0.008% | 10.00% | 2.26% | PASS | 0.378% | PASS |
| carry_fxrv_lowvol_only_63d | 1.50 | static | -0.007% | 10.00% | 2.65% | PASS | 0.404% | PASS |
| fxrv_ew_21d | 3.17 | daily | 0.092% | 1.31% | 5.00% | PASS | -0.038% | PASS |

## Promote / 1%/mo bar

**Unscaled joint promote:** **NO**. **Scaled primary (`carry_fxrv_cool_21d`) promote:** **NO**. Do not claim 1%/mo unless earned. Locked sleeve untouched; Yahoo ≠ FTMO MT5; no go-live. Distinct from VIX-only regime (§8).

### Honest gaps

- Yahoo D1 abs-return average ≠ OTC tick FX vol / option-implied FXVIX.
- G10 subset only (7 USD majors); no EM carries in the RV basket.
- Innovation proxy is RV − trailing mean, not a full ARMA residual.
- Menkhoff risk *premium* is about pricing carry crashes — tradable USD-tilt / cool rules are research proxies, not paper replications.

Artifacts: `scholarly_fx_rv_*.csv`, `scholarly_fx_rv_meta.json`.
