# Scholarly FX combo wave (carry + mom + dollar TSMOM × GPR/VIX)

**data_source:** `approximate_non_ftmo` + FRED rates + VIX + Caldara–Iacoviello GPR

**signal_lag:** 1 | **rates pub_lag:** 1m | **VIX lag:** 1d | **GPR:** monthly 1m / daily 1d for regime+events

**Priors (fixed, no HO tuning):** EW 1/3 carry / mom / dollar_tsmom; carry cool=0.35 when max(z_VIX,z_GPR)≥1.0; USD tilt≤0.15 on GPR z; mom 63d/skip21; n=2/2.

## Full-sample monthly moments

| Factor | n_mo | mean_mo | t OLS | t NW | NW L | boot 95% CI | %pos | top3 | Sharpe |
|--------|-----:|--------:|------:|-----:|----:|------------|-----:|-----:|-------:|
| carry_rank | 180 | -0.001% | -0.02 | -0.02 | 4 | [-0.0015, 0.0014] | 51.1% | 10% | -0.04 |
| fx_momentum | 180 | -0.064% | -0.90 | -0.92 | 4 | [-0.0020, 0.0007] | 45.6% | 12% | -0.21 |
| dollar_tsmom | 180 | 0.012% | 0.09 | 0.11 | 4 | [-0.0023, 0.0026] | 50.0% | 11% | 0.03 |
| combo_ew_raw | 180 | -0.016% | -0.30 | -0.34 | 4 | [-0.0012, 0.0009] | 51.1% | 11% | -0.09 |
| carry_cooled | 180 | 0.008% | 0.14 | 0.15 | 4 | [-0.0010, 0.0012] | 52.2% | 12% | -0.02 |
| scholarly_combo | 180 | 0.028% | 0.36 | 0.38 | 4 | [-0.0012, 0.0018] | 52.2% | 12% | 0.08 |

## Calendar / holdout windows

| Factor | Window | mean_mo | %pos | top3 | t OLS | t NW | Sharpe | gates | 1% bar | role |
|--------|--------|--------:|-----:|-----:|------:|-----:|-------:|:-----:|:------:|------|
| carry_rank | IS_2018_2023 | -0.061% | 46.5% | 23% | -0.57 | -0.66 | -0.21 | PASS | no | eval |
| carry_rank | year_2024 | 0.148% | 63.6% | 59% | 0.56 | 0.64 | 0.76 | PASS | no | eval |
| carry_rank | year_2025 | 0.125% | 63.6% | 63% | 0.86 | 0.71 | 0.21 | PASS | no | eval |
| carry_rank | year_2026 | 0.128% | 62.5% | 80% | 0.70 | 0.74 | 1.01 | PASS | no | eval |
| carry_rank | holdout_365d | 0.310% | 75.0% | 53% | 2.12 | 1.43 | 1.35 | PASS | no | confirm_only |
| fx_momentum | IS_2018_2023 | -0.094% | 45.1% | 25% | -0.79 | -0.83 | -0.28 | PASS | no | eval |
| fx_momentum | year_2024 | -0.171% | 45.5% | 70% | -0.51 | -0.63 | -0.64 | PASS | no | eval |
| fx_momentum | year_2025 | 0.437% | 90.9% | 51% | 4.37 | 1.92 | 1.28 | PASS | no | eval |
| fx_momentum | year_2026 | 0.344% | 62.5% | 76% | 1.79 | 1.56 | 1.65 | PASS | no | eval |
| fx_momentum | holdout_365d | 0.384% | 75.0% | 50% | 2.86 | 1.95 | 2.05 | PASS | no | confirm_only |
| dollar_tsmom | IS_2018_2023 | -0.072% | 47.9% | 25% | -0.33 | -0.35 | -0.03 | FAIL | no | eval |
| dollar_tsmom | year_2024 | 0.803% | 63.6% | 75% | 1.92 | 1.64 | 1.26 | PASS | no | eval |
| dollar_tsmom | year_2025 | -0.387% | 45.5% | 93% | -0.87 | -1.01 | -0.71 | PASS | no | eval |
| dollar_tsmom | year_2026 | 0.005% | 37.5% | 100% | 0.01 | 0.02 | 0.59 | PASS | no | eval |
| dollar_tsmom | holdout_365d | 0.330% | 58.3% | 78% | 0.73 | 0.96 | 0.27 | PASS | no | confirm_only |
| combo_ew_raw | IS_2018_2023 | -0.074% | 47.9% | 26% | -0.85 | -0.84 | -0.24 | PASS | no | eval |
| combo_ew_raw | year_2024 | 0.262% | 81.8% | 76% | 0.99 | 1.23 | 1.05 | PASS | no | eval |
| combo_ew_raw | year_2025 | 0.063% | 54.5% | 79% | 0.36 | 0.46 | 0.04 | PASS | no | eval |
| combo_ew_raw | year_2026 | 0.160% | 62.5% | 89% | 1.14 | 1.27 | 1.59 | PASS | no | eval |
| combo_ew_raw | holdout_365d | 0.342% | 75.0% | 59% | 2.45 | 1.60 | 1.59 | PASS | no | confirm_only |
| carry_cooled | IS_2018_2023 | -0.057% | 45.1% | 26% | -0.78 | -0.82 | -0.28 | PASS | no | eval |
| carry_cooled | year_2024 | 0.139% | 72.7% | 68% | 0.69 | 0.81 | 1.12 | PASS | no | eval |
| carry_cooled | year_2025 | 0.154% | 63.6% | 74% | 1.41 | 1.27 | 0.65 | PASS | no | eval |
| carry_cooled | year_2026 | 0.051% | 75.0% | 74% | 0.38 | 0.50 | 0.65 | PASS | no | eval |
| carry_cooled | holdout_365d | 0.200% | 83.3% | 56% | 1.72 | 1.36 | 1.11 | PASS | no | confirm_only |
| scholarly_combo | IS_2018_2023 | -0.020% | 49.3% | 27% | -0.14 | -0.15 | -0.01 | PASS | no | eval |
| scholarly_combo | year_2024 | 0.403% | 72.7% | 57% | 1.38 | 1.09 | 1.66 | PASS | no | eval |
| scholarly_combo | year_2025 | 0.097% | 63.6% | 84% | 0.37 | 0.42 | 0.18 | PASS | no | eval |
| scholarly_combo | year_2026 | 0.358% | 62.5% | 84% | 1.00 | 1.39 | 0.82 | PASS | no | eval |
| scholarly_combo | holdout_365d | 0.295% | 66.7% | 74% | 1.18 | 1.64 | 0.84 | PASS | no | confirm_only |

## GPR top-decile event study (USD vs risk FX)

n_events=206 thr≈176.35 (top-decile lagged GPR, ≥5d separation)

| h | USD cum | risk FX cum | USD−risk |
|--:|--------:|------------:|---------:|
| -5 | 0.103% | -0.179% | 0.283% |
| -4 | 0.088% | -0.101% | 0.189% |
| -3 | 0.058% | -0.085% | 0.143% |
| -2 | 0.061% | -0.080% | 0.141% |
| -1 | 0.027% | -0.052% | 0.079% |
| 0 | 0.006% | -0.010% | 0.016% |
| 1 | 0.028% | -0.015% | 0.043% |
| 2 | 0.019% | 0.002% | 0.017% |
| 3 | 0.068% | -0.011% | 0.079% |
| 4 | 0.073% | -0.020% | 0.093% |
| 5 | 0.099% | -0.052% | 0.152% |
| 6 | 0.098% | 0.001% | 0.098% |
| 7 | 0.091% | 0.036% | 0.055% |
| 8 | 0.129% | 0.033% | 0.096% |
| 9 | 0.107% | 0.065% | 0.042% |
| 10 | 0.120% | 0.065% | 0.055% |

At h=+5: USD−risk ≈ **0.152%** (positive ⇒ USD outperformed risk FX after GPR spike).

At h=+10: USD−risk ≈ **0.055%** (positive ⇒ USD outperformed risk FX after GPR spike).

## Consistency bar vs ~1%/mo FTMO goal

- Any single window clears (≥1% mean, ≥70% pos, top3≤55%, gates PASS)? **NO**
- Joint year_2024+2025+2026+holdout clear for `scholarly_combo`? **NO**
- **Do not claim 1%/mo** unless joint clear — this wave does not promote.
- Still `approximate_non_ftmo`; need FTMO MT5 CSVs + optional news NLP for next unlock.
