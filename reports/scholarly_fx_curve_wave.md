# Scholarly FX: term-structure / yield-curve factors

**Prior (fixed):** Relative yield-curve slope and level vs USD predict FX (Chen–Tsang / Ang–Chen style); slope×carry interaction; secondary UIP on short / 3m differentials. pub_lag=1m + signal_lag=1m; n_long=n_short=2; costs=1.5 bps/side. No HO tuning.

**Data:** `approximate_non_ftmo` D1 FX + FRED OECD LT (`IRLTLT01*`) − immediate (`IRSTCI01*`) slope. LT cols: AUD, CAD, CHF, EUR, GBP, JPY, NZD, USD. IR3M secondary: yes.

## Factor board (full sample, unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| curve_slope_xs | 0.045% | 0.71 | 0.71 | 47.8% | 13% | 0.18 |
| curve_lt_xs | -0.006% | -0.07 | -0.08 | 50.0% | 11% | -0.05 |
| curve_slope_z_xs | 0.039% | 0.64 | 0.75 | 48.3% | 12% | 0.17 |
| slope_x_carry | -0.024% | -0.37 | -0.42 | 51.1% | 13% | -0.04 |
| uip_st_xs | -0.033% | -0.42 | -0.51 | 51.7% | 10% | -0.13 |
| uip_ir3m_xs | -0.033% | -0.42 | -0.49 | 52.8% | 10% | -0.13 |
| curve_ew | 0.020% | 0.34 | 0.42 | 50.0% | 14% | 0.06 |

## Calendar / holdout windows (unscaled)

| Strategy | Window | mean_mo | %pos | top3 | t NW | Sharpe | gates | 1% bar | role |
|----------|--------|--------:|-----:|-----:|-----:|-------:|:-----:|:------:|------|
| curve_slope_xs | full_sample | 0.045% | 47.8% | 13% | 0.71 | 0.18 | PASS | no | eval |
| curve_slope_xs | year_2024 | -0.112% | 45.5% | 87% | -0.55 | -1.00 | PASS | no | eval |
| curve_slope_xs | year_2025 | -0.140% | 27.3% | 100% | -1.04 | -0.29 | PASS | no | eval |
| curve_slope_xs | year_2026 | 0.189% | 50.0% | 94% | 0.79 | 0.23 | PASS | no | eval |
| curve_slope_xs | holdout_365d | -0.089% | 33.3% | 94% | -0.41 | -0.19 | PASS | no | confirm_only |
| curve_lt_xs | full_sample | -0.006% | 50.0% | 11% | -0.08 | -0.05 | FAIL | no | eval |
| curve_lt_xs | year_2024 | -0.039% | 63.6% | 55% | -0.12 | -0.12 | PASS | no | eval |
| curve_lt_xs | year_2025 | -0.009% | 63.6% | 69% | -0.05 | -0.09 | PASS | no | eval |
| curve_lt_xs | year_2026 | 0.212% | 62.5% | 76% | 1.18 | 1.02 | PASS | no | eval |
| curve_lt_xs | holdout_365d | 0.320% | 75.0% | 44% | 1.70 | 1.03 | PASS | no | confirm_only |
| curve_slope_z_xs | full_sample | 0.039% | 48.3% | 12% | 0.75 | 0.17 | PASS | no | eval |
| curve_slope_z_xs | year_2024 | 0.087% | 54.5% | 89% | 0.52 | -0.04 | PASS | no | eval |
| curve_slope_z_xs | year_2025 | -0.135% | 27.3% | 100% | -0.96 | -0.43 | PASS | no | eval |
| curve_slope_z_xs | year_2026 | 0.210% | 50.0% | 100% | 1.18 | 0.37 | PASS | no | eval |
| curve_slope_z_xs | holdout_365d | 0.083% | 50.0% | 88% | 0.69 | 0.46 | PASS | no | confirm_only |
| slope_x_carry | full_sample | -0.024% | 51.1% | 13% | -0.42 | -0.04 | FAIL | no | eval |
| slope_x_carry | year_2024 | -0.070% | 54.5% | 72% | -0.40 | -0.01 | PASS | no | eval |
| slope_x_carry | year_2025 | 0.007% | 36.4% | 96% | 0.05 | -0.20 | PASS | no | eval |
| slope_x_carry | year_2026 | 0.207% | 50.0% | 92% | 1.39 | 1.52 | PASS | no | eval |
| slope_x_carry | holdout_365d | 0.181% | 50.0% | 81% | 1.46 | 0.81 | PASS | no | confirm_only |
| uip_st_xs | full_sample | -0.033% | 51.7% | 10% | -0.51 | -0.13 | FAIL | no | eval |
| uip_st_xs | year_2024 | 0.063% | 72.7% | 64% | 0.24 | 0.39 | PASS | no | eval |
| uip_st_xs | year_2025 | -0.068% | 63.6% | 74% | -0.43 | -0.29 | PASS | no | eval |
| uip_st_xs | year_2026 | 0.028% | 50.0% | 79% | 0.20 | 0.34 | PASS | no | eval |
| uip_st_xs | holdout_365d | 0.166% | 66.7% | 48% | 1.18 | 0.42 | PASS | no | confirm_only |
| uip_ir3m_xs | full_sample | -0.033% | 52.8% | 10% | -0.49 | -0.13 | FAIL | no | eval |
| uip_ir3m_xs | year_2024 | 0.063% | 72.7% | 64% | 0.24 | 0.39 | PASS | no | eval |
| uip_ir3m_xs | year_2025 | 0.101% | 72.7% | 59% | 0.57 | 0.15 | PASS | no | eval |
| uip_ir3m_xs | year_2026 | 0.190% | 75.0% | 69% | 0.92 | 0.96 | PASS | no | eval |
| uip_ir3m_xs | holdout_365d | 0.308% | 83.3% | 43% | 1.50 | 1.17 | PASS | no | confirm_only |
| curve_ew | full_sample | 0.020% | 50.0% | 14% | 0.42 | 0.06 | PASS | no | eval |
| curve_ew | year_2024 | -0.075% | 63.6% | 67% | -0.53 | -0.78 | PASS | no | eval |
| curve_ew | year_2025 | -0.073% | 45.5% | 85% | -1.15 | -0.22 | PASS | no | eval |
| curve_ew | year_2026 | 0.201% | 87.5% | 70% | 1.54 | 1.35 | PASS | no | eval |
| curve_ew | holdout_365d | 0.116% | 75.0% | 60% | 1.34 | 1.02 | PASS | no | confirm_only |

## FTMO risk sweep (IS → OOS confirm)

IS = [2011-09-18 … 2025-09-14]; OOS = [2025-09-15 … 2026-09-15]. Run because ≥1 factor had positive IS mean.

| Strategy | scale | bind | IS mean_mo | IS static | IS daily | IS gate | OOS mean_mo | OOS gate |
|----------|------:|:----:|-----------:|----------:|---------:|:-------:|-----------:|:--------:|
| curve_slope_xs | 2.13 | daily | 0.106% | 7.52% | 5.00% | PASS | -0.189% | PASS |
| curve_lt_xs | 0.86 | static | -0.021% | 10.00% | 2.89% | PASS | 0.274% | PASS |
| curve_slope_z_xs | 3.75 | daily | 0.129% | 8.24% | 5.00% | PASS | 0.319% | PASS |
| slope_x_carry | 0.98 | static | -0.034% | 10.00% | 1.79% | PASS | 0.178% | PASS |
| uip_st_xs | 0.84 | static | -0.035% | 10.00% | 2.57% | PASS | 0.139% | PASS |
| uip_ir3m_xs | 0.68 | static | -0.038% | 10.00% | 2.06% | PASS | 0.210% | PASS |
| curve_ew | 1.47 | static | 0.019% | 10.00% | 3.12% | PASS | 0.170% | PASS |

## Promote / 1%/mo bar

**Unscaled joint promote:** **NO**. **Scaled primary (`curve_slope_xs`) promote:** **NO**. Do not claim 1%/mo unless earned. Locked sleeve untouched; Yahoo/FRED ≠ FTMO MT5; no go-live.

### Honest gaps

- OECD LT − immediate short is a **simple slope**, not Nelson–Siegel / Svensson level-slope-curvature from a full zero curve.
- EUR EZ LT can lag; DE bund fills gaps (documented).
- No paid swap / forward points — UIP legs use cash rate differentials only.
- Academic curve–FX premia are not prop-firm 1%/mo consistency claims.

Artifacts: `scholarly_fx_curve_*.csv`, `scholarly_fx_curve_meta.json`.
