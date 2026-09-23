# Scholarly FX: ICE BofA IG OAS / credit risk-appetite wave (§36)

**Path:** ICE BofA IG OAS (`BAMLC0A0CM`) USD-haven / carry-conditioned risk-appetite channel (BNP / Menkhoff) — **distinct** from funding-liq BAA10Y/NFCI (§20), VIX, GPR, FX-RV, EPU, crash-skew (§26), house-price (§35), BIS credit-gap (§32).
**Data:** `approximate_non_ftmo` + free FRED ICE BofA OAS. **PIT:** daily_pub_lag=1d, signal_lag=1d, z_window=252, z_high=1.0, cool=0.35, usd_tilt=0.5. **Note:** FRED fredgraph.csv ICE BofA OAS truncated to ~3y without API key.
**Primary:** `ig_oas_usd` (long USD when lagged z(IG OAS) ≥ z_high). Locked sleeve untouched (no cooler overlay).

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| ig_oas_usd | -0.009% | -0.82 | -0.94 | 1% | 100% | -0.19 |
| ig_oas_lvl_usd | -0.010% | -0.84 | -0.99 | 3% | 93% | -0.21 |
| oas_stress_fx | +0.008% | +0.79 | +0.93 | 1% | 100% | +0.18 |
| hy_oas_usd | -0.012% | -0.92 | -1.06 | 1% | 100% | -0.26 |
| ig_oas_chg_usd | -0.017% | -1.05 | -0.96 | 6% | 62% | -0.42 |
| carry_ig_oas_cool | -0.003% | -0.04 | -0.05 | 52% | 10% | -0.04 |
| carry_ig_oas_loose | +0.024% | +1.18 | +0.91 | 11% | 26% | +0.27 |
| oas_ew | -0.013% | -0.98 | -1.00 | 7% | 58% | -0.33 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| ig_oas_usd | year_2024 | +0.00% | 0% | PASS | no |
| ig_oas_usd | year_2025 | -0.14% | 9% | PASS | no |
| ig_oas_usd | year_2026 | +0.00% | 0% | PASS | no |
| ig_oas_usd | holdout_365d | +0.00% | 0% | PASS | no |
| ig_oas_lvl_usd | year_2024 | -0.00% | 0% | PASS | no |
| ig_oas_lvl_usd | year_2025 | -0.16% | 18% | PASS | no |
| ig_oas_lvl_usd | year_2026 | -0.00% | 50% | PASS | no |
| ig_oas_lvl_usd | holdout_365d | -0.00% | 33% | PASS | no |
| oas_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| oas_stress_fx | year_2025 | +0.13% | 18% | PASS | no |
| oas_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| oas_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| hy_oas_usd | year_2024 | -0.00% | 0% | PASS | no |
| hy_oas_usd | year_2025 | -0.17% | 9% | PASS | no |
| hy_oas_usd | year_2026 | -0.02% | 12% | PASS | no |
| hy_oas_usd | holdout_365d | -0.02% | 8% | PASS | no |
| ig_oas_chg_usd | year_2024 | +0.05% | 55% | PASS | no |
| ig_oas_chg_usd | year_2025 | -0.31% | 18% | PASS | no |
| ig_oas_chg_usd | year_2026 | -0.04% | 25% | PASS | no |
| ig_oas_chg_usd | holdout_365d | -0.04% | 25% | PASS | no |
| carry_ig_oas_cool | year_2024 | +0.12% | 64% | PASS | no |
| carry_ig_oas_cool | year_2025 | +0.11% | 64% | PASS | no |
| carry_ig_oas_cool | year_2026 | +0.16% | 75% | PASS | no |
| carry_ig_oas_cool | holdout_365d | +0.33% | 83% | PASS | no |
| carry_ig_oas_loose | year_2024 | +0.06% | 55% | PASS | no |
| carry_ig_oas_loose | year_2025 | +0.13% | 55% | PASS | no |
| carry_ig_oas_loose | year_2026 | +0.25% | 75% | PASS | no |
| carry_ig_oas_loose | holdout_365d | +0.39% | 83% | PASS | no |
| oas_ew | year_2024 | +0.02% | 55% | PASS | no |
| oas_ew | year_2025 | -0.21% | 27% | PASS | no |
| oas_ew | year_2026 | -0.02% | 25% | PASS | no |
| oas_ew | holdout_365d | -0.02% | 25% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_ig_oas_*.csv`, `scholarly_fx_ig_oas_meta.json`.
