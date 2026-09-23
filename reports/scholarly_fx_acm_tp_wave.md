# Scholarly FX: NY Fed ACM term premium → USD risk-appetite wave (§44)

**Path:** ACM `THREEFYTP10` USD-haven / carry-conditioned risk-appetite channel (Adrian–Crump–Moench / Lustig–Stathopoulos–Verdelhan / Hofmann–Shim–Shin) — **distinct** from yield-curve slope (§16), real-rate/TIPS (§23), funding-liq (§20), IG OAS (§36).
**Data:** `approximate_non_ftmo` + free FRED ACM THREEFYTP10/05. **PIT:** daily_pub_lag=1d, signal_lag=1d, z_window=252, z_high=1.0, cool=0.35, usd_tilt=0.5. **Note:** FRED fredgraph.csv THREEFYTP10 ACM 10y term premium ~1990–present live.
**Primary:** `acm_tp_usd` (long USD when lagged z(THREEFYTP10) ≥ z_high). Locked sleeve untouched (no cooler overlay).

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| acm_tp_usd | +0.007% | +0.18 | +0.20 | 22% | 26% | +0.05 |
| acm_tp_lvl_usd | +0.040% | +0.94 | +1.05 | 32% | 18% | +0.24 |
| acm_tp_stress_fx | -0.014% | -0.35 | -0.39 | 17% | 27% | -0.09 |
| acm_tp5_usd | +0.040% | +0.98 | +1.12 | 25% | 21% | +0.27 |
| acm_tp_chg_usd | +0.008% | +0.27 | +0.27 | 39% | 22% | +0.07 |
| carry_acm_tp_cool | +0.007% | +0.11 | +0.13 | 52% | 13% | -0.02 |
| carry_acm_tp_loose | +0.008% | +0.14 | +0.18 | 33% | 16% | +0.03 |
| acm_tp_ew | +0.018% | +0.61 | +0.67 | 46% | 19% | +0.16 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| acm_tp_usd | year_2024 | +0.14% | 36% | PASS | no |
| acm_tp_usd | year_2025 | -0.08% | 0% | PASS | no |
| acm_tp_usd | year_2026 | +0.03% | 62% | PASS | no |
| acm_tp_usd | holdout_365d | -0.13% | 42% | PASS | no |
| acm_tp_lvl_usd | year_2024 | +0.19% | 45% | PASS | no |
| acm_tp_lvl_usd | year_2025 | -0.21% | 9% | PASS | no |
| acm_tp_lvl_usd | year_2026 | +0.07% | 62% | PASS | no |
| acm_tp_lvl_usd | holdout_365d | -0.10% | 42% | PASS | no |
| acm_tp_stress_fx | year_2024 | -0.14% | 18% | PASS | no |
| acm_tp_stress_fx | year_2025 | +0.07% | 9% | PASS | no |
| acm_tp_stress_fx | year_2026 | -0.05% | 38% | PASS | no |
| acm_tp_stress_fx | holdout_365d | +0.12% | 33% | PASS | no |
| acm_tp5_usd | year_2024 | +0.12% | 27% | PASS | no |
| acm_tp5_usd | year_2025 | -0.08% | 0% | PASS | no |
| acm_tp5_usd | year_2026 | +0.08% | 62% | PASS | no |
| acm_tp5_usd | holdout_365d | -0.10% | 42% | PASS | no |
| acm_tp_chg_usd | year_2024 | -0.06% | 45% | PASS | no |
| acm_tp_chg_usd | year_2025 | -0.22% | 18% | PASS | no |
| acm_tp_chg_usd | year_2026 | +0.09% | 50% | PASS | no |
| acm_tp_chg_usd | holdout_365d | +0.00% | 42% | PASS | no |
| carry_acm_tp_cool | year_2024 | +0.06% | 64% | PASS | no |
| carry_acm_tp_cool | year_2025 | +0.13% | 64% | PASS | no |
| carry_acm_tp_cool | year_2026 | +0.12% | 75% | PASS | no |
| carry_acm_tp_cool | holdout_365d | +0.25% | 83% | PASS | no |
| carry_acm_tp_loose | year_2024 | -0.14% | 36% | PASS | no |
| carry_acm_tp_loose | year_2025 | +0.26% | 55% | PASS | no |
| carry_acm_tp_loose | year_2026 | +0.05% | 25% | PASS | no |
| carry_acm_tp_loose | holdout_365d | +0.16% | 42% | PASS | no |
| acm_tp_ew | year_2024 | +0.06% | 55% | PASS | no |
| acm_tp_ew | year_2025 | -0.12% | 18% | PASS | no |
| acm_tp_ew | year_2026 | +0.07% | 62% | PASS | no |
| acm_tp_ew | holdout_365d | -0.07% | 50% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_acm_tp_*.csv`, `scholarly_fx_acm_tp_meta.json`.
