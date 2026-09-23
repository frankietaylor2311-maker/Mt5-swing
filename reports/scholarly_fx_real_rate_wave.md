# Scholarly FX: real-rate / breakeven differential wave

**Path:** Frankel / Meese–Rogoff real interest differential + US TIPS/BE state — **distinct** from nominal yield-curve (§16), PPP/CPI real FX, and CB-BS QE (§22).
**Data:** `approximate_non_ftmo` + free FRED `DFII10` / `T10YIE` / `DGS10` + OECD LT − CPI YoY proxies. **PIT:** daily pub_lag=1d, monthly pub_lag=1m + signal_lag=1d / 1m.
**Primary:** `us_real_usd` (high US real → long USD). Foreign real = LT−CPI proxy (not linkers). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| us_real_usd | +0.012% | +0.31 | +0.30 | 21% | 23% | +0.08 |
| us_real_chg_usd | +0.084% | +2.46 | +2.24 | 28% | 20% | +0.65 |
| us_be_fx | -0.062% | -1.81 | -1.97 | 16% | 35% | -0.51 |
| us_be_usd | +0.053% | +1.54 | +1.73 | 19% | 25% | +0.44 |
| rr_xs | -0.068% | -1.04 | -1.16 | 50% | 10% | -0.22 |
| rr_z_xs | -0.056% | -0.94 | -1.04 | 49% | 14% | -0.15 |
| rr_chg_xs | -0.034% | -0.51 | -0.48 | 50% | 12% | -0.10 |
| rr_ew | -0.027% | -0.75 | -0.70 | 48% | 12% | -0.16 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| us_real_usd | year_2024 | -0.09% | 9% | PASS | no |
| us_real_usd | year_2025 | -0.08% | 0% | PASS | no |
| us_real_usd | year_2026 | +0.04% | 25% | PASS | no |
| us_real_usd | holdout_365d | +0.03% | 17% | PASS | no |
| us_real_chg_usd | year_2024 | +0.13% | 36% | PASS | no |
| us_real_chg_usd | year_2025 | +0.01% | 18% | PASS | no |
| us_real_chg_usd | year_2026 | +0.02% | 38% | PASS | no |
| us_real_chg_usd | holdout_365d | +0.03% | 42% | PASS | no |
| us_be_fx | year_2024 | -0.11% | 9% | PASS | no |
| us_be_fx | year_2025 | +0.05% | 18% | PASS | no |
| us_be_fx | year_2026 | +0.02% | 12% | PASS | no |
| us_be_fx | holdout_365d | +0.02% | 8% | PASS | no |
| us_be_usd | year_2024 | +0.10% | 18% | PASS | no |
| us_be_usd | year_2025 | -0.07% | 18% | PASS | no |
| us_be_usd | year_2026 | -0.05% | 12% | PASS | no |
| us_be_usd | holdout_365d | -0.03% | 8% | PASS | no |
| rr_xs | year_2024 | -0.10% | 73% | PASS | no |
| rr_xs | year_2025 | -0.19% | 45% | PASS | no |
| rr_xs | year_2026 | +0.14% | 62% | PASS | no |
| rr_xs | holdout_365d | +0.04% | 50% | PASS | no |
| rr_z_xs | year_2024 | +0.03% | 55% | PASS | no |
| rr_z_xs | year_2025 | -0.21% | 45% | PASS | no |
| rr_z_xs | year_2026 | +0.14% | 62% | PASS | no |
| rr_z_xs | holdout_365d | +0.04% | 50% | PASS | no |
| rr_chg_xs | year_2024 | -0.08% | 73% | PASS | no |
| rr_chg_xs | year_2025 | -0.08% | 36% | PASS | no |
| rr_chg_xs | year_2026 | -0.15% | 25% | PASS | no |
| rr_chg_xs | holdout_365d | -0.10% | 33% | PASS | no |
| rr_ew | year_2024 | -0.10% | 55% | PASS | no |
| rr_ew | year_2025 | -0.13% | 36% | PASS | no |
| rr_ew | year_2026 | +0.09% | 38% | PASS | no |
| rr_ew | holdout_365d | +0.03% | 33% | PASS | no |

**Board:** n=8 soft_nw_pos=2 hard_nw_pos=1 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_real_rate_*.csv`, `scholarly_fx_real_rate_meta.json`.
