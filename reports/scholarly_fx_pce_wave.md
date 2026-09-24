# Scholarly FX: OECD/FRED real private final consumption differential wave (§65)

**Path:** FRED OECD MEI **real private final consumption YoY** (`NAEXKP02*Q657S` QoQ→YoY; full G10; EUR=Germany DEQ proxy; not BEA PCE) — PCE YoY XS — **distinct** from GDP (§58), GFCF (§64), retail (§46), cars (§60), household credit (§63), construction (§59), IP (§42), CLI/CCI/BCI, emp/labour (§41/§54–57).
**Data:** `approximate_non_ftmo` + free FRED NAEXKP02 YoY (from Q657S QoQ) after pub_lag. EUR EA EZQ657S stale ~2023-01 → DEQ657S Germany proxy; full G10 mapped. `n_long=n_short=2` on available foreign panel. **PIT:** pub_lag_months=3 (quarterly PCE/NA lag a priori) + signal_lag_months=1 + 1d weight lag. Score basis = **real private final consumption YoY %** (OECD NAEXKP02, not BEA PCE).
**Primary:** `high_pce_xs` (long high relative PCE YoY / short low — consumption boom / economic-momentum → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_pce_xs | -0.025% | -0.33 | -0.44 | 51% | 11% | -0.13 |
| low_pce_xs | +0.019% | +0.25 | +0.33 | 48% | 12% | +0.11 |
| high_pce_z_xs | +0.044% | +0.67 | +0.85 | 52% | 12% | +0.13 |
| pce_chg_xs | +0.019% | +0.28 | +0.30 | 50% | 12% | +0.04 |
| us_pce_stress_fx | +0.018% | +0.90 | +0.94 | 4% | 59% | +0.26 |
| us_pce_haven_usd | -0.018% | -0.92 | -0.95 | 6% | 53% | -0.26 |
| pce_ew | +0.004% | +0.09 | +0.11 | 51% | 13% | -0.02 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_pce_xs | year_2024 | -0.03% | 55% | PASS | no |
| high_pce_xs | year_2025 | +0.23% | 73% | PASS | no |
| high_pce_xs | year_2026 | +0.23% | 62% | PASS | no |
| high_pce_xs | holdout_365d | +0.33% | 75% | PASS | no |
| low_pce_xs | year_2024 | +0.02% | 45% | PASS | no |
| low_pce_xs | year_2025 | -0.24% | 27% | PASS | no |
| low_pce_xs | year_2026 | -0.23% | 38% | PASS | no |
| low_pce_xs | holdout_365d | -0.34% | 25% | PASS | no |
| high_pce_z_xs | year_2024 | -0.11% | 27% | PASS | no |
| high_pce_z_xs | year_2025 | +0.10% | 36% | PASS | no |
| high_pce_z_xs | year_2026 | +0.19% | 75% | PASS | no |
| high_pce_z_xs | holdout_365d | +0.07% | 58% | PASS | no |
| pce_chg_xs | year_2024 | -0.01% | 27% | PASS | no |
| pce_chg_xs | year_2025 | -0.08% | 45% | PASS | no |
| pce_chg_xs | year_2026 | +0.35% | 75% | PASS | no |
| pce_chg_xs | holdout_365d | +0.32% | 67% | PASS | no |
| us_pce_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_pce_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_pce_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_pce_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_pce_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_pce_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_pce_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_pce_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| pce_ew | year_2024 | -0.01% | 36% | PASS | no |
| pce_ew | year_2025 | +0.05% | 64% | PASS | no |
| pce_ew | year_2026 | +0.19% | 75% | PASS | no |
| pce_ew | holdout_365d | +0.22% | 83% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_pce_*.csv`, `scholarly_fx_pce_meta.json`.
