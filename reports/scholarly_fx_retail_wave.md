# Scholarly FX: Retail-sales (SLRTTO / RSAFS) differential wave (§46)

**Path:** OECD MEI retail volume growth via FRED (`SLRTTO01*Q657S`) + USD `RSAFS` / CAD `CANWSCNDW01IXOBSAM`→YoY — retail YoY XS — **distinct** from IP (§42), employment (§41), building-permits (§43), OECD CLI/CCI/BCI, macro_diff EW, CA/TB, fiscal/debt, house-price, money, equity, IG OAS, commodity, GPR, ACM TP, WUI, EPU/TPU.
**Data:** `approximate_non_ftmo` + free FRED SLRTTO / RSAFS (quarterly + monthly → YoY after pub_lag). EUR = `SLRTTO01DEQ657S` (Germany proxy). CAD gap ~2024-04. Full G10 mapped. `n_long=n_short=2` a priori. **PIT:** pub_lag_months=2 (retail a priori) + signal_lag_months=0 + 1d weight lag. Score basis = **YoY / growth %**.
**Primary:** `high_retail_xs` (long high relative retail YoY / short low — Dahlquist–Hasseltoft → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_retail_xs | +0.008% | +0.13 | +0.14 | 46% | 12% | +0.02 |
| low_retail_xs | -0.029% | -0.45 | -0.50 | 51% | 12% | -0.09 |
| high_retail_z_xs | +0.059% | +1.00 | +1.08 | 53% | 9% | +0.22 |
| retail_chg_xs | +0.062% | +1.00 | +1.14 | 53% | 11% | +0.25 |
| us_retail_stress_fx | -0.021% | -1.04 | -0.80 | 8% | 60% | -0.23 |
| us_retail_haven_usd | +0.020% | +0.97 | +0.75 | 7% | 40% | +0.21 |
| retail_ew | +0.017% | +0.42 | +0.46 | 52% | 12% | +0.10 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_retail_xs | year_2024 | +0.25% | 36% | PASS | no |
| high_retail_xs | year_2025 | +0.24% | 64% | PASS | no |
| high_retail_xs | year_2026 | +0.06% | 50% | PASS | no |
| high_retail_xs | holdout_365d | +0.23% | 67% | PASS | no |
| low_retail_xs | year_2024 | -0.26% | 64% | PASS | no |
| low_retail_xs | year_2025 | -0.25% | 36% | PASS | no |
| low_retail_xs | year_2026 | -0.07% | 50% | PASS | no |
| low_retail_xs | holdout_365d | -0.24% | 33% | PASS | no |
| high_retail_z_xs | year_2024 | +0.20% | 45% | PASS | no |
| high_retail_z_xs | year_2025 | +0.34% | 82% | PASS | no |
| high_retail_z_xs | year_2026 | -0.04% | 38% | PASS | no |
| high_retail_z_xs | holdout_365d | +0.18% | 58% | PASS | no |
| retail_chg_xs | year_2024 | +0.34% | 55% | PASS | no |
| retail_chg_xs | year_2025 | +0.10% | 64% | PASS | no |
| retail_chg_xs | year_2026 | +0.03% | 50% | PASS | no |
| retail_chg_xs | holdout_365d | +0.02% | 58% | PASS | no |
| us_retail_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_retail_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_retail_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_retail_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_retail_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_retail_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_retail_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_retail_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| retail_ew | year_2024 | +0.19% | 45% | PASS | no |
| retail_ew | year_2025 | +0.11% | 64% | PASS | no |
| retail_ew | year_2026 | +0.03% | 38% | PASS | no |
| retail_ew | holdout_365d | +0.08% | 50% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_retail_*.csv`, `scholarly_fx_retail_meta.json`.

Locked `fx4plus_gbpcad_d1_voltarget_0025` **untouched**; re-verify PASS (see `quest_locked_verify_retail.md`).
