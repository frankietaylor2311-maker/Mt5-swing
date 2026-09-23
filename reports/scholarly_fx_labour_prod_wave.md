# Scholarly FX: OECD labour-productivity (ULQELP01) differential wave (§49)

**Path:** OECD MEI early-estimate labour-productivity YoY via FRED (`ULQELP01{ISO2}Q657S`) — LP YoY XS productivity — **companion** to ULC (§48); **distinct** from employment (§41), IP (§42), ULC (§48), Balassa–Samuelson IP-levels (§14), retail (§46), DSR (§47).
**Data:** `approximate_non_ftmo` + free FRED ULQELP01 Q657S YoY (quarterly → monthly after pub_lag). **STALE** — all G10 end ~2023-04..07 on free FRED. EUR = `ULQELP01DEQ657S` (Germany proxy). Full G10 mapped. `n_long=n_short=2`. **PIT:** pub_lag_months=3 (LP labour a priori) + signal_lag_months=1 + 1d weight lag. Score basis = **YoY growth as reported**.
**Primary:** `high_lp_xs` (long high relative LP YoY / short low — productivity → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_lp_xs | -0.056% | -0.91 | -1.04 | 47% | 14% | -0.26 |
| low_lp_xs | +0.044% | +0.71 | +0.82 | 52% | 9% | +0.22 |
| high_lp_z_xs | -0.125% | -1.99 | -2.22 | 42% | 15% | -0.52 |
| lp_chg_xs | -0.081% | -1.27 | -1.40 | 44% | 12% | -0.33 |
| us_lp_stress_fx | -0.010% | -0.47 | -0.45 | 6% | 62% | -0.13 |
| us_lp_haven_usd | +0.010% | +0.43 | +0.42 | 7% | 50% | +0.12 |
| lp_ew | -0.049% | -1.32 | -1.49 | 47% | 15% | -0.35 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_lp_xs | year_2024 | +0.06% | 55% | PASS | no |
| high_lp_xs | year_2025 | -0.34% | 27% | PASS | no |
| high_lp_xs | year_2026 | +0.05% | 50% | PASS | no |
| high_lp_xs | holdout_365d | -0.20% | 33% | PASS | no |
| low_lp_xs | year_2024 | -0.06% | 45% | PASS | no |
| low_lp_xs | year_2025 | +0.34% | 73% | PASS | no |
| low_lp_xs | year_2026 | -0.05% | 50% | PASS | no |
| low_lp_xs | holdout_365d | +0.20% | 67% | PASS | no |
| high_lp_z_xs | year_2024 | -0.10% | 55% | PASS | no |
| high_lp_z_xs | year_2025 | -0.46% | 9% | PASS | no |
| high_lp_z_xs | year_2026 | -0.03% | 50% | PASS | no |
| high_lp_z_xs | holdout_365d | -0.21% | 33% | PASS | no |
| lp_chg_xs | year_2024 | +0.18% | 64% | PASS | no |
| lp_chg_xs | year_2025 | +0.09% | 64% | PASS | no |
| lp_chg_xs | year_2026 | +0.19% | 50% | PASS | no |
| lp_chg_xs | holdout_365d | +0.04% | 50% | PASS | no |
| us_lp_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_lp_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_lp_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_lp_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_lp_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_lp_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_lp_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_lp_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| lp_ew | year_2024 | +0.08% | 64% | PASS | no |
| lp_ew | year_2025 | -0.08% | 45% | PASS | no |
| lp_ew | year_2026 | +0.08% | 50% | PASS | no |
| lp_ew | holdout_365d | -0.05% | 42% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_labour_prod_*.csv`, `scholarly_fx_labour_prod_meta.json`.

**Stale honesty:** free FRED ULQELP01 panel ends ~2023-04..07 for all G10 — post-2023 ranks are ffilled frozen. IMF IFS UV/ToT fallback 404 on FRED.

Sweep primary `high_lp_xs` scale≈**0.918** bind=**static**; scaled HO clear **NO**. Soft-best-ish = `low_lp_xs` (~+4.4 bp/mo, NW t≈+0.82).

Locked `fx4plus_gbpcad_d1_voltarget_0025` **untouched PASS** (see `quest_locked_verify_labour_prod.md`).

**No go-live** under `approximate_non_ftmo`.
