# Scholarly FX: World Uncertainty Index (WUI) differential wave (§40)

**Path:** Ahir–Bloom–Furceri country WUI via FRED (`WUIUSA` / `WUIDEU` / …) — uncertainty XS on **levels** — **distinct** from EPU/TPU, GPR/AI-GPR, OECD CLI (§37) / CCI (§38) / BCI (§39), IG OAS (§36), coincident macro-diff CPI/IP/UR, CA/TB, fiscal/debt, BIS REER/credit, reserves, house-price, money-growth, equity-diff, crash-skew, dollar-beta.
**Data:** `approximate_non_ftmo` + free FRED country WUI (quarterly → monthly after pub_lag). EUR = `WUIDEU` (Germany proxy). Full G10 mapped. `n_long=n_short=2` a priori. **PIT:** pub_lag_months=4 (conservative quarterly EIU/WUI) + signal_lag_months=0 + 1d weight lag. Score basis = **levels** (WUI is already an index).
**Primary:** `low_wui_xs` (long low relative WUI / short high — calmer uncertainty → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| low_wui_xs | -0.071% | -1.11 | -1.07 | 49% | 12% | -0.27 |
| high_wui_xs | +0.061% | +0.96 | +0.93 | 52% | 13% | +0.24 |
| low_wui_z_xs | +0.073% | +1.17 | +1.23 | 57% | 9% | +0.25 |
| wui_chg_xs | +0.023% | +0.37 | +0.39 | 56% | 8% | +0.05 |
| us_wui_stress_fx | +0.018% | +0.54 | +0.67 | 13% | 27% | +0.14 |
| us_wui_haven_usd | -0.019% | -0.57 | -0.71 | 14% | 29% | -0.14 |
| wui_ew | -0.010% | -0.25 | -0.25 | 51% | 11% | -0.09 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| low_wui_xs | year_2024 | +0.00% | 64% | PASS | no |
| low_wui_xs | year_2025 | -0.24% | 18% | PASS | no |
| low_wui_xs | year_2026 | +0.21% | 62% | PASS | no |
| low_wui_xs | holdout_365d | +0.25% | 58% | PASS | no |
| high_wui_xs | year_2024 | +0.01% | 55% | PASS | no |
| high_wui_xs | year_2025 | +0.23% | 82% | PASS | no |
| high_wui_xs | year_2026 | -0.21% | 38% | PASS | no |
| high_wui_xs | holdout_365d | -0.25% | 42% | PASS | no |
| low_wui_z_xs | year_2024 | -0.05% | 45% | PASS | no |
| low_wui_z_xs | year_2025 | +0.10% | 73% | PASS | no |
| low_wui_z_xs | year_2026 | +0.24% | 62% | PASS | no |
| low_wui_z_xs | holdout_365d | +0.31% | 67% | PASS | no |
| wui_chg_xs | year_2024 | -0.19% | 45% | PASS | no |
| wui_chg_xs | year_2025 | -0.23% | 36% | PASS | no |
| wui_chg_xs | year_2026 | +0.31% | 62% | PASS | no |
| wui_chg_xs | holdout_365d | +0.36% | 67% | PASS | no |
| us_wui_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_wui_stress_fx | year_2025 | -0.00% | 27% | PASS | no |
| us_wui_stress_fx | year_2026 | -0.19% | 38% | PASS | no |
| us_wui_stress_fx | holdout_365d | -0.01% | 42% | PASS | no |
| us_wui_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_wui_haven_usd | year_2025 | +0.00% | 36% | PASS | no |
| us_wui_haven_usd | year_2026 | +0.19% | 62% | PASS | no |
| us_wui_haven_usd | holdout_365d | +0.01% | 58% | PASS | no |
| wui_ew | year_2024 | -0.06% | 45% | PASS | no |
| wui_ew | year_2025 | -0.16% | 36% | PASS | no |
| wui_ew | year_2026 | +0.11% | 75% | PASS | no |
| wui_ew | holdout_365d | +0.20% | 67% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_wui_*.csv`, `scholarly_fx_wui_meta.json`.
