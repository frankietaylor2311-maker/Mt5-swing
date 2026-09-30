# Scholarly FX: WUI-conditioned soft-signal EW wave (§85)

**Path:** Ahir–Bloom–Furceri **US WUI** (FRED WUIUSA) stress × Dahlquist–Hasseltoft ``soft_ew_macro5`` gate/cool — **distinct** from raw WUI XS (§40), soft_ew (§66 ungated), soft-stack CIP enrichment (§69), CIP-conditioned soft (§71), VIX/GPR soft (§72), EPU/TPU soft (§76), CIP/VIX/EPU × carry/mom/value/REER (§68/§73–§75/§77–§84), capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED WUIUSA + FRED/OECD soft legs (§66 PIT). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** WUI pub_lag_months=4 + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 mirror) + weight_lag_days=1; soft legs source-wave pub/signal lags. Costs 1.5 bps/side inside source factors (haven applies own costs).
**Primary:** `soft_low_wui` (trade soft only when lagged US WUI z ≤ 0). Companions: soft_wui_cool / soft_raw / soft_high_wui / us_wui_haven_usd / soft_wui_stack / soft_wui_ew / soft_wui_regime. Cool z_high=1.0 (§71–§84 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| soft_low_wui | +0.078% | +3.64 | +3.27 | 33% | 19% | +0.88 |
| soft_wui_cool | +0.101% | +4.18 | +3.54 | 61% | 13% | +1.01 |
| soft_raw | +0.118% | +4.26 | +3.63 | 61% | 11% | +1.02 |
| soft_high_wui | +0.023% | +1.85 | +1.37 | 17% | 33% | +0.44 |
| us_wui_haven_usd | -0.026% | -0.76 | -0.87 | 14% | 29% | -0.20 |
| soft_wui_stack | +0.078% | +3.64 | +3.27 | 33% | 19% | +0.88 |
| soft_wui_ew | +0.051% | +2.70 | +2.68 | 57% | 13% | +0.67 |
| soft_wui_regime | +0.052% | +1.26 | +1.41 | 46% | 12% | +0.32 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| soft_raw | year_2024 | +0.08% | 55% | PASS | no |
| soft_raw | year_2025 | +0.04% | 64% | PASS | no |
| soft_raw | year_2026 | +0.14% | 88% | PASS | no |
| soft_raw | holdout_365d | +0.06% | 67% | PASS | no |
| soft_wui_cool | year_2024 | +0.08% | 55% | PASS | no |
| soft_wui_cool | year_2025 | +0.03% | 64% | PASS | no |
| soft_wui_cool | year_2026 | +0.05% | 88% | PASS | no |
| soft_wui_cool | holdout_365d | +0.02% | 67% | PASS | no |
| soft_low_wui | year_2024 | +0.04% | 18% | PASS | no |
| soft_low_wui | year_2025 | -0.01% | 18% | PASS | no |
| soft_low_wui | year_2026 | +0.00% | 0% | PASS | no |
| soft_low_wui | holdout_365d | +0.00% | 0% | PASS | no |
| soft_high_wui | year_2024 | +0.00% | 0% | PASS | no |
| soft_high_wui | year_2025 | +0.01% | 36% | PASS | no |
| soft_high_wui | year_2026 | +0.14% | 88% | PASS | no |
| soft_high_wui | holdout_365d | +0.06% | 67% | PASS | no |
| us_wui_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_wui_haven_usd | year_2025 | +0.09% | 36% | PASS | no |
| us_wui_haven_usd | year_2026 | +0.19% | 62% | PASS | no |
| us_wui_haven_usd | holdout_365d | +0.01% | 58% | PASS | no |
| soft_wui_stack | year_2024 | +0.04% | 18% | PASS | no |
| soft_wui_stack | year_2025 | -0.01% | 18% | PASS | no |
| soft_wui_stack | year_2026 | +0.00% | 0% | PASS | no |
| soft_wui_stack | holdout_365d | +0.00% | 0% | PASS | no |
| soft_wui_ew | year_2024 | +0.04% | 55% | PASS | no |
| soft_wui_ew | year_2025 | +0.04% | 64% | PASS | no |
| soft_wui_ew | year_2026 | +0.08% | 62% | PASS | no |
| soft_wui_ew | holdout_365d | +0.01% | 58% | PASS | no |
| soft_wui_regime | year_2024 | +0.04% | 18% | PASS | no |
| soft_wui_regime | year_2025 | +0.08% | 55% | PASS | no |
| soft_wui_regime | year_2026 | +0.19% | 62% | PASS | no |
| soft_wui_regime | holdout_365d | +0.01% | 58% | PASS | no |

**Board:** n=8 soft_nw_pos=5 hard_nw_pos=5 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US WUI [1952-05-01..2026-08-01]. Gate z≤0 / cool z≥1 fixed (§71–§84 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). Single-stress design (like CIP soft §71), not dual EPU+TPU.

Artifacts: `reports/scholarly_fx_wui_conditioned_soft_*.csv`, `scholarly_fx_wui_conditioned_soft_meta.json`, `quest_locked_verify_wui_conditioned_soft.md`.
