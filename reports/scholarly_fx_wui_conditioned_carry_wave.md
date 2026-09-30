# Scholarly FX: WUI-conditioned Lustig–Verdelhan carry wave (§86)

**Path:** Ahir–Bloom–Furceri **US WUI** (FRED WUIUSA) stress × Lustig–Verdelhan IR3M carry gate/cool — **distinct** from raw WUI XS (§40), WUI soft (§85), soft_ew (§66), CIP×carry (§68), VIX/GPR×carry (§73), EPU/TPU×carry (§78), CIP/VIX/EPU soft/mom/value/REER (§71–§77/§79–§84), capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED WUIUSA + FRED IR3M rates (pub_lag=1m). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** WUI pub_lag_months=4 + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI soft §85 mirror) + weight_lag_days=1; carry signal_lag=1d. Costs 1.5 bps/side.
**Primary:** `carry_low_wui` (trade carry only when lagged US WUI z ≤ 0). Companions: carry_wui_cool / carry_raw / carry_high_wui / us_wui_haven_usd / carry_wui_stack / carry_wui_ew / carry_wui_regime. Cool z_high=1.0 (§71–§85 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| carry_low_wui | -0.018% | -0.37 | -0.41 | 23% | 21% | -0.14 |
| carry_wui_cool | -0.008% | -0.13 | -0.15 | 51% | 13% | -0.07 |
| carry_raw | -0.002% | -0.02 | -0.03 | 51% | 10% | -0.04 |
| carry_high_wui | +0.003% | +0.09 | +0.08 | 16% | 31% | +0.02 |
| us_wui_haven_usd | -0.026% | -0.76 | -0.87 | 14% | 29% | -0.20 |
| carry_wui_stack | -0.018% | -0.37 | -0.41 | 23% | 21% | -0.14 |
| carry_wui_ew | -0.017% | -0.47 | -0.53 | 48% | 14% | -0.15 |
| carry_wui_regime | -0.045% | -0.74 | -0.84 | 36% | 14% | -0.22 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| carry_raw | year_2024 | +0.15% | 64% | PASS | no |
| carry_raw | year_2025 | +0.12% | 64% | PASS | no |
| carry_raw | year_2026 | +0.13% | 62% | PASS | no |
| carry_raw | holdout_365d | +0.31% | 75% | PASS | no |
| carry_wui_cool | year_2024 | +0.14% | 64% | PASS | no |
| carry_wui_cool | year_2025 | -0.01% | 64% | PASS | no |
| carry_wui_cool | year_2026 | +0.04% | 62% | PASS | no |
| carry_wui_cool | holdout_365d | +0.11% | 75% | PASS | no |
| carry_low_wui | year_2024 | +0.03% | 18% | PASS | no |
| carry_low_wui | year_2025 | -0.07% | 0% | PASS | no |
| carry_low_wui | year_2026 | +0.00% | 0% | PASS | no |
| carry_low_wui | holdout_365d | +0.00% | 0% | PASS | no |
| carry_high_wui | year_2024 | +0.00% | 0% | PASS | no |
| carry_high_wui | year_2025 | +0.22% | 45% | PASS | no |
| carry_high_wui | year_2026 | +0.13% | 62% | PASS | no |
| carry_high_wui | holdout_365d | +0.31% | 75% | PASS | no |
| us_wui_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_wui_haven_usd | year_2025 | +0.09% | 36% | PASS | no |
| us_wui_haven_usd | year_2026 | +0.19% | 62% | PASS | no |
| us_wui_haven_usd | holdout_365d | +0.01% | 58% | PASS | no |
| carry_wui_stack | year_2024 | +0.03% | 18% | PASS | no |
| carry_wui_stack | year_2025 | -0.07% | 0% | PASS | no |
| carry_wui_stack | year_2026 | +0.00% | 0% | PASS | no |
| carry_wui_stack | holdout_365d | +0.00% | 0% | PASS | no |
| carry_wui_ew | year_2024 | +0.06% | 64% | PASS | no |
| carry_wui_ew | year_2025 | +0.00% | 55% | PASS | no |
| carry_wui_ew | year_2026 | +0.08% | 50% | PASS | no |
| carry_wui_ew | holdout_365d | +0.04% | 50% | PASS | no |
| carry_wui_regime | year_2024 | +0.03% | 18% | PASS | no |
| carry_wui_regime | year_2025 | +0.02% | 36% | PASS | no |
| carry_wui_regime | year_2026 | +0.19% | 62% | PASS | no |
| carry_wui_regime | holdout_365d | +0.01% | 58% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US WUI [1952-05-01..2026-08-01]. Gate z≤0 / cool z≥1 fixed (§71–§85 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). Single-stress design (like CIP×carry §68), not dual EPU+TPU.

Artifacts: `reports/scholarly_fx_wui_conditioned_carry_*.csv`, `scholarly_fx_wui_conditioned_carry_meta.json`, `quest_locked_verify_wui_conditioned_carry.md`.
