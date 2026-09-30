# Scholarly FX: WUI-conditioned Menkhoff FX momentum wave (§87)

**Path:** Ahir–Bloom–Furceri **US WUI** (FRED WUIUSA) stress × Menkhoff (2012 JFE) FX momentum gate/cool — **distinct** from raw WUI XS (§40), WUI soft (§85), WUI carry (§86), soft_ew (§66), CIP×mom (§82), VIX/GPR×mom (§74), EPU/TPU×mom (§79), CIP/VIX/EPU soft–carry–mom–value–REER (§68–§84), capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED WUIUSA. G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** WUI pub_lag_months=4 + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI soft §85 / WUI carry §86 mirror) + weight_lag_days=1; mom formation=63d skip=21d signal_lag=1d. Costs 1.5 bps/side.
**Primary:** `mom_low_wui` (trade mom only when lagged US WUI z ≤ 0). Companions: mom_wui_cool / mom_raw / mom_high_wui / us_wui_haven_usd / mom_wui_stack / mom_wui_ew / mom_wui_regime. Cool z_high=1.0 (§71–§86 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| mom_low_wui | -0.000% | -0.00 | -0.00 | 23% | 24% | +0.00 |
| mom_wui_cool | -0.054% | -0.93 | -1.08 | 45% | 16% | -0.22 |
| mom_raw | -0.076% | -1.08 | -1.10 | 45% | 12% | -0.25 |
| mom_high_wui | -0.044% | -1.32 | -1.02 | 13% | 23% | -0.29 |
| us_wui_haven_usd | -0.026% | -0.76 | -0.87 | 14% | 29% | -0.20 |
| mom_wui_stack | -0.000% | -0.00 | -0.00 | 23% | 24% | +0.00 |
| mom_wui_ew | -0.027% | -0.75 | -0.92 | 45% | 17% | -0.18 |
| mom_wui_regime | -0.026% | -0.44 | -0.56 | 36% | 16% | -0.11 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| mom_raw | year_2024 | -0.19% | 45% | PASS | no |
| mom_raw | year_2025 | +0.43% | 91% | PASS | no |
| mom_raw | year_2026 | +0.33% | 62% | PASS | no |
| mom_raw | holdout_365d | +0.37% | 75% | PASS | no |
| mom_wui_cool | year_2024 | -0.17% | 45% | PASS | no |
| mom_wui_cool | year_2025 | +0.24% | 91% | PASS | no |
| mom_wui_cool | year_2026 | +0.12% | 62% | PASS | no |
| mom_wui_cool | holdout_365d | +0.13% | 75% | PASS | no |
| mom_low_wui | year_2024 | +0.01% | 27% | PASS | no |
| mom_low_wui | year_2025 | +0.08% | 18% | PASS | no |
| mom_low_wui | year_2026 | +0.00% | 0% | PASS | no |
| mom_low_wui | holdout_365d | +0.00% | 0% | PASS | no |
| mom_high_wui | year_2024 | +0.00% | 0% | PASS | no |
| mom_high_wui | year_2025 | +0.25% | 55% | PASS | no |
| mom_high_wui | year_2026 | +0.33% | 62% | PASS | no |
| mom_high_wui | holdout_365d | +0.37% | 75% | PASS | no |
| us_wui_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_wui_haven_usd | year_2025 | +0.09% | 36% | PASS | no |
| us_wui_haven_usd | year_2026 | +0.19% | 62% | PASS | no |
| us_wui_haven_usd | holdout_365d | +0.01% | 58% | PASS | no |
| mom_wui_stack | year_2024 | +0.01% | 27% | PASS | no |
| mom_wui_stack | year_2025 | +0.08% | 18% | PASS | no |
| mom_wui_stack | year_2026 | +0.00% | 0% | PASS | no |
| mom_wui_stack | holdout_365d | +0.00% | 0% | PASS | no |
| mom_wui_ew | year_2024 | -0.06% | 55% | PASS | no |
| mom_wui_ew | year_2025 | +0.14% | 73% | PASS | no |
| mom_wui_ew | year_2026 | +0.10% | 62% | PASS | no |
| mom_wui_ew | holdout_365d | +0.05% | 58% | PASS | no |
| mom_wui_regime | year_2024 | +0.01% | 27% | PASS | no |
| mom_wui_regime | year_2025 | +0.17% | 55% | PASS | no |
| mom_wui_regime | year_2026 | +0.19% | 62% | PASS | no |
| mom_wui_regime | holdout_365d | +0.01% | 58% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US WUI [1952-05-01..2026-08-01]. Gate z≤0 / cool z≥1 fixed (§71–§86 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). Single-stress design (like CIP×mom §82), not dual EPU+TPU.

Artifacts: `reports/scholarly_fx_wui_conditioned_mom_*.csv`, `scholarly_fx_wui_conditioned_mom_meta.json`, `quest_locked_verify_wui_conditioned_mom.md`.
