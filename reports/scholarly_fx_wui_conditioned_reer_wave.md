# Scholarly FX: WUI-conditioned BIS REER HML-FX value wave (§89)

**Path:** Ahir–Bloom–Furceri **US WUI** (FRED WUIUSA) stress × BIS multilateral REER undervaluation / HML-FX value (`reer_cheap_xs` §31) gate/cool — **distinct** from raw WUI XS (§40), WUI soft (§85), WUI carry (§86), WUI mom (§87), WUI PPP value (§88), raw bis_reer (§31), CIP×REER (§84), VIX/GPR×REER (§81), EPU×REER (§77), CIP/VIX/EPU×PPP, soft–carry–mom–value stacks, capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED WUIUSA + FRED BIS REER (pub_lag=2m). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** WUI pub_lag_months=4 + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI soft §85 / WUI carry §86 / WUI mom §87 / WUI value §88 mirror) + weight_lag_days=1; REER pub_lag=2m signal_lag=1m + weight lag 1d. Costs 1.5 bps/side.
**Primary:** `reer_low_wui` (trade REER-value only when lagged US WUI z ≤ 0). Companions: reer_wui_cool / reer_raw / reer_high_wui / us_wui_haven_usd / reer_wui_stack / reer_wui_ew / reer_wui_regime. Cool z_high=1.0 (§71–§88 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| reer_low_wui | +0.027% | +0.59 | +0.66 | 29% | 23% | +0.18 |
| reer_wui_cool | +0.020% | +0.38 | +0.45 | 53% | 16% | +0.13 |
| reer_raw | +0.034% | +0.51 | +0.60 | 52% | 13% | +0.15 |
| reer_high_wui | +0.020% | +0.48 | +0.51 | 15% | 36% | +0.13 |
| us_wui_haven_usd | -0.023% | -0.68 | -0.80 | 14% | 29% | -0.17 |
| reer_wui_stack | +0.027% | +0.59 | +0.66 | 29% | 23% | +0.18 |
| reer_wui_ew | +0.008% | +0.23 | +0.27 | 51% | 17% | +0.09 |
| reer_wui_regime | +0.003% | +0.06 | +0.06 | 43% | 15% | +0.05 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| reer_raw | year_2024 | -0.11% | 36% | PASS | no |
| reer_raw | year_2025 | -0.27% | 27% | PASS | no |
| reer_raw | year_2026 | +0.16% | 62% | PASS | no |
| reer_raw | holdout_365d | +0.08% | 58% | PASS | no |
| reer_wui_cool | year_2024 | -0.10% | 36% | PASS | no |
| reer_wui_cool | year_2025 | -0.18% | 27% | PASS | no |
| reer_wui_cool | year_2026 | +0.06% | 62% | PASS | no |
| reer_wui_cool | holdout_365d | +0.03% | 58% | PASS | no |
| reer_low_wui | year_2024 | +0.00% | 27% | PASS | no |
| reer_low_wui | year_2025 | -0.06% | 18% | PASS | no |
| reer_low_wui | year_2026 | +0.00% | 0% | PASS | no |
| reer_low_wui | holdout_365d | +0.00% | 0% | PASS | no |
| reer_high_wui | year_2024 | +0.00% | 0% | PASS | no |
| reer_high_wui | year_2025 | -0.10% | 18% | PASS | no |
| reer_high_wui | year_2026 | +0.16% | 62% | PASS | no |
| reer_high_wui | holdout_365d | +0.08% | 58% | PASS | no |
| us_wui_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_wui_haven_usd | year_2025 | +0.10% | 36% | PASS | no |
| us_wui_haven_usd | year_2026 | +0.19% | 62% | PASS | no |
| us_wui_haven_usd | holdout_365d | +0.01% | 58% | PASS | no |
| reer_wui_stack | year_2024 | +0.00% | 27% | PASS | no |
| reer_wui_stack | year_2025 | -0.06% | 18% | PASS | no |
| reer_wui_stack | year_2026 | +0.00% | 0% | PASS | no |
| reer_wui_stack | holdout_365d | +0.00% | 0% | PASS | no |
| reer_wui_ew | year_2024 | -0.03% | 36% | PASS | no |
| reer_wui_ew | year_2025 | -0.05% | 36% | PASS | no |
| reer_wui_ew | year_2026 | +0.08% | 62% | PASS | no |
| reer_wui_ew | holdout_365d | +0.01% | 50% | PASS | no |
| reer_wui_regime | year_2024 | +0.00% | 27% | PASS | no |
| reer_wui_regime | year_2025 | +0.03% | 55% | PASS | no |
| reer_wui_regime | year_2026 | +0.19% | 62% | PASS | no |
| reer_wui_regime | holdout_365d | +0.01% | 58% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US WUI [1952-05-01..2026-08-01]. REER [1994-03-01..2026-09-01]. Gate z≤0 / cool z≥1 fixed (§71–§88 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). REER pub_lag=2m signal_lag=1m. Single-stress design (like CIP×REER §84), not dual EPU+TPU.

Artifacts: `reports/scholarly_fx_wui_conditioned_reer_*.csv`, `scholarly_fx_wui_conditioned_reer_meta.json`, `quest_locked_verify_wui_conditioned_reer.md`.
