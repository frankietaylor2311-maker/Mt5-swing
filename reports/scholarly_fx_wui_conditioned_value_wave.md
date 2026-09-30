# Scholarly FX: WUI-conditioned Rogoff PPP / real-FX value wave (§88)

**Path:** Ahir–Bloom–Furceri **US WUI** (FRED WUIUSA) stress × Rogoff (1996) PPP / real-FX value gate/cool — **distinct** from raw WUI XS (§40), WUI soft (§85), WUI carry (§86), WUI mom (§87), CIP×PPP (§83), VIX/GPR×PPP (§75), EPU×PPP (§80), raw PPP, BIS REER (§31), soft–carry–mom–value–REER (§68–§84), capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + FRED WUIUSA + FRED CPI levels (pub_lag=1m). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** WUI pub_lag_months=4 + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI soft §85 / WUI carry §86 / WUI mom §87 mirror) + weight_lag_days=1; PPP lookback=60m ppp_signal_lag=1m + weight lag 1d. Costs 1.5 bps/side.
**Primary:** `value_low_wui` (trade value only when lagged US WUI z ≤ 0). Companions: value_wui_cool / value_raw / value_high_wui / us_wui_haven_usd / value_wui_stack / value_wui_ew / value_wui_regime. Cool z_high=1.0 (§71–§87 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| value_low_wui | -0.044% | -1.03 | -0.98 | 21% | 25% | -0.23 |
| value_wui_cool | -0.034% | -0.70 | -0.66 | 41% | 14% | -0.15 |
| value_raw | -0.014% | -0.22 | -0.22 | 42% | 12% | -0.05 |
| value_high_wui | +0.019% | +0.54 | +0.67 | 13% | 34% | +0.13 |
| us_wui_haven_usd | -0.026% | -0.76 | -0.87 | 14% | 29% | -0.20 |
| value_wui_stack | -0.044% | -1.03 | -0.98 | 21% | 25% | -0.23 |
| value_wui_ew | -0.035% | -1.12 | -1.07 | 42% | 16% | -0.25 |
| value_wui_regime | -0.070% | -1.28 | -1.29 | 34% | 14% | -0.30 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| value_raw | year_2024 | -0.10% | 64% | PASS | no |
| value_raw | year_2025 | -0.30% | 27% | PASS | no |
| value_raw | year_2026 | +0.15% | 75% | PASS | no |
| value_raw | holdout_365d | +0.10% | 67% | PASS | no |
| value_wui_cool | year_2024 | -0.09% | 64% | PASS | no |
| value_wui_cool | year_2025 | -0.24% | 18% | PASS | no |
| value_wui_cool | year_2026 | +0.05% | 75% | PASS | no |
| value_wui_cool | holdout_365d | +0.03% | 67% | PASS | no |
| value_low_wui | year_2024 | -0.06% | 36% | PASS | no |
| value_low_wui | year_2025 | -0.12% | 9% | PASS | no |
| value_low_wui | year_2026 | +0.00% | 0% | PASS | no |
| value_low_wui | holdout_365d | +0.00% | 0% | PASS | no |
| value_high_wui | year_2024 | +0.00% | 0% | PASS | no |
| value_high_wui | year_2025 | -0.04% | 27% | PASS | no |
| value_high_wui | year_2026 | +0.15% | 75% | PASS | no |
| value_high_wui | holdout_365d | +0.10% | 67% | PASS | no |
| us_wui_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_wui_haven_usd | year_2025 | +0.09% | 36% | PASS | no |
| us_wui_haven_usd | year_2026 | +0.19% | 62% | PASS | no |
| us_wui_haven_usd | holdout_365d | +0.01% | 58% | PASS | no |
| value_wui_stack | year_2024 | -0.06% | 36% | PASS | no |
| value_wui_stack | year_2025 | -0.12% | 9% | PASS | no |
| value_wui_stack | year_2026 | +0.00% | 0% | PASS | no |
| value_wui_stack | holdout_365d | +0.00% | 0% | PASS | no |
| value_wui_ew | year_2024 | -0.05% | 64% | PASS | no |
| value_wui_ew | year_2025 | -0.09% | 27% | PASS | no |
| value_wui_ew | year_2026 | +0.08% | 62% | PASS | no |
| value_wui_ew | holdout_365d | +0.02% | 58% | PASS | no |
| value_wui_regime | year_2024 | -0.06% | 36% | PASS | no |
| value_wui_regime | year_2025 | -0.02% | 45% | PASS | no |
| value_wui_regime | year_2026 | +0.19% | 62% | PASS | no |
| value_wui_regime | holdout_365d | +0.01% | 58% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US WUI [1952-05-01..2026-08-01]. CPI [1914-02-01..2026-09-01]. Gate z≤0 / cool z≥1 fixed (§71–§87 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). PPP lookback=60m (honesty 120m in raw ppp wave). AU/NZ CPI quarterly ffilled. Single-stress design (like CIP×PPP §83), not dual EPU+TPU.

Artifacts: `reports/scholarly_fx_wui_conditioned_value_*.csv`, `scholarly_fx_wui_conditioned_value_meta.json`, `quest_locked_verify_wui_conditioned_value.md`.
