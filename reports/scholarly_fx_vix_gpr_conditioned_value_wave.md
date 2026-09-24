# Scholarly FX: VIX/GPR-conditioned Rogoff PPP / real-FX value wave (§75)

**Path:** Menkhoff **VIX** (global FX-vol proxy) + Caldara–Iacoviello **aggregate GPR** stress × Rogoff PPP / real-FX cross-sectional value gate/cool — **distinct** from raw ppp_wave / bis_reer, combo (§8), soft EW (§66), VIX/GPR soft (§72), VIX/GPR carry (§73), VIX/GPR mom (§74), CIP waves, capital-sleeve (§53/§70).
**Data:** `approximate_non_ftmo` + `vix_yahoo.csv` + `gpr_daily.csv` (bar_lag=1) + FRED CPI levels (pub_lag=1m). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. `lookback=60m` `n_long=n_short=2`. **PIT:** VIX/GPR bar_lag=1 + signal_lag=1d on trailing z (252d); CPI pub_lag + ppp_signal_lag=1m + weight lag 1d.
**Primary:** `value_low_vix` (trade value only when lagged VIX z ≤ 0). Honesty `value_raw` + `value_high_vix`. Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| value_raw | -0.014% | -0.22 | -0.22 | 42% | 12% | -0.05 |
| value_vix_cool | -0.021% | -0.46 | -0.46 | 39% | 13% | -0.10 |
| value_gpr_cool | -0.027% | -0.53 | -0.57 | 38% | 14% | -0.11 |
| value_low_vix | -0.024% | -0.61 | -0.60 | 31% | 17% | -0.13 |
| value_low_gpr | -0.074% | -1.63 | -1.69 | 34% | 18% | -0.36 |
| value_high_vix | -0.046% | -1.11 | -1.21 | 12% | 39% | -0.27 |
| value_vix_gpr_stack | -0.028% | -0.68 | -0.70 | 37% | 14% | -0.15 |
| value_vix_gpr_ew | -0.036% | -0.88 | -0.91 | 39% | 14% | -0.19 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| value_raw | year_2024 | -0.10% | 64% | PASS | no |
| value_raw | year_2025 | -0.30% | 27% | PASS | no |
| value_raw | year_2026 | +0.15% | 75% | PASS | no |
| value_raw | holdout_365d | +0.10% | 67% | PASS | no |
| value_vix_cool | year_2024 | -0.09% | 64% | PASS | no |
| value_vix_cool | year_2025 | -0.24% | 18% | PASS | no |
| value_vix_cool | year_2026 | +0.16% | 75% | PASS | no |
| value_vix_cool | holdout_365d | +0.09% | 58% | PASS | no |
| value_gpr_cool | year_2024 | -0.11% | 45% | PASS | no |
| value_gpr_cool | year_2025 | -0.26% | 27% | PASS | no |
| value_gpr_cool | year_2026 | +0.06% | 75% | PASS | no |
| value_gpr_cool | holdout_365d | +0.02% | 67% | PASS | no |
| value_low_vix | year_2024 | -0.05% | 45% | PASS | no |
| value_low_vix | year_2025 | -0.16% | 27% | PASS | no |
| value_low_vix | year_2026 | +0.13% | 75% | PASS | no |
| value_low_vix | holdout_365d | +0.07% | 58% | PASS | no |
| value_low_gpr | year_2024 | -0.25% | 36% | PASS | no |
| value_low_gpr | year_2025 | -0.30% | 18% | PASS | no |
| value_low_gpr | year_2026 | -0.13% | 38% | PASS | no |
| value_low_gpr | holdout_365d | -0.15% | 33% | PASS | no |
| value_high_vix | year_2024 | -0.08% | 27% | PASS | no |
| value_high_vix | year_2025 | -0.11% | 18% | PASS | no |
| value_high_vix | year_2026 | -0.06% | 0% | PASS | no |
| value_high_vix | holdout_365d | -0.04% | 8% | PASS | no |
| value_vix_gpr_stack | year_2024 | -0.10% | 45% | PASS | no |
| value_vix_gpr_stack | year_2025 | -0.20% | 18% | PASS | no |
| value_vix_gpr_stack | year_2026 | +0.08% | 75% | PASS | no |
| value_vix_gpr_stack | holdout_365d | +0.02% | 58% | PASS | no |
| value_vix_gpr_ew | year_2024 | -0.13% | 45% | PASS | no |
| value_vix_gpr_ew | year_2025 | -0.24% | 9% | PASS | no |
| value_vix_gpr_ew | year_2026 | +0.06% | 75% | PASS | no |
| value_vix_gpr_ew | holdout_365d | +0.01% | 58% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** VIX ends **2026-09-16**; GPR ends **2026-09-14**; CPI ends **2026-09-01**. Gate z≤0 / cool z≥1 fixed (§71–§74 symmetry; lit GPR often 1.5 — documented, not HO-tuned). z_window=252 daily (NOT monthly CIP 60). PPP lookback=60m (honesty 120m in raw ppp wave). AU/NZ CPI quarterly ffilled.

Artifacts: `reports/scholarly_fx_vix_gpr_conditioned_value_*.csv`, `scholarly_fx_vix_gpr_conditioned_value_meta.json`, `quest_locked_verify_vix_gpr_conditioned_value.md`.
