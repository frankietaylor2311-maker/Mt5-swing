# Scholarly FX: bilateral AI-GPR role decompositions wave

**Path:** Caldara–Iacoviello AI-GPR threats/acts / oil-region roles — **distinct** from aggregate GPR (§8), country-GPRC sorts (§9), news events (§10), and CRR/ToT commodity FX.
**Data:** `approximate_non_ftmo` + free `ai_gpr_daily.csv`. **PIT:** pub_lag=1d + signal_lag=1d; trailing z=252d; binary z≥1.0; costs 1.5 bps/side.
**Primary:** `ai_threats_usd` (high threats z → long USD / short risk FX). Locked sleeve untouched (no cooler overlay).

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| ai_threats_usd | -0.012% | -0.40 | -0.39 | 43% | 21% | -0.12 |
| ai_acts_usd | +0.001% | +0.05 | +0.06 | 37% | 24% | +0.01 |
| ai_gpr_usd | -0.045% | -1.52 | -1.55 | 33% | 28% | -0.43 |
| oil_gpr_usd | +0.020% | +0.76 | +0.71 | 41% | 17% | +0.20 |
| oil_threats_usd | -0.043% | -1.59 | -1.48 | 39% | 19% | -0.41 |
| oil_me_vs_non | -0.027% | -1.05 | -1.17 | 36% | 21% | -0.30 |
| carry_ai_threats_cool | -0.062% | -0.95 | -1.03 | 49% | 12% | -0.25 |
| ai_ew | -0.009% | -0.42 | -0.43 | 44% | 24% | -0.12 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| ai_threats_usd | year_2024 | -0.03% | 36% | PASS | no |
| ai_threats_usd | year_2025 | -0.30% | 27% | PASS | no |
| ai_threats_usd | year_2026 | +0.04% | 62% | PASS | no |
| ai_threats_usd | holdout_365d | +0.07% | 58% | PASS | no |
| ai_acts_usd | year_2024 | +0.05% | 64% | PASS | no |
| ai_acts_usd | year_2025 | -0.03% | 27% | PASS | no |
| ai_acts_usd | year_2026 | +0.01% | 50% | PASS | no |
| ai_acts_usd | holdout_365d | +0.01% | 50% | PASS | no |
| ai_gpr_usd | year_2024 | -0.06% | 18% | PASS | no |
| ai_gpr_usd | year_2025 | -0.33% | 36% | PASS | no |
| ai_gpr_usd | year_2026 | -0.02% | 38% | PASS | no |
| ai_gpr_usd | holdout_365d | +0.02% | 50% | PASS | no |
| oil_gpr_usd | year_2024 | -0.05% | 36% | PASS | no |
| oil_gpr_usd | year_2025 | -0.12% | 36% | PASS | no |
| oil_gpr_usd | year_2026 | +0.10% | 50% | PASS | no |
| oil_gpr_usd | holdout_365d | +0.03% | 42% | PASS | no |
| oil_threats_usd | year_2024 | +0.01% | 55% | PASS | no |
| oil_threats_usd | year_2025 | -0.25% | 27% | PASS | no |
| oil_threats_usd | year_2026 | +0.14% | 62% | PASS | no |
| oil_threats_usd | holdout_365d | +0.10% | 58% | PASS | no |
| oil_me_vs_non | year_2024 | +0.06% | 55% | PASS | no |
| oil_me_vs_non | year_2025 | -0.02% | 18% | PASS | no |
| oil_me_vs_non | year_2026 | -0.01% | 62% | PASS | no |
| oil_me_vs_non | holdout_365d | -0.08% | 42% | PASS | no |
| carry_ai_threats_cool | year_2024 | -0.04% | 64% | PASS | no |
| carry_ai_threats_cool | year_2025 | +0.21% | 73% | PASS | no |
| carry_ai_threats_cool | year_2026 | +0.01% | 50% | PASS | no |
| carry_ai_threats_cool | holdout_365d | +0.15% | 67% | PASS | no |
| ai_ew | year_2024 | -0.02% | 45% | PASS | no |
| ai_ew | year_2025 | -0.20% | 27% | PASS | no |
| ai_ew | year_2026 | +0.03% | 62% | PASS | no |
| ai_ew | holdout_365d | +0.03% | 58% | PASS | no |

**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_ai_gpr_*.csv`, `scholarly_fx_ai_gpr_meta.json`.
