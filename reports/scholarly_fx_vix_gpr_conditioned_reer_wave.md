# Scholarly FX: VIX/GPR-conditioned BIS REER HML-FX value wave (§81)

**Path:** Menkhoff (2012 JF) **VIX** / FX-vol + Caldara–Iacoviello / Liu–Zhang **aggregate GPR** stress × BIS multilateral REER undervaluation / HML-FX value (``reer_cheap_xs`` §31) gate/cool — **distinct** from raw bis_reer (§31), raw ppp, VIX/GPR×PPP value (§75), EPU/TPU×REER (§77), EPU soft/carry/mom/value (§76–§80), CIP waves, capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + yfinance VIX + Caldara–Iacoviello GPR + FRED BIS RB*BIS (pub_lag=2m). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** VIX/GPR bar_lag=1 + signal_lag=1d on trailing **daily** z (z_window=252, min_periods=60 — §71–§75 symmetry); REER pub_lag=2 + signal_lag=1m + weight lag 1d. Costs 1.5 bps/side.
**Primary:** `reer_low_vix` (trade REER-value only when lagged VIX z ≤ 0). Companions: reer_vix_cool / reer_low_gpr / reer_gpr_cool / reer_raw / reer_high_vix / reer_vix_gpr_stack / reer_vix_gpr_ew. Cool z_high=1.0 for both VIX and GPR (§71–§75 symmetry; lit GPR often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| reer_low_vix | +0.015% | +0.32 | +0.39 | 41% | 17% | +0.08 |
| reer_vix_cool | +0.028% | +0.50 | +0.59 | 52% | 14% | +0.15 |
| reer_low_gpr | -0.099% | -2.00 | -2.26 | 39% | 17% | -0.49 |
| reer_gpr_cool | -0.010% | -0.18 | -0.21 | 50% | 15% | -0.01 |
| reer_raw | +0.034% | +0.51 | +0.60 | 52% | 13% | +0.15 |
| reer_high_vix | -0.020% | -0.64 | -0.73 | 18% | 33% | -0.16 |
| reer_vix_gpr_stack | -0.015% | -0.33 | -0.38 | 49% | 16% | -0.04 |
| reer_vix_gpr_ew | -0.017% | -0.36 | -0.43 | 49% | 16% | -0.06 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| reer_raw | year_2024 | -0.11% | 36% | PASS | no |
| reer_raw | year_2025 | -0.27% | 27% | PASS | no |
| reer_raw | year_2026 | +0.16% | 62% | PASS | no |
| reer_raw | holdout_365d | +0.08% | 58% | PASS | no |
| reer_vix_cool | year_2024 | -0.12% | 36% | PASS | no |
| reer_vix_cool | year_2025 | -0.17% | 36% | PASS | no |
| reer_vix_cool | year_2026 | +0.15% | 75% | PASS | no |
| reer_vix_cool | holdout_365d | +0.05% | 67% | PASS | no |
| reer_gpr_cool | year_2024 | -0.18% | 36% | PASS | no |
| reer_gpr_cool | year_2025 | -0.28% | 27% | PASS | no |
| reer_gpr_cool | year_2026 | +0.10% | 62% | PASS | no |
| reer_gpr_cool | holdout_365d | -0.00% | 50% | PASS | no |
| reer_low_vix | year_2024 | -0.17% | 18% | PASS | no |
| reer_low_vix | year_2025 | -0.08% | 36% | PASS | no |
| reer_low_vix | year_2026 | +0.07% | 50% | PASS | no |
| reer_low_vix | holdout_365d | -0.02% | 42% | PASS | no |
| reer_low_gpr | year_2024 | -0.16% | 45% | PASS | no |
| reer_low_gpr | year_2025 | -0.28% | 18% | PASS | no |
| reer_low_gpr | year_2026 | -0.06% | 50% | PASS | no |
| reer_low_gpr | holdout_365d | -0.17% | 33% | PASS | no |
| reer_high_vix | year_2024 | -0.02% | 27% | PASS | no |
| reer_high_vix | year_2025 | -0.15% | 18% | PASS | no |
| reer_high_vix | year_2026 | -0.06% | 12% | PASS | no |
| reer_high_vix | holdout_365d | -0.03% | 25% | PASS | no |
| reer_vix_gpr_stack | year_2024 | -0.17% | 36% | PASS | no |
| reer_vix_gpr_stack | year_2025 | -0.18% | 27% | PASS | no |
| reer_vix_gpr_stack | year_2026 | +0.08% | 62% | PASS | no |
| reer_vix_gpr_stack | holdout_365d | -0.02% | 50% | PASS | no |
| reer_vix_gpr_ew | year_2024 | -0.16% | 45% | PASS | no |
| reer_vix_gpr_ew | year_2025 | -0.20% | 27% | PASS | no |
| reer_vix_gpr_ew | year_2026 | +0.06% | 62% | PASS | no |
| reer_vix_gpr_ew | holdout_365d | -0.04% | 50% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** VIX [1990-01-03..2026-09-16]; GPR [1985-01-02..2026-09-14]; REER [1994-03-01..2026-09-01]. Gate z≤0 / cool z≥1 fixed for both (§71–§75 symmetry; lit GPR often 1.5 — documented, not HO-tuned). Daily z_window=252 (not monthly EPU 60m).

Artifacts: `reports/scholarly_fx_vix_gpr_conditioned_reer_*.csv`, `scholarly_fx_vix_gpr_conditioned_reer_meta.json`, `quest_locked_verify_vix_gpr_conditioned_reer.md`.
