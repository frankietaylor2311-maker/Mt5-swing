# Scholarly FX: VIX/GPR-conditioned soft-signal EW wave (§72)

**Path:** Menkhoff **VIX** (global FX-vol proxy) + Caldara–Iacoviello **aggregate GPR** stress × Dahlquist–Hasseltoft ``soft_ew_macro5`` gate/cool — **distinct** from soft_ew (§66 ungated), soft-stack CIP enrichment (§69), CIP-conditioned carry (§68), CIP XS (§67), CIP-conditioned soft (§71 CIP stress), capital-sleeve (§53/§70), funding_liq (§20), combo (§8), gpr_regime standalone, AI-GPR (§25), country-GPR (§45). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + `vix_yahoo.csv` + `gpr_daily.csv` + FRED/OECD soft legs (§66 PIT). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** VIX/GPR bar_lag=1 + signal_lag=1d on trailing z; soft legs source-wave pub/signal lags. Costs 1.5 bps/side inside source factors.
**Primary:** `soft_low_vix` (trade soft only when lagged VIX z ≤ 0). Companions: soft_vix_cool / soft_low_gpr / soft_gpr_cool / soft_raw / soft_high_vix / soft_vix_gpr_stack / soft_vix_gpr_ew. Cool z_high=1.0 for both VIX and GPR (§71 symmetry; lit GPR often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| soft_low_vix | +0.058% | +2.92 | +2.38 | 48% | 14% | +0.69 |
| soft_vix_cool | +0.091% | +3.96 | +3.21 | 59% | 11% | +0.93 |
| soft_low_gpr | +0.044% | +1.98 | +2.16 | 52% | 18% | +0.53 |
| soft_gpr_cool | +0.089% | +3.78 | +3.41 | 59% | 12% | +0.90 |
| soft_raw | +0.118% | +4.26 | +3.63 | 61% | 11% | +1.02 |
| soft_high_vix | +0.029% | +2.27 | +2.07 | 24% | 35% | +0.60 |
| soft_vix_gpr_stack | +0.069% | +3.47 | +2.98 | 57% | 12% | +0.81 |
| soft_vix_gpr_ew | +0.071% | +3.55 | +3.07 | 57% | 12% | +0.85 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| soft_raw | year_2024 | +0.08% | 55% | PASS | no |
| soft_raw | year_2025 | +0.04% | 64% | PASS | no |
| soft_raw | year_2026 | +0.14% | 88% | PASS | no |
| soft_raw | holdout_365d | +0.06% | 67% | PASS | no |
| soft_vix_cool | year_2024 | +0.07% | 55% | PASS | no |
| soft_vix_cool | year_2025 | +0.01% | 64% | PASS | no |
| soft_vix_cool | year_2026 | +0.10% | 88% | PASS | no |
| soft_vix_cool | holdout_365d | +0.04% | 67% | PASS | no |
| soft_gpr_cool | year_2024 | +0.10% | 45% | PASS | no |
| soft_gpr_cool | year_2025 | +0.06% | 73% | PASS | no |
| soft_gpr_cool | year_2026 | +0.10% | 75% | PASS | no |
| soft_gpr_cool | holdout_365d | +0.03% | 58% | PASS | no |
| soft_low_vix | year_2024 | +0.09% | 55% | PASS | no |
| soft_low_vix | year_2025 | -0.01% | 36% | PASS | no |
| soft_low_vix | year_2026 | +0.04% | 50% | PASS | no |
| soft_low_vix | holdout_365d | -0.02% | 33% | PASS | no |
| soft_low_gpr | year_2024 | +0.10% | 73% | PASS | no |
| soft_low_gpr | year_2025 | +0.07% | 64% | PASS | no |
| soft_low_gpr | year_2026 | +0.08% | 62% | PASS | no |
| soft_low_gpr | holdout_365d | +0.03% | 50% | PASS | no |
| soft_high_vix | year_2024 | +0.00% | 36% | PASS | no |
| soft_high_vix | year_2025 | +0.06% | 36% | PASS | no |
| soft_high_vix | year_2026 | +0.06% | 38% | PASS | no |
| soft_high_vix | holdout_365d | +0.04% | 33% | PASS | no |
| soft_vix_gpr_stack | year_2024 | +0.09% | 55% | PASS | no |
| soft_vix_gpr_stack | year_2025 | +0.03% | 64% | PASS | no |
| soft_vix_gpr_stack | year_2026 | +0.08% | 62% | PASS | no |
| soft_vix_gpr_stack | holdout_365d | +0.01% | 50% | PASS | no |
| soft_vix_gpr_ew | year_2024 | +0.09% | 45% | PASS | no |
| soft_vix_gpr_ew | year_2025 | +0.03% | 64% | PASS | no |
| soft_vix_gpr_ew | year_2026 | +0.08% | 75% | PASS | no |
| soft_vix_gpr_ew | holdout_365d | +0.02% | 58% | PASS | no |

**Board:** n=8 soft_nw_pos=8 hard_nw_pos=8 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** VIX [1990-01-03..2026-09-16]; GPR [1985-01-02..2026-09-14]. Gate z≤0 / cool z≥1 fixed for both (§71 symmetry; lit GPR often 1.5 — documented, not HO-tuned).

Artifacts: `reports/scholarly_fx_vix_gpr_conditioned_soft_*.csv`, `scholarly_fx_vix_gpr_conditioned_soft_meta.json`, `quest_locked_verify_vix_gpr_conditioned_soft.md`.
