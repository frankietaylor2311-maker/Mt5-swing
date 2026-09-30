# Scholarly FX: OFR FSI-conditioned soft-signal EW wave (§110)

**Path:** Office of Financial Research **US OFR FSI** (official daily CSV) stress × Dahlquist–Hasseltoft ``soft_ew_macro5`` gate/cool — **distinct** from funding_liquidity (§20), soft_ew (§66 ungated), NFCI soft (§90), ANFCI soft (§95), STLFSI soft (§100), KCFSI soft (§105), soft-stack CIP enrichment (§69), CIP-conditioned soft (§71), VIX/GPR soft (§72), EPU/TPU soft (§76), WUI soft (§85), CIP/VIX/EPU/WUI × carry/mom/value/REER (§68/§73–§89), capital-sleeve (§53/§70), combo (§8). Explicit: do **not** overlay coolers on locked fx4plus.
**Data:** `approximate_non_ftmo` + OFR FSI CSV + FRED/OECD soft legs (§66 PIT). G10 trade: EUR/GBP/JPY/AUD/CAD/CHF/NZD. **PIT:** OFR FSI pub_lag_days=2 + signal_lag_months=1 on trailing **monthly** z (z_window=60m, CIP §71 / EPU §76 / WUI §85 / NFCI/ANFCI/STLFSI/KCFSI soft mirror) + weight_lag_days=1; soft legs source-wave pub/signal lags. Costs 1.5 bps/side inside source factors (haven applies own costs).
**Primary:** `soft_low_ofr` (trade soft only when lagged US OFR FSI z ≤ 0). Companions: soft_ofr_cool / soft_raw / soft_high_ofr / us_ofr_haven_usd / soft_ofr_stack / soft_ofr_ew / soft_ofr_regime. Cool z_high=1.0 (§71–§109 symmetry; lit often 1.5). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| soft_low_ofr | +0.081% | +3.81 | +2.96 | 48% | 14% | +0.87 |
| soft_ofr_cool | +0.110% | +4.39 | +3.58 | 61% | 11% | +1.05 |
| soft_raw | +0.118% | +4.26 | +3.63 | 61% | 11% | +1.02 |
| soft_high_ofr | +0.012% | +1.70 | +1.55 | 6% | 57% | +0.36 |
| us_ofr_haven_usd | -0.010% | -0.33 | -0.25 | 4% | 71% | -0.09 |
| soft_ofr_stack | +0.081% | +3.81 | +2.96 | 48% | 14% | +0.87 |
| soft_ofr_ew | +0.060% | +3.40 | +2.60 | 59% | 12% | +0.82 |
| soft_ofr_regime | +0.071% | +1.96 | +1.48 | 53% | 18% | +0.49 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| soft_raw | year_2024 | +0.08% | 55% | PASS | no |
| soft_raw | year_2025 | +0.04% | 64% | PASS | no |
| soft_raw | year_2026 | +0.14% | 88% | PASS | no |
| soft_raw | holdout_365d | +0.06% | 67% | PASS | no |
| soft_ofr_cool | year_2024 | +0.08% | 55% | PASS | no |
| soft_ofr_cool | year_2025 | +0.04% | 64% | PASS | no |
| soft_ofr_cool | year_2026 | +0.14% | 88% | PASS | no |
| soft_ofr_cool | holdout_365d | +0.06% | 67% | PASS | no |
| soft_low_ofr | year_2024 | +0.08% | 55% | PASS | no |
| soft_low_ofr | year_2025 | +0.02% | 64% | PASS | no |
| soft_low_ofr | year_2026 | +0.14% | 75% | PASS | no |
| soft_low_ofr | holdout_365d | +0.06% | 58% | PASS | no |
| soft_high_ofr | year_2024 | +0.00% | 0% | PASS | no |
| soft_high_ofr | year_2025 | +0.00% | 0% | PASS | no |
| soft_high_ofr | year_2026 | +0.00% | 0% | PASS | no |
| soft_high_ofr | holdout_365d | +0.00% | 0% | PASS | no |
| us_ofr_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_ofr_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_ofr_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_ofr_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| soft_ofr_stack | year_2024 | +0.08% | 55% | PASS | no |
| soft_ofr_stack | year_2025 | +0.02% | 64% | PASS | no |
| soft_ofr_stack | year_2026 | +0.14% | 75% | PASS | no |
| soft_ofr_stack | holdout_365d | +0.06% | 58% | PASS | no |
| soft_ofr_ew | year_2024 | +0.06% | 55% | PASS | no |
| soft_ofr_ew | year_2025 | +0.02% | 64% | PASS | no |
| soft_ofr_ew | year_2026 | +0.09% | 88% | PASS | no |
| soft_ofr_ew | holdout_365d | +0.04% | 67% | PASS | no |
| soft_ofr_regime | year_2024 | +0.08% | 55% | PASS | no |
| soft_ofr_regime | year_2025 | +0.02% | 64% | PASS | no |
| soft_ofr_regime | year_2026 | +0.14% | 75% | PASS | no |
| soft_ofr_regime | holdout_365d | +0.06% | 58% | PASS | no |

**Board:** n=8 soft_nw_pos=6 hard_nw_pos=5 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

**Data caveat:** US OFR FSI [2000-01-05..2026-09-30]. Monthly. Gate z≤0 / cool z≥1 fixed (§71–§99 symmetry; lit often other cutoffs — documented, not HO-tuned). Monthly z_window=60m (not 252d daily). Single-stress design (like CIP soft §71 / WUI soft §85 / KCFSI soft §105). Distinct from funding_liq §20.

Artifacts: `reports/scholarly_fx_ofr_conditioned_soft_*.csv`, `scholarly_fx_ofr_conditioned_soft_meta.json`, `quest_locked_verify_ofr_conditioned_soft.md`.
