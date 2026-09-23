# Scholarly FX: Brunnermeier–Nagel–Pedersen crash-risk / skewness wave

**Path:** BNP (2008) currency crash-risk premium via trailing return skewness / left-tail shortfall XS sorts — **distinct** from Menkhoff FX-RV (§15), Lustig carry, and AI-GPR roles (§25).
**Data:** `approximate_non_ftmo` Yahoo D1 USD majors (OHLC only; no paid IV/RR). **PIT:** skip=1d + signal_lag=1d; formation 63/126d; left-tail q=0.05; costs 1.5 bps/side.
**Primary:** `crash_skew_xs` (long more-negative skew / high crash risk). Locked sleeve untouched (no cooler overlay).

**Board:** n=6 soft_nw_pos=0 hard_nw_pos=0 promote=0

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| crash_skew_xs | -0.151% | -2.31 | -2.02 | 43% | 14% | -0.54 |
| crash_skew_xs_126 | -0.066% | -0.99 | -1.04 | 47% | 19% | -0.22 |
| left_tail_xs | +0.070% | +0.94 | +1.07 | 51% | 15% | +0.23 |
| mom_skew_regime | -0.086% | -1.30 | -1.17 | 43% | 19% | -0.25 |
| carry_crash_cool | +0.002% | +0.03 | +0.04 | 51% | 12% | -0.03 |
| crash_ew | -0.057% | -1.37 | -1.39 | 45% | 20% | -0.32 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | top3 | gates | 1% bar |
|----------|--------|--------:|-----:|-----:|:-----:|:------:|
| crash_skew_xs | year_2024 | -0.23% | 45% | 89% | PASS | no |
| crash_skew_xs | year_2025 | -0.06% | 64% | 80% | PASS | no |
| crash_skew_xs | year_2026 | +0.03% | 75% | 79% | PASS | no |
| crash_skew_xs | holdout_365d | +0.03% | 75% | 60% | PASS | no |
| crash_skew_xs_126 | year_2024 | -0.11% | 45% | 87% | PASS | no |
| crash_skew_xs_126 | year_2025 | -0.10% | 55% | 77% | PASS | no |
| crash_skew_xs_126 | year_2026 | -0.16% | 50% | 94% | PASS | no |
| crash_skew_xs_126 | holdout_365d | +0.00% | 58% | 81% | PASS | no |
| left_tail_xs | year_2024 | -0.22% | 45% | 78% | PASS | no |
| left_tail_xs | year_2025 | -0.21% | 27% | 100% | PASS | no |
| left_tail_xs | year_2026 | -0.07% | 50% | 85% | PASS | no |
| left_tail_xs | holdout_365d | -0.08% | 42% | 72% | PASS | no |
| mom_skew_regime | year_2024 | +0.27% | 82% | 70% | PASS | no |
| mom_skew_regime | year_2025 | +0.12% | 55% | 85% | PASS | no |
| mom_skew_regime | year_2026 | +0.34% | 75% | 65% | PASS | no |
| mom_skew_regime | holdout_365d | +0.42% | 83% | 48% | PASS | no |
| carry_crash_cool | year_2024 | +0.02% | 64% | 63% | PASS | no |
| carry_crash_cool | year_2025 | +0.04% | 64% | 69% | PASS | no |
| carry_crash_cool | year_2026 | +0.16% | 62% | 79% | PASS | no |
| carry_crash_cool | holdout_365d | +0.32% | 75% | 51% | PASS | no |
| crash_ew | year_2024 | -0.07% | 55% | 59% | PASS | no |
| crash_ew | year_2025 | -0.06% | 45% | 77% | PASS | no |
| crash_ew | year_2026 | +0.04% | 62% | 86% | PASS | no |
| crash_ew | holdout_365d | +0.09% | 58% | 71% | PASS | no |

**Risk sweep run:** YES.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_crash_skew_*.csv`, `scholarly_fx_crash_skew_meta.json`.
