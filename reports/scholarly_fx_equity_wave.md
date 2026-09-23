# Scholarly FX: Hau–Rey equity differential (local−US → FX)

**Prior (fixed):** Hau & Rey (2006) — relative local equity outperformance vs US associates with local FX appreciation (portfolio / risk-appetite channel).
**PIT:** pub_lag=1d + signal_lag=1d; formation=21d, skip=0d; n_long=n_short=2; costs=1.5 bps/side.
**Data:** `approximate_non_ftmo` Yahoo D1 FX + Yahoo equity (`{'USD': '^GSPC', 'EUR': '^GDAXI', 'GBP': '^FTSE', 'JPY': '^N225', 'CAD': '^GSPTSE', 'AUD': '^AXJO', 'CHF': '^SSMI'}`). Calendar-date normalize-before-join (commodity-style session fix). NZD omitted (no free broad Yahoo index).
**Primary:** `eq_diff_xs`. Locked sleeve untouched. No HO tuning.

**Board:** n=4 soft=0 hard=0 promote=0

### Construction note (XS identity)

For equal-weight cross-sectional rank sorts, score=(local_mom−US_mom) ranks identically to local_mom (US term is common). eq_diff_xs ≡ eq_mom_xs by construction; distinctive Hau–Rey content is eq_diff_ts (sign of differential) and carry_eq_cool (US equity risk-off cool).

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| eq_diff_xs | -0.066% | -1.09 | -1.18 | 48% | 12% | -0.20 |
| eq_diff_ts | -0.019% | -0.26 | -0.30 | 51% | 14% | -0.07 |
| eq_mom_xs | -0.066% | -1.09 | -1.18 | 48% | 12% | -0.20 |
| carry_eq_cool | -0.024% | -0.39 | -0.48 | 50% | 12% | -0.13 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | top3 | gates | 1% bar |
|----------|--------|--------:|-----:|-----:|:-----:|:------:|
| eq_diff_xs | year_2024 | +0.04% | 55% | 88% | PASS | no |
| eq_diff_xs | year_2025 | -0.08% | 45% | 91% | PASS | no |
| eq_diff_xs | year_2026 | -0.18% | 50% | 87% | PASS | no |
| eq_diff_xs | holdout_365d | -0.14% | 50% | 72% | PASS | no |
| eq_diff_ts | year_2024 | +0.25% | 64% | 68% | PASS | no |
| eq_diff_ts | year_2025 | -0.29% | 36% | 95% | PASS | no |
| eq_diff_ts | year_2026 | -0.08% | 50% | 95% | PASS | no |
| eq_diff_ts | holdout_365d | -0.10% | 42% | 87% | PASS | no |
| eq_mom_xs | year_2024 | +0.04% | 55% | 88% | PASS | no |
| eq_mom_xs | year_2025 | -0.08% | 45% | 91% | PASS | no |
| eq_mom_xs | year_2026 | -0.18% | 50% | 87% | PASS | no |
| eq_mom_xs | holdout_365d | -0.14% | 50% | 72% | PASS | no |
| carry_eq_cool | year_2024 | +0.12% | 73% | 60% | PASS | no |
| carry_eq_cool | year_2025 | +0.18% | 64% | 72% | PASS | no |
| carry_eq_cool | year_2026 | -0.03% | 38% | 100% | PASS | no |
| carry_eq_cool | holdout_365d | +0.19% | 58% | 63% | PASS | no |

## Risk sweep

Skipped: **no positive IS mean** on any factor (scaling would worsen return). Best unscaled full-sample mean is `eq_diff_ts` ≈ −0.02%/mo.

**Unscaled promote:** NO. **Scaled primary promote:** NO.

### Honest read

On this Yahoo equity+FX sample, the Hau–Rey **tradable** legs do **not** clear the FTMO ~1%/mo bar. Primary XS (`eq_diff_xs` ≡ `eq_mom_xs` under EW ranks) full-sample mean_mo ≈ -0.066% (NW t ≈ -1.18); TS differential and US-equity-cooled carry are also near zero / slightly negative. Year windows occasionally pass FTMO gates at tiny means but %pos and top3 fail consistency. Indices ≠ Hau–Rey portfolio-flow data; no go-live; locked sleeve untouched.

Artifacts: `reports/scholarly_fx_equity_*.csv`, `scholarly_fx_equity_meta.json`.
