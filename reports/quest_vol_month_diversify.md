# Quest wave: vol/equity/month-aware flatten + dual-year diversify + H1

**data_source:** `approximate_non_ftmo`  **RF:** 8%  **per-leg:** 1.60%  **warmup:** 250  **signal_lag=1**
**Dual-year gate:** 2024 AND 2025 mean_mo ≥ 0.8% + dual-confirm + corr≤0.55 (before add).
**Holdout:** confirmation only (not used for overlay hyperparams).

## Baseline (locked fx4plus_gbpcad_d1_voltarget_0025)

| Window | Mean mo | %pos | Top3 | Gates |
|--------|--------:|-----:|-----:|:-----:|
| 2024 | 0.42% | 55% | 74% | PASS |
| 2025 | 1.63% | 73% | 69% | PASS |
| 2026 | 2.30% | 88% | 87% | PASS |
| holdout_365d | 1.52% | 75% | 81% | PASS |
| roll12_m6 | 1.13% | 67% | 80% | PASS |

## Overlay IS tune (2024) — top scores

| Spec | Mean mo | %pos | Top3 | P2T | Score |
|------|--------:|-----:|-----:|----:|------:|
| eq_only_0.006 | 0.10% | 73% | 65% | 3.5% | 0.459 |
| vt+eqflat0.008 | 0.12% | 73% | 66% | 5.2% | 0.440 |
| vt+eqflat0.01 | 0.16% | 73% | 66% | 5.2% | 0.416 |
| eq_only_0.01 | 0.17% | 73% | 67% | 3.5% | 0.367 |
| ma_only_sm0.02 | 0.16% | 73% | 67% | 3.3% | 0.356 |
| ma_only_sm0.015 | 0.16% | 73% | 67% | 3.4% | 0.355 |
| ma_only_sm0.02 | 0.17% | 73% | 67% | 3.0% | 0.355 |
| ma_only_sm0.025 | 0.18% | 73% | 67% | 3.5% | 0.354 |
| ma_only_sm0.03 | 0.18% | 73% | 67% | 3.5% | 0.354 |
| vtflat0.002_hi1.0 | 0.18% | 73% | 67% | 3.5% | 0.354 |
| vtflat0.0025_hi1.0 | 0.18% | 73% | 67% | 3.5% | 0.354 |
| vtflat0.003_hi1.0 | 0.18% | 73% | 67% | 3.5% | 0.354 |

## Overlay confirmation (key windows mean_mo %)

| Spec | 2024 | 2025 | 2026 | HO | HO top3 | Promote? |
|------|-----:|-----:|-----:|---:|--------:|:--------:|
| eq_only_0.006 | 0.10 | 0.32 | 0.51 | 0.36 | 75% | no |
| vt+eqflat0.008 | 0.12 | 0.44 | 0.72 | 0.45 | 72% | no |
| vt+eqflat0.01 | 0.16 | 0.52 | 0.79 | 0.58 | 74% | no |
| eq_only_0.01 | 0.17 | 0.49 | 0.68 | 0.56 | 80% | no |
| ma_only_sm0.02 | 0.16 | 0.52 | 0.92 | 0.69 | 83% | no |
| ma_only_sm0.015 | 0.16 | 0.46 | 0.92 | 0.69 | 83% | no |
| ma_only_sm0.025 | 0.18 | 0.54 | 0.93 | 0.69 | 83% | no |
| baseline_vt0025 | 0.42 | 1.63 | 2.30 | 1.52 | 81% | no |
| vt_eq_ma_combo | -0.10 | 0.90 | 0.69 | 0.21 | 70% | no |

## Dual-year screen (unused dual-confirm)

Passers (≥0.8% both years, gates, corr≤0.55): **0**

| Symbol | TF | Strat | mo2024 | mo2025 | corr | Pass |
|--------|----|-------|-------:|-------:|-----:|:----:|
| GBPJPY | H4 | squeeze_breakout | 1.42% | -1.20% | -0.30 | n |
| EURGBP | D1 | cci_reversion | 0.71% | -0.17% | -0.03 | n |
| EURAUD | H4 | breakout_donchian | 0.59% | -0.91% | -0.24 | n |
| USDJPY | D1 | hybrid_regime | 0.37% | -0.04% | 0.03 | n |
| EURUSD | D1 | cci_reversion | 0.32% | 0.11% | 0.15 | n |
| AUDJPY | D1 | mean_reversion_regime | 0.30% | 0.36% | -0.00 | n |
| EURCAD | D1 | bbands_reversion | 0.11% | 0.03% | 0.20 | n |
| NZDCAD | H4 | mean_reversion_regime | 0.10% | 0.63% | 0.21 | n |
| AUDUSD | H4 | bbands_reversion | 0.03% | 0.00% | 0.24 | n |
| EURJPY | D1 | ema_pullback | -0.28% | 0.10% | 0.09 | n |
| NZDJPY | D1 | mean_reversion_regime | -0.30% | 0.24% | 0.08 | n |
| EURCHF | H4 | cci_reversion | -0.47% | 1.35% | 0.04 | n |

## H1 probe

Downloaded H1 for: ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY', 'AUDUSD', 'USDCHF']. Yahoo ~730d — 2024 often incomplete.

| Symbol | Strat | mo2025 | mo HO | gates25 |
|--------|-------|-------:|------:|:-------:|
| EURUSD | bbands_reversion | 0.00% | 0.00% | Y |
| EURUSD | mean_reversion_regime | 0.00% | 0.00% | Y |
| EURUSD | breakout_donchian | 0.00% | 0.00% | n |
| EURUSD | squeeze_breakout | 0.00% | 0.00% | Y |
| EURUSD | cci_reversion | 0.00% | 0.00% | Y |
| GBPUSD | bbands_reversion | 0.00% | 0.00% | Y |
| GBPUSD | mean_reversion_regime | 0.00% | 0.00% | Y |
| GBPUSD | breakout_donchian | 0.00% | 0.00% | n |
| GBPUSD | squeeze_breakout | 0.00% | 0.00% | Y |
| GBPUSD | cci_reversion | 0.00% | 0.00% | Y |
| USDJPY | bbands_reversion | 0.00% | 0.00% | Y |
| USDJPY | mean_reversion_regime | 0.00% | 0.00% | Y |
| USDJPY | breakout_donchian | 0.00% | 0.00% | Y |
| USDJPY | squeeze_breakout | 0.00% | 0.00% | Y |
| USDJPY | cci_reversion | 0.00% | -0.89% | Y |

## Promotion board

| Idea | 2024 | 2025 | 2026 | HO | HO%pos | HO top3 | Promote | Why |
|------|-----:|-----:|-----:|---:|-------:|--------:|:-------:|-----|
| baseline_vt0025 | 0.42% | 1.63% | 2.30% | 1.52% | 75% | 81% | no | 2024 mean_mo=0.42% <1% |
| ov::baseline_vt0025 | 0.42% | 1.63% | 2.30% | 1.52% | 75% | 81% | no | 2024 mean_mo=0.42% <1% |
| ov::ma_only_sm0.025 | 0.18% | 0.54% | 0.93% | 0.69% | 75% | 83% | no | 2024 mean_mo=0.18% <1% |
| ov::eq_only_0.01 | 0.17% | 0.49% | 0.68% | 0.56% | 75% | 80% | no | 2024 mean_mo=0.17% <1% |
| ov::ma_only_sm0.015 | 0.16% | 0.46% | 0.92% | 0.69% | 75% | 83% | no | 2024 mean_mo=0.16% <1% |
| ov::vt+eqflat0.01 | 0.16% | 0.52% | 0.79% | 0.58% | 75% | 74% | no | 2024 mean_mo=0.16% <1% |
| ov::ma_only_sm0.02 | 0.16% | 0.52% | 0.92% | 0.69% | 75% | 83% | no | 2024 mean_mo=0.16% <1% |
| ov::vt+eqflat0.008 | 0.12% | 0.44% | 0.72% | 0.45% | 75% | 72% | no | 2024 mean_mo=0.12% <1% |
| ov::eq_only_0.006 | 0.10% | 0.32% | 0.51% | 0.36% | 75% | 75% | no | 2024 mean_mo=0.10% <1% |
| ov::vt_eq_ma_combo | -0.10% | 0.90% | 0.69% | 0.21% | 75% | 70% | no | 2024 mean_mo=-0.10% <1% |

**PROMOTE:** none — official tag unchanged `fx4plus_gbpcad_d1_voltarget_0025`.


## Failures (honest)

- Month-aware / equity-curve / runup overlays that only flatten tend to **cut 2024 further** below 0.42% or fail to lift it to 1%.
- Dual-year ≥0.8% gate is intentionally harsh after 2024-only add-ons wrecked 2025 last wave.
- H1 Yahoo history is short (~2y); incomplete 2024 calendar — H1 used as diversity probe only.
- No leverage increase (RF fixed; new flatten scales ≤1).

## Extra: vol clip_hi confirmation (eval_windowed)

| Spec | 2024 mo/%pos/top3 | 2025 mo/%pos | HO mo/%pos/top3 |
|------|------------------:|-------------:|----------------:|
| vt0025 clip_hi=1.5 | 0.25 / 73 / 67 | 0.82 / 73 | 1.00 / 75 / 80 |
| vt0025 clip_hi=2.0 | 0.28 / 64 / 71 | 1.09 / 73 | 1.27 / 83 / 79 |
| vt0025 clip_hi=1.0 | 0.18 / 73 / 67 | 0.54 / 73 | 0.72 / 75 / 82 |

## Extra: soft dual-year explore (not promote — gate was 0.8%)

| Basket | 2024 mo/%pos/top3 | 2025 mo/%pos | HO mo/%pos/top3 | Consistency note |
|--------|------------------:|-------------:|----------------:|------------------|
| +AUDJPY | 0.46 / 73 / 71 | 1.60 / 64 | 1.54 / 83 / 80 | 2025 %pos regresses |
| +NZDCAD | 0.42 / 55 / 76 | 1.64 / 64 | 1.14 / 67 / 83 | HO %pos regresses |
| +both | 0.46 / 73 / 77 | 1.56 / 64 | 1.12 / 67 / 80 | burstier / weaker HO |

**Rubric:** monthly consistency first — reject burst boosters and %pos regressions even if mean looks fine.
