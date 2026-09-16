# FTMO 2-Step Marathon Report (approximate_non_ftmo)

**As of:** 2026-09-16 ~10:30 Europe/London

## Preferred (OOS-best dual FX4)

| Basket | Holdout | Sharpe | Gates | OOS |
|--------|--------:|-------:|------:|----:|
| Task baseline @2% | +1.81% | — | PASS | — |
| Prior oos_sharpe FX4 | +2.88% | 1.23 | PASS | — |
| **Current FX4 equal CHF/GBP/CADJPY/AUDCAD @2.5%** | **4.82%** | **1.91** | **PASS** | **1.70%** |

**Δ vs +1.81%:** **+3.01 pp**

**Legs:** USDCHF H4 bbands · GBPUSD H4 breakout (vt) · CADJPY H4 MR (vt+mh16) · AUDCAD H4 MR (IS ATR exits)

**Config:** `fx4_equal_chf_gbp_cadjpy_audcad_rf025`

## Methodology / infra
- signal_lag=1; IS-only; OOS basket construction; holdout confirmation
- JPY pip_value fix; willr strategy; equal weights
- No golive without data/ftmo/

## Blockers
No FTMO exports; Yahoo≠CFD; below 10% challenge target; pure Sharpe-first fails holdout.
