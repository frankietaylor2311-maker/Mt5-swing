# FTMO 2-Step Marathon Report (approximate_non_ftmo)

**As of:** 2026-09-16 ~10:30 Europe/London

## Preferred

| Basket | Holdout | Sharpe | Gates | OOS |
|--------|--------:|-------:|------:|----:|
| Task baseline @2% | +1.81% | — | PASS | — |
| **Current FX4 OOS-Sharpe CHF/GBP/CADJPY/AUDCAD @2.5%** | **4.39%** | **1.92** | **PASS** | **3.27%** |

**Δ vs +1.81%:** **+2.58 pp**

**Legs / weights:** see `configs/best_interim_approximate.yaml` (`fx4_oos_sharpe_chf_gbp_cadjpy_audcad_rf025`)

## Blockers
No FTMO exports; Yahoo≠CFD; below 10% challenge target.
