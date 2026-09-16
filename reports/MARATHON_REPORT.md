# FTMO 2-Step Marathon Report (approximate_non_ftmo)

**As of:** 2026-09-16 ~10:45 Europe/London  
**Session deadline:** ~11:40 Europe/London

## Preferred basket (current best)

| Metric | Value |
|--------|------:|
| **Holdout return @2.5%** | **+4.39%** |
| Sharpe | 1.92 |
| Static / daily loss | 0.03% / 0.40% |
| Gates (5%/10% Prague) | **PASS** |
| OOS basket proxy | **+3.27%** |
| **Δ vs task baseline +1.81%** | **+2.58 pp** |

**Tag:** `fx4_oos_sharpe_chf_gbp_cadjpy_audcad_rf025`  
**Config:** `configs/best_interim_approximate.yaml`

| Leg | TF | Strategy | Weight | Notes |
|-----|----|----------|-------:|-------|
| USDCHF | H4 | bbands_reversion | 0.273 | |
| GBPUSD | H4 | breakout_donchian | 0.082 | vt; tp5/stop1.5/mh24 |
| CADJPY | H4 | mean_reversion_regime | 0.332 | vt; mh16 (IS) |
| AUDCAD | H4 | mean_reversion_regime | 0.312 | IS atr tp5/stop1.5/mh24 |

**Construction path (a priori / dual):** JPY pip fix → CADJPY dual → replace AUDUSD → add AUDCAD / drop USDJPY by OOS → OOS-Sharpe weights beat equal on OOS.

Risk stress PASS: 1%→+2.68% · 1.5%→+3.96% · 2%→+4.42% · **2.5%→+4.39%**

## Notable this session
- JPY `pip_value_per_lot` fix; `willr_reversion`; merge-protect research runner
- Equal then OOS-Sharpe promotes; CADJPY + AUDCAD dual legs
- Basket builder overlays locked IS exits

## Blockers
1. No `data/ftmo/` → approximate_non_ftmo; no golive
2. Far from FTMO 10% challenge target
3. Yahoo FX ≠ broker CFD
4. Pure Sharpe-first weaker than interim path on holdout
