# FTMO 2-Step Marathon Report (approximate_non_ftmo)

**Session end:** 2026-09-16 ~11:40 Europe/London

## Preferred basket — NEW BEST

| Metric | Value |
|--------|------:|
| **Holdout @2.5% risk** | **+4.39%** |
| Sharpe | 1.92 |
| Static / daily (Prague) | 0.03% / 0.40% |
| Gates 5%/10% | **PASS** |
| OOS basket proxy | **+3.27%** |
| **Δ vs task baseline +1.81%** | **+2.58 pp** |

Prior session best was ~+1.81% @2% / later +2.88% oos_sharpe FX4 with AUDUSD.

**Config:** `configs/best_interim_approximate.yaml`  
**Tag:** `fx4_oos_sharpe_chf_gbp_cadjpy_audcad_rf025`

| Leg | W | Exits |
|-----|--:|-------|
| USDCHF H4 bbands_reversion | 0.273 | default |
| GBPUSD H4 breakout_donchian | 0.082 | vt, tp5, stop1.5, mh24 |
| CADJPY H4 mean_reversion_regime | 0.332 | vt, mh16 |
| AUDCAD H4 mean_reversion_regime | 0.312 | atr tp5/stop1.5/mh24 |

Equal-weight same legs: HO ~+4.82% but OOS only +1.70% → not selected (OOS-first).

## Key commits
- `209891e` JPY pip_value fix + CADJPY promote
- `4ac0c44` OOS-Sharpe weights → +4.39%
- `701578d` session end (HEAD)

## Gates / blockers
- signal_lag=1; IS-only; holdout confirmation; pytest **32** green
- **No `data/ftmo/`** → approximate_non_ftmo; no golive
- Yahoo ≠ CFD; below FTMO 10% challenge target
