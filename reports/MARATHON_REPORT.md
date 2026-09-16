# FTMO 2-Step Swing Research Marathon — Status

**Wave through run32:** multi-pair expansion, new strats, OOS-Sharpe weights, IS exits, IS vol_target.  
**Repo:** https://github.com/frankietaylor2311-maker/Mt5-swing  

## FTMO exports
**Still absent.** All = `approximate_non_ftmo`. **`ftmo_golive_candidate` = 0**.

## Preferred vs prior +1.81%@2% equal-weight
| Basket | Holdout | Sharpe | Gates |
|--------|--------:|-------:|:-----:|
| Prior FX4 equal @2% | +1.81% | 0.92 | PASS |
| **Current FX4 @2.5% OOS-Sharpe + IS exits + IS vol_target** | **+2.88%** | **1.23** | PASS |

**Δ vs +1.81%:** **+1.07 pp** holdout (same dual FX4 legs; construction/exits improved a priori).

**Legs:** USDCHF H4 bbands · USDJPY D1 hybrid · GBPUSD H4 breakout (vol_target) · AUDUSD D1 hybrid  
**Weights ∝ OOS Sharpe:** ~0.455 / 0.300 / 0.137 / 0.108  

Risk stress (PASS 5%/10% Prague): 1%→+1.68% · 1.5%→+2.39% · 2%→+2.81% · **2.5%→+2.88%**

## Methodology (HARD gates)
- `signal_lag=1`; IS-only param + exit + vol_target selection; holdout confirmation only
- Dual FX ≤1/symbol interim set; calendar-honest overlapping join
- Pure Sharpe-first dual (no interim lock) fails holdout (~−0.7%) — documented
- pytest 30/30 green

## Data / strats added this wave
- EURGBP, AUDJPY, EURCHF, CADJPY, NZDJPY, EURCAD, AUDCAD, GBPCAD, EURAUD, NZDCAD
- stoch_reversion, atr_channel_breakout, cci_reversion; bbands max_hold
- Notable: GBPCAD D1 bbands OOS +0.84% Sh 2.50 (not promoted — FX5 dilutes vs FX4)

## Blockers
1. No FTMO MT5 exports — no go-live  
2. Pure a priori Sharpe-first basket fails holdout  
3. Far from FTMO 10%/5% phase targets  
4. Yahoo ≠ broker CFD  
