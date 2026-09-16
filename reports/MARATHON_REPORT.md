# FTMO 2-Step Swing Research Marathon — Status

**Wave:** OOS-Sharpe weights + IS exits v2 + IS vol_target.  
**Repo:** https://github.com/frankietaylor2311-maker/Mt5-swing  

## FTMO exports
**Still absent.** All = `approximate_non_ftmo`. **`ftmo_golive_candidate` = 0**.

## Anti-overfit / anti-lookahead
- IS-only grids + IS-only exit/vol_target refine; holdout confirmation only; `signal_lag=1`
- pytest 30/30 green
- Basket: overlapping dates; interim dual FX4; weights ∝ OOS Sharpe; risk=2.5%

## Preferred vs prior +1.81%@2%
| Basket | Holdout | Sharpe | Gates |
|--------|--------:|-------:|:-----:|
| Prior FX4 equal @2% | +1.81% | 0.92 | PASS |
| FX4 OOS-Sharpe w @2.5% | +2.28% | 0.98 | PASS |
| FX4 + IS exits v2 | +2.84% | 1.20 | PASS |
| **FX4 + IS exits + IS vol_target** | **+2.88%** | **1.23** | PASS |

**Legs:** USDCHF H4 bbands · USDJPY D1 hybrid · GBPUSD H4 breakout (vol_target) · AUDUSD D1 hybrid  

Risk stress (PASS): 1.0%→+1.68% · 1.5%→+2.39% · 2.0%→+2.81% · **2.5%→+2.88%**

## Added
- Pairs: EURGBP/AUDJPY/EURCHF/CADJPY/NZDJPY
- Strats: stoch_reversion, atr_channel_breakout, cci_reversion; bbands max_hold (not preferred)
- Preferred builder with OOS-Sharpe weights + IS exit overrides

## Blockers
1. No FTMO MT5 exports  
2. Pure Sharpe-first dual fails holdout  
3. Far from FTMO 10%/5% targets  
4. Yahoo ≠ broker CFD  
