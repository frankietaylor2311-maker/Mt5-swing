# FTMO 2-Step Swing Research Marathon — Status

**Wave:** pairs/strats + OOS-Sharpe weights + IS ATR exits v2.  
**Repo:** https://github.com/frankietaylor2311-maker/Mt5-swing  
**HEAD:** see latest main  

## FTMO exports
**Still absent** (`data/ftmo/` README only). All = `approximate_non_ftmo`. **`ftmo_golive_candidate` = 0**.

## Anti-overfit / anti-lookahead
- IS-only param grids + IS-only exit refine; holdout confirmation only; `signal_lag=1`
- pytest green (30+)
- Basket: overlapping dates; interim dual FX4; weights ∝ OOS Sharpe; risk=2.5% (=ftmo_2step default)

## Preferred vs prior +1.81%@2%
| Basket | Holdout | Sharpe | Gates |
|--------|--------:|-------:|:-----:|
| Prior FX4 equal @2% | +1.81% | 0.92 | PASS |
| FX4 OOS-Sharpe w @2.5% | +2.28% | 0.98 | PASS |
| **FX4 OOS-Sharpe w + IS exits v2 @2.5%** | **+2.84%** | **1.20** | PASS |

**Legs:** USDCHF H4 bbands · USDJPY D1 hybrid · GBPUSD H4 breakout · AUDUSD D1 hybrid  
**Weights (OOS Sh):** ~0.455 / 0.300 / 0.137 / 0.108  
**IS exits:** USDJPY trail1.5/stop1.5/hold16 · GBPUSD tp5/stop1.5/hold24 · AUDUSD tp2/stop2.5/hold16

Risk stress (all PASS): 1.0%→+1.69% · 1.5%→+2.37% · 2.0%→+2.81% · **2.5%→+2.84%**

## Data / strategies added
- EURGBP, AUDJPY, EURCHF, CADJPY, NZDJPY
- stoch_reversion, atr_channel_breakout, cci_reversion
- CAD/NZD JPY: no dual OOS≥0.1%; EURUSD D1 CCI strong OOS but holdout fails

## Blockers
1. No FTMO MT5 exports — stay `approximate_non_ftmo`  
2. Pure Sharpe-first dual basket fails holdout (~−0.7%)  
3. Far from FTMO 10%/5% phase targets  
4. Yahoo ≠ broker CFD  
