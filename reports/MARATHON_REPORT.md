# FTMO 2-Step Swing Research Marathon — Status

**Wave:** pairs/strats + OOS-Sharpe weights + IS ATR exits.  
**Repo:** https://github.com/frankietaylor2311-maker/Mt5-swing  

## FTMO exports
**Still absent.** All = `approximate_non_ftmo`. **`ftmo_golive_candidate` = 0**.

## Anti-overfit / anti-lookahead
- IS-only grids + IS-only exit refine; holdout confirmation only; `signal_lag=1`
- pytest green
- Basket: overlapping dates; interim dual FX4; weights ∝ OOS Sharpe; risk=2.5%

## Preferred vs prior +1.81%@2%
| Basket | Holdout | Sharpe | Gates |
|--------|--------:|-------:|:-----:|
| Prior FX4 equal @2% | +1.81% | 0.92 | PASS |
| FX4 OOS-Sharpe w @2.5% | +2.28% | 0.98 | PASS |
| **FX4 OOS-Sharpe w + IS exits @2.5%** | **+2.83%** | **1.21** | PASS |

**Legs:** USDCHF H4 bbands · USDJPY D1 hybrid (trail1.5/tp4) · GBPUSD H4 breakout (tp4) · AUDUSD D1 hybrid (tp4)

## Added this wave
- EURGBP/AUDJPY/EURCHF/CADJPY/NZDJPY data
- stoch_reversion, atr_channel_breakout, cci_reversion
- Preferred builder + OOS-Sharpe weights + IS exit overrides

## Blockers
1. No FTMO MT5 exports  
2. Pure Sharpe-first dual basket fails holdout  
3. Far from FTMO 10%/5% targets  
4. Yahoo ≠ broker CFD  
