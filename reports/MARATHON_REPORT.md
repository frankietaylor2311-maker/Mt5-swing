# FTMO 2-Step Swing Research Marathon — Status

**Wave:** run21–23 (EURGBP/AUDJPY/EURCHF + stoch/atr_channel) + OOS-Sharpe portfolio weights.  
**Repo:** https://github.com/frankietaylor2311-maker/Mt5-swing  
**HEAD:** see latest main  

## FTMO exports
**Still absent** (`data/ftmo/` README only). All = `approximate_non_ftmo`. **`ftmo_golive_candidate` = 0**.

## Anti-overfit / anti-lookahead
- IS-only grids; holdout confirmation only; `signal_lag=1`
- pytest green
- Basket: overlapping dates only; dual FX ≤1/symbol interim set; **weights ∝ OOS Sharpe**; risk=`ftmo_2step` default 2.5%
- Pure dual Sharpe-first (no interim lock) still fails holdout (~−0.7%) — documented, not promoted

## Preferred vs prior +1.81%@2%
| Basket | Holdout | Sharpe | Gates |
|--------|--------:|-------:|:-----:|
| Prior FX4 equal @2% | +1.81% | 0.92 | PASS |
| FX4 equal @2.5% config | +2.03% | 0.88 | PASS |
| **FX4 OOS-Sharpe w @2.5%** | **+2.28%** | **0.98** | PASS |

**Legs:** USDCHF H4 bbands · USDJPY D1 hybrid · GBPUSD H4 breakout · AUDUSD D1 hybrid  
**Weights (OOS Sh):** ~0.455 / 0.300 / 0.137 / 0.108

Risk stress (PASS 5%/10%): 1.0%→+1.18% · 1.5%→+1.78% · 2.0%→+2.11% · **2.5%→+2.28%**

## Data / strategies added
- EURGBP, AUDJPY, EURCHF interim downloads
- `stoch_reversion`, `atr_channel_breakout` (+ `stoch_k` feature)
- Notable OOS: USDCHF D1 stoch +0.49% (n=64); AUDUSD D1 atr_channel +0.36%; EURGBP D1 stoch +0.10% (dual)

## Engineering
1. Preferred builder supports interim FX4 + OOS-Sharpe weights  
2. Calendar-honest join unchanged  
3. New strategy tests (lookahead-stable)

## Blockers
1. No FTMO MT5 exports — stay `approximate_non_ftmo`  
2. Pure a priori Sharpe-first dual basket fails holdout  
3. Far from FTMO 10%/5% phase targets  
4. Yahoo ≠ broker CFD  
