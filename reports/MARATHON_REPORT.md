# FTMO 2-Step Swing Research Marathon — Status

**Wave:** continued past ~68798c4 through run20 (exits, multi-pair expansion, Sharpe-first basket).  
**Repo:** https://github.com/frankietaylor2311-maker/Mt5-swing  
**HEAD:** see latest main  

## FTMO exports
**Still absent** (`data/ftmo/` README only). All = `approximate_non_ftmo`. **`ftmo_golive_candidate` = 0**.

## Anti-overfit / anti-lookahead
- IS-only grids; holdout confirmation only; `signal_lag=1`
- **28/28 pytest green**
- Basket: overlapping dates only; dual FX ≤1/symbol; OOS Sharpe-first; **min OOS return 0.1%**

## Preferred vs prior ~+1.7–2%
| Basket | Holdout | Sharpe | Gates |
|--------|--------:|-------:|:-----:|
| Prior wave | ~+1.69–1.7% | ~0.97 | PASS |
| **This wave FX4 @2%** | **+1.81%** | **0.92** | PASS |

**Legs:** USDCHF H4 bbands · USDJPY D1 hybrid · GBPUSD H4 breakout · AUDUSD D1 hybrid  

Risk stress (all PASS 5%/10%): 1.0%→+0.98% · 1.5%→+1.47% · **2.0%→+1.81%**

## Data / duals added
- NZDUSD, USDCHF, EURJPY, GBPJPY interim downloads
- New duals: USDCHF H4/D1 bbands; EURJPY H4 breakout (OOS edge <0.1% floor → excluded from preferred)

## Engineering
1. ATR trail + max_hold_bars exits  
2. macd_trend (no dual)  
3. hybrid max_hold IS grid  
4. Calendar-honest basket join + Sharpe-first + min-OOS filters  
5. RISK_FRACTION / ATR_TRAIL_MULT env + risk-stress script  

## Blockers
1. No FTMO MT5 exports — stay `approximate_non_ftmo`  
2. OOS-only a priori basket still fails holdout  
3. Far from FTMO 10%/5% phase targets  
4. Yahoo ≠ broker CFD  

## Key commits this wave
- `40ea118` ATR trail + macd  
- `eafc1b3` calendar-honest FX3  
- `cc2dcf0` NZD/CHF + FX4 +1.81%  
- `5cea9cd` Sharpe-first lock  
- follow-up: EURJPY/GBPJPY + min-OOS floor  
