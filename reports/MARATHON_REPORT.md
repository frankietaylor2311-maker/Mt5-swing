# FTMO 2-Step Swing Research Marathon — Status

**Wave:** continued past ~68798c4 through run18 (NZD/CHF + calendar-honest baskets).  
**Repo:** https://github.com/frankietaylor2311-maker/Mt5-swing  

## FTMO exports
**Still absent.** All results = `approximate_non_ftmo`. **`ftmo_golive_candidate` = 0**.

## Anti-overfit / anti-lookahead
- IS-only grids; holdout confirmation only; `signal_lag=1`
- **28/28 pytest green**
- Basket equity: overlapping dates only (`dropna any`)

## Engineering this wave
1. ATR trail + max_hold_bars exits
2. macd_trend (no dual yet)
3. Risk stress 1–2%; calendar-honest joins
4. Downloaded NZDUSD + USDCHF interim; run18 WF
5. New duals: **USDCHF H4/D1 bbands_reversion**

## Preferred basket vs prior ~+1.7%
| Basket | Holdout | Sharpe | Gates |
|--------|--------:|-------:|:-----:|
| **Prior wave (reported)** | ~+1.69–1.7% | ~0.97 | PASS |
| FX3 ≤1/symbol @2% (calendar-honest) | +1.50% | 0.52 | PASS |
| **FX4 +USDCHF H4 bbands @2% (NEW BEST)** | **+1.81%** | **0.92** | PASS |

Legs (FX4): USDJPY D1 hybrid · AUDUSD D1 hybrid · GBPUSD H4 breakout · USDCHF H4 bbands  
Risk stress: 1.0%→+0.98%, 1.5%→+1.47%, **2.0%→+1.81%** — all PASS 5%/10%.

## Top duals
| Combo | OOS | n | Holdout |
|-------|----:|--:|--------:|
| USDJPY D1 hybrid_regime | 0.80% | 35 | +1.83% |
| XAUUSD H4 breakout_donchian | 0.63% | 60 | +23.3%* |
| AUDUSD D1 hybrid_regime | 0.25% | 96 | +1.55% |
| USDJPY D1 bbands_reversion | 0.24% | 18 | +5.2% |
| USDCHF D1 bbands_reversion | 0.22% | 21 | +0.77% |
| USDCHF H4 bbands_reversion | 0.21% | 26 | +3.60% |
| GBPUSD H4 breakout_donchian | 0.19% | 104 | +3.3% |

## Blockers
1. No FTMO MT5 exports
2. OOS-only a priori basket still fails holdout
3. Far from 10%/5% phase targets
4. Yahoo ≠ FTMO CFD

## Commits
- `40ea118` ATR trail + macd
- `eafc1b3` calendar-honest FX3 +1.50%
- follow-up: NZD/CHF data + FX4 preferred **+1.81%**
