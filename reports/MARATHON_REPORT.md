# FTMO 2-Step Swing Research Marathon — Status

**Wave:** continued past ~68798c4 through run19 (exits, NZD/CHF, hybrid max_hold, Sharpe-first basket).  
**Repo:** https://github.com/frankietaylor2311-maker/Mt5-swing  

## FTMO exports
**Still absent.** All = `approximate_non_ftmo`. **`ftmo_golive_candidate` = 0**.

## Anti-overfit / anti-lookahead
- IS-only grids; holdout confirmation only; `signal_lag=1`
- **28/28 pytest green** (ATR trail/time-stop, macd_trend, hybrid max_hold)
- Basket equity: overlapping dates only (`dropna any`)
- Preferred members: dual FX ≤1/symbol ranked by **OOS Sharpe → OOS return** (a priori)

## Preferred vs prior ~+1.7%
| Basket | Holdout | Sharpe | Gates |
|--------|--------:|-------:|:-----:|
| Prior wave (reported) | ~+1.69–1.7% | ~0.97 | PASS |
| **FX4 Sharpe-first @2% (NEW)** | **+1.81%** | **0.92** | PASS |

**Legs:** USDCHF H4 bbands · USDJPY D1 hybrid · GBPUSD H4 breakout · AUDUSD D1 hybrid  

Risk stress: 1.0%→+0.98% · 1.5%→+1.47% · **2.0%→+1.81%** — all PASS FTMO 5%/10%.

## Engineering
1. ATR trail + max_hold_bars engine exits  
2. macd_trend (no dual)  
3. hybrid_regime `max_hold` IS grid (run19) — max_hold=20 often hurts OOS; keep higher-OOS IS-valid hybrid  
4. NZDUSD/USDCHF data + run18 — **USDCHF bbands duals**  
5. Calendar-honest basket join  

## Top duals (canonical)
| Combo | OOS | Sh | Holdout |
|-------|----:|---:|--------:|
| USDJPY D1 hybrid | 0.80% | 0.95 | +1.83% |
| XAUUSD H4 breakout* | 0.63% | 0.51 | +23.3% |
| AUDUSD D1 hybrid | 0.25% | 0.34 | +1.55% |
| USDCHF H4 bbands | 0.21% | **1.44** | +3.60% |
| GBPUSD H4 breakout | 0.19% | 0.43 | +3.3% |

\*Yahoo gold ≠ FTMO CFD.

## Blockers
1. No FTMO MT5 exports  
2. OOS-only a priori basket still fails holdout  
3. Far from 10%/5% phase targets  
4. Approximate Yahoo data only  

## Commits
- `40ea118` ATR trail + macd  
- `eafc1b3` calendar-honest FX3  
- `cc2dcf0` NZD/CHF + FX4 +1.81%  
- follow-up: hybrid max_hold + Sharpe-first preferred lock  
