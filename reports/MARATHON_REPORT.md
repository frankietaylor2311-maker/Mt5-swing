# FTMO 2-Step Swing Research Marathon — Status

**Wave:** continued past HEAD ~68798c4 through run17 (exits / macd / calendar-honest baskets).  
**Repo:** https://github.com/frankietaylor2311-maker/Mt5-swing  

## FTMO exports
**Still absent** (`data/ftmo/` only README). All results = `approximate_non_ftmo`.  
**`ftmo_golive_candidate` = 0** everywhere — never set go-live.

## Anti-overfit / anti-lookahead (green)
- IS-only param grids; holdout never for tuning
- `signal_lag=1`; **28/28 pytest green** (macd_trend, ATR trail/time-stop)
- `min_trades≥8`; basket ≤1/symbol among duals for confirmation
- Basket equity uses **overlapping dates only** (`dropna how=any`) — fixes H4/D1 calendar bias that previously inflated mixed-TF portfolios

## Engineering this wave
1. ATR trailing stop + max_hold_bars in engine
2. macd_trend strategy (+ hold_while_hist); run15–17 — no meaningful dual yet
3. RISK_FRACTION / ATR_TRAIL_MULT env knobs + `run_preferred_risk_stress.py`
4. Run14 @1.5%: AUDUSD D1 keltner dual artifact (OOS 0.35% / hold +1.98%)
5. Calendar-honest basket aggregation

## Best duals (canonical ~2.5%)
| Combo | OOS | n | Holdout |
|-------|----:|--:|--------:|
| USDJPY D1 hybrid_regime | **0.80%** | 35 | +1.83% |
| XAUUSD H4 breakout_donchian | **0.63%** | 60 | +23.3%* |
| AUDUSD D1 hybrid_regime | **0.25%** | 96 | +1.55% |
| USDJPY D1 bbands_reversion | 0.24% | 18 | +5.2% |
| GBPUSD H4 breakout_donchian | 0.19% | 104 | +3.3% |
| USDJPY H4 squeeze_breakout | 0.12% | 44 | +0.93% |

\*Yahoo gold ≠ FTMO CFD — exploratory.

## Preferred basket (confirmation) vs prior ~+1.7%
**Prior wave reported ~+1.69–1.7%** dual-confirm mix (likely with loose calendar join).

**This wave (calendar-honest, overlapping dates):**
| Basket | Holdout | Sharpe | Gates | Notes |
|--------|--------:|-------:|:-----:|-------|
| **FX3 ≤1/symbol @2%** (USDJPY hybrid, AUDUSD hybrid, GBPUSD H4 breakout) | **+1.50%** | 0.52 | PASS | Primary preferred |
| FX3 USDJPY-heavy D1 @2% | +1.41% | 2.14 | PASS | Thin bbands holdout n=4 |
| +XAU exploratory @2% | +1.76% | — | PASS | Not FTMO-transferable |
| Trail=1.5 on FX3 primary | −3.97% | — | — | Do **not** use trail on this mix |

Risk stress (FX3 primary, no trail): 1.0%→+0.80%, 1.5%→+1.21%, **2.0%→+1.50%** — all PASS 5%/10%.

## Honest OOS-only a priori basket
Still fails holdout (~−0.8%) — GBPUSD D1 breakout OOS mirage.

## Blockers
1. FTMO MT5 exports still missing
2. Dual filter still needed; OOS-only basket fails holdout
3. Far from 10%/5% phase targets
4. macd_trend: no dual winners
5. Yahoo ≠ FTMO CFD (esp. gold)

## Commits this wave
- `40ea118` — ATR trail, macd_trend, initial FX3@2% (pre-calendar-fix)
- follow-up — calendar-honest preferred + basket join fix
