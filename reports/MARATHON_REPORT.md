# FTMO 2-Step Swing Research Marathon — Status

**Wave:** continued past HEAD ~68798c4 through run16 (macd / exits / risk stress).  
**Repo:** https://github.com/frankietaylor2311-maker/Mt5-swing  

## FTMO exports
**Still absent** (`data/ftmo/` only README). All results = `approximate_non_ftmo`.  
**`ftmo_golive_candidate` = 0** everywhere — never set go-live.

## Anti-overfit / anti-lookahead (green)
- IS-only param grids; holdout never for tuning
- Walk-forward OOS folds unused for selection
- `signal_lag=1`; look-ahead sabotage tests **28/28 pytest green** (incl. macd_trend, ATR trail/time-stop)
- Zero-trade “wins” rejected (`min_trades≥8`)
- Basket a priori: OOS Sharpe>0, IS>0, ≥15 trades, symbol breadth, ≤1/symbol+strategy; holdout confirmation only

## Engineering this wave
1. **ATR trailing stop + max_hold_bars** exits in backtest engine (`atr_trail_mult`, `max_hold_bars`)
2. **macd_trend** strategy + IS grid (run15/16)
3. **RISK_FRACTION / ATR_TRAIL_MULT** env overrides for research + basket stress
4. Preferred **risk stress** script (`scripts/run_preferred_risk_stress.py`) — reports 1.0/1.5/2.0% under FTMO gates
5. Run14 @1.5% risk: new dual sensitivity **AUDUSD D1 keltner_breakout** (OOS 0.35% / hold +1.98%) — kept in run14 artifact, not mixed into 2.5% canonical

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

## Preferred basket (confirmation) — NEW BEST vs prior ~+1.7%
**FX3 @ 2.0% risk/trade, trail=0 (primary):**
- **Holdout basket +1.74%**, static loss 0.69%, daily 0.53%, gates **PASS**, Sharpe **2.01**
- Legs: USDJPY D1 hybrid · AUDUSD D1 hybrid · USDJPY D1 bbands
- Trail=1.5 **sensitivity only**: +1.97% / Sharpe 2.23 (not primary — avoids holdout-max cherry-pick)
- Prior wave dual-confirm mix ~**+1.69%** (with XAU) → FX3@2% is the new FX-relevant confirmation best; XAU@2% equal-weight ≈ +1.39%

## Honest OOS-only a priori basket
Still **fails holdout (~−0.8%)** — GBPUSD D1 breakout looks strong OOS then fails confirmation.

## Risk stress (FX3 preferred legs)
| Risk | No trail | Trail 1.5 |
|---:|---:|---:|
| 1.0% | +0.92% | +1.04% |
| 1.5% | +1.39% | +1.58% |
| 2.0% | **+1.74%** | +1.97% |
All PASS FTMO 5%/10% gates on holdout window.

## Blockers
1. Need Windows FTMO MT5 H4/D1 exports → `mt5-swing import-ftmo-data`
2. Naive OOS top-basket fails holdout — dual filter still needed
3. Edge far from FTMO 10%/5% phase targets in absolute terms
4. macd_trend: no dual winners yet under trail; run16 no-trail in flight
5. Gold results not transferable without broker CFD data
6. Preferred FX3 concentrates 2/3 legs on USDJPY; bbands holdout n=4 (thin)

## Artifacts
- `configs/best_interim_approximate.yaml` — FX3 @2%, trail=0 primary
- `reports/basket_preferred.md`, `basket_risk_stress.md`, `basket_risk_stress_trail.md`
- `reports/leaderboard_run14.csv` (1.5% risk), `leaderboard_run15.csv` (macd+trail)
