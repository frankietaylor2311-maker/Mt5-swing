# FTMO 2-Step Swing Research Marathon — Status

**Elapsed:** ~2h10m autonomous (13+ WF research passes).  
**Repo:** https://github.com/frankietaylor2311-maker/Mt5-swing  

## FTMO exports
**Still absent** (`data/ftmo/` only README). All results = `approximate_non_ftmo`.  
**`ftmo_golive_candidate` = 0** everywhere.

## Anti-overfit / anti-lookahead (green)
- IS-only param grids; holdout never for tuning
- Walk-forward OOS folds unused for selection  
- `signal_lag=1`; look-ahead sabotage tests **26/26 pytest green** (incl. squeeze)
- Zero-trade “wins” rejected (`min_trades≥8`)
- Basket now uses OOS stability filters (Sharpe>0, IS>0, ≤1/symbol) without holdout leak

## Engineering (latest)
1. **squeeze_breakout** strategy + ATR exits (run13) — 1 new dual winner (USDJPY H4)
2. Merged run12+run13 leaderboards (68 rows, **9 duals**)
3. FX-only basket path (`FX_ONLY=1`) for FTMO-relevant reporting

## Best duals (OOS+≥10 trades+gates+holdout)
| Combo | OOS | n | Holdout |
|-------|----:|--:|--------:|
| USDJPY D1 hybrid_regime | **0.80%** | 35 | +1.83% |
| XAUUSD H4 breakout_donchian | **0.63%** | 60 | +23.3%* |
| AUDUSD D1 hybrid_regime | **0.25%** | 96 | +1.55% |
| USDJPY D1 bbands_reversion | 0.24% | 18 | +5.2% |
| GBPUSD H4 breakout_donchian | 0.19% | 104 | +3.3% |
| USDJPY H4 squeeze_breakout | 0.12% | 44 | +0.93% |

\*Yahoo gold ≠ FTMO CFD — exploratory.

## Honest blockers
1. Need Windows FTMO MT5 H4/D1 exports → `mt5-swing import-ftmo-data`
2. Naive OOS top-basket fails holdout (GBPUSD D1 breakout etc.) — dual filter still needed for confirmation
3. Edge still far from FTMO 10%/5% phase targets in absolute terms
4. Gold results not transferable without broker CFD data
