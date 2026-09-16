# FTMO 2-Step Swing Research Marathon — Final Report

**Elapsed:** ~66 minutes autonomous (12 WF research passes).  
**HEAD:** `(see git log)`  
**Repo:** https://github.com/frankietaylor2311-maker/Mt5-swing  

## FTMO exports
**Still absent** (`data/ftmo/` only README). All results = `approximate_non_ftmo`.  
**`ftmo_golive_candidate` = 0** everywhere.

## Anti-overfit / anti-lookahead (green)
- IS-only param grids; holdout never for tuning
- Walk-forward OOS folds unused for selection  
- `signal_lag=1`; look-ahead sabotage tests **25/25 pytest green**
- Zero-trade “wins” rejected (`min_trades≥8`)

## Engineering breakthroughs
1. **Donchian prior-N fix** — breakouts were structurally impossible (0 trades)
2. Strategies added: `ema_pullback`, `hybrid_regime`, `bbands_reversion`, `keltner_breakout`
3. ATR stop/target exits for trend/breakout family; MR/BB keep signal mid-exits
4. Causal HTF SMA filter; `vol_target=false`; risk up to **2.5%**/trade
5. Expanded universe: EUR/GBP/JPY + **XAUUSD** (Yahoo GC=F) + **AUDUSD/USDCAD**

## Best duals (OOS+≥10 trades+gates+holdout) — run12
| Combo | OOS | n | Holdout |
|-------|----:|--:|--------:|
| USDJPY D1 hybrid_regime | **0.80%** | 35 | +1.83% |
| XAUUSD H4 breakout_donchian | **0.63%** | 60 | +23.3%* |
| AUDUSD D1 hybrid_regime | **0.25%** | 96 | +1.55% |
| USDJPY D1 bbands_reversion | 0.24% | 18 | +5.2% |
| GBPUSD H4 breakout_donchian | 0.19% | 104 | +3.3% |

\*Yahoo gold ≠ FTMO CFD — exploratory.

## Preferred basket (holdout confirmation)
A priori recipe → OOS-ranked legs: **+1.7% to +2.0%** holdout, **gates PASS**, Sharpe ~**1.0**  
See `reports/basket_preferred.md`.

## vs starting point
| Metric | Before | After |
|--------|--------|-------|
| Zero-trade OOS “passes” | many | **0** (meaningful n) |
| OOS profitable+gates | 6/18 | **29/60** |
| Holdout ok | 4/18 | **19/60** |
| Max dual OOS | ~0.2% | **~0.8%** FX / **0.63%** gold |
| Go-live candidates | 0 | **0** (no FTMO data) |

## Honest blockers
1. Need Windows FTMO MT5 H4/D1 exports → `mt5-swing import-ftmo-data`
2. Interim Yahoo edge still far from FTMO 10%/5% phase targets in absolute $ terms
3. Gold results not transferable without broker CFD data
4. Higher risk (2.5%) can fail OOS gates on overtrading gold D1 breakouts

## Artifacts
- `reports/leaderboard.md` / `walk_forward_summary.csv`
- `configs/best_interim_approximate.yaml`
- Run archives: `reports/walk_forward_summary_run{1..12}.csv`
