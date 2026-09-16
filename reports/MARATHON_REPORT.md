# FTMO 2-Step Swing Research Marathon — Report

**Wall time:** ~45–50 minutes autonomous iteration (9+ WF passes).  
**Repo:** https://github.com/frankietaylor2311-maker/Mt5-swing  
**Branch tip:** see `git log -1`  
**FTMO exports:** **NONE** (`data/ftmo/` empty) → **`ftmo_golive_candidate` always false**.

## Discipline (all passes)
- IS-only constrained grids; OOS never used for param selection
- Untouched holdout confirmation only
- `signal_lag=1`; look-ahead tests green (**25 pytest**)
- Zero-trade “wins” rejected (`min_trades≥8`)

## Key engineering wins
1. **Donchian prior-window fix** — current-bar channel made breakouts never fire
2. New strategies: `ema_pullback`, `hybrid_regime`, `bbands_reversion`, `keltner_breakout`
3. ATR stop/target exits (trend/breakout family only); MR/BB keep mid exits
4. Causal HTF SMA proxy filter; `vol_target` off; risk up to **2%**/trade
5. Approximate **XAUUSD** from Yahoo `GC=F` (labeled `approximate_non_ftmo`)

## Best dual results (OOS profitable + ≥10 trades + gates + holdout ok)
| Combo | OOS | n | Holdout |
|-------|----:|--:|--------:|
| USDJPY D1 `hybrid_regime` | **0.79%** | 35 | +1.32% |
| XAUUSD H4 `breakout_donchian` | **0.56%** | 56 | +15.5%* |
| XAUUSD D1 `breakout_donchian` | **0.49%** | 436 | +10.8%* |
| USDJPY D1 `bbands_reversion` | 0.28% | 18 | +5.2% |

\*Yahoo gold ≠ FTMO CFD — exploratory only.

## Preferred basket (a priori recipe, OOS-ranked)
**XAU breakout + USDJPY D1 hybrid/BB** → holdout about **+1.4%**, gates **PASS**, Sharpe ~**0.95**  
(`reports/basket_preferred.md`)

## Commits (newest → older)
`3b90683` `b35641c` `3e6c0fa` `4841c39` `12f2149` `4fae3e8` `64978cf` `fc74033`

## Blockers
1. No `ftmo_mt5_export` OHLC — cannot claim go-live
2. Interim Yahoo FX edge is thin vs FTMO 10%/5% targets
3. Gold results not transferable without FTMO terminal exports
4. Some high-OOS FX breakouts fail holdout (overfit risk)

## Config pointer
`configs/best_interim_approximate.yaml` + `reports/leaderboard.md`
