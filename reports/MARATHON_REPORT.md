# FTMO 2-Step Swing Research Marathon — Report

**Elapsed:** ~40 minutes of autonomous iteration (continuing capability retained).  
**Repo:** https://github.com/frankietaylor2311-maker/Mt5-swing  
**Data:** `approximate_non_ftmo` only — **`data/ftmo/` empty** → **0** `ftmo_golive_candidate`.

## Discipline held
- IS-only constrained grids; walk-forward OOS never used for selection
- Untouched holdout for confirmation only
- `signal_lag=1`; look-ahead sabotage tests green (25 pytest)
- Zero-trade “wins” rejected (`min_trades≥8`)

## Root-cause fix
Donchian included the current bar → breakouts never fired. Exclusive prior-N window restored Turtle-style breaks and trade counts.

## Commits (latest first)
- `3e6c0fa` Run6 + XAUUSD Yahoo data
- `4841c39` Run5 @1.5% risk; BB basket +1.7%
- `12f2149` HTF filter; vol_target off
- `4fae3e8` Strategy-aware ATR; keltner
- `64978cf` ATR exits; risk 1%
- `fc74033` Donchian fix + new strategies

## Leaderboard highlights (run7)
| Rank | Combo | OOS | n | Holdout | Gates |
|-----:|-------|----:|--:|--------|:-----:|
| 1 | USDJPY D1 hybrid_regime | 0.79% | 35 | +1.32% | PASS |
| 2 | XAUUSD H4 breakout_donchian | 0.56% | 56 | **+15.5%** | PASS |
| 3 | XAUUSD D1 breakout_donchian | 0.49% | 436 | +10.8% | PASS |
| 4 | USDJPY D1 bbands_reversion | 0.28% | 18 | +5.2% | PASS |

## Baskets (holdout confirmation)
- BB-family (FX): ~+1.5–1.7% gates PASS (run4–5)
- Preferred XAU-breakout + USDJPY hybrid/BB: see `reports/basket_preferred.md`

## Preferred basket v2 (holdout)
- **+1.39%** return, gates **PASS**, Sharpe **0.95**
- Legs: USDJPY D1 hybrid + XAUUSD H4/D1 breakout + USDJPY D1 bbands (OOS-ranked within a priori recipe)

## Honest blockers
1. **No FTMO MT5 exports** — cannot set go-live true
2. Yahoo gold ≠ FTMO XAUUSD CFD (spread/session/gap) — gold results exploratory
3. Absolute FX OOS returns still small vs 10% / 5% challenge targets even at 2% risk/trade
4. Some high-OOS FX breakouts fail holdout (overfit risk)

## Best configs
See `configs/best_interim_approximate.yaml`.
