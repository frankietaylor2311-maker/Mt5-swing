# Marathon progress (interim)

**UTC start:** see marathon_start_utc.txt  
**Commit pushed:** `fc74033` — Donchian prior-window fix + 3 new strategies + min-trade IS scoring.

## Diagnosis fixed
- `breakout_donchian` had **0 trades** because Donchian included the current bar (`close > upper` impossible).
- Exclusive prior-N window restores Turtle-style breaks; OOS now shows **dozens** of trades on H4.
- Zero-trade “wins” rejected via `min_trades=8` in `constrained_grid_search`.

## New strategies
- `ema_pullback`, `hybrid_regime`, `bbands_reversion` (registered; look-ahead tests green).

## Early run1 snippets (approximate_non_ftmo)
| Combo | OOS ret | OOS n | Gates | Holdout |
|---|---:|---:|:---:|:---:|
| EURUSD H4 bbands | +0.07% | 37 | PASS | NO |
| EURUSD H4 ema_pullback | +0.01% | 22 | PASS | NO |
| EURUSD H4 MR | +0.04% | 8 | PASS | NO |
| EURUSD D1 bbands | +0.15% | 30 | PASS | NO |

FTMO exports: still **absent**. `ftmo_golive_candidate` stays false.
