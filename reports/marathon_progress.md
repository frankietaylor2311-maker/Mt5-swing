# Marathon progress

**Elapsed:** ~15–20 min into multi-hour push (continuing).

## Commits
- `fc74033` Donchian prior-window fix + ema_pullback/hybrid/bbands + min-trade IS scoring
- `64978cf` ATR stop/target exits + risk_fraction 0.01 + run1 leaderboard
- (latest) Run2 archive + strategy-aware ATR + keltner

## Hard gates discipline
- IS-only param grids; holdout never for tuning
- signal_lag=1 intact; look-ahead tests green (25 pytest)
- Zero-trade “wins” rejected (`min_trades=8`)

## Results (approximate_non_ftmo — NOT go-live)
| Pass | OOS prof+gates | Holdout ok | Zero-trade OOS | Max OOS ret | Mean OOS n |
|------|---------------:|-----------:|---------------:|------------:|-----------:|
| Pre  | 6/18 | 4/18 | many | ~0.20% | ~0–50 |
| Run1 | 23/36 | 15/36 | **0** | ~0.27% | ~26 |
| Run2 | 24/36 | 9/36 | **0** | ~0.55% | ~46 |

**FTMO exports:** still empty (`data/ftmo/`). `ftmo_golive_candidate` remains false.

## Best dual (OOS+holdout) so far
- GBPUSD D1 trend_ma_adx / GBPUSD H4 bbands / USDJPY H4+D1 bbands / hybrid D1
- Absolute returns still small vs FTMO 10%/5% targets — edge thin on Yahoo interim data.

## Next
- Run3: strategy-aware ATR + keltner
- Basket of OOS-ranked legs (holdout confirmation only)
