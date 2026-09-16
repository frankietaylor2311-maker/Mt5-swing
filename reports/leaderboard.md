# Research leaderboard

Ranked by holdout_profitable_and_gates → meaningful OOS (≥10 trades + gates) → OOS return/Sharpe.
Params selected on IS only; holdout never used for tuning. Zero-trade rows are not wins.

| Rank | Symbol | TF | Strategy | OOS ret | OOS Sh | OOS n | Holdout | Source |
|---:|---|---|---|---:|---:|---:|:---:|---|
| 1 | GBPUSD | D1 | trend_ma_adx | 0.18% | 1.53 | 33 | YES | `approximate_non_ftmo` |
| 2 | USDJPY | D1 | hybrid_regime | 0.15% | 0.32 | 20 | YES | `approximate_non_ftmo` |
| 3 | USDJPY | H4 | bbands_reversion | 0.14% | 0.38 | 20 | YES | `approximate_non_ftmo` |
| 4 | GBPUSD | H4 | bbands_reversion | 0.11% | 0.91 | 41 | YES | `approximate_non_ftmo` |
| 5 | GBPUSD | H4 | breakout_donchian | 0.08% | 0.27 | 19 | YES | `approximate_non_ftmo` |
| 6 | USDJPY | D1 | bbands_reversion | 0.08% | 0.82 | 18 | YES | `approximate_non_ftmo` |
| 7 | GBPUSD | D1 | bbands_reversion | 0.01% | 0.27 | 28 | YES | `approximate_non_ftmo` |
| 8 | GBPUSD | H4 | ema_pullback | 0.00% | -0.11 | 20 | YES | `approximate_non_ftmo` |
| 9 | USDJPY | D1 | mean_reversion_regime | -0.02% | -0.61 | 18 | YES | `approximate_non_ftmo` |
| 10 | EURUSD | D1 | ema_pullback | -0.07% | -0.82 | 13 | YES | `approximate_non_ftmo` |
| 11 | GBPUSD | H4 | mean_reversion_regime | -0.08% | 0.07 | 22 | YES | `approximate_non_ftmo` |
| 12 | USDJPY | H4 | breakout_donchian | -0.10% | 0.09 | 23 | YES | `approximate_non_ftmo` |
