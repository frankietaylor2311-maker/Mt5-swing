# Research leaderboard

Ranked by holdout_profitable_and_gates → meaningful OOS (≥10 trades + gates) → OOS return/Sharpe.
Params selected on IS only; holdout never used for tuning. Zero-trade rows are not wins.

| Rank | Symbol | TF | Strategy | OOS ret | OOS Sh | OOS n | Holdout | Source |
|---:|---|---|---|---:|---:|---:|:---:|---|
| 1 | USDJPY | D1 | hybrid_regime | 0.79% | 0.97 | 35 | YES | `approximate_non_ftmo` |
| 2 | XAUUSD | H4 | breakout_donchian | 0.56% | 0.66 | 56 | YES | `approximate_non_ftmo` |
| 3 | XAUUSD | D1 | breakout_donchian | 0.49% | 0.32 | 436 | YES | `approximate_non_ftmo` |
| 4 | USDJPY | D1 | bbands_reversion | 0.28% | 0.67 | 18 | YES | `approximate_non_ftmo` |
| 5 | XAUUSD | D1 | hybrid_regime | 0.19% | -0.02 | 207 | YES | `approximate_non_ftmo` |
| 6 | EURUSD | H4 | breakout_donchian | 0.04% | 0.12 | 94 | YES | `approximate_non_ftmo` |
| 7 | USDJPY | D1 | mean_reversion_regime | -0.04% | -0.63 | 18 | YES | `approximate_non_ftmo` |
| 8 | USDJPY | H4 | mean_reversion_regime | -0.06% | -0.24 | 28 | YES | `approximate_non_ftmo` |
| 9 | GBPUSD | D1 | bbands_reversion | -0.17% | -0.11 | 23 | YES | `approximate_non_ftmo` |
| 10 | GBPUSD | H4 | mean_reversion_regime | -0.20% | -0.00 | 22 | YES | `approximate_non_ftmo` |
| 11 | GBPUSD | D1 | mean_reversion_regime | -0.43% | 0.35 | 15 | YES | `approximate_non_ftmo` |
| 12 | XAUUSD | H4 | hybrid_regime | -0.46% | -0.94 | 46 | YES | `approximate_non_ftmo` |
