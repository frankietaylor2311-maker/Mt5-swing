# Research leaderboard

Ranked by holdout_profitable_and_gates → meaningful OOS (≥10 trades + gates) → OOS return/Sharpe.
Params selected on IS only; holdout never used for tuning. Zero-trade rows are not wins.

| Rank | Symbol | TF | Strategy | OOS ret | OOS Sh | OOS n | Holdout | Source |
|---:|---|---|---|---:|---:|---:|:---:|---|
| 1 | USDJPY | D1 | hybrid_regime | 0.50% | 1.03 | 35 | YES | `approximate_non_ftmo` |
| 2 | USDJPY | D1 | bbands_reversion | 0.26% | 0.74 | 18 | YES | `approximate_non_ftmo` |
| 3 | USDJPY | H4 | breakout_donchian | 0.15% | 0.39 | 87 | YES | `approximate_non_ftmo` |
| 4 | USDJPY | D1 | mean_reversion_regime | 0.07% | -0.62 | 18 | YES | `approximate_non_ftmo` |
| 5 | USDJPY | H4 | mean_reversion_regime | -0.07% | -0.25 | 28 | YES | `approximate_non_ftmo` |
| 6 | GBPUSD | D1 | bbands_reversion | -0.10% | -0.14 | 23 | YES | `approximate_non_ftmo` |
| 7 | GBPUSD | H4 | mean_reversion_regime | -0.17% | -0.00 | 22 | YES | `approximate_non_ftmo` |
| 8 | GBPUSD | D1 | mean_reversion_regime | -0.25% | 0.35 | 15 | YES | `approximate_non_ftmo` |
| 9 | GBPUSD | D1 | breakout_donchian | 0.58% | 1.03 | 47 | NO | `approximate_non_ftmo` |
| 10 | GBPUSD | H4 | hybrid_regime | 0.31% | 0.29 | 91 | NO | `approximate_non_ftmo` |
| 11 | EURUSD | D1 | bbands_reversion | 0.26% | 0.73 | 17 | NO | `approximate_non_ftmo` |
| 12 | USDJPY | D1 | breakout_donchian | 0.23% | 0.41 | 30 | NO | `approximate_non_ftmo` |
