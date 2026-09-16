# Research leaderboard

Ranked by holdout_profitable_and_gates → meaningful OOS (≥10 trades + gates) → OOS return/Sharpe.
Params selected on IS only; holdout never used for tuning. Zero-trade rows are not wins.

| Rank | Symbol | TF | Strategy | OOS ret | OOS Sh | OOS n | Holdout | Source |
|---:|---|---|---|---:|---:|---:|:---:|---|
| 1 | USDJPY | D1 | bbands_reversion | 0.16% | 0.86 | 31 | YES | `approximate_non_ftmo` |
| 2 | GBPUSD | H4 | breakout_donchian | 0.14% | 0.28 | 106 | YES | `approximate_non_ftmo` |
| 3 | USDJPY | H4 | bbands_reversion | 0.14% | 0.68 | 38 | YES | `approximate_non_ftmo` |
| 4 | GBPUSD | H4 | bbands_reversion | 0.06% | 0.70 | 75 | YES | `approximate_non_ftmo` |
| 5 | EURUSD | H4 | bbands_reversion | 0.05% | 0.67 | 24 | YES | `approximate_non_ftmo` |
| 6 | GBPUSD | D1 | bbands_reversion | -0.10% | -0.16 | 51 | YES | `approximate_non_ftmo` |
| 7 | USDJPY | H4 | mean_reversion_regime | -0.17% | -0.44 | 31 | YES | `approximate_non_ftmo` |
| 8 | GBPUSD | D1 | mean_reversion_regime | -0.37% | -0.63 | 47 | YES | `approximate_non_ftmo` |
| 9 | USDJPY | D1 | mean_reversion_regime | 0.11% | 0.35 | 7 | YES | `approximate_non_ftmo` |
| 10 | GBPUSD | D1 | breakout_donchian | 0.55% | 1.11 | 47 | NO | `approximate_non_ftmo` |
| 11 | USDJPY | D1 | hybrid_regime | 0.49% | 1.08 | 35 | NO | `approximate_non_ftmo` |
| 12 | GBPUSD | H4 | hybrid_regime | 0.43% | 0.99 | 71 | NO | `approximate_non_ftmo` |
