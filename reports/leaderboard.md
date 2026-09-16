# Research leaderboard

Ranked by holdout_profitable_and_gates → meaningful OOS (≥10 trades + gates) → OOS return/Sharpe.
Params selected on IS only; holdout never used for tuning. Zero-trade rows are not wins.

| Rank | Symbol | TF | Strategy | OOS ret | OOS Sh | OOS n | Holdout | Source |
|---:|---|---|---|---:|---:|---:|:---:|---|
| 1 | USDJPY | D1 | hybrid_regime | 0.80% | 0.95 | 35 | YES | `approximate_non_ftmo` |
| 2 | XAUUSD | H4 | breakout_donchian | 0.63% | 0.51 | 60 | YES | `approximate_non_ftmo` |
| 3 | USDJPY | D1 | bbands_reversion | 0.24% | 0.66 | 18 | YES | `approximate_non_ftmo` |
| 4 | USDJPY | D1 | mean_reversion_regime | 0.23% | -0.20 | 12 | YES | `approximate_non_ftmo` |
| 5 | GBPUSD | H4 | breakout_donchian | 0.19% | 0.43 | 104 | YES | `approximate_non_ftmo` |
| 6 | XAUUSD | D1 | hybrid_regime | 0.17% | -0.17 | 214 | YES | `approximate_non_ftmo` |
| 7 | USDJPY | H4 | mean_reversion_regime | -0.06% | -0.24 | 28 | YES | `approximate_non_ftmo` |
| 8 | GBPUSD | D1 | bbands_reversion | -0.17% | -0.10 | 23 | YES | `approximate_non_ftmo` |
| 9 | GBPUSD | H4 | mean_reversion_regime | -0.20% | -0.00 | 22 | YES | `approximate_non_ftmo` |
| 10 | GBPUSD | D1 | mean_reversion_regime | -0.41% | 0.37 | 15 | YES | `approximate_non_ftmo` |
| 11 | XAUUSD | H4 | hybrid_regime | -0.65% | -0.57 | 54 | YES | `approximate_non_ftmo` |
| 12 | XAUUSD | D1 | mean_reversion_regime | -0.66% | -0.76 | 12 | YES | `approximate_non_ftmo` |
