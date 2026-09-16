# Research leaderboard

Ranked by holdout_profitable_and_gates → meaningful OOS (≥10 trades + gates) → OOS return/Sharpe.
Params selected on IS only; holdout never used for tuning. Zero-trade rows are not wins.

| Rank | Symbol | TF | Strategy | OOS ret | OOS Sh | OOS n | Holdout | Source |
|---:|---|---|---|---:|---:|---:|:---:|---|
| 1 | GBPCAD | D1 | bbands_reversion | 0.84% | 2.50 | 40 | YES | `approximate_non_ftmo` |
| 2 | GBPCAD | D1 | cci_reversion | 0.68% | 1.87 | 30 | YES | `approximate_non_ftmo` |
| 3 | GBPCAD | D1 | stoch_reversion | 0.63% | 1.78 | 56 | YES | `approximate_non_ftmo` |
| 4 | AUDCAD | D1 | cci_reversion | 0.36% | 0.56 | 29 | YES | `approximate_non_ftmo` |
| 5 | AUDCAD | H4 | cci_reversion | 0.15% | 0.65 | 40 | YES | `approximate_non_ftmo` |
| 6 | AUDCAD | H4 | bbands_reversion | 0.13% | 1.10 | 42 | YES | `approximate_non_ftmo` |
| 7 | AUDCAD | H4 | stoch_reversion | 0.02% | 0.15 | 57 | YES | `approximate_non_ftmo` |
| 8 | AUDCAD | D1 | stoch_reversion | -0.06% | 0.08 | 52 | YES | `approximate_non_ftmo` |
| 9 | GBPCAD | H4 | bbands_reversion | -0.07% | -0.09 | 11 | YES | `approximate_non_ftmo` |
| 10 | GBPCAD | H4 | cci_reversion | -0.23% | -0.87 | 22 | YES | `approximate_non_ftmo` |
| 11 | GBPCAD | H4 | stoch_reversion | -0.28% | -1.07 | 43 | YES | `approximate_non_ftmo` |
| 12 | GBPCAD | D1 | breakout_donchian | -1.13% | -2.54 | 41 | YES | `approximate_non_ftmo` |
