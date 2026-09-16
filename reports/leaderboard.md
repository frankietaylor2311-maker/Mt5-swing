# Research leaderboard

Ranked by holdout_profitable_and_gates → meaningful OOS (≥10 trades + gates) → OOS return/Sharpe.
Params selected on IS only; holdout never used for tuning. Zero-trade rows are not wins.

| Rank | Symbol | TF | Strategy | OOS ret | OOS Sh | OOS n | Holdout | Source |
|---:|---|---|---|---:|---:|---:|:---:|---|
| 1 | GBPCAD | D1 | bbands_reversion | 0.84% | 2.50 | 40 | YES | `approximate_non_ftmo` |
| 2 | USDJPY | D1 | hybrid_regime | 0.80% | 0.95 | 35 | YES | `approximate_non_ftmo` |
| 3 | GBPCAD | D1 | cci_reversion | 0.68% | 1.87 | 30 | YES | `approximate_non_ftmo` |
| 4 | GBPCAD | D1 | stoch_reversion | 0.63% | 1.78 | 56 | YES | `approximate_non_ftmo` |
| 5 | XAUUSD | H4 | breakout_donchian | 0.63% | 0.51 | 60 | YES | `approximate_non_ftmo` |
| 6 | AUDUSD | D1 | atr_channel_breakout | 0.39% | 0.08 | 120 | YES | `approximate_non_ftmo` |
| 7 | AUDCAD | D1 | cci_reversion | 0.36% | 0.56 | 29 | YES | `approximate_non_ftmo` |
| 8 | EURGBP | D1 | mean_reversion_regime | 0.33% | 0.64 | 10 | YES | `approximate_non_ftmo` |
| 9 | EURGBP | D1 | cci_reversion | 0.26% | 1.10 | 14 | YES | `approximate_non_ftmo` |
| 10 | AUDUSD | D1 | hybrid_regime | 0.25% | 0.34 | 96 | YES | `approximate_non_ftmo` |
| 11 | USDJPY | D1 | bbands_reversion | 0.24% | 0.66 | 18 | YES | `approximate_non_ftmo` |
| 12 | USDCHF | D1 | bbands_reversion | 0.23% | 1.36 | 30 | YES | `approximate_non_ftmo` |
