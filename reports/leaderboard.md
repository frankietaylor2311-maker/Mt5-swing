# Research leaderboard

Ranked by holdout_profitable_and_gates → meaningful OOS (≥10 trades + gates) → OOS return/Sharpe.
Params selected on IS only; holdout never used for tuning. Zero-trade rows are not wins.

| Rank | Symbol | TF | Strategy | OOS ret | OOS Sh | OOS n | Holdout | Source |
|---:|---|---|---|---:|---:|---:|:---:|---|
| 1 | USDJPY | D1 | hybrid_regime | 0.80% | 0.95 | 35 | YES | `approximate_non_ftmo` |
| 2 | XAUUSD | H4 | breakout_donchian | 0.63% | 0.51 | 60 | YES | `approximate_non_ftmo` |
| 3 | AUDUSD | D1 | hybrid_regime | 0.25% | 0.34 | 96 | YES | `approximate_non_ftmo` |
| 4 | USDJPY | D1 | bbands_reversion | 0.24% | 0.66 | 18 | YES | `approximate_non_ftmo` |
| 5 | GBPUSD | H4 | breakout_donchian | 0.19% | 0.43 | 104 | YES | `approximate_non_ftmo` |
| 6 | XAUUSD | D1 | hybrid_regime | 0.17% | -0.17 | 214 | YES | `approximate_non_ftmo` |
| 7 | USDJPY | H4 | squeeze_breakout | 0.12% | 0.50 | 44 | YES | `approximate_non_ftmo` |
| 8 | AUDUSD | D1 | bbands_reversion | -0.10% | 0.50 | 37 | YES | `approximate_non_ftmo` |
| 9 | GBPUSD | D1 | bbands_reversion | -0.17% | -0.10 | 23 | YES | `approximate_non_ftmo` |
| 10 | AUDUSD | D1 | squeeze_breakout | -0.42% | -0.60 | 181 | YES | `approximate_non_ftmo` |
| 11 | XAUUSD | H4 | hybrid_regime | -0.65% | -0.57 | 54 | YES | `approximate_non_ftmo` |
| 12 | XAUUSD | D1 | breakout_donchian | 0.40% | 0.21 | 428 | YES | `approximate_non_ftmo` |
