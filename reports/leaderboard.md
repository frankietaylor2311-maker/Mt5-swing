# Research leaderboard

Ranked by holdout_profitable_and_gates → meaningful OOS (≥10 trades + gates) → OOS return/Sharpe.
Params selected on IS only; holdout never used for tuning. Zero-trade rows are not wins.

| Rank | Symbol | TF | Strategy | OOS ret | OOS Sh | OOS n | Holdout | Source |
|---:|---|---|---|---:|---:|---:|:---:|---|
| 1 | USDJPY | D1 | hybrid_regime | 0.79% | 0.97 | 35 | YES | `approximate_non_ftmo` |
| 2 | USDJPY | D1 | bbands_reversion | 0.28% | 0.67 | 18 | YES | `approximate_non_ftmo` |
| 3 | XAUUSD | D1 | hybrid_regime | 0.19% | -0.02 | 207 | YES | `approximate_non_ftmo` |
| 4 | XAUUSD | H4 | breakout_donchian | 0.16% | 0.41 | 52 | YES | `approximate_non_ftmo` |
| 5 | XAUUSD | H4 | hybrid_regime | -0.46% | -0.94 | 46 | YES | `approximate_non_ftmo` |
| 6 | XAUUSD | D1 | breakout_donchian | 0.19% | 0.07 | 119 | YES | `approximate_non_ftmo` |
| 7 | USDJPY | D1 | breakout_donchian | 0.41% | 0.61 | 17 | NO | `approximate_non_ftmo` |
| 8 | USDJPY | H4 | hybrid_regime | 0.28% | 0.56 | 89 | NO | `approximate_non_ftmo` |
| 9 | USDJPY | H4 | breakout_donchian | -0.12% | 0.14 | 87 | NO | `approximate_non_ftmo` |
| 10 | XAUUSD | H4 | bbands_reversion | -0.33% | 0.12 | 12 | NO | `approximate_non_ftmo` |
| 11 | USDJPY | H4 | bbands_reversion | 0.00% | 0.00 | 0 | NO | `approximate_non_ftmo` |
| 12 | XAUUSD | D1 | bbands_reversion | 0.00% | 0.00 | 0 | NO | `approximate_non_ftmo` |
