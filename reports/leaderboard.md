# Research leaderboard

Ranked by holdout_profitable_and_gates → meaningful OOS (≥10 trades + gates) → OOS return/Sharpe.
Params selected on IS only; holdout never used for tuning. Zero-trade rows are not wins.

| Rank | Symbol | TF | Strategy | OOS ret | OOS Sh | OOS n | Holdout | Source |
|---:|---|---|---|---:|---:|---:|:---:|---|
| 1 | GBPCAD | D1 | bbands_reversion | 0.84% | 2.50 | 40 | YES | `approximate_non_ftmo` |
| 2 | USDJPY | D1 | hybrid_regime | 0.80% | 0.95 | 35 | YES | `approximate_non_ftmo` |
| 3 | GBPCAD | D1 | cci_reversion | 0.68% | 1.87 | 30 | YES | `approximate_non_ftmo` |
| 4 | GBPCAD | D1 | stoch_reversion | 0.63% | 1.78 | 56 | YES | `approximate_non_ftmo` |
| 5 | GBPCAD | D1 | willr_reversion | 0.50% | 1.88 | 44 | YES | `approximate_non_ftmo` |
| 6 | CADJPY | H4 | willr_reversion | 0.48% | 1.40 | 61 | YES | `approximate_non_ftmo` |
| 7 | EURJPY | D1 | ema_pullback | 0.41% | 2.00 | 19 | YES | `approximate_non_ftmo` |
| 8 | AUDJPY | D1 | squeeze_breakout | 0.39% | 0.20 | 80 | YES | `approximate_non_ftmo` |
| 9 | AUDUSD | D1 | atr_channel_breakout | 0.39% | 0.08 | 120 | YES | `approximate_non_ftmo` |
| 10 | CADJPY | D1 | breakout_donchian | 0.39% | 0.22 | 20 | YES | `approximate_non_ftmo` |
| 11 | AUDCAD | D1 | cci_reversion | 0.36% | 0.56 | 29 | YES | `approximate_non_ftmo` |
| 12 | EURGBP | D1 | mean_reversion_regime | 0.33% | 0.64 | 10 | YES | `approximate_non_ftmo` |
