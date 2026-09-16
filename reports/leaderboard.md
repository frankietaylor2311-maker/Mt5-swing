# Research leaderboard

Ranked by holdout_profitable_and_gates → meaningful OOS (≥10 trades + gates) → OOS return/Sharpe.
Params selected on IS only; holdout never used for tuning. Zero-trade rows are not wins.

| Rank | Symbol | TF | Strategy | OOS ret | OOS Sh | OOS n | Holdout | Source |
|---:|---|---|---|---:|---:|---:|:---:|---|
| 1 | USDJPY | D1 | hybrid_regime | 0.79% | 0.97 | 35 | YES | `approximate_non_ftmo` |
| 2 | USDJPY | D1 | bbands_reversion | 0.28% | 0.67 | 18 | YES | `approximate_non_ftmo` |
| 3 | EURUSD | H4 | breakout_donchian | 0.04% | 0.12 | 94 | YES | `approximate_non_ftmo` |
| 4 | USDJPY | D1 | mean_reversion_regime | -0.04% | -0.63 | 18 | YES | `approximate_non_ftmo` |
| 5 | USDJPY | H4 | mean_reversion_regime | -0.06% | -0.24 | 28 | YES | `approximate_non_ftmo` |
| 6 | GBPUSD | D1 | bbands_reversion | -0.17% | -0.11 | 23 | YES | `approximate_non_ftmo` |
| 7 | GBPUSD | H4 | mean_reversion_regime | -0.20% | -0.00 | 22 | YES | `approximate_non_ftmo` |
| 8 | GBPUSD | D1 | mean_reversion_regime | -0.43% | 0.35 | 15 | YES | `approximate_non_ftmo` |
| 9 | GBPUSD | D1 | breakout_donchian | 1.17% | 1.08 | 47 | NO | `approximate_non_ftmo` |
| 10 | USDJPY | D1 | breakout_donchian | 0.42% | 0.44 | 30 | NO | `approximate_non_ftmo` |
| 11 | USDJPY | H4 | trend_ma_adx | 0.40% | 0.93 | 97 | NO | `approximate_non_ftmo` |
| 12 | EURUSD | D1 | mean_reversion_regime | 0.34% | 0.71 | 11 | NO | `approximate_non_ftmo` |
