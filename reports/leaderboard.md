# Research leaderboard

Ranked by holdout_profitable_and_gates → meaningful OOS (≥10 trades + gates) → OOS return/Sharpe.
Params selected on IS only; holdout never used for tuning.

| Rank | Symbol | TF | Strategy | OOS ret | OOS Sh | OOS n | Holdout | Source |
|---:|---|---|---|---:|---:|---:|:---:|---|
| 1 | USDJPY | D1 | hybrid_regime | 0.80% | 0.95 | 35 | YES | `approximate_non_ftmo` |
| 2 | XAUUSD | H4 | breakout_donchian | 0.63% | 0.51 | 60 | YES | `approximate_non_ftmo` |
| 3 | AUDUSD | D1 | atr_channel_breakout | 0.36% | 0.13 | 120 | YES | `approximate_non_ftmo` |
| 4 | EURGBP | D1 | mean_reversion_regime | 0.33% | 0.64 | 10 | YES | `approximate_non_ftmo` |
| 5 | AUDUSD | D1 | hybrid_regime | 0.25% | 0.34 | 96 | YES | `approximate_non_ftmo` |
| 6 | USDJPY | D1 | bbands_reversion | 0.24% | 0.66 | 18 | YES | `approximate_non_ftmo` |
| 7 | USDJPY | D1 | mean_reversion_regime | 0.23% | -0.20 | 12 | YES | `approximate_non_ftmo` |
| 8 | USDCHF | D1 | bbands_reversion | 0.22% | 1.17 | 21 | YES | `approximate_non_ftmo` |
| 9 | USDCHF | H4 | bbands_reversion | 0.21% | 1.44 | 26 | YES | `approximate_non_ftmo` |
| 10 | GBPUSD | H4 | breakout_donchian | 0.19% | 0.43 | 104 | YES | `approximate_non_ftmo` |
| 11 | USDJPY | H4 | squeeze_breakout | 0.12% | 0.50 | 44 | YES | `approximate_non_ftmo` |
| 12 | EURGBP | D1 | stoch_reversion | 0.10% | 0.50 | 51 | YES | `approximate_non_ftmo` |
| 13 | USDCAD | H4 | hybrid_regime | 0.08% | 0.27 | 73 | YES | `approximate_non_ftmo` |
| 14 | USDCHF | H4 | breakout_donchian | 0.06% | -0.26 | 83 | YES | `approximate_non_ftmo` |
