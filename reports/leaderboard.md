# Research leaderboard

Ranked by holdout_profitable_and_gates → meaningful OOS (≥10 trades + gates) → OOS return/Sharpe.
Params selected on IS only; holdout never used for tuning. Zero-trade rows are not wins.

| Rank | Symbol | TF | Strategy | OOS ret | OOS Sh | OOS n | Holdout | Source |
|---:|---|---|---|---:|---:|---:|:---:|---|
| 1 | USDJPY | D1 | hybrid_regime | 0.69% | 1.00 | 35 | YES | `approximate_non_ftmo` |
| 2 | USDJPY | D1 | bbands_reversion | 0.30% | 0.69 | 18 | YES | `approximate_non_ftmo` |
| 3 | USDJPY | D1 | mean_reversion_regime | 0.04% | -0.64 | 18 | YES | `approximate_non_ftmo` |
| 4 | USDJPY | H4 | mean_reversion_regime | -0.05% | -0.22 | 28 | YES | `approximate_non_ftmo` |
| 5 | GBPUSD | D1 | bbands_reversion | -0.15% | -0.12 | 23 | YES | `approximate_non_ftmo` |
| 6 | GBPUSD | H4 | mean_reversion_regime | -0.20% | -0.00 | 22 | YES | `approximate_non_ftmo` |
| 7 | USDJPY | H4 | breakout_donchian | -0.23% | -0.09 | 103 | YES | `approximate_non_ftmo` |
| 8 | GBPUSD | D1 | mean_reversion_regime | -0.38% | 0.35 | 15 | YES | `approximate_non_ftmo` |
| 9 | GBPUSD | D1 | breakout_donchian | 0.88% | 1.04 | 47 | NO | `approximate_non_ftmo` |
| 10 | USDJPY | H4 | trend_ma_adx | 0.41% | 0.97 | 97 | NO | `approximate_non_ftmo` |
| 11 | EURUSD | D1 | bbands_reversion | 0.33% | 0.76 | 17 | NO | `approximate_non_ftmo` |
| 12 | USDJPY | D1 | breakout_donchian | 0.30% | 0.41 | 30 | NO | `approximate_non_ftmo` |
