# Research leaderboard

Ranked by holdout_profitable_and_gates → meaningful OOS (≥10 trades + gates) → OOS return/Sharpe.
Params selected on IS only; holdout never used for tuning. Zero-trade rows are not wins.

| Rank | Symbol | TF | Strategy | OOS ret | OOS Sh | OOS n | Holdout | Source |
|---:|---|---|---|---:|---:|---:|:---:|---|
| 1 | EURGBP | D1 | cci_reversion | 0.26% | 1.10 | 14 | YES | `approximate_non_ftmo` |
| 2 | USDJPY | D1 | cci_reversion | 0.19% | 0.88 | 23 | YES | `approximate_non_ftmo` |
| 3 | USDCHF | H4 | cci_reversion | 0.00% | 0.00 | 0 | YES | `approximate_non_ftmo` |
| 4 | EURUSD | D1 | cci_reversion | 0.58% | 1.54 | 24 | NO | `approximate_non_ftmo` |
| 5 | EURUSD | H4 | cci_reversion | 0.09% | 0.37 | 28 | NO | `approximate_non_ftmo` |
| 6 | GBPUSD | H4 | cci_reversion | 0.09% | 0.29 | 33 | NO | `approximate_non_ftmo` |
| 7 | AUDUSD | H4 | cci_reversion | -0.07% | 0.05 | 40 | NO | `approximate_non_ftmo` |
| 8 | EURGBP | H4 | cci_reversion | -0.16% | -1.12 | 24 | NO | `approximate_non_ftmo` |
| 9 | USDJPY | H4 | cci_reversion | -0.22% | -0.60 | 24 | NO | `approximate_non_ftmo` |
| 10 | AUDUSD | D1 | cci_reversion | -0.41% | -0.01 | 56 | NO | `approximate_non_ftmo` |
| 11 | GBPUSD | D1 | cci_reversion | 0.00% | 0.00 | 0 | NO | `approximate_non_ftmo` |
| 12 | USDCHF | D1 | cci_reversion | 0.00% | 0.00 | 0 | NO | `approximate_non_ftmo` |
