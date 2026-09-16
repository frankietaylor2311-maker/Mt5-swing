# Research leaderboard

Ranked by holdout_profitable_and_gates → meaningful OOS (≥10 trades + gates) → OOS return/Sharpe.
Params selected on IS only; holdout never used for tuning. Zero-trade rows are not wins.

| Rank | Symbol | TF | Strategy | OOS ret | OOS Sh | OOS n | Holdout | Source |
|---:|---|---|---|---:|---:|---:|:---:|---|
| 1 | USDCHF | D1 | bbands_reversion | 0.23% | 1.36 | 30 | YES | `approximate_non_ftmo` |
| 2 | USDCHF | H4 | bbands_reversion | 0.20% | 1.43 | 27 | YES | `approximate_non_ftmo` |
