# Research leaderboard

Ranked by holdout_profitable_and_gates → meaningful OOS (≥10 trades + gates) → OOS return/Sharpe.
Params selected on IS only; holdout never used for tuning. Zero-trade rows are not wins.

| Rank | Symbol | TF | Strategy | OOS ret | OOS Sh | OOS n | Holdout | Source |
|---:|---|---|---|---:|---:|---:|:---:|---|
| 1 | EURJPY | H4 | atr_channel_breakout | 0.00% | 0.57 | 93 | YES | `approximate_non_ftmo` |
| 2 | NZDUSD | H4 | atr_channel_breakout | -0.01% | -0.19 | 128 | YES | `approximate_non_ftmo` |
| 3 | EURJPY | H4 | cci_reversion | 0.00% | 0.00 | 0 | YES | `approximate_non_ftmo` |
| 4 | EURJPY | H4 | stoch_reversion | 0.00% | 0.00 | 0 | YES | `approximate_non_ftmo` |
| 5 | GBPJPY | H4 | cci_reversion | 0.00% | 0.00 | 0 | YES | `approximate_non_ftmo` |
| 6 | USDCAD | D1 | stoch_reversion | 0.05% | 0.44 | 133 | NO | `approximate_non_ftmo` |
| 7 | GBPJPY | H4 | atr_channel_breakout | 0.00% | -0.29 | 106 | NO | `approximate_non_ftmo` |
| 8 | GBPJPY | D1 | atr_channel_breakout | -0.00% | -0.54 | 63 | NO | `approximate_non_ftmo` |
| 9 | EURJPY | D1 | atr_channel_breakout | -0.01% | -1.76 | 25 | NO | `approximate_non_ftmo` |
| 10 | NZDUSD | H4 | cci_reversion | -0.05% | -0.02 | 38 | NO | `approximate_non_ftmo` |
| 11 | USDCAD | H4 | cci_reversion | -0.20% | -0.58 | 33 | NO | `approximate_non_ftmo` |
| 12 | USDCAD | H4 | stoch_reversion | -0.24% | -1.27 | 59 | NO | `approximate_non_ftmo` |
