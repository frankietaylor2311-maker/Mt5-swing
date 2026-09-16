# Research leaderboard

Ranked by holdout_profitable_and_gates → meaningful OOS (≥10 trades + gates) → OOS return/Sharpe.
Params selected on IS only; holdout never used for tuning. Zero-trade rows are not wins.

| Rank | Symbol | TF | Strategy | OOS ret | OOS Sh | OOS n | Holdout | Source |
|---:|---|---|---|---:|---:|---:|:---:|---|
| 1 | NZDCAD | D1 | bbands_reversion | 0.18% | 0.32 | 23 | YES | `approximate_non_ftmo` |
| 2 | EURAUD | H4 | breakout_donchian | 0.15% | 0.49 | 65 | YES | `approximate_non_ftmo` |
| 3 | EURAUD | D1 | hybrid_regime | -0.04% | -0.23 | 47 | YES | `approximate_non_ftmo` |
| 4 | NZDCAD | D1 | cci_reversion | -0.41% | -0.48 | 27 | YES | `approximate_non_ftmo` |
| 5 | NZDCAD | D1 | stoch_reversion | -1.04% | -2.00 | 41 | YES | `approximate_non_ftmo` |
| 6 | NZDCAD | D1 | breakout_donchian | 0.40% | 0.95 | 32 | NO | `approximate_non_ftmo` |
| 7 | NZDCAD | H4 | bbands_reversion | 0.15% | 0.78 | 31 | NO | `approximate_non_ftmo` |
| 8 | NZDCAD | H4 | cci_reversion | 0.08% | 0.58 | 35 | NO | `approximate_non_ftmo` |
| 9 | NZDCAD | H4 | stoch_reversion | -0.03% | -0.01 | 49 | NO | `approximate_non_ftmo` |
| 10 | EURAUD | H4 | hybrid_regime | -0.07% | -0.93 | 78 | NO | `approximate_non_ftmo` |
| 11 | EURAUD | D1 | breakout_donchian | -0.16% | -0.47 | 19 | NO | `approximate_non_ftmo` |
| 12 | NZDCAD | D1 | hybrid_regime | -0.24% | -0.15 | 43 | NO | `approximate_non_ftmo` |
