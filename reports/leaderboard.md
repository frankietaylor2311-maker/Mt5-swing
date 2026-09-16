# Research leaderboard

Ranked by holdout_profitable_and_gates → meaningful OOS (≥10 trades + gates) → OOS return/Sharpe.
Params selected on IS only; holdout never used for tuning. Zero-trade rows are not wins.

| Rank | Symbol | TF | Strategy | OOS ret | OOS Sh | OOS n | Holdout | Source |
|---:|---|---|---|---:|---:|---:|:---:|---|
| 1 | EURCAD | D1 | bbands_reversion | 0.21% | 0.92 | 23 | YES | `approximate_non_ftmo` |
| 2 | EURCAD | H4 | bbands_reversion | 0.12% | 0.69 | 25 | YES | `approximate_non_ftmo` |
| 3 | EURCAD | D1 | cci_reversion | 0.11% | 0.20 | 26 | YES | `approximate_non_ftmo` |
| 4 | EURCAD | H4 | mean_reversion_regime | 0.08% | 0.73 | 11 | YES | `approximate_non_ftmo` |
| 5 | EURCAD | D1 | macd_trend | -0.01% | -0.69 | 43 | YES | `approximate_non_ftmo` |
| 6 | EURCAD | D1 | squeeze_breakout | -0.85% | -2.03 | 75 | YES | `approximate_non_ftmo` |
| 7 | EURCAD | D1 | ema_pullback | 0.04% | 0.12 | 7 | YES | `approximate_non_ftmo` |
| 8 | EURCAD | H4 | trend_ma_adx | 0.15% | 0.48 | 85 | NO | `approximate_non_ftmo` |
| 9 | EURCAD | H4 | squeeze_breakout | 0.12% | 0.66 | 61 | NO | `approximate_non_ftmo` |
| 10 | EURCAD | H4 | cci_reversion | 0.09% | 0.29 | 20 | NO | `approximate_non_ftmo` |
| 11 | EURCAD | H4 | hybrid_regime | 0.08% | 0.03 | 87 | NO | `approximate_non_ftmo` |
| 12 | EURCAD | H4 | stoch_reversion | 0.08% | -0.38 | 40 | NO | `approximate_non_ftmo` |
