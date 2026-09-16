# Preferred basket risk stress (confirmation)

- config: `best_interim_approximate.yaml`
- data_source: approximate_non_ftmo
- fx_only: False
- legs: 4
- atr_trail_mult: 0.0
- Note: params from IS grids; holdout never used to choose risk.

| Risk/trade | Basket ret | Static loss | Daily loss | Gates | Sharpe |
|---:|---:|---:|---:|:---:|---:|
| 1.0% | 0.70% | 0.33% | 0.22% | PASS | 0.96 |
| 1.5% | 1.06% | 0.47% | 0.33% | PASS | 0.96 |
| 2.0% | 1.39% | 0.64% | 0.44% | PASS | 0.95 |

## Legs (params IS-selected)

- USDJPY D1 hybrid_regime: OOS=0.80% hold=1.83%
- XAUUSD H4 breakout_donchian: OOS=0.63% hold=23.25%
- AUDUSD D1 hybrid_regime: OOS=0.25% hold=1.55%
- USDJPY D1 bbands_reversion: OOS=0.24% hold=5.24%

