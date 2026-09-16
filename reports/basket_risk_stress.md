# Preferred basket risk stress (confirmation)

- config: `best_interim_approximate.yaml`
- data_source: approximate_non_ftmo
- fx_only: True
- legs: 4
- atr_trail_mult: 0.0
- weights: oos_sharpe
- Note: params from IS grids; holdout never used to choose risk/weights.

| Risk/trade | Basket ret | Static loss | Daily loss | Gates | Sharpe |
|---:|---:|---:|---:|:---:|---:|
| 1.0% | 1.18% | 0.29% | 0.26% | PASS | 1.09 |
| 1.5% | 1.78% | 0.44% | 0.39% | PASS | 1.09 |
| 2.0% | 2.11% | 0.57% | 0.50% | PASS | 1.02 |
| 2.5% | 2.28% | 0.62% | 0.51% | PASS | 0.98 |

## Legs (params IS-selected)

- USDCHF H4 bbands_reversion: OOS=0.21% hold=3.76% w=0.455
- USDJPY D1 hybrid_regime: OOS=0.80% hold=-0.03% w=0.300
- GBPUSD H4 breakout_donchian: OOS=0.19% hold=3.66% w=0.137
- AUDUSD D1 hybrid_regime: OOS=0.25% hold=0.72% w=0.108
