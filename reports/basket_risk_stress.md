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
| 1.0% | 2.68% | 0.08% | 0.24% | PASS | 2.02 |
| 1.5% | 3.96% | 0.09% | 0.34% | PASS | 2.05 |
| 2.0% | 4.42% | 0.03% | 0.38% | PASS | 2.00 |
| 2.5% | 4.39% | 0.03% | 0.40% | PASS | 1.92 |

## Legs (params IS-selected)

- USDCHF H4 bbands_reversion: OOS=0.21% hold=3.76% w=0.273
- GBPUSD H4 breakout_donchian: OOS=0.19% hold=7.13% w=0.082
- CADJPY H4 mean_reversion_regime: OOS=0.33% hold=7.91% w=0.332
- AUDCAD H4 mean_reversion_regime: OOS=0.16% hold=0.46% w=0.312

