# Preferred basket risk stress (confirmation)

- config: `best_interim_approximate.yaml`
- data_source: approximate_non_ftmo
- fx_only: True
- legs: 3
- atr_trail_mult: 1.5
- Note: params from IS grids; holdout never used to choose risk.

| Risk/trade | Basket ret | Static loss | Daily loss | Gates | Sharpe |
|---:|---:|---:|---:|:---:|---:|
| 1.0% | 1.04% | 0.42% | 0.13% | PASS | 2.16 |
| 1.5% | 1.58% | 0.63% | 0.20% | PASS | 2.17 |
| 2.0% | 1.97% | 0.70% | 0.23% | PASS | 2.23 |

## Legs (params IS-selected)

- USDJPY D1 hybrid_regime: OOS=0.80% hold=1.83%
- AUDUSD D1 hybrid_regime: OOS=0.25% hold=1.55%
- USDJPY D1 bbands_reversion: OOS=0.24% hold=5.24%

