# Preferred basket risk stress (confirmation)

- config: `best_interim_approximate.yaml`
- data_source: approximate_non_ftmo
- fx_only: True
- legs: 3
- atr_trail_mult: 0.0
- Note: params from IS grids; holdout never used to choose risk.

| Risk/trade | Basket ret | Static loss | Daily loss | Gates | Sharpe |
|---:|---:|---:|---:|:---:|---:|
| 1.0% | 0.80% | 0.74% | 0.25% | PASS | 0.50 |
| 1.5% | 1.21% | 1.10% | 0.37% | PASS | 0.51 |
| 2.0% | 1.50% | 1.32% | 0.48% | PASS | 0.52 |

## Legs (params IS-selected)

- USDJPY D1 hybrid_regime: OOS=0.80% hold=1.83%
- AUDUSD D1 hybrid_regime: OOS=0.25% hold=1.55%
- GBPUSD H4 breakout_donchian: OOS=0.19% hold=3.26%

