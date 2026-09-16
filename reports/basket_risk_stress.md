# Preferred basket risk stress (confirmation)

- config: `best_interim_approximate.yaml`
- data_source: approximate_non_ftmo
- fx_only: True
- legs: 4
- atr_trail_mult: 0.0
- Note: params from IS grids; holdout never used to choose risk.

| Risk/trade | Basket ret | Static loss | Daily loss | Gates | Sharpe |
|---:|---:|---:|---:|:---:|---:|
| 1.0% | 0.98% | 0.41% | 0.19% | PASS | 0.97 |
| 1.5% | 1.47% | 0.61% | 0.28% | PASS | 0.96 |
| 2.0% | 1.81% | 0.82% | 0.36% | PASS | 0.92 |

## Legs (params IS-selected)

- USDJPY D1 hybrid_regime: OOS=0.80% hold=1.83%
- AUDUSD D1 hybrid_regime: OOS=0.25% hold=1.55%
- GBPUSD H4 breakout_donchian: OOS=0.19% hold=3.26%
- USDCHF H4 bbands_reversion: OOS=0.21% hold=3.60%

