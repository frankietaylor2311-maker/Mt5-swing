# Preferred basket risk stress (confirmation)

- config: `best_interim_approximate.yaml`
- weights: oos_sharpe
- IS exits on trend legs
- legs: 4

| Risk/trade | Basket ret | Static loss | Daily loss | Gates | Sharpe |
|---:|---:|---:|---:|:---:|---:|
| 1.0% | 1.44% | 0.17% | 0.24% | PASS | 1.31 |
| 1.5% | 2.16% | 0.25% | 0.37% | PASS | 1.31 |
| 2.0% | 2.59% | 0.35% | 0.47% | PASS | 1.23 |
| 2.5% | 2.83% | 0.44% | 0.48% | PASS | 1.21 |

## Legs

- USDCHF H4 bbands_reversion: OOS=0.21% w=0.455 hold=3.76% exits=default
- USDJPY D1 hybrid_regime: OOS=0.80% w=0.300 hold=0.74% exits={'atr_trail_mult': 1.5, 'atr_target_mult': 4.0, 'atr_stop_mult': 2.0, 'max_hold_bars': 0}
- GBPUSD H4 breakout_donchian: OOS=0.19% w=0.137 hold=5.36% exits={'atr_trail_mult': 0.0, 'atr_target_mult': 4.0, 'atr_stop_mult': 2.0, 'max_hold_bars': 0}
- AUDUSD D1 hybrid_regime: OOS=0.25% w=0.108 hold=1.54% exits={'atr_trail_mult': 0.0, 'atr_target_mult': 4.0, 'atr_stop_mult': 2.0, 'max_hold_bars': 0}
