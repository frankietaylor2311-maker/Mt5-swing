# Preferred basket (run12 @ 0.025)

A priori multi-asset recipe (XAU breakout + USDJPY D1 hybrid/BB + AUDUSD D1 hybrid if OOS-qualified).
Holdout confirmation only. All approximate_non_ftmo.

- return: 1.69%
- static loss: 0.72%
- daily loss: 0.53%
- gates: PASS
- sharpe: 0.97

## Legs
- USDJPY D1 hybrid_regime: OOS=0.80% hold=-0.03% n=9 gates=True
- XAUUSD H4 breakout_donchian: OOS=0.63% hold=5.09% n=118 gates=True
- AUDUSD D1 hybrid_regime: OOS=0.25% hold=0.72% n=9 gates=True
- USDJPY D1 bbands_reversion: OOS=0.24% hold=3.40% n=4 gates=True
