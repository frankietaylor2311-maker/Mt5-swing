# Preferred basket (run11 @ risk_fraction=0.025)

A priori: profitable XAU breakout + USDJPY D1 hybrid/BB. OOS-ranked. Holdout confirmation only.
XAUUSD Yahoo = approximate_non_ftmo.

- return: 1.99%
- static loss: 0.77%
- daily loss: 0.53%
- gates: PASS
- sharpe: 1.06

## Legs
- USDJPY D1 hybrid_regime: OOS=0.80% hold=0.24% n=9 gates=True
- XAUUSD H4 breakout_donchian: OOS=0.63% hold=6.40% n=118 gates=True
- USDJPY D1 bbands_reversion: OOS=0.24% hold=3.73% n=4 gates=True
