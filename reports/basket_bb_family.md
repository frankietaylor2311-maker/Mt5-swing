# BB-family basket holdout (run5, risk_fraction=0.015)

Members: OOS-ranked BB/MR family only (a priori). Holdout never for selection.

- return: 1.70%
- static loss: 0.16%
- daily loss: 0.20%
- gates: PASS
- sharpe: 1.13

## Legs
- EURUSD D1 bbands_reversion: OOS=0.33% hold=-0.21% n=5 gates=True
- USDJPY D1 bbands_reversion: OOS=0.30% hold=2.15% n=4 gates=True
- EURUSD H4 bbands_reversion: OOS=0.09% hold=-1.26% n=44 gates=True
- USDJPY D1 mean_reversion_regime: OOS=0.04% hold=3.17% n=6 gates=True
