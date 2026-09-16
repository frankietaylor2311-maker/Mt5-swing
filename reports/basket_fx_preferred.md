# Preferred FX3 basket @2% (confirmation)

Post-hoc dual winners (OOS+holdout gates) for confirmation. Params IS-only. Not go-live.

- return: 1.74%
- static loss: 0.69%
- daily loss: 0.53%
- gates: PASS
- sharpe: 2.01
- risk_fraction: 2.0%
- atr_trail_mult: 0.0 (primary)
- trail=1.5 sensitivity return: 1.97% sharpe=2.23

## Legs

- USDJPY D1 hybrid_regime: OOS=0.80% hold_leg=0.03% n=9 gates=True
- AUDUSD D1 hybrid_regime: OOS=0.25% hold_leg=0.75% n=9 gates=True
- USDJPY D1 bbands_reversion: OOS=0.24% hold_leg=3.46% n=4 gates=True
