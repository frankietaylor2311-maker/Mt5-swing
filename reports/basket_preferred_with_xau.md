# Preferred basket +XAU @2% (exploratory)

Post-hoc dual winners (OOS+holdout gates) for confirmation. Params IS-only. Not go-live.

- return: 1.39%
- static loss: 0.64%
- daily loss: 0.44%
- gates: PASS
- sharpe: 0.95
- risk_fraction: 2.0%
- atr_trail_mult: 0.0 (primary)
- Yahoo gold ≠ FTMO CFD — exploratory only

## Legs

- USDJPY D1 hybrid_regime: OOS=0.80% hold_leg=-0.07% n=9 gates=True
- XAUUSD H4 breakout_donchian: OOS=0.63% hold_leg=3.43% n=118 gates=True
- AUDUSD D1 hybrid_regime: OOS=0.25% hold_leg=0.57% n=9 gates=True
- USDJPY D1 bbands_reversion: OOS=0.24% hold_leg=2.85% n=4 gates=True
