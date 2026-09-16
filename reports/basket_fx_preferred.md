# Preferred basket (fx4_chf_h4) @ 2%

Dual-confirm ≤1/symbol; params IS-only; overlapping holdout dates. Not go-live.

- return: 1.81%
- static loss: 0.82%
- daily loss: 0.36%
- gates: PASS
- sharpe: 0.92
- risk_fraction: 2.0%
- atr_trail_mult: 0.0

## Legs

- USDJPY D1 hybrid_regime: OOS=0.80% hold_leg=-0.07% n=9 gates=True
- AUDUSD D1 hybrid_regime: OOS=0.25% hold_leg=0.57% n=9 gates=True
- GBPUSD H4 breakout_donchian: OOS=0.19% hold_leg=3.13% n=111 gates=True
- USDCHF H4 bbands_reversion: OOS=0.21% hold_leg=3.62% n=43 gates=True
