# Preferred FX3 (≤1/symbol) @2%

Dual-confirm ≤1/symbol; params IS-only; overlapping holdout dates only. Not go-live.

- return: 1.50%
- static loss: 1.32%
- daily loss: 0.48%
- gates: PASS
- sharpe: 0.52
- risk_fraction: 2.0%
- atr_trail_mult: 0.0
- methodology: equal-weight on intersection of leg dates (dropna any)
- trail=1.5 sensitivity: -3.97%
- alt USDJPY-heavy D1-only: 1.41% (thin bbands n=4 holdout)
- prior-wave-style +XAU (corrected calendar): 1.76%

## Legs

- USDJPY D1 hybrid_regime: OOS=0.80% hold_leg=0.03% n=9 gates=True
- AUDUSD D1 hybrid_regime: OOS=0.25% hold_leg=0.75% n=9 gates=True
- GBPUSD H4 breakout_donchian: OOS=0.19% hold_leg=3.72% n=111 gates=True
