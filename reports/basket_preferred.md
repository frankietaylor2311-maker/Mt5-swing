# Preferred basket holdout (a priori: XAU breakout + USDJPY hybrid/BB)

OOS-ranked within a pre-declared strategy/symbol recipe. Holdout never for selection.
XAUUSD = Yahoo GC=F approximate_non_ftmo — not FTMO CFD.

- return: -0.07%
- static loss: 1.11%
- daily loss: 0.93%
- gates: PASS
- sharpe: -0.04

## Legs
- USDJPY D1 hybrid_regime: OOS=0.79% hold=-0.07% n=9 gates=True
- XAUUSD H4 breakout_donchian: OOS=0.56% hold=7.14% n=107 gates=True
- XAUUSD D1 breakout_donchian: OOS=0.49% hold=2.63% n=17 gates=True
- USDJPY H4 hybrid_regime: OOS=0.28% hold=-10.00% n=72 gates=False
