# Preferred FX4 @2% (Sharpe-first, min OOS 0.1%)

Dual-confirm ≤1/symbol; OOS Sharpe→return; **min OOS return 0.1%** (excludes noise duals). Overlapping dates. Not go-live.

- return: 1.81%
- static loss: 0.82%
- daily loss: 0.36%
- gates: PASS
- sharpe: 0.92
- risk_fraction: 2.0%

## Risk stress

- 1.0%: ret=0.98% static=0.41% daily=0.19% gates=PASS sharpe=0.97
- 1.5%: ret=1.47% static=0.61% daily=0.28% gates=PASS sharpe=0.96
- 2.0%: ret=1.81% static=0.82% daily=0.36% gates=PASS sharpe=0.92

## Legs

- USDCHF H4 bbands_reversion: OOS=0.21% Sh=1.44 hold_leg=3.62% n=43
- USDJPY D1 hybrid_regime: OOS=0.80% Sh=0.95 hold_leg=-0.07% n=9
- GBPUSD H4 breakout_donchian: OOS=0.19% Sh=0.43 hold_leg=3.13% n=111
- AUDUSD D1 hybrid_regime: OOS=0.25% Sh=0.34 hold_leg=0.57% n=9
