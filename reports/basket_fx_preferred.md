# Preferred FX4 @2.5% equal + IS exits + IS vol_target

Equal weights selected vs oos_sharpe on OOS basket only; IS exits unchanged; holdout confirmation.

- return: 3.06%
- static loss: 0.62%
- daily loss: 0.34%
- gates: PASS
- sharpe: 1.34
- risk_fraction: 2.5%
- weight_mode: equal

## Legs

- USDCHF H4 bbands_reversion: oos_leg=-1.55% n=29 ho_leg=3.76%
- USDJPY D1 hybrid_regime: oos_leg=1.04% n=92 ho_leg=0.25%
- GBPUSD H4 breakout_donchian: oos_leg=-3.87% n=88 ho_leg=7.13%
- AUDUSD D1 hybrid_regime: oos_leg=-0.69% n=110 ho_leg=1.11%

## Risk stress (same exits/weights)

- 1.0%: ret=1.89% static=0.40% daily=0.19% gates=PASS sharpe=1.43
- 1.5%: ret=2.58% static=0.57% daily=0.29% gates=PASS sharpe=1.39
- 2.0%: ret=3.06% static=0.61% daily=0.33% gates=PASS sharpe=1.41
- 2.5%: ret=3.06% static=0.62% daily=0.34% gates=PASS sharpe=1.34
- 3.0%: ret=3.09% static=0.61% daily=0.34% gates=PASS sharpe=1.32
