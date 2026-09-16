# Preferred FX4 @2.5% equal (CHF/GBP/CADJPY/AUDCAD)

USDJPY→AUDCAD via OOS basket (+1.70% vs FX5 +0.99% / FX4-CADJPY +0.77%). Holdout confirmation.

- return: 4.82%
- static loss: 0.22%
- daily loss: 0.46%
- gates: PASS
- sharpe: 1.91
- OOS proxy: 1.70%
- risk_fraction: 2.5%
- weight_mode: equal

## Legs

- USDCHF H4 bbands_reversion: ho_leg=3.76% w=0.25 vt=False ex={}
- GBPUSD H4 breakout_donchian: ho_leg=7.13% w=0.25 vt=True ex={'atr_trail_mult': 0.0, 'atr_target_mult': 5.0, 'atr_stop_mult': 1.5, 'max_hold_bars': 24}
- CADJPY H4 mean_reversion_regime: ho_leg=7.91% w=0.25 vt=True ex={'max_hold_bars': 16}
- AUDCAD H4 mean_reversion_regime: ho_leg=0.46% w=0.25 vt=False ex={'atr_target_mult': 5.0, 'atr_stop_mult': 1.5, 'max_hold_bars': 24}

## Risk stress

- 1.0%: ret=3.07% static=0.23% daily=0.25% gates=PASS sharpe=2.00
- 1.5%: ret=4.35% static=0.30% daily=0.37% gates=PASS sharpe=2.00
- 2.0%: ret=4.93% static=0.21% daily=0.45% gates=PASS sharpe=2.01
- 2.5%: ret=4.82% static=0.22% daily=0.46% gates=PASS sharpe=1.91
- 3.0%: ret=4.73% static=0.22% daily=0.46% gates=PASS sharpe=1.86
