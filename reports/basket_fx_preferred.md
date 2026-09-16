# Preferred FX4 @2.5% equal + CADJPY MR (JPY pip fix)

AUDUSD→CADJPY H4 mean_reversion selected by OOS basket vs FX4-equal; IS vol_target; holdout confirmation.

- return: 4.90%
- static loss: 0.08%
- daily loss: 0.46%
- gates: PASS
- sharpe: 1.83
- OOS proxy: 0.73% (sh 0.40)
- risk_fraction: 2.5%
- weight_mode: equal

## Legs

- USDCHF H4 bbands_reversion: OOS_sh=1.44 ho_leg=3.76% w=0.250 vt=False ex={}
- USDJPY D1 hybrid_regime: OOS_sh=0.95 ho_leg=0.25% w=0.250 vt=False ex={'atr_trail_mult': 1.5, 'atr_target_mult': 0.0, 'atr_stop_mult': 1.5, 'max_hold_bars': 16}
- GBPUSD H4 breakout_donchian: OOS_sh=0.43 ho_leg=7.13% w=0.250 vt=True ex={'atr_trail_mult': 0.0, 'atr_target_mult': 5.0, 'atr_stop_mult': 1.5, 'max_hold_bars': 24}
- CADJPY H4 mean_reversion_regime: OOS_sh=1.75 ho_leg=8.46% w=0.250 vt=True ex={}

## Risk stress

- 1.0%: ret=3.32% static=0.06% daily=0.23% gates=PASS sharpe=2.06
- 1.5%: ret=4.65% static=0.06% daily=0.32% gates=PASS sharpe=2.02
- 2.0%: ret=5.04% static=0.04% daily=0.41% gates=PASS sharpe=1.94
- 2.5%: ret=4.90% static=0.08% daily=0.46% gates=PASS sharpe=1.83
- 3.0%: ret=4.89% static=0.07% daily=0.46% gates=PASS sharpe=1.80
