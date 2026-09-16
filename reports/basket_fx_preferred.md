# Preferred FX4 @2.5% equal + CADJPY MR (vt+mh16)

CADJPY IS exit upgraded vol_target+max_hold=16 (OOS basket +0.77% vs +0.73%). Holdout confirmation.

- return: 4.76%
- static loss: 0.36%
- daily loss: 0.46%
- gates: PASS
- sharpe: 1.78
- OOS proxy: 0.77%
- risk_fraction: 2.5%
- weight_mode: equal

## Legs

- USDCHF H4 bbands_reversion: ho_leg=3.76% w=0.25 vt=False ex={}
- USDJPY D1 hybrid_regime: ho_leg=0.25% w=0.25 vt=False ex={'atr_trail_mult': 1.5, 'atr_target_mult': 0.0, 'atr_stop_mult': 1.5, 'max_hold_bars': 16}
- GBPUSD H4 breakout_donchian: ho_leg=7.13% w=0.25 vt=True ex={'atr_trail_mult': 0.0, 'atr_target_mult': 5.0, 'atr_stop_mult': 1.5, 'max_hold_bars': 24}
- CADJPY H4 mean_reversion_regime: ho_leg=7.91% w=0.25 vt=True ex={'max_hold_bars': 16}

## Risk stress

- 1.0%: ret=3.04% static=0.29% daily=0.24% gates=PASS sharpe=1.90
- 1.5%: ret=4.31% static=0.39% daily=0.37% gates=PASS sharpe=1.90
- 2.0%: ret=4.88% static=0.33% daily=0.45% gates=PASS sharpe=1.89
- 2.5%: ret=4.76% static=0.36% daily=0.46% gates=PASS sharpe=1.78
- 3.0%: ret=4.71% static=0.36% daily=0.46% gates=PASS sharpe=1.74
