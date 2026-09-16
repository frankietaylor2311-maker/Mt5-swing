# Preferred FX5 @2.5% equal + CADJPY + AUDCAD

AUDCAD H4 mean_reversion added by OOS basket (+1.00% vs FX4 +0.77%). Holdout confirmation.

- return: 3.96%
- static loss: 0.17%
- daily loss: 0.36%
- gates: PASS
- sharpe: 1.89
- OOS proxy: 0.99%
- risk_fraction: 2.5%
- weight_mode: equal

## Legs

- USDCHF H4 bbands_reversion: ho_leg=3.62% w=0.20 vt=False ex={}
- USDJPY D1 hybrid_regime: ho_leg=0.05% w=0.20 vt=False ex={'atr_trail_mult': 1.5, 'atr_target_mult': 0.0, 'atr_stop_mult': 1.5, 'max_hold_bars': 16}
- GBPUSD H4 breakout_donchian: ho_leg=7.68% w=0.20 vt=True ex={'atr_trail_mult': 0.0, 'atr_target_mult': 5.0, 'atr_stop_mult': 1.5, 'max_hold_bars': 24}
- CADJPY H4 mean_reversion_regime: ho_leg=8.18% w=0.20 vt=True ex={'max_hold_bars': 16}
- AUDCAD H4 mean_reversion_regime: ho_leg=0.25% w=0.20 vt=False ex={'atr_target_mult': 5.0, 'atr_stop_mult': 1.5, 'max_hold_bars': 24}

## Risk stress

- 1.0%: ret=2.04% static=0.15% daily=0.16% gates=PASS sharpe=1.95
- 1.5%: ret=2.87% static=0.23% daily=0.24% gates=PASS sharpe=1.87
- 2.0%: ret=3.64% static=0.22% daily=0.31% gates=PASS sharpe=1.90
- 2.5%: ret=3.96% static=0.17% daily=0.36% gates=PASS sharpe=1.89
- 3.0%: ret=3.91% static=0.18% daily=0.37% gates=PASS sharpe=1.81
