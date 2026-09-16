# Preferred FX4 @2.5% OOS-Sharpe (CHF/GBP/CADJPY/AUDCAD)

Weights ∝ OOS Sharpe selected vs equal on OOS basket (+3.27% vs +1.70%). Holdout confirmation.

- return: 4.39%
- static loss: 0.03%
- daily loss: 0.40%
- gates: PASS
- sharpe: 1.92
- OOS proxy: 3.27%
- risk_fraction: 2.5%
- weight_mode: oos_sharpe

## Legs

- USDCHF H4 bbands_reversion: ho_leg=3.76% w=0.273 vt=False ex={}
- GBPUSD H4 breakout_donchian: ho_leg=7.13% w=0.082 vt=True ex={'atr_trail_mult': 0.0, 'atr_target_mult': 5.0, 'atr_stop_mult': 1.5, 'max_hold_bars': 24}
- CADJPY H4 mean_reversion_regime: ho_leg=7.91% w=0.332 vt=True ex={'max_hold_bars': 16}
- AUDCAD H4 mean_reversion_regime: ho_leg=0.46% w=0.312 vt=False ex={'atr_target_mult': 5.0, 'atr_stop_mult': 1.5, 'max_hold_bars': 24}

## Risk stress

- 1.0%: ret=2.68% static=0.08% daily=0.24% gates=PASS sharpe=2.02
- 1.5%: ret=3.96% static=0.09% daily=0.34% gates=PASS sharpe=2.05
- 2.0%: ret=4.42% static=0.03% daily=0.38% gates=PASS sharpe=2.00
- 2.5%: ret=4.39% static=0.03% daily=0.40% gates=PASS sharpe=1.92
- 3.0%: ret=4.33% static=0.03% daily=0.40% gates=PASS sharpe=1.88
