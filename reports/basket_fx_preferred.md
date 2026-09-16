# Preferred FX4 @2.5% OOS-Sharpe + IS exits + IS vol_target

IS exits + IS vol_target (GBPUSD); OOS-Sharpe weights; risk 2.5%. Not go-live.

- return: 2.88%
- static loss: 0.36%
- daily loss: 0.48%
- gates: PASS
- sharpe: 1.23
- risk_fraction: 2.5%

## Risk stress

- 1.0%: ret=1.68% static=0.22% daily=0.26% gates=PASS sharpe=1.43
- 1.5%: ret=2.39% static=0.31% daily=0.39% gates=PASS sharpe=1.38
- 2.0%: ret=2.81% static=0.31% daily=0.49% gates=PASS sharpe=1.32
- 2.5%: ret=2.88% static=0.36% daily=0.48% gates=PASS sharpe=1.23

## Legs

- USDCHF H4 bbands_reversion: OOS=0.21% Sh=1.44 hold_leg=3.76% n=43 vt=False exits=default w=0.455
- USDJPY D1 hybrid_regime: OOS=0.80% Sh=0.95 hold_leg=0.25% n=15 vt=False exits={'atr_trail_mult': 1.5, 'atr_target_mult': 0.0, 'atr_stop_mult': 1.5, 'max_hold_bars': 16} w=0.300
- GBPUSD H4 breakout_donchian: OOS=0.19% Sh=0.43 hold_leg=7.13% n=119 vt=True exits={'atr_trail_mult': 0.0, 'atr_target_mult': 5.0, 'atr_stop_mult': 1.5, 'max_hold_bars': 24} w=0.137
- AUDUSD D1 hybrid_regime: OOS=0.25% Sh=0.34 hold_leg=1.11% n=10 vt=False exits={'atr_trail_mult': 0.0, 'atr_target_mult': 2.0, 'atr_stop_mult': 2.5, 'max_hold_bars': 16} w=0.108
