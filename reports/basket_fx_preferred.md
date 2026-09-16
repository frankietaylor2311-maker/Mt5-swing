# Preferred FX4 @2.5% OOS-Sharpe weights + IS exits

Dual FX4 interim; **OOS-Sharpe weights**; **IS-selected ATR exits** on trend legs; risk 2.5%. Holdout confirmation only. Not go-live.

- return: 2.83%
- static loss: 0.44%
- daily loss: 0.48%
- gates: PASS
- sharpe: 1.21
- risk_fraction: 2.5%

## Risk stress

- 1.0%: ret=1.44% static=0.17% daily=0.24% gates=PASS sharpe=1.31
- 1.5%: ret=2.16% static=0.25% daily=0.37% gates=PASS sharpe=1.31
- 2.0%: ret=2.59% static=0.35% daily=0.47% gates=PASS sharpe=1.23
- 2.5%: ret=2.83% static=0.44% daily=0.48% gates=PASS sharpe=1.21

## Legs

- USDCHF H4 bbands_reversion: OOS=0.21% Sh=1.44 hold_leg=3.76% n=43 exits=default w=0.455
- USDJPY D1 hybrid_regime: OOS=0.80% Sh=0.95 hold_leg=0.74% n=15 exits={'atr_trail_mult': 1.5, 'atr_target_mult': 4.0, 'atr_stop_mult': 2.0, 'max_hold_bars': 0} w=0.300
- GBPUSD H4 breakout_donchian: OOS=0.19% Sh=0.43 hold_leg=5.36% n=76 exits={'atr_trail_mult': 0.0, 'atr_target_mult': 4.0, 'atr_stop_mult': 2.0, 'max_hold_bars': 0} w=0.137
- AUDUSD D1 hybrid_regime: OOS=0.25% Sh=0.34 hold_leg=1.54% n=5 exits={'atr_trail_mult': 0.0, 'atr_target_mult': 4.0, 'atr_stop_mult': 2.0, 'max_hold_bars': 0} w=0.108
