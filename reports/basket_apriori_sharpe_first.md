# Preferred apriori_sharpe_first @2.5% (weights=equal)

Interim dual FX4 or pure Sharpe-first; weights from OOS only. Holdout confirmation only.

- return: -0.78%
- static loss: 1.50%
- daily loss: 0.44%
- gates: PASS
- sharpe: -0.40
- risk_fraction: 2.5%
- weight_mode: equal

## Legs

- USDCHF H4 bbands_reversion: OOS=0.21% Sh=1.44 hold_leg=3.76% n=43 w=0.250
- GBPUSD D1 breakout_donchian: OOS=1.34% Sh=1.07 hold_leg=-1.64% n=11 w=0.250
- AUDUSD H4 bbands_reversion: OOS=0.12% Sh=0.97 hold_leg=-5.21% n=36 w=0.250
- USDJPY D1 hybrid_regime: OOS=0.80% Sh=0.95 hold_leg=-0.03% n=9 w=0.250

