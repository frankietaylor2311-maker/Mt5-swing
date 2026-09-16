# Preferred verify_rebuild @2.0% (weights=oos_sharpe)

Interim dual FX4 or pure Sharpe-first; weights from OOS only. Holdout confirmation only.

- return: 4.62%
- static loss: 0.01%
- daily loss: 0.36%
- gates: PASS
- sharpe: 2.05
- risk_fraction: 2.0%
- weight_mode: oos_sharpe

## Legs

- USDCHF H4 bbands_reversion: OOS=0.21% Sh=1.44 hold_leg=3.62% n=43 w=0.273
- GBPUSD H4 breakout_donchian: OOS=0.19% Sh=0.43 hold_leg=3.13% n=111 w=0.082
- CADJPY H4 mean_reversion_regime: OOS=0.33% Sh=1.75 hold_leg=9.01% n=31 w=0.332
- AUDCAD H4 mean_reversion_regime: OOS=0.16% Sh=1.64 hold_leg=1.22% n=12 w=0.312

