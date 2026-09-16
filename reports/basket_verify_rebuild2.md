# Preferred verify_rebuild2 @2.0% (weights=oos_sharpe)

Interim dual FX4 or pure Sharpe-first; weights from OOS only. Holdout confirmation only.

- return: 4.42%
- static loss: 0.03%
- daily loss: 0.38%
- gates: PASS
- sharpe: 2.00
- risk_fraction: 2.0%
- weight_mode: oos_sharpe

## Legs

- USDCHF H4 bbands_reversion: OOS=0.21% Sh=1.44 hold_leg=3.62% n=43 w=0.273
- GBPUSD H4 breakout_donchian: OOS=0.19% Sh=0.43 hold_leg=7.68% n=119 w=0.082
- CADJPY H4 mean_reversion_regime: OOS=0.33% Sh=1.75 hold_leg=8.18% n=51 w=0.332
- AUDCAD H4 mean_reversion_regime: OOS=0.16% Sh=1.64 hold_leg=0.25% n=26 w=0.312

