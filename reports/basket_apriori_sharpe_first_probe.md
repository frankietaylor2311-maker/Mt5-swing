# Preferred apriori_sharpe_first_probe @2.0% (weights=oos_sharpe)

Interim dual FX4 or pure Sharpe-first; weights from OOS only. Holdout confirmation only.

- return: 2.54%
- static loss: 0.02%
- daily loss: 0.24%
- gates: PASS
- sharpe: 1.79
- risk_fraction: 2.0%
- weight_mode: oos_sharpe

## Legs

- GBPCAD D1 bbands_reversion: OOS=0.84% Sh=2.50 hold_leg=0.23% n=8 w=0.317
- EURJPY D1 ema_pullback: OOS=0.41% Sh=2.00 hold_leg=0.85% n=7 w=0.254
- CADJPY H4 mean_reversion_regime: OOS=0.33% Sh=1.75 hold_leg=9.01% n=31 w=0.221
- AUDCAD H4 mean_reversion_regime: OOS=0.16% Sh=1.64 hold_leg=1.22% n=12 w=0.208

