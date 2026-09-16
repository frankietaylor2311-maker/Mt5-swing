# Basket oos_all (confirmation only)

Members chosen by **OOS research** only (profitable+gates, ≥15 trades, OOS Sharpe>0, IS return>0, ≤1 leg/symbol). Holdout never used for selection.

- data_source: approximate_non_ftmo (unless FTMO exports present)
- fx_only: False
- legs: 3
- holdout basket return: -0.81%
- static loss: 1.56%
- max daily loss: 0.41%
- gates: PASS
- sharpe: -1.39

## Legs

- GBPUSD D1 breakout_donchian: OOS=1.34% n=47 holdout_leg=-2.20% gates=True
- USDJPY D1 hybrid_regime: OOS=0.80% n=35 holdout_leg=0.24% gates=True
- EURUSD D1 bbands_reversion: OOS=0.35% n=17 holdout_leg=-0.48% gates=True
