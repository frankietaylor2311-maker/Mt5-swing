# Basket fx_preferred (confirmation only)

Members chosen by **OOS research** only (profitable+gates, ≥15 trades, OOS Sharpe>0, IS return>0, ≤1 leg/symbol). Holdout never used for selection.

- data_source: approximate_non_ftmo (unless FTMO exports present)
- fx_only: True
- legs: 4
- holdout basket return: -0.68%
- static loss: 1.24%
- max daily loss: 0.40%
- gates: PASS
- sharpe: -1.20

## Legs

- GBPUSD D1 breakout_donchian: OOS=1.34% n=47 holdout_leg=-1.64% gates=True
- USDJPY D1 hybrid_regime: OOS=0.80% n=35 holdout_leg=-0.03% gates=True
- EURUSD D1 bbands_reversion: OOS=0.35% n=17 holdout_leg=-0.36% gates=True
- AUDUSD D1 breakout_donchian: OOS=0.34% n=61 holdout_leg=-0.85% gates=True
