# Marathon progress

## Best so far (approximate_non_ftmo)
- **Preferred FX4 @2.5% with OOS-Sharpe weights: +2.28% holdout**, Sharpe 0.98, gates PASS
  - Same dual legs: USDCHF H4 bbands · USDJPY D1 hybrid · GBPUSD H4 breakout · AUDUSD D1 hybrid
  - Weights ∝ OOS Sharpe (a priori); holdout confirmation only
- Prior lock equal-weight @2%: +1.81% / Sh 0.92 (still valid baseline)
- Board: **210 rows**, **11 dual FX symbols**; `ftmo_golive_candidate=0`
- Runs 21–23: EURGBP/EURCHF/AUDJPY + stoch_reversion / atr_channel_breakout
- FTMO exports absent — no go-live; pytest **29/29** green
