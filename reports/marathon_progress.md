# Marathon progress

## Best so far (approximate_non_ftmo)
- **Preferred FX4 @2.5% OOS-Sharpe weights: +2.28% holdout**, Sharpe 0.98, gates PASS
  - USDCHF H4 bbands · USDJPY D1 hybrid · GBPUSD H4 breakout · AUDUSD D1 hybrid
  - Weights ∝ OOS Sharpe (research only); risk=ftmo_2step default 2.5%
  - Prior equal-weight @2%: +1.81%
- Risk stress: 1%→+1.18%, 1.5%→+1.78%, 2%→+2.11%, 2.5%→+2.28% (all PASS)
- Added EURGBP/AUDJPY/EURCHF; stoch_reversion + atr_channel_breakout
- Pure Sharpe-first dual (no interim lock) holdout ~−0.7% — not preferred
- FTMO exports absent — no go-live; pytest green
