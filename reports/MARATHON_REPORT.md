# FTMO 2-Step Swing Research Marathon — Status

**Wave:** runs 21–23 + OOS-Sharpe basket weighting.
**Repo:** https://github.com/frankietaylor2311-maker/Mt5-swing

## FTMO exports
**Still absent.** All = `approximate_non_ftmo`. **`ftmo_golive_candidate` = 0**.

## Anti-overfit / anti-lookahead
- IS-only grids; holdout confirmation only; `signal_lag=1`
- **29/29 pytest green**
- Dual FX ≤1/symbol; OOS Sharpe-first; min OOS return 0.1%; weights ∝ OOS Sharpe

## Preferred
| Basket | Holdout | Sharpe | Gates |
|--------|--------:|-------:|:-----:|
| **FX4 @2.5% OOS-Sharpe w** | **+2.28%** | **0.98** | PASS |
| Prior equal @2% | +1.81% | 0.92 | PASS |

**Legs:** USDCHF H4 bbands · USDJPY D1 hybrid · GBPUSD H4 breakout · AUDUSD D1 hybrid

## This keep-alive (~08:50 London)
- Merged run21–23 into board (210 rows / 11 dual symbols)
- New strats + EURGBP/EURCHF/AUDJPY interim data
- Basket weighting upgrade (OOS Sharpe); WRITE_CFG hardened default off
- Still working toward higher OOS expectancy under gates

## Blockers
1. No FTMO MT5 exports
2. Far from FTMO 10%/5% phase targets
3. Yahoo ≠ broker CFD
