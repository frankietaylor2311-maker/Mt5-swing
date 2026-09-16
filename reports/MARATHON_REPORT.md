# FTMO 2-Step Marathon Report (approximate_non_ftmo)

**As of:** 2026-09-16 ~10:25 Europe/London

## Preferred basket (current best by OOS proxy)

| Basket | Holdout | Sharpe | Gates | OOS proxy |
|--------|--------:|-------:|-------|----------:|
| Task baseline @2% | +1.81% | — | PASS | — |
| FX4 oos_sharpe+IS @2.5% | +2.88% | 1.23 | PASS | — |
| FX4 equal AUDUSD | +3.06% | 1.34 | PASS | −1.31% |
| FX4 equal CADJPY vt+mh16 | +4.76% | 1.78 | PASS | +0.77% |
| **Current FX5 equal +AUDCAD MR @2.5%** | **+3.96%** | **1.89** | **PASS** | **+0.99%** |

**Δ vs +1.81%:** **+2.15 pp** (OOS-best construction; FX4 CADJPY remains stronger on holdout confirmation at +4.76%)

**Legs (dual ≤1/symbol):**
1. USDCHF H4 bbands_reversion
2. USDJPY D1 hybrid_regime (trail/stop 1.5, mh 16)
3. GBPUSD H4 breakout_donchian (tp5/stop1.5/mh24, vol_target)
4. CADJPY H4 mean_reversion_regime (vol_target, mh 16)
5. AUDCAD H4 mean_reversion_regime (tp5/stop1.5/mh24 IS)

**Config:** `fx5_equal_cadjpy_audcad_rf025`

## Methodology

- signal_lag=1; IS-only; OOS basket selects among dual constructions; holdout confirmation only
- JPY pip_value fix enabled CADJPY research
- No golive without data/ftmo/

## Blockers

No FTMO exports; Yahoo≠CFD; short of 10% challenge target; pure Sharpe-first fails holdout.
