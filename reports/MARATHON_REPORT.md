# FTMO 2-Step Marathon Report (approximate_non_ftmo)

**Wave through run37:** JPY pip fix → CADJPY dual; equal FX4; IS mh16 on CADJPY.  
**As of:** 2026-09-16 ~10:15 Europe/London

## Preferred basket (current best)

| Basket | Holdout | Sharpe | Gates |
|--------|--------:|-------:|-------|
| Task baseline equal @2% | +1.81% | — | PASS |
| FX4 oos_sharpe + IS exits + VT @2.5% | +2.88% | 1.23 | PASS |
| FX4 equal (AUDUSD) @2.5% | +3.06% | 1.34 | PASS |
| FX4 equal CADJPY MR vt @2.5% | +4.90% | 1.83 | PASS |
| **Current FX4 equal CADJPY MR vt+mh16 @2.5%** | **+4.76%** | **1.78** | **PASS** |

**Δ vs +1.81%:** **+2.95 pp** (OOS proxy improved +0.73%→+0.77% with mh16; holdout confirmation slightly lower)

**Legs:**
- USDCHF H4 `bbands_reversion` (w=0.25)
- USDJPY D1 `hybrid_regime` (w=0.25) — trail 1.5 / stop 1.5 / max_hold 16
- GBPUSD H4 `breakout_donchian` (w=0.25) — tp 5 / stop 1.5 / max_hold 24 / vol_target
- CADJPY H4 `mean_reversion_regime` (w=0.25) — **vol_target + max_hold 16** (IS)

**Config:** `configs/best_interim_approximate.yaml` (`fx4_equal_cadjpy_mr_vt_mh16_rf025`)

## Gates / methodology

- `signal_lag=1`; IS-only; holdout confirmation only; FTMO 5%/10% Prague
- No go-live without `data/ftmo/`

## Infra

- Fixed `pip_value_per_lot` for quote=JPY (was ~1000 → 0.01 lots)
- `willr_reversion` strategy; merge-protect research runner

## Blockers

1. No FTMO MT5 exports
2. Short of challenge 10% target
3. Yahoo ≠ broker CFD
4. Pure Sharpe-first without interim lock fails holdout historically
