# FTMO 2-Step Marathon Report (approximate_non_ftmo)

**Wave through run36:** JPY pip-value sizing fix; CADJPY dual unlocked; equal-weight FX4 promote.  
**As of:** 2026-09-16 ~10:15 Europe/London

## Preferred basket (current best)

| Basket | Holdout | Sharpe | Gates |
|--------|--------:|-------:|-------|
| Task baseline equal @2% | +1.81% | — | PASS |
| FX4 oos_sharpe + IS exits + VT @2.5% | +2.88% | 1.23 | PASS |
| FX4 equal (AUDUSD) @2.5% | +3.06% | 1.34 | PASS |
| **Current FX4 equal CADJPY MR @2.5%** | **+4.90%** | **1.83** | **PASS** |

**Δ vs +1.81%:** **+3.09 pp**

**Legs (interim dual FX4 ≤1/symbol):**
- USDCHF H4 `bbands_reversion` (w=0.25)
- USDJPY D1 `hybrid_regime` (w=0.25) — trail 1.5 / stop 1.5 / max_hold 16
- GBPUSD H4 `breakout_donchian` (w=0.25) — tp 5 / stop 1.5 / max_hold 24 / **vol_target**
- **CADJPY H4 `mean_reversion_regime` (w=0.25)** — **vol_target** (IS); replaces AUDUSD

**Config:** `configs/best_interim_approximate.yaml` (`basket_tag: fx4_equal_cadjpy_mr_is_vt_rf025`)

**Selection:** After fixing `pip_value_per_lot` for *JPY crosses, CADJPY became dual-rich. Among FX4/FX5 constructions ranked by **OOS basket return**, replace-AUDUSD→CADJPY MR equal won (OOS **+0.73%** vs prior equal **−1.31%**). Holdout confirmation **+4.90%** (not used for selection).

Risk stress (PASS 5%/10% Prague): 1%→+3.32% · 1.5%→+4.65% · **2%→+5.04%** · 2.5%→+4.90% · 3%→+4.89%  
(Config risk stays 2.5% default; 2% stress is confirmation only — not promoted as tune.)

## Gates / methodology

- `signal_lag=1`; IS-only param + exit + vol_target; holdout confirmation only
- FTMO 5%/10% static Europe/Prague; calendar-honest joins
- No go-live without `data/ftmo/`

## Notable infra

- **Bugfix:** EURJPY/GBPJPY/CADJPY pip_value was ~1000 (treated as USD) → 0.01 lots; now `raw/price` for quote=JPY
- **willr_reversion** strategy added
- Research runner merges summaries; protects interim legs

## Blockers

1. No FTMO MT5 exports → `approximate_non_ftmo` only
2. Still short of challenge 10% target on ~1y holdout
3. Yahoo ≠ broker CFD
4. Pure Sharpe-first (no interim) historically fails holdout
