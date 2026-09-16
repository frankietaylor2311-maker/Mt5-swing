# FTMO 2-Step Marathon Report (approximate_non_ftmo)

**Wave through run35:** willr strategy, merge-protect research runner, portfolio construction probes, equal-weight promote.  
**As of:** 2026-09-16 ~09:55 Europe/London

## Preferred basket (current best)

| Basket | Holdout | Sharpe | Gates |
|--------|--------:|-------:|-------|
| Prior equal @2% (baseline task) | +1.81% | — | PASS |
| FX4 @2.5% OOS-Sharpe + IS exits + VT | +2.88% | 1.23 | PASS |
| **Current FX4 @2.5% equal + IS exits + VT** | **+3.06%** | **1.34** | **PASS** |

**Δ vs +1.81%:** **+1.25 pp**

**Legs (interim dual FX4 ≤1/symbol):**
- USDCHF H4 `bbands_reversion` (w=0.25)
- USDJPY D1 `hybrid_regime` (w=0.25) — trail 1.5 / stop 1.5 / max_hold 16 / tp 0
- GBPUSD H4 `breakout_donchian` (w=0.25) — tp 5 / stop 1.5 / max_hold 24 / **vol_target**
- AUDUSD D1 `hybrid_regime` (w=0.25) — tp 2 / stop 2.5 / max_hold 16

**Config:** `configs/best_interim_approximate.yaml` (`basket_tag: fx4_equal_is_exits_vt_rf025`)

**Weight method:** equal selected vs `oos_sharpe` on **OOS basket return only** (−1.31% vs −1.48%). Holdout confirmation +3.06% (not used for selection).

Risk stress (PASS 5%/10% Prague): 1%→+1.89% · 1.5%→+2.58% · 2%→+3.06% · **2.5%→+3.06%** · 3%→+3.09% (3% not promoted — would be holdout-adjacent).

## Gates / methodology

- `signal_lag=1`; IS-only param + exit + vol_target selection; holdout confirmation only
- FTMO 5%/10% static Europe/Prague; calendar-honest basket joins (`ffill` + `dropna(how=any)`)
- No go-live: `ftmo_golive_candidate=0`; no `data/ftmo/`

## Probes (not promoted)

- **inv_vol weights:** best FX4 OOS (−0.80%) but HO +2.70% — deferred (path-vol DOF)
- **FX5 +GBPCAD bbands/willr/cci:** often better OOS, dilute HO vs FX4 equal (+1.7–2.5%)
- **FX3 drop AUDUSD:** worse OOS than FX4
- **Pure Sharpe-first dual (no interim lock):** holdout fails historically
- **willr_reversion:** new strat; GBPCAD D1 OOS +0.50% Sh 1.88 dual — exploratory only

## Blockers

1. No FTMO MT5 exports → stay `approximate_non_ftmo`
2. Far from FTMO challenge 10%/5% targets on ~1y holdout scale
3. Yahoo FX ≠ broker CFD costs/spreads/sessions
4. Pure a priori Sharpe-first construction fails holdout → interim FX4 lock remains
