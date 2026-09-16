# One-percent per month quest — status

**Data:** `approximate_non_ftmo` (Yahoo via yfinance). **No `data/ftmo/` exports — never golive.**
**Gates:** FTMO 2-Step static max loss 10%, daily 5% (Europe/Prague). `signal_lag=1`. IS-only grids. Holdout never for selection.

Session: 2026-09-16 evening Europe/London (wave: regime / diversification).

## Target vs result

| Criterion | Target | Locked candidate | Met? |
|-----------|--------|------------------|:----:|
| Mean monthly on eval windows | ≥ ~1.0% | Holdout **+1.52%**; 2025 **+1.63%**; 2026 **+2.30%**; roll12_m6 **+1.13%**; **2024 +0.42%** | **Mostly — 2024 still short** |
| % positive months | ≥ ~65–70% | Holdout **75%**; 2026 **88%**; 2025 **73%**; 2024 **55%** | Holdout/2025/2026 yes |
| Top-3 months share of gains | ≲ 50% | Holdout **~81%** | **No — still bursty** |
| Gates | PASS | All listed windows **PASS** | **Yes** |
| Multi-window, warmup, fixed params | required | Independent windows + 250-bar warmup; params frozen | **Yes** |

**Verdict:** Holdout / 2025 / 2026 / trailing-12m clear ~1%/mo with gates PASS. Calendar **2024 remains ~0.4%/mo**. Burstiness not solved. **Target not fully met.**

## Locked candidate (unchanged — still best overall)

**Tag:** `fx4plus_gbpcad_d1_voltarget_0025`  
**Config:** `configs/quest_one_pct_candidate.yaml`  

Verified this wave via `scripts/eval_windowed_consistency.py` (WEIGHTS=oos_sharpe, PORT_VOL_TARGET=0.0025):

| Window | Return | Mean mo | %pos | Top3 | Gates | P2T |
|--------|-------:|--------:|-----:|-----:|:-----:|----:|
| 2024 | +5.38% | **0.42%** | 55% | 74% | PASS | 7.42% |
| 2025 | +22.04% | **1.63%** | 73% | 69% | PASS | 3.86% |
| 2026 YTD | +20.03% | **2.30%** | 88% | 87% | PASS | 6.38% |
| holdout_365d | +21.55% | **1.52%** | 75% | 81% | PASS | 6.83% |
| roll12_m6 | +16.45% | **1.13%** | 67% | 80% | PASS | 3.87% |

## Wave: regime / diversification (new)

Script: `scripts/quest_regime_diversify.py` — **per-window** BT + warmup (full-history kill-switch contamination avoided). Hyperparams on **2024 only**.

| Idea | 2024 mo | HO mo | HO %pos | HO top3 | Notes |
|------|--------:|------:|--------:|--------:|-------|
| baseline_vt0025 | 0.42% | **1.52%** | 75% | 81% | locked reference |
| **hotstreak** | **0.51%** | **1.56%** | 75% | **78%** | best soft overlay; still not material |
| adx_tilt | 0.46% | 1.58% | 75% | 79% | soft MR/TR tilt by EURUSD ADX |
| corr_throttle+vt | 0.47% | 1.52% | 75% | 79% | peer-corr scale |
| regime_sleeves | **0.66%** | **−0.04%** | 58% | 77% | IS win / HO fail — reject |
| roll_is_sharpe | 0.24% | −0.26% | 25% | 100% | reject |

**Promote?** **No.** Soft overlays only nudge 2024 (+0.1 pp) and top3 (−3 pp). Sleeve switching overfits research year.

Details: `reports/quest_regime_diversify.md`, `configs/quest_regime_selected.json`.

## Ideas tried (cumulative)

| Idea | Result |
|------|--------|
| New strats TSMOM/vol-breakout/carry/Kalman/ORB | Weak OOS; many holdout fails; carry ~0 trades |
| Two-leg pairs MR | Negative once real spreads+ATR sizing |
| FX5/FX6 H4 & mix @8% | HO ~0.5–0.8%/mo; weak 2024 |
| Risk-parity / higher risk 10–12% | Flatten or non-monotonic vs kill-switch |
| Preferred FX4 + vt without GBPCAD | 2025 gates FAIL |
| Clip vol-scale 1.5–2.0 | Smoother %pos but worse 2024 mean_mo |
| **Regime sleeves (this wave)** | 2024↑ HO↓ — rejected |
| **Corr / hotstreak / ADX tilt (this wave)** | Marginal vs locked; not promoted |

## Gap remaining

1. **2024-like windows ~0.4%/mo** — overlays cannot mint alpha; need uncorrelated legs that actually work in that regime (IS/OOS only).
2. **Burstiness** — top3 still ~78–81% of HO gains; need more independent return streams or hard concurrent/risk caps that do not kill mean.
3. **No FTMO MT5 exports** — all `approximate_non_ftmo`.
4. Next wave ideas: (a) IS-screened 2024-positive duals as *add-only* legs with corr gate; (b) D1+H4 clock diversification with fixed risk budgets; (c) causal max-concurrent + recycle idle risk; (d) finish new-strat WF only on dual-confirm symbols.

## Process / tests

- pytest: run after this wave.
- `signal_lag=1`; IS grids only; holdout confirmation.
- Overlay hyperparams picked on 2024 only.
