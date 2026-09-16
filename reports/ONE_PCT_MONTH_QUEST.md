# One-percent per month quest — status

**Data:** `approximate_non_ftmo` (Yahoo via yfinance). **No `data/ftmo/` exports — never golive.**
**Gates:** FTMO 2-Step static max loss 10%, daily 5% (Europe/Prague). `signal_lag=1`. IS-only grids. Holdout never for selection.

Session: 2026-09-16 evening Europe/London (wave: add-only duals / D1-H4 budgets / causal max-concurrent recycle).

## Target vs result

| Criterion | Target | Locked candidate | Met? |
|-----------|--------|------------------|:----:|
| Mean monthly on eval windows | ≥ ~1.0% | Holdout **+1.52%**; 2025 **+1.63%**; 2026 **+2.30%**; roll12_m6 **+1.13%**; **2024 +0.42%** | **Mostly — 2024 still short on the locked tag** |
| % positive months | ≥ ~65–70% | Holdout **75%**; 2026 **88%**; 2025 **73%**; 2024 **55%** | Holdout/2025/2026 yes |
| Top-3 months share of gains | ≲ 50% | Holdout **~81%** | **No — still bursty** |
| Gates | PASS | All listed windows **PASS** | **Yes** |
| Multi-window, warmup, fixed params | required | Independent windows + 250-bar warmup; params frozen | **Yes** |

**Verdict:** Official locked tag unchanged. A **clock + idle-recycle overlay** on the same five legs lifts **2024 to +1.18%/mo** and keeps 2025/2026/HO ≥1.8% with ≥73% positive months and gates PASS, but **holdout top3 rises 81%→90%**, so it is **not promoted**. 2024-positive dual add-only **overfits 2024 and wrecks 2025**. **Target not fully met.**

## Locked candidate (unchanged — still official)

**Tag:** `fx4plus_gbpcad_d1_voltarget_0025`  
**Config:** `configs/quest_one_pct_candidate.yaml`

Verified this wave via `scripts/quest_addonly_clock_recycle.py` baseline (WEIGHTS=oos_sharpe, PORT_VOL_TARGET=0.0025, RF=8%):

| Window | Return | Mean mo | %pos | Top3 | Gates | P2T |
|--------|-------:|--------:|-----:|-----:|:-----:|----:|
| 2024 | +5.38% | **0.42%** | 55% | 74% | PASS | 7.4% |
| 2025 | +22.04% | **1.63%** | 73% | 69% | PASS | 3.9% |
| 2026 YTD | +20.03% | **2.30%** | 88% | 87% | PASS | 6.4% |
| holdout_365d | +21.55% | **1.52%** | 75% | 81% | PASS | 6.8% |
| roll12_m6 | +16.45% | **1.13%** | 67% | 80% | PASS | 3.9% |

## Wave: add-only duals / clock budgets / causal recycle (new)

Script: `scripts/quest_addonly_clock_recycle.py`. Per-window BT + warmup. Hyperparams on **2024 only**. Dual-confirm = meaningful OOS on **both** H4 and D1. Add-only unused symbols, corr vs locked 2024 daily returns. Recycle exposure **clipped to 1** (idle weight only; RF/leg frozen at 1.60%).

| Idea | 2024 mo | 2025 mo | HO mo | HO %pos | HO top3 | Notes |
|------|--------:|--------:|------:|--------:|--------:|-------|
| baseline_vt0025 | 0.42% | **1.63%** | **1.52%** | 75% | 81% | locked reference |
| **locked_clock_recycle** | **1.18%** | **1.86%** | **1.80%** | **75%** | **90%** | H4 share 55% + max_k=3 + recycle cap 2 + vt; years ≥1% but burstier HO |
| locked_h4_055_cap15 | 0.97% | 1.63% | 1.63% | **58%** | 90% | gentler recycle; HO %pos fails |
| addonly_plain (3 duals) | **1.15%** | **0.53%** | 1.49% | 75% | 73% | GBPJPY+EURGBP+EURAUD; 2025 fail |
| selected (add3+recycle) | **2.42%** | **0.18%** | 2.06% | 83% | 74% | 2024 IS win / 2025 collapse — reject |
| add_gbpjpy_plain | 0.97% | 0.85% | 1.33% | 75% | 79% | best single dual; 2025 still <1% |

**Promote?** **No.** Add-only duals overfit 2024. Clock+recycle on locked legs is the only confirmation-passing way to get 2024 ≥1%/mo across 2024–2026, but it **worsens top3**. Official tag stays `fx4plus_gbpcad_d1_voltarget_0025`.

Details: `reports/quest_addonly_clock_recycle.md`, `configs/quest_addonly_selected.json`.

### 2024 IS-screened duals (add-only pool)

Best unused duals with 2024 mean_mo>0, gates, corr gate: GBPJPY H4 squeeze (1.42%/mo, corr −0.30), EURGBP D1 cci (0.71%, −0.03), EURAUD H4 donchian (0.59%, −0.24). All **hurt 2025** once added.

## New-strat WF (dual-confirm symbols only)

Prior board already had tsmom/vol_breakout/carry_proxy/kalman_trend/session_orb on 13 duals (mostly weak OOS / holdout fail; carry ~0 trades). This wave filled the **remaining** duals: EURAUD, EURCAD, EURGBP, GBPCAD, NZDCAD (H4+D1) — 50 combos, board now 674 rows. `approximate_non_ftmo`.

Meaningful OOS (sharpe>0.3, trades≥10, profit): GBPCAD H4 kalman (OOS sh 0.76, holdout fail), EURCAD D1 vol_breakout (0.48, holdout fail), NZDCAD D1 kalman (0.40, holdout fail), EURGBP H4 kalman (0.39, holdout fail). Carry still ~0 trades. **No new-strat dual promoted** (holdout fails; GBPCAD already in locked basket as D1 bbands).

## Ideas tried (cumulative)

| Idea | Result |
|------|--------|
| New strats TSMOM/vol-breakout/carry/Kalman/ORB | Weak OOS; many holdout fails; carry ~0 trades |
| Two-leg pairs MR | Negative once real spreads+ATR sizing |
| FX5/FX6 H4 & mix @8% | HO ~0.5–0.8%/mo; weak 2024 |
| Risk-parity / higher risk 10–12% | Flatten or non-monotonic vs kill-switch |
| Preferred FX4 + vt without GBPCAD | 2025 gates FAIL |
| Clip vol-scale 1.5–2.0 | Smoother %pos but worse 2024 mean_mo |
| Regime sleeves | 2024↑ HO↓ — rejected |
| Corr / hotstreak / ADX tilt | Marginal vs locked; not promoted |
| **2024-positive dual add-only (this wave)** | 2024↑ 2025↓ — rejected |
| **D1/H4 clock budgets + causal max-k recycle (this wave)** | 2024→1.18%/mo and all years ≥1% on locked legs; HO top3 worse — **not promoted** |

## Gap remaining

1. **Official locked 2024 still ~0.4%/mo.** Recycle overlay can print 2024 ≥1% without extra leverage, but it concentrates gains (HO top3 90%). Need uncorrelated legs that survive **2025-like** years, not just 2024 IS.
2. **Burstiness** — locked HO top3 ~81%; recycle overlay ~90%. Still far from ≲50%.
3. **No FTMO MT5 exports** — all `approximate_non_ftmo`.
4. Next wave: (a) reject 2024-only duals that fail a second IS year (e.g. 2023 or rolling pre-holdout); (b) milder recycle (cap 1.25–1.4) with a top3 penalty; (c) finish dual-only new-strat WF and only add if 2024 **and** 2023 (pre-holdout) are both positive.

## Process / tests

- pytest: run after this wave.
- `signal_lag=1`; IS grids only; holdout confirmation.
- Overlay / add-only hyperparams picked on 2024 only.
- Recycle occupancy lagged one bar; exposure ≤ 1.
