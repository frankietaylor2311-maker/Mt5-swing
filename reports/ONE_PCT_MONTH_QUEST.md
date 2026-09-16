# One-percent per month quest — status

**Data:** `approximate_non_ftmo` (Yahoo via yfinance). **No `data/ftmo/` exports — never golive.**
**Gates:** FTMO 2-Step static max loss 10%, daily 5% (Europe/Prague). `signal_lag=1`. IS-only grids. Holdout never for selection.

Session: 2026-09-16 19:05 BST Europe/London (wave: **equity-curve TSMOM + monthly-budget VT + pairs residual**).

## Scoring rubric this wave (primary)

1. High **% positive months** (prefer ≥70–75%)
2. Lower **top-3 gain concentration** (prefer ≤~50–55%; reject if HO top3 worsens vs locked)
3. Mean monthly **~1% stable** across 2024, 2025, 2026, holdout — **2024 cannot stay at 0.42%**
4. Prefer **lower monthly variance / less lumpy** mean over higher bursty mean
5. **Reject** candidates that boost mean via bursts even if gates PASS
6. **No leverage increase** (RF fixed at 8%; overlay `hi≤1`)

**Selection objective:** maximize `min(IS year mean_mo)` for `{2024, 2025_IS}` subject to %pos≥70% and top3≤55% (soft≤70%). Holdout / full 2025–2026 **confirmation only** — promote only if holdout **also** clears ≥1% mean and ≥70% pos **and** each of 2024/2025/2026 clears the same.

## Target vs result

| Criterion | Target | Locked candidate | Met? |
|-----------|--------|------------------|:----:|
| Mean monthly on eval windows | ≥ ~1.0% | Holdout **+1.52%**; 2025 **+1.63%**; 2026 **+2.30%**; roll12_m6 **+1.13%**; **2024 +0.42%** | **Mostly — 2024 still short** |
| % positive months | ≥ ~70–75% | Holdout **75%**; 2026 **88%**; 2025 **73%**; **2024 55%** | 2024 fails consistency |
| Top-3 months share of gains | ≲ 50–55% | Holdout **~81%**; 2024 **74%** | **No — still bursty** |
| Gates | PASS | All listed windows **PASS** | **Yes** |
| Multi-window, warmup, fixed params | required | Independent windows + 250-bar warmup; params frozen | **Yes** |

**Verdict:** Official locked tag **unchanged**. New structure (equity TSMOM / monthly-budget VT / pairs residual / D1 expand) — **204** candidates, **118** soft IS passers, **0** hard top3≤55% passers. Closest pairs Δz proxies reach ~1%/mo IS with high %pos but **holdout or 2026 dips just under 1%**; equity-TSMOM overlays **crush mean**. Prior real two-leg pairs (ATR/spreads) already negative — Δz proxy **not** promote-eligible. **Target not met on `approximate_non_ftmo`.**

## Locked candidate (unchanged — still official)

**Tag:** `fx4plus_gbpcad_d1_voltarget_0025`  
**Config:** `configs/quest_one_pct_candidate.yaml`

Verified this wave (`RF=0.08`, `PORT_VOL_TARGET=0.0025`, weights oos_sharpe; **GBPCAD_D1 not overwritten** by yfinance expand):

| Window | Return | Mean mo | %pos | Top3 | Gates | P2T |
|--------|-------:|--------:|-----:|-----:|:-----:|----:|
| 2024 | +5.38% | **0.42%** | 55% | 74% | PASS | ~7% |
| 2025 | +22.04% | **1.63%** | 73% | 69% | PASS | ~4% |
| 2026 YTD | +20.0%+ | **2.30%** | 88% | 87% | PASS | ~6% |
| holdout_365d | +21.5%+ | **1.52%** | 75% | 81% | PASS | ~7% |
| roll12_m6 | +16.4%+ | **1.13%** | 67% | 80% | PASS | ~4% |

## Wave: equity TSMOM + monthly-budget VT + pairs residual (new)

Scripts/helpers:
- `scripts/quest_equity_tsmom_pairs_wf.py`
- `src/mt5_swing/portfolio/equity_tsmom.py`
- `src/mt5_swing/portfolio/pairs_residual.py`

### Design (not another same-board basket grid)

1. **Equity-curve TSMOM** on locked diversified sleeve (lagged lookback return → scale next bar; `hi≤1`) then/or **causal monthly-budget VT** (~1%/mo → daily vol map; flattener `hi≤1`).
2. **Pairs/spread residual MR** on cointegrated FX (IS OLS hedge + rolling z); hard risk caps (`RF/n`, bar-return clip); monthly distribution score.
3. **D1 history expand** via yfinance to ~15y for pairs symbols; **skip locked D1 (GBPCAD)**; H4 Yahoo 1h still ~730d — irreducible.
4. Score = `min(2024, 2025_IS mean_mo)` with %pos/top3; holdout confirmation only.

### Board summary

| Family | N | Soft IS | Hard IS | Best IS min_mo |
|--------|--:|--------:|--------:|---------------:|
| equity overlays | 30 | 5 | 0 | ~0.42% (budget on locked VT; %pos fail) / soft ~0.18% |
| pairs residual (Δz proxy) | 48 | 36 | 0 | **~1.05%** |
| blend lock+pairs | 126 | 77 | 0 | ~0.96% |
| **Total** | **204** | **118** | **0** | |

### Best IS soft (holdout excluded from score)

`pairs_n3` entry=2.0 sleeve=1.0 (EURUSD/GBPUSD, EURCHF/USDCHF, EURUSD/USDCHF; win=40) — IS min_mo **1.01%**, max top3 **62%**, min %pos **75%**.

### Confirmation (promote gate)

| Idea | 2024 mo/%pos | 2025 | 2026 | HO mo/%pos | Promote | Reason |
|------|-------------:|-----:|-----:|-----------:|:-------:|--------|
| pairs_n3 e2.0 | 1.35/100 | 1.12/91 | **0.88**/88 | **0.95**/92 | no | HO&2026 &lt;1% |
| pairs_n4 e1.5 | 1.79/100 | 1.01/100 | 1.07/100 | **0.99**/100 | no | HO 0.99% &lt;1% |
| pairs_n2 e2.0 | 1.47/100 | 1.06/91 | **0.98**/88 | 1.06/92 | no | 2026 &lt;1% |
| eq budget_mo / tsmom | ≤0.18 soft | — | — | — | no | mean collapse |
| locked baseline | 0.42/55 | 1.63/73 | 2.30/88 | 1.52/75 | — | official |

**Promote?** **No** — official tag unchanged.

### Evidence: impossible (this structure) on approximate_non_ftmo

1. **Equity TSMOM / monthly-budget VT** on the locked sleeve raises 2024 %pos only when they **cut mean far below 1%** (soft passers ≈0.17–0.18%/mo). On locked VT base they keep ~0.42% but fail %pos. No overlay clears dual-year ≥1% with ≥70% pos.
2. **Pairs Δz residual proxy** can print ~1%/mo IS with excellent %pos / lower top3, but:
   - Holdout or calendar-2026 consistently lands **0.88–0.99%** — just under the ≥1% joint gate.
   - Proxy maps `risk_frac · Δz` → return; **prior real two-leg ATR/spread basket was negative** (`pairs_two_leg_quest`) — so proxy **cannot** be promoted without FTMO-realistic fills.
3. **Blends** lock+pairs sit between (~0.75–0.96% IS min) — inherit lock 2024 weakness or pairs HO shortfall.
4. **H4 cannot be extended** past Yahoo ~730d 1h; D1 expand helps pairs fit history but does not unlock dual-year ≥1% under promote rules.
5. **0 / 204** hard top3≤55% on both IS windows.

**Conclusion:** Under no-look-ahead, no RF hike, consistency &gt; peak mean, and holdout-as-confirmation-only, **clean ≥1% mean monthly on EACH of 2024, 2025, 2026, holdout with ≥70% pos is not achieved** on `approximate_non_ftmo` with this wave’s structures. Closest approaches fail either HO/2026 by a few bp or rely on a non-tradable Δz proxy.

Details: `reports/quest_equity_tsmom_pairs_wf.md`, `reports/quest_equity_tsmom_pairs_board.csv`, `reports/quest_equity_tsmom_pairs_promote.csv`, `configs/quest_equity_tsmom_pairs_selected.json`.

## Ideas tried (cumulative)

| Idea | Result |
|------|--------|
| New strats TSMOM/vol-breakout/carry/Kalman/ORB | Weak OOS; many holdout fails; carry ~0 trades |
| Two-leg pairs MR (real fills) | Negative once real spreads+ATR sizing |
| FX5/FX6 H4 & mix @8% | HO ~0.5–0.8%/mo; weak 2024 |
| Risk-parity / higher risk 10–12% | Flatten or non-monotonic vs kill-switch |
| Clip vol-scale 1.5–2.0 | **%pos↑ on 2024** but mean↓ — not enough for 1% |
| Regime sleeves | 2024↑ HO↓ — rejected |
| Corr / hotstreak / ADX tilt | Marginal vs locked; not promoted |
| 2024-positive dual add-only | 2024↑ 2025↓ — rejected |
| D1/H4 clock + max-k recycle | 2024→1.18%/mo but HO top3 worse — not promoted |
| Month-aware / equity-curve / runup flatten | Consistency↑ mean↓ — rejected |
| Dual-year ≥0.8% add-ons | **0 passers** |
| Soft +AUDJPY | 2024 %pos↑ 2025 %pos↓ — rejected |
| H1 Yahoo probe | Flat / no edge — rejected |
| Smooth many-leg + IS min(year) WF | Best IS ~0.81%/mo; HO %pos collapses — rejected |
| Metals dual-confirm | None eligible |
| **Equity TSMOM + monthly-budget VT (this wave)** | Soft ~0.18%/mo; mean collapse — **rejected** |
| **Pairs residual Δz + hard caps (this wave)** | IS ~1.0%+ / high %pos; HO or 2026 &lt;1%; proxy≠real fills — **rejected** |
| **Blend lock+pairs (this wave)** | IS min &lt;1% — **rejected** |
| **D1 yfinance expand (this wave)** | ~15y D1 for pairs; H4 still capped; locked GBPCAD preserved |

## Gap remaining / irreducible Yahoo limits

1. **2024 mean still ~0.4%/mo** on official locked; diversifiers / overlays that lift 2024 either hurt HO %pos or cut mean below 1%.
2. **Burstiness** — locked HO top3 ~81%; hard top3≤55% on both IS years: **zero** passers this wave (204-grid) and prior smooth 528-grid.
3. **Pairs:** realistic two-leg negative; Δz proxy near-miss on HO/2026 — not a promote path.
4. **No dual-confirm metals**; **no index history** (US30/NAS100/SPX) in repo.
5. **No FTMO MT5 exports** — all `approximate_non_ftmo`. Spreads/swap/sessions/Yahoo FX ≠ FTMO CFD book.
6. **H4 Yahoo ~730d** — cannot add more H4 years via yfinance.
7. **What FTMO `data/ftmo/` would unlock:** true spreads/commission/swap, session filters, deeper H1/M15, index CFDs if offered, and any go-live path (`ftmo_mt5_export`).

## Process / tests

- pytest: `tests/test_equity_tsmom_pairs.py` (+ smooth_select / month_equity / lookahead) — 19 passed.
- `signal_lag=1`; IS grids only (`2024`, `2025_IS`); holdout confirmation.
- RF forced to **8%**; overlay/pair caps `hi≤1` / RF/n; no leverage hike.
