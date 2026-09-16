# Quest wave — add-only duals / D1-H4 budgets / causal max-concurrent recycle

**data_source:** `approximate_non_ftmo` — never golive without FTMO MT5 exports.
**risk_fraction:** 8%  **per-leg (locked ref):** 1.60%  **warmup:** 250  **signal_lag=1**.
**Selection:** 2024 only (IS-screen + corr gate + overlay grid). Holdout confirmation only.
**Leverage:** recycle exposure clipped to 1; per-leg risk frozen at locked RF/5 — not raised.

## Dual-confirm universe

Symbols with meaningful OOS+gates on **both** H4 and D1: `['AUDCAD', 'AUDJPY', 'AUDUSD', 'CADJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURUSD', 'GBPCAD', 'GBPJPY', 'GBPUSD', 'NZDCAD', 'NZDJPY', 'USDCHF', 'USDJPY']`

## 2024 IS screen (add-only unused symbols)

| Symbol | TF | Strategy | OOS Sh | 2024 mo | Gates | n | corr vs locked | pass |
|--------|----|----------|-------:|--------:|:-----:|--:|---------------:|:----:|
| EURJPY | D1 | ema_pullback | 2.00 | -0.28% | PASS | 8 | 0.09 | NO |
| EURUSD | D1 | cci_reversion | 1.54 | 0.32% | PASS | 9 | 0.15 | YES |
| GBPJPY | H4 | squeeze_breakout | 1.44 | 1.42% | PASS | 57 | -0.30 | YES |
| NZDCAD | H4 | mean_reversion_regime | 1.35 | 0.10% | PASS | 11 | 0.21 | YES |
| EURCHF | H4 | cci_reversion | 1.20 | -0.47% | PASS | 10 | 0.04 | NO |
| EURGBP | D1 | cci_reversion | 1.10 | 0.71% | PASS | 8 | -0.03 | YES |
| AUDUSD | H4 | bbands_reversion | 0.97 | 0.03% | PASS | 34 | 0.24 | YES |
| USDJPY | D1 | hybrid_regime | 0.95 | 0.37% | PASS | 23 | 0.03 | YES |
| EURCAD | D1 | bbands_reversion | 0.92 | 0.11% | PASS | 9 | 0.20 | YES |
| AUDJPY | D1 | mean_reversion_regime | 0.90 | 0.30% | PASS | 6 | -0.00 | YES |
| NZDJPY | D1 | mean_reversion_regime | 0.81 | -0.30% | PASS | 4 | 0.08 | NO |
| EURAUD | H4 | breakout_donchian | 0.49 | 0.59% | PASS | 89 | -0.24 | YES |

## Selected (IS=2024)
```json
{
  "add_set": "corr0.25_n3",
  "extras": [
    "GBPJPY|H4|squeeze_breakout",
    "EURGBP|D1|cci_reversion",
    "EURAUD|H4|breakout_donchian"
  ],
  "clock": "h4=0.55",
  "h4_share": 0.55,
  "max_k": 3,
  "recycle_cap": 2.0,
  "vt": 0.0025,
  "n_legs": 8
}
```

| Idea | Window | Mean mo | Med mo | %pos | Top3 | Gates | P2T | Static | Daily | Legs | Trades | Overlap |
|------|--------|--------:|-------:|-----:|-----:|:-----:|----:|-------:|------:|-----:|-------:|---------|
| baseline_vt0025 | 2024 | 0.42% | 0.12% | 55% | 74% | PASS | 7.4% | 4.4% | 2.9% | 5 | 130 | research/WF |
| selected | 2024 | 2.42% | 1.47% | 91% | 59% | PASS | 5.2% | 0.0% | 2.9% | 8 | 284 | research/WF |
| selected_no_recycle | 2024 | 1.21% | 1.57% | 73% | 56% | PASS | 5.0% | 0.0% | 1.9% | 8 | 284 | research/WF |
| selected_no_clock | 2024 | 2.15% | 1.47% | 82% | 62% | PASS | 4.7% | 0.0% | 3.3% | 8 | 284 | research/WF |
| addonly_plain | 2024 | 1.15% | 0.64% | 82% | 61% | PASS | 5.5% | 0.0% | 2.1% | 8 | 284 | research/WF |
| locked_clock_recycle | 2024 | 1.18% | 0.87% | 73% | 71% | PASS | 6.4% | 3.0% | 2.8% | 5 | 130 | research/WF |
| locked_h4_055_cap15 | 2024 | 0.97% | 0.91% | 73% | 67% | PASS | 7.9% | 5.1% | 2.8% | 5 | 130 | research/WF |
| locked_h4_055_k3_cap20 | 2024 | 1.18% | 0.87% | 73% | 71% | PASS | 6.4% | 3.0% | 2.8% | 5 | 130 | research/WF |
| locked_oos_cap15 | 2024 | 0.86% | 0.54% | 73% | 76% | PASS | 8.0% | 5.2% | 3.0% | 5 | 130 | research/WF |
| add_gbpjpy_plain | 2024 | 0.97% | 0.87% | 73% | 69% | PASS | 5.9% | 0.0% | 2.3% | 6 | 187 | research/WF |
| add_gbpjpy_cap15 | 2024 | 1.55% | 1.46% | 82% | 67% | PASS | 6.9% | 0.0% | 3.3% | 6 | 187 | research/WF |
| baseline_vt0025 | 2025 | 1.63% | 1.00% | 73% | 69% | PASS | 3.9% | 1.2% | 3.6% | 5 | 74 | mixed |
| selected | 2025 | 0.18% | 0.25% | 55% | 74% | PASS | 9.1% | 1.4% | 3.0% | 8 | 195 | mixed |
| selected_no_recycle | 2025 | 0.45% | 0.50% | 73% | 69% | PASS | 4.1% | 0.7% | 2.0% | 8 | 195 | mixed |
| selected_no_clock | 2025 | 0.37% | 0.80% | 64% | 62% | PASS | 11.0% | 1.7% | 3.6% | 8 | 195 | mixed |
| addonly_plain | 2025 | 0.53% | 0.99% | 73% | 68% | PASS | 4.7% | 0.9% | 2.5% | 8 | 195 | mixed |
| locked_clock_recycle | 2025 | 1.86% | 0.91% | 73% | 73% | PASS | 4.6% | 2.7% | 4.4% | 5 | 74 | mixed |
| locked_h4_055_cap15 | 2025 | 1.63% | 0.68% | 82% | 72% | PASS | 3.9% | 1.5% | 3.4% | 5 | 74 | mixed |
| locked_h4_055_k3_cap20 | 2025 | 1.86% | 0.91% | 73% | 73% | PASS | 4.6% | 2.7% | 4.4% | 5 | 74 | mixed |
| locked_oos_cap15 | 2025 | 1.93% | 1.02% | 82% | 70% | PASS | 3.4% | 1.6% | 4.3% | 5 | 74 | mixed |
| add_gbpjpy_plain | 2025 | 0.85% | 1.31% | 73% | 62% | PASS | 5.4% | 1.0% | 3.0% | 6 | 137 | mixed |
| add_gbpjpy_cap15 | 2025 | 0.79% | 0.82% | 73% | 60% | PASS | 9.1% | 1.7% | 3.4% | 6 | 137 | mixed |
| baseline_vt0025 | 2026 | 2.30% | 1.34% | 88% | 87% | PASS | 6.4% | 3.4% | 2.5% | 5 | 122 | pure holdout |
| selected | 2026 | 2.37% | 1.23% | 75% | 86% | PASS | 6.3% | 4.7% | 2.6% | 8 | 195 | pure holdout |
| selected_no_recycle | 2026 | 1.68% | 0.97% | 75% | 82% | PASS | 4.5% | 1.7% | 2.0% | 8 | 195 | pure holdout |
| selected_no_clock | 2026 | 2.62% | 0.85% | 62% | 93% | PASS | 5.6% | 4.5% | 2.7% | 8 | 195 | pure holdout |
| addonly_plain | 2026 | 1.89% | 0.97% | 75% | 86% | PASS | 5.1% | 2.5% | 1.9% | 8 | 195 | pure holdout |
| locked_clock_recycle | 2026 | 2.56% | 0.85% | 75% | 93% | PASS | 5.7% | 3.0% | 2.4% | 5 | 122 | pure holdout |
| locked_h4_055_cap15 | 2026 | 2.43% | 0.95% | 75% | 91% | PASS | 5.2% | 2.9% | 2.5% | 5 | 122 | pure holdout |
| locked_h4_055_k3_cap20 | 2026 | 2.56% | 0.85% | 75% | 93% | PASS | 5.7% | 3.0% | 2.4% | 5 | 122 | pure holdout |
| locked_oos_cap15 | 2026 | 2.41% | 0.94% | 75% | 90% | PASS | 5.5% | 3.3% | 2.6% | 5 | 122 | pure holdout |
| add_gbpjpy_plain | 2026 | 1.83% | 0.81% | 75% | 91% | PASS | 7.4% | 5.5% | 2.3% | 6 | 138 | pure holdout |
| add_gbpjpy_cap15 | 2026 | 1.97% | 0.78% | 62% | 93% | PASS | 7.1% | 5.1% | 2.8% | 6 | 138 | pure holdout |
| baseline_vt0025 | holdout_365d | 1.52% | 1.05% | 75% | 81% | PASS | 6.8% | 2.5% | 3.0% | 5 | 146 | pure holdout |
| selected | holdout_365d | 2.06% | 1.06% | 83% | 74% | PASS | 4.7% | 1.0% | 2.8% | 8 | 236 | pure holdout |
| selected_no_recycle | holdout_365d | 1.40% | 0.93% | 75% | 71% | PASS | 4.1% | 2.1% | 2.1% | 8 | 236 | pure holdout |
| selected_no_clock | holdout_365d | 1.96% | 0.68% | 67% | 84% | PASS | 4.4% | 0.9% | 2.9% | 8 | 236 | pure holdout |
| addonly_plain | holdout_365d | 1.49% | 0.97% | 75% | 73% | PASS | 4.7% | 2.7% | 2.4% | 8 | 236 | pure holdout |
| locked_clock_recycle | holdout_365d | 1.80% | 0.32% | 75% | 90% | PASS | 5.2% | 1.0% | 3.6% | 5 | 146 | pure holdout |
| locked_h4_055_cap15 | holdout_365d | 1.63% | 0.19% | 58% | 90% | PASS | 5.5% | 1.0% | 3.5% | 5 | 146 | pure holdout |
| locked_h4_055_k3_cap20 | holdout_365d | 1.80% | 0.32% | 75% | 90% | PASS | 5.2% | 1.0% | 3.6% | 5 | 146 | pure holdout |
| locked_oos_cap15 | holdout_365d | 1.49% | 0.44% | 58% | 89% | PASS | 6.4% | 2.2% | 3.6% | 5 | 146 | pure holdout |
| add_gbpjpy_plain | holdout_365d | 1.33% | 1.03% | 75% | 79% | PASS | 6.2% | 3.0% | 2.7% | 6 | 149 | pure holdout |
| add_gbpjpy_cap15 | holdout_365d | 1.28% | 0.48% | 58% | 87% | PASS | 6.8% | 3.7% | 3.6% | 6 | 149 | pure holdout |
| baseline_vt0025 | roll12_end | 1.52% | 1.05% | 75% | 81% | PASS | 6.8% | 2.5% | 3.0% | 5 | 146 | pure holdout |
| selected | roll12_end | 2.06% | 1.06% | 83% | 74% | PASS | 4.7% | 1.0% | 2.8% | 8 | 236 | pure holdout |
| selected_no_recycle | roll12_end | 1.40% | 0.93% | 75% | 71% | PASS | 4.1% | 2.1% | 2.1% | 8 | 236 | pure holdout |
| selected_no_clock | roll12_end | 1.96% | 0.68% | 67% | 84% | PASS | 4.4% | 0.9% | 2.9% | 8 | 236 | pure holdout |
| addonly_plain | roll12_end | 1.49% | 0.97% | 75% | 73% | PASS | 4.7% | 2.7% | 2.4% | 8 | 236 | pure holdout |
| locked_clock_recycle | roll12_end | 1.80% | 0.32% | 75% | 90% | PASS | 5.2% | 1.0% | 3.6% | 5 | 146 | pure holdout |
| locked_h4_055_cap15 | roll12_end | 1.63% | 0.19% | 58% | 90% | PASS | 5.5% | 1.0% | 3.5% | 5 | 146 | pure holdout |
| locked_h4_055_k3_cap20 | roll12_end | 1.80% | 0.32% | 75% | 90% | PASS | 5.2% | 1.0% | 3.6% | 5 | 146 | pure holdout |
| locked_oos_cap15 | roll12_end | 1.49% | 0.44% | 58% | 89% | PASS | 6.4% | 2.2% | 3.6% | 5 | 146 | pure holdout |
| add_gbpjpy_plain | roll12_end | 1.33% | 1.03% | 75% | 79% | PASS | 6.2% | 3.0% | 2.7% | 6 | 149 | pure holdout |
| add_gbpjpy_cap15 | roll12_end | 1.28% | 0.48% | 58% | 87% | PASS | 6.8% | 3.7% | 3.6% | 6 | 149 | pure holdout |
| baseline_vt0025 | roll12_m6 | 1.13% | 0.30% | 67% | 80% | PASS | 3.9% | 1.8% | 3.4% | 5 | 79 | mixed |
| selected | roll12_m6 | 0.45% | 0.24% | 67% | 79% | PASS | 5.3% | 2.6% | 2.9% | 8 | 146 | mixed |
| selected_no_recycle | roll12_m6 | 0.36% | 0.31% | 58% | 80% | PASS | 4.6% | 2.9% | 1.9% | 8 | 146 | mixed |
| selected_no_clock | roll12_m6 | 0.63% | 0.47% | 67% | 72% | PASS | 5.4% | 2.6% | 3.5% | 8 | 146 | mixed |
| addonly_plain | roll12_m6 | 0.35% | 0.23% | 58% | 80% | PASS | 5.1% | 3.4% | 2.4% | 8 | 146 | mixed |
| locked_clock_recycle | roll12_m6 | 1.44% | 0.86% | 75% | 73% | PASS | 4.6% | 1.3% | 4.4% | 5 | 79 | mixed |
| locked_h4_055_cap15 | roll12_m6 | 1.29% | 0.65% | 75% | 73% | PASS | 3.5% | 1.1% | 3.4% | 5 | 79 | mixed |
| locked_h4_055_k3_cap20 | roll12_m6 | 1.44% | 0.86% | 75% | 73% | PASS | 4.6% | 1.3% | 4.4% | 5 | 79 | mixed |
| locked_oos_cap15 | roll12_m6 | 1.43% | 0.61% | 75% | 76% | PASS | 3.6% | 1.7% | 4.1% | 5 | 79 | mixed |
| add_gbpjpy_plain | roll12_m6 | 0.55% | 0.20% | 58% | 83% | PASS | 5.0% | 3.2% | 2.8% | 6 | 107 | mixed |
| add_gbpjpy_cap15 | roll12_m6 | 0.67% | 0.25% | 58% | 89% | PASS | 4.8% | 2.9% | 3.3% | 6 | 107 | mixed |

## Honest verdict

- **baseline_vt0025**: 2024=0.42%/55%pos/top3=74% gates=True; 2025=1.63%/73%pos/top3=69% gates=True; 2026=2.30%/88%pos/top3=87% gates=True; HO=1.52%/75%pos/top3=81% gates=True
- **selected**: 2024=2.42%/91%pos/top3=59% gates=True; 2025=0.18%/55%pos/top3=74% gates=True; 2026=2.37%/75%pos/top3=86% gates=True; HO=2.06%/83%pos/top3=74% gates=True
- **selected_no_recycle**: 2024=1.21%/73%pos/top3=56% gates=True; 2025=0.45%/73%pos/top3=69% gates=True; 2026=1.68%/75%pos/top3=82% gates=True; HO=1.40%/75%pos/top3=71% gates=True
- **selected_no_clock**: 2024=2.15%/82%pos/top3=62% gates=True; 2025=0.37%/64%pos/top3=62% gates=True; 2026=2.62%/62%pos/top3=93% gates=True; HO=1.96%/67%pos/top3=84% gates=True
- **addonly_plain**: 2024=1.15%/82%pos/top3=61% gates=True; 2025=0.53%/73%pos/top3=68% gates=True; 2026=1.89%/75%pos/top3=86% gates=True; HO=1.49%/75%pos/top3=73% gates=True
- **locked_clock_recycle**: 2024=1.18%/73%pos/top3=71% gates=True; 2025=1.86%/73%pos/top3=73% gates=True; 2026=2.56%/75%pos/top3=93% gates=True; HO=1.80%/75%pos/top3=90% gates=True
- **locked_h4_055_cap15**: 2024=0.97%/73%pos/top3=67% gates=True; 2025=1.63%/82%pos/top3=72% gates=True; 2026=2.43%/75%pos/top3=91% gates=True; HO=1.63%/58%pos/top3=90% gates=True
- **locked_h4_055_k3_cap20**: 2024=1.18%/73%pos/top3=71% gates=True; 2025=1.86%/73%pos/top3=73% gates=True; 2026=2.56%/75%pos/top3=93% gates=True; HO=1.80%/75%pos/top3=90% gates=True
- **locked_oos_cap15**: 2024=0.86%/73%pos/top3=76% gates=True; 2025=1.93%/82%pos/top3=70% gates=True; 2026=2.41%/75%pos/top3=90% gates=True; HO=1.49%/58%pos/top3=89% gates=True
- **add_gbpjpy_plain**: 2024=0.97%/73%pos/top3=69% gates=True; 2025=0.85%/73%pos/top3=62% gates=True; 2026=1.83%/75%pos/top3=91% gates=True; HO=1.33%/75%pos/top3=79% gates=True
- **add_gbpjpy_cap15**: 2024=1.55%/82%pos/top3=67% gates=True; 2025=0.79%/73%pos/top3=60% gates=True; 2026=1.97%/62%pos/top3=93% gates=True; HO=1.28%/58%pos/top3=87% gates=True

### vs ≥1%/mo & ≥65–70% pos & smoother top3

- Baseline locked+vt: 2024 mo=0.42%; HO mo=1.52% top3=81%
- Selected: 2024 mo=2.42%; HO mo=2.06% pos=83% top3=74%
- HO ≥1% & ≥65% pos & gates: **YES**
- 2024 ~1%: **YES**
- 2025 ~1% (confirmation): **NO**
- Materially smoother vs baseline: **YES**; better 2024: **YES**
- **Overall target met:** **NO**
- **Promote over locked candidate:** **NO**

## Methodology
- Per-window BT + 250-bar warmup; FTMO static 10% / daily 5% Europe/Prague.
- Dual-confirm = meaningful OOS (trades≥10, gates, sharpe>0.3) on **both** H4 and D1.
- Add-only: unused dual symbols, 2024 mean_mo>0 + gates; corr vs locked 2024 daily returns.
- Clock budgets: within-H4 / within-D1 relative OOS-Sharpe, then H4/D1 share.
- Max-concurrent + recycle: lagged occupancy, a-priori priority, exposure ≤ 1.
- Yahoo ≠ FTMO.

## Takeaways

- **2024-positive dual add-only overfits the research year.** GBPJPY H4 squeeze (2024 +1.42%/mo, corr −0.30) plus EURGBP D1 / EURAUD H4 lifts 2024 to 1.15–2.42%/mo but **2025 falls to 0.18–0.53%/mo**. Holdout looks fine only because 2026 is strong. **Not promoted.**
- **Clock + causal recycle on the locked FX4+GBPCAD legs** (`h4_share=0.55`, `max_k=3`, `recycle_cap=2`, `vt=0.0025`) is the only idea that keeps **every calendar year ≥~1%/mo** with ≥73% positive months and gates PASS: 2024 **1.18%**, 2025 **1.86%**, 2026 **2.56%**, HO **1.80%**. Exposure clipped to 1 (idle weight recycled, not extra leverage). **HO top3 90% vs locked 81% — burstiness worse, so not promoted over the locked candidate.**
- Conservative recycle (`cap=1.5`) gets 2024 to 0.86–0.97%/mo but holdout %pos drops to 58%.
- New-strat WF restricted to dual-confirm symbols (this wave): remaining EURAUD/EURCAD/EURGBP/GBPCAD/NZDCAD; prior board already covered 13 duals. Carry still ~0 trades; tsmom/kalman OOS mixed and mostly holdout-fail.

Yahoo ≠ FTMO.

## New-strat WF remaining duals (this wave)

EURAUD / EURCAD / EURGBP / GBPCAD / NZDCAD × {tsmom, vol_breakout, carry_proxy, kalman_trend, session_orb} × {H4, D1} = 50. Merged board 674 rows.

Meaningful OOS did **not** hold out: GBPCAD H4 kalman, EURCAD D1 vol_breakout, NZDCAD D1 kalman, EURGBP H4 kalman. Carry ~0 trades. Not added to the basket (and GBPCAD is already a locked D1 leg).
