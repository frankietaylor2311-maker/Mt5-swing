# Quest wave — regime / diversification overlays

**data_source:** `approximate_non_ftmo` — never golive without FTMO MT5 exports.
**risk_fraction:** 8%  **warmup:** 250  **signal_lag=1**  **per-window BT**.
**Selection:** overlay hyperparams on **2024 only**; holdout confirmation only.

## Ideas

1. **regime_sleeves** — MR vs Trend H4 sleeves; EURUSD ADX gate (full capital mix).
2. **corr_throttle** — scale locked FX4+GBPCAD-D1 when peer-corr high; +vt.
3. **hotstreak** — anti-chase cool/heat on trailing portfolio returns (+vt base).
4. **roll_is_sharpe** — causal rolling IS Sharpe weights.
5. **adx_tilt** — soft MR/TR weight tilt on locked basket by EURUSD ADX (+vt).

## Window table

| Idea | Window | Mean mo | Med mo | %pos | Top3 | Gates | P2T | Daily | Trades | Overlap |
|------|--------|--------:|-------:|-----:|-----:|:-----:|----:|------:|-------:|---------|
| baseline_vt0025 | 2024 | 0.42% | 0.12% | 55% | 74% | PASS | 7.4% | 2.9% | 130 | research/WF |
| regime_sleeves | 2024 | 0.66% | 0.37% | 55% | 69% | PASS | 4.0% | 1.1% | 257 | research/WF |
| corr_throttle | 2024 | 0.47% | 0.17% | 55% | 73% | PASS | 7.2% | 2.8% | 130 | research/WF |
| hotstreak | 2024 | 0.51% | 0.41% | 64% | 72% | PASS | 8.0% | 2.9% | 130 | research/WF |
| roll_is_sharpe | 2024 | 0.24% | 0.17% | 64% | 75% | PASS | 8.7% | 3.1% | 130 | research/WF |
| baseline_vt0025 | 2025 | 1.63% | 1.00% | 73% | 69% | PASS | 3.9% | 3.6% | 74 | mixed |
| regime_sleeves | 2025 | 0.03% | 0.38% | 64% | 69% | PASS | 6.7% | 1.2% | 258 | mixed |
| corr_throttle | 2025 | 1.52% | 1.18% | 73% | 66% | PASS | 3.8% | 2.9% | 74 | mixed |
| hotstreak | 2025 | 1.63% | 1.19% | 73% | 68% | PASS | 3.9% | 3.9% | 74 | mixed |
| roll_is_sharpe | 2025 | 1.39% | 0.26% | 64% | 75% | PASS | 7.8% | 3.3% | 74 | mixed |
| baseline_vt0025 | 2026 | 2.30% | 1.34% | 88% | 87% | PASS | 6.4% | 2.5% | 122 | pure holdout |
| regime_sleeves | 2026 | 0.32% | 0.29% | 75% | 90% | PASS | 4.8% | 1.3% | 149 | pure holdout |
| corr_throttle | 2026 | 2.30% | 1.43% | 88% | 86% | PASS | 6.3% | 2.5% | 122 | pure holdout |
| hotstreak | 2026 | 2.26% | 1.28% | 88% | 86% | PASS | 6.3% | 2.5% | 122 | pure holdout |
| roll_is_sharpe | 2026 | 0.01% | -0.35% | 38% | 100% | PASS | 6.6% | 3.1% | 122 | pure holdout |
| baseline_vt0025 | holdout_365d | 1.52% | 1.05% | 75% | 81% | PASS | 6.8% | 3.0% | 146 | pure holdout |
| regime_sleeves | holdout_365d | -0.04% | 0.04% | 58% | 77% | PASS | 4.7% | 0.9% | 143 | pure holdout |
| corr_throttle | holdout_365d | 1.52% | 1.08% | 75% | 79% | PASS | 6.5% | 3.0% | 146 | pure holdout |
| hotstreak | holdout_365d | 1.56% | 0.93% | 75% | 78% | PASS | 6.5% | 3.4% | 146 | pure holdout |
| roll_is_sharpe | holdout_365d | -0.26% | -0.63% | 25% | 100% | PASS | 8.6% | 1.3% | 146 | pure holdout |
| baseline_vt0025 | roll12_end | 1.52% | 1.05% | 75% | 81% | PASS | 6.8% | 3.0% | 146 | pure holdout |
| regime_sleeves | roll12_end | -0.04% | 0.04% | 58% | 77% | PASS | 4.7% | 0.9% | 143 | pure holdout |
| corr_throttle | roll12_end | 1.52% | 1.08% | 75% | 79% | PASS | 6.5% | 3.0% | 146 | pure holdout |
| hotstreak | roll12_end | 1.56% | 0.93% | 75% | 78% | PASS | 6.5% | 3.4% | 146 | pure holdout |
| roll_is_sharpe | roll12_end | -0.26% | -0.63% | 25% | 100% | PASS | 8.6% | 1.3% | 146 | pure holdout |
| baseline_vt0025 | roll12_m6 | 1.13% | 0.30% | 67% | 80% | PASS | 3.9% | 3.4% | 79 | mixed |
| regime_sleeves | roll12_m6 | 0.17% | 0.18% | 75% | 68% | PASS | 4.8% | 1.0% | 215 | mixed |
| corr_throttle | roll12_m6 | 1.02% | 0.31% | 67% | 78% | PASS | 3.9% | 2.8% | 79 | mixed |
| hotstreak | roll12_m6 | 1.09% | 0.30% | 67% | 79% | PASS | 3.9% | 3.7% | 79 | mixed |
| roll_is_sharpe | roll12_m6 | 0.53% | -0.05% | 50% | 96% | PASS | 7.2% | 3.2% | 79 | mixed |
| adx_tilt | 2024 | 0.46% | 0.11% | 55% | 74% | PASS | 7.2% | 2.7% | 130 | research/WF |
| adx_tilt | 2025 | 1.51% | 1.10% | 73% | 67% | PASS | 3.8% | 3.0% | 74 | mixed |
| adx_tilt | 2026 | 2.37% | 1.47% | 88% | 85% | PASS | 6.3% | 2.6% | 122 | pure holdout |
| adx_tilt | holdout_365d | 1.58% | 1.07% | 75% | 79% | PASS | 6.5% | 2.9% | 146 | pure holdout |
| adx_tilt | roll12_end | 1.58% | 1.07% | 75% | 79% | PASS | 6.5% | 2.9% | 146 | pure holdout |
| adx_tilt | roll12_m6 | 0.96% | 0.29% | 67% | 79% | PASS | 4.4% | 2.8% | 79 | mixed |

## Honest verdict

- **regime_sleeves**: 2024=0.66%/55%pos; 2025=0.03%; 2026=0.32%; HO=-0.04%/58%pos/top3=77% gates=True
- **corr_throttle**: 2024=0.47%/55%pos; 2025=1.52%; 2026=2.30%; HO=1.52%/75%pos/top3=79% gates=True
- **hotstreak**: 2024=0.51%/64%pos; 2025=1.63%; 2026=2.26%; HO=1.56%/75%pos/top3=78% gates=True
- **roll_is_sharpe**: 2024=0.24%/64%pos; 2025=1.39%; 2026=0.01%; HO=-0.26%/25%pos/top3=100% gates=True
- **adx_tilt**: 2024=0.46%/55%pos; 2025=1.51%; 2026=2.37%; HO=1.58%/75%pos/top3=79% gates=True
- **baseline_vt0025**: 2024=0.42%/55%pos; 2025=1.63%; 2026=2.30%; HO=1.52%/75%pos/top3=81% gates=True

### vs ≥1%/mo & ≥65–70% pos & smoother

- Best new by HO: **adx_tilt** (HO mo=1.58%, pos=75%, top3=79%; 2024 mo=0.46%)
- Baseline locked+vt: HO mo=1.52% top3=81%; 2024 mo=0.42%
- HO ≥1% & ≥65% pos & gates: **YES**
- 2024 ~1%: **NO**
- Burstiness fixed (top3≤55%): **NO**
- **Overall target met: NO**
- **Promote over locked candidate: NO** — hotstreak/adx_tilt are only marginal vs baseline; regime sleeves overfit 2024 and fail HO.

## Takeaways

- Full sleeve switching (Idea A) improves research-year 2024 but **destroys holdout** (trend overweight).
- Soft overlays (corr / hotstreak / ADX tilt) preserve HO ~1.5%/mo but only nudge 2024 (~0.42→~0.51%) and top3 (~81→~78%).
- Gap remains: **independent alpha in 2024-like regimes** + **uncorrelated legs** to cut burstiness — not more leverage.

Yahoo ≠ FTMO.

