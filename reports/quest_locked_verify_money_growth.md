# One-pct month quest — locked_verify_money_growth

**data_source:** approximate_non_ftmo (no ftmo exports in data/ftmo/)
**risk_fraction:** 8.0%  **max_lot:** 50.0  **warmup:** 250  **weights:** oos_sharpe
**legs:** 5

| Window | Return | Mean mo | %pos mo | Top3 share | Gates | P2T DD | Overlap |
|--------|-------:|--------:|--------:|-----------:|:-----:|-------:|---------|
| 2024 | 5.38% | 0.42% | 55% | 74% | PASS | 7.42% | research/WF overlap |
| 2025 | 22.04% | 1.63% | 73% | 69% | PASS | 3.86% | mixed research→holdout |
| 2026 | 20.03% | 2.30% | 88% | 87% | PASS | 6.38% | pure holdout |
| holdout_365d | 21.55% | 1.52% | 75% | 81% | PASS | 6.83% | pure holdout |
| roll12_end | 21.55% | 1.52% | 75% | 81% | PASS | 6.83% | pure holdout |
| roll12_m6 | 16.45% | 1.13% | 67% | 80% | PASS | 3.87% | mixed |

## vs 1%/mo target

- Holdout mean monthly: **1.52%** (target ≥1.0%) — gap **-0.52 pp**
- Holdout % positive months: **75%** (prefer ≥70%)
- Holdout top-3 share of gains: **81%** (prefer ≤50%)
- Holdout gates: **PASS**
- **Target met (relaxed):** YES
