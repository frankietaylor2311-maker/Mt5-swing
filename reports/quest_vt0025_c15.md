# One-pct month quest — vt0025_c15

**data_source:** approximate_non_ftmo (no ftmo exports in data/ftmo/)
**risk_fraction:** 8.0%  **max_lot:** 50.0  **warmup:** 250  **weights:** oos_sharpe
**legs:** 5

| Window | Return | Mean mo | %pos mo | Top3 share | Gates | P2T DD | Overlap |
|--------|-------:|--------:|--------:|-----------:|:-----:|-------:|---------|
| 2024 | 3.62% | 0.25% | 73% | 67% | PASS | 5.16% | research/WF overlap |
| 2025 | 10.54% | 0.82% | 73% | 67% | PASS | 2.42% | mixed research→holdout |
| 2026 | 11.89% | 1.37% | 75% | 87% | PASS | 3.38% | pure holdout |
| holdout_365d | 13.31% | 1.00% | 75% | 80% | PASS | 3.37% | pure holdout |
| roll12_end | 13.31% | 1.00% | 75% | 80% | PASS | 3.37% | pure holdout |
| roll12_m6 | 8.55% | 0.56% | 67% | 79% | PASS | 2.53% | mixed |

## vs 1%/mo target

- Holdout mean monthly: **1.00%** (target ≥1.0%) — gap **0.00 pp**
- Holdout % positive months: **75%** (prefer ≥70%)
- Holdout top-3 share of gains: **80%** (prefer ≤50%)
- Holdout gates: **PASS**
- **Target met (relaxed):** YES
