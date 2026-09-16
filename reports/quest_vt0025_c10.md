# One-pct month quest — vt0025_c10

**data_source:** approximate_non_ftmo (no ftmo exports in data/ftmo/)
**risk_fraction:** 8.0%  **max_lot:** 50.0  **warmup:** 250  **weights:** oos_sharpe
**legs:** 5

| Window | Return | Mean mo | %pos mo | Top3 share | Gates | P2T DD | Overlap |
|--------|-------:|--------:|--------:|-----------:|:-----:|-------:|---------|
| 2024 | 2.57% | 0.18% | 73% | 67% | PASS | 3.47% | research/WF overlap |
| 2025 | 6.67% | 0.54% | 73% | 66% | PASS | 1.61% | mixed research→holdout |
| 2026 | 8.54% | 0.99% | 75% | 90% | PASS | 2.26% | pure holdout |
| holdout_365d | 9.28% | 0.72% | 75% | 82% | PASS | 2.25% | pure holdout |
| roll12_end | 9.28% | 0.72% | 75% | 82% | PASS | 2.25% | pure holdout |
| roll12_m6 | 5.72% | 0.37% | 67% | 79% | PASS | 1.69% | mixed |

## vs 1%/mo target

- Holdout mean monthly: **0.72%** (target ≥1.0%) — gap **0.28 pp**
- Holdout % positive months: **75%** (prefer ≥70%)
- Holdout top-3 share of gains: **82%** (prefer ≤50%)
- Holdout gates: **PASS**
- **Target met (relaxed):** NO
