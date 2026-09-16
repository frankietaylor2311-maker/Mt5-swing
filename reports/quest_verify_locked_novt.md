# One-pct month quest — verify_locked_novt

**data_source:** approximate_non_ftmo (no ftmo exports in data/ftmo/)
**risk_fraction:** 8.0%  **max_lot:** 50.0  **warmup:** 250  **weights:** locked
**legs:** 5

| Window | Return | Mean mo | %pos mo | Top3 share | Gates | P2T DD | Overlap |
|--------|-------:|--------:|--------:|-----------:|:-----:|-------:|---------|
| 2024 | 2.33% | 0.16% | 64% | 71% | PASS | 3.93% | research/WF overlap |
| 2025 | 7.43% | 0.60% | 73% | 65% | PASS | 1.91% | mixed research→holdout |
| 2026 | 9.97% | 1.16% | 62% | 91% | PASS | 2.68% | pure holdout |
| holdout_365d | 10.75% | 0.83% | 67% | 85% | PASS | 2.55% | pure holdout |
| roll12_end | 10.75% | 0.83% | 67% | 85% | PASS | 2.55% | pure holdout |
| roll12_m6 | 6.02% | 0.40% | 67% | 80% | PASS | 1.93% | mixed |

## vs 1%/mo target

- Holdout mean monthly: **0.83%** (target ≥1.0%) — gap **0.17 pp**
- Holdout % positive months: **67%** (prefer ≥70%)
- Holdout top-3 share of gains: **85%** (prefer ≤50%)
- Holdout gates: **PASS**
- **Target met (relaxed):** NO
