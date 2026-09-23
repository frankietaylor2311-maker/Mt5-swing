# One-pct month quest — locked_verify_cap_util

**data_source:** approximate_non_ftmo (no ftmo exports in data/ftmo/)
**risk_fraction:** 7.5%  **max_lot:** 50.0  **warmup:** 250  **weights:** locked
**legs:** 5

| Window | Return | Mean mo | %pos mo | Top3 share | Gates | P2T DD | Overlap |
|--------|-------:|--------:|--------:|-----------:|:-----:|-------:|---------|
| 2024 | 4.43% | 0.35% | 55% | 78% | PASS | 7.50% | research/WF overlap |
| 2025 | 33.17% | 2.44% | 73% | 60% | PASS | 5.23% | mixed research→holdout |
| 2026 | 20.63% | 2.37% | 75% | 87% | PASS | 6.99% | pure holdout |
| holdout_365d | 21.31% | 1.51% | 67% | 83% | PASS | 7.14% | pure holdout |
| roll12_end | 21.31% | 1.51% | 67% | 83% | PASS | 7.14% | pure holdout |
| roll12_m6 | 21.75% | 1.54% | 67% | 63% | PASS | 6.22% | mixed |

## vs 1%/mo target

- Holdout mean monthly: **1.51%** (target ≥1.0%) — gap **-0.51 pp**
- Holdout % positive months: **67%** (prefer ≥70%)
- Holdout top-3 share of gains: **83%** (prefer ≤50%)
- Holdout gates: **PASS**
- **Target met (relaxed):** YES
