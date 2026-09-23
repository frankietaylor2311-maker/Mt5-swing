# One-pct month quest — locked_verify_ppi

**data_source:** approximate_non_ftmo (no ftmo exports in data/ftmo/)
**risk_fraction:** 7.5%  **max_lot:** 50.0  **warmup:** 250  **weights:** locked
**legs:** 5

| Window | Return | Mean mo | %pos mo | Top3 share | Gates | P2T DD | Overlap |
|--------|-------:|--------:|--------:|-----------:|:-----:|-------:|---------|
| 2024 | 5.29% | 0.40% | 55% | 75% | PASS | 7.15% | research/WF overlap |
| 2025 | 30.55% | 2.25% | 82% | 60% | PASS | 5.02% | mixed research→holdout |
| 2026 | 19.33% | 2.22% | 88% | 86% | PASS | 6.29% | pure holdout |
| holdout_365d | 21.10% | 1.50% | 75% | 80% | PASS | 6.63% | pure holdout |
| roll12_end | 21.10% | 1.50% | 75% | 80% | PASS | 6.63% | pure holdout |
| roll12_m6 | 21.39% | 1.49% | 83% | 64% | PASS | 6.36% | mixed |

## vs 1%/mo target

- Holdout mean monthly: **1.50%** (target ≥1.0%) — gap **-0.50 pp**
- Holdout % positive months: **75%** (prefer ≥70%)
- Holdout top-3 share of gains: **80%** (prefer ≤50%)
- Holdout gates: **PASS**
- **Target met (relaxed):** YES
