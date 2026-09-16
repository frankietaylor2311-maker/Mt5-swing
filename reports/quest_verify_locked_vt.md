# One-pct month quest — verify_locked_vt

**data_source:** approximate_non_ftmo (no ftmo exports in data/ftmo/)
**risk_fraction:** 8.0%  **max_lot:** 50.0  **warmup:** 250  **weights:** locked
**legs:** 5

| Window | Return | Mean mo | %pos mo | Top3 share | Gates | P2T DD | Overlap |
|--------|-------:|--------:|--------:|-----------:|:-----:|-------:|---------|
| 2024 | 4.62% | 0.36% | 55% | 78% | PASS | 7.58% | research/WF overlap |
| 2025 | 24.70% | 1.84% | 73% | 68% | PASS | 4.21% | mixed research→holdout |
| 2026 | 21.12% | 2.43% | 75% | 88% | PASS | 7.05% | pure holdout |
| holdout_365d | 21.57% | 1.54% | 67% | 84% | PASS | 7.29% | pure holdout |
| roll12_end | 21.57% | 1.54% | 67% | 84% | PASS | 7.29% | pure holdout |
| roll12_m6 | 17.69% | 1.24% | 67% | 82% | PASS | 3.84% | mixed |

## vs 1%/mo target

- Holdout mean monthly: **1.54%** (target ≥1.0%) — gap **-0.54 pp**
- Holdout % positive months: **67%** (prefer ≥70%)
- Holdout top-3 share of gains: **84%** (prefer ≤50%)
- Holdout gates: **PASS**
- **Target met (relaxed):** YES
