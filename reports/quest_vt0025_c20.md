# One-pct month quest — vt0025_c20

**data_source:** approximate_non_ftmo (no ftmo exports in data/ftmo/)
**risk_fraction:** 8.0%  **max_lot:** 50.0  **warmup:** 250  **weights:** oos_sharpe
**legs:** 5

| Window | Return | Mean mo | %pos mo | Top3 share | Gates | P2T DD | Overlap |
|--------|-------:|--------:|--------:|-----------:|:-----:|-------:|---------|
| 2024 | 4.12% | 0.28% | 64% | 71% | PASS | 6.22% | research/WF overlap |
| 2025 | 14.31% | 1.09% | 73% | 67% | PASS | 3.21% | mixed research→holdout |
| 2026 | 15.22% | 1.76% | 88% | 85% | PASS | 4.52% | pure holdout |
| holdout_365d | 17.29% | 1.27% | 83% | 79% | PASS | 4.57% | pure holdout |
| roll12_end | 17.29% | 1.27% | 83% | 79% | PASS | 4.57% | pure holdout |
| roll12_m6 | 11.40% | 0.75% | 67% | 79% | PASS | 3.35% | mixed |

## vs 1%/mo target

- Holdout mean monthly: **1.27%** (target ≥1.0%) — gap **-0.27 pp**
- Holdout % positive months: **83%** (prefer ≥70%)
- Holdout top-3 share of gains: **79%** (prefer ≤50%)
- Holdout gates: **PASS**
- **Target met (relaxed):** YES
