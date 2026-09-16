# One-pct month quest — baseline_fx4

**data_source:** approximate_non_ftmo (no ftmo exports in data/ftmo/)
**risk_fraction:** 7.5%  **max_lot:** 50.0  **warmup:** 250  **weights:** locked
**legs:** 4

| Window | Return | Mean mo | %pos mo | Top3 share | Gates | P2T DD | Overlap |
|--------|-------:|--------:|--------:|-----------:|:-----:|-------:|---------|
| 2024 | 2.67% | 0.19% | 64% | 68% | PASS | 5.63% | research/WF overlap |
| 2025 | 11.73% | 0.91% | 55% | 66% | PASS | 2.87% | mixed research→holdout |
| 2026 | 14.56% | 1.68% | 75% | 94% | PASS | 3.48% | pure holdout |
| holdout_365d | 1.47% | 0.04% | 50% | 87% | PASS | 4.22% | pure holdout |
| roll12_end | 1.47% | 0.04% | 50% | 87% | PASS | 4.22% | pure holdout |
| roll12_m6 | 8.50% | 0.57% | 42% | 83% | PASS | 2.81% | mixed |

## vs 1%/mo target

- Holdout mean monthly: **0.04%** (target ≥1.0%) — gap **0.96 pp**
- Holdout % positive months: **50%** (prefer ≥70%)
- Holdout top-3 share of gains: **87%** (prefer ≤50%)
- Holdout gates: **PASS**
- **Target met (relaxed):** NO
