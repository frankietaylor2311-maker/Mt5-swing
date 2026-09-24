# One-pct month quest — locked_verify_gfcf

**data_source:** approximate_non_ftmo (no ftmo exports in data/ftmo/)
**risk_fraction:** 8.0%  **max_lot:** 50.0  **warmup:** 250  **weights:** oos_sharpe
**legs:** 5

| Window | Return | Mean mo | %pos mo | Top3 share | Gates | P2T DD | Overlap |
|--------|-------:|--------:|--------:|-----------:|:-----:|-------:|---------|
| 2024 | 5.38% | 0.42% | 55% | 74% | PASS | 7.42% | research/WF overlap |
| 2025 | 22.04% | 1.63% | 73% | 69% | PASS | 3.86% | mixed research→holdout |
| 2026 | 20.03% | 2.30% | 88% | 87% | PASS | 6.38% | pure holdout |
| holdout_365d | 21.55% | 1.52% | 75% | 81% | PASS | 6.83% | pure holdout |

Locked sleeve `fx4plus_gbpcad_d1_voltarget_0025` **untouched** — config sha `bafed8eff5aa7541a42b99a486def2402f530b288a1ad6e7d273a56b2feb1790` unchanged (§64 GFCF investment wave).
