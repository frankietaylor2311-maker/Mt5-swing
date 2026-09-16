# HO-robust consistency (locked sleeve)

**data_source:** `approximate_non_ftmo`  **RF:** 8%  **PORT_VOL_TARGET:** 0.0025  **signal_lag:** 1
**Locked tag (unchanged unless promote):** `fx4plus_gbpcad_d1_voltarget_0025`
**IS windows:** ['2024', '2025_IS']  **Confirm:** ['2025', '2026', 'holdout_365d']
**Selection:** nested on 2024 (weights: max min(2024,2025_IS)); HO never for selection.
**Families:** baseline / monthly_budget_vt / daily_budget_vt / equity_tsmom / dense wrebal / wrebal+mtd / wrebal+budget / optional combos. hi≤1.

## Board

| Family | N | Soft IS | Hard IS | Promote |
|--------|--:|--------:|--------:|--------:|
| baseline | 1 | 0 | 0 | 0 |
| daily_budget_vt | 1 | 0 | 0 | 0 |
| equity_tsmom | 1 | 0 | 0 | 0 |
| monthly_budget_vt | 1 | 0 | 0 | 0 |
| weight_rebalance | 1 | 0 | 0 | 0 |
| wrebal_combo_two | 1 | 1 | 0 | 0 |
| wrebal_plus_daily_budget | 1 | 0 | 0 | 0 |
| wrebal_plus_monthly_budget | 1 | 0 | 0 | 0 |
| wrebal_plus_mtd | 1 | 1 | 0 | 0 |

**Totals:** soft=2 hard=0 promote=0 (board n=9)

**2024 monthly_budget pick:** {'target_mo': 0.015, 'lookback_months': 4, 'assumed_sharpe': 0.8}
**2024 daily_budget pick:** {'target_mo': 0.008, 'assumed_sharpe': 0.8, 'look': 42}
**2024 equity_tsmom pick:** {'lookback': 42, 'neg_scale': 0.2, 'pos_scale': 0.85, 'flat_band': 0.0}
**IS weight rebalance:** [0.3282, 0.0291, 0.2413, 0.1776, 0.2238] score=(0.006947717475139892, 0.75, -0.6958783537057224)
**wrebal+MTD pick:** {'tau': 0.015, 'after': 0.5}
**wrebal+monthly_budget pick:** {'target_mo': 0.015, 'lookback_months': 6, 'assumed_sharpe': 0.8}
**wrebal+daily_budget pick:** {'target_mo': 0.008, 'assumed_sharpe': 0.8, 'look': 42}

## Soft / hard / promote

**No promote.**

**Best soft IS:**

### `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5` (wrebal_plus_mtd) — note=holdout_fail

| Window | Mean mo | %pos | Top3 |
|--------|--------:|-----:|-----:|
| 2024 (IS) | 1.20% | 82% | 45% |
| 2025_IS | 1.74% | 75% | 64% |
| 2025 (cal) | 1.11% | 73% | 61% |
| 2026 | 2.41% | 75% | 80% |
| holdout | 1.69% | 67% | 71% |

soft=True hard=False promote=False min_is_mo=1.20% max_is_top3=64%

### Top-8 by min_is_mo (diagnostic)

| idea | family | min_is_mo | pos24 | pos25is | top3_max | note |
|------|--------|----------:|------:|--------:|---------:|------|
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5` | wrebal_plus_mtd | 1.20% | 82% | 75% | 64% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mb_tm0.015_lb6_sh0.8+mtd_t0.015_a0.5` | wrebal_combo_two | 1.13% | 82% | 75% | 64% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+db_tm0.008_sh0.8_lk42` | wrebal_plus_daily_budget | 0.69% | 82% | 75% | 70% | soft_mean_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224` | weight_rebalance | 0.69% | 82% | 75% | 70% | soft_mean_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mb_tm0.015_lb6_sh0.8` | wrebal_plus_monthly_budget | 0.60% | 82% | 75% | 67% | soft_mean_fail |
| `lock_db_tm0.008_sh0.8_lk42` | daily_budget_vt | 0.42% | 55% | 75% | 74% | is_fail |
| `baseline_locked` | baseline | 0.42% | 55% | 75% | 74% | is_fail |
| `lock_mb_tm0.015_lb4_sh0.8` | monthly_budget_vt | 0.38% | 55% | 75% | 73% | is_fail |

## Verdict

- **Promote:** NO
- Locked tag remains `fx4plus_gbpcad_d1_voltarget_0025`
- Still blocked on FTMO CSVs: **YES** (`data/ftmo/` empty of CSVs)

Artifacts: `reports/quest_ho_robust_board.csv`, `reports/quest_ho_robust_consistency.md`, `configs/quest_ho_robust_selected.json`

