# Loss-halt / after-loss consistency (locked sleeve)

**data_source:** `approximate_non_ftmo`  **RF:** 8%  **PORT_VOL_TARGET:** 0.0025  **signal_lag:** 1
**Locked tag (unchanged unless promote):** `fx4plus_gbpcad_d1_voltarget_0025`
**IS windows:** ['2024', '2025_IS']  **Confirm:** ['2025', '2026', 'holdout_365d']
**Selection:** nested on 2024 only; HO never for selection.
**Families:** baseline / mtd_loss_halt / after_loss_throttle / combos (lh+al, lh+mtd, al+mtd) / prior mtd±lh. hi≤1.
**Soft mean floor:** 0.0095 on BOTH 2024 and 2025_IS.

## Board

| Family | N | Soft IS | Hard IS | Promote |
|--------|--:|--------:|--------:|--------:|
| after_loss_throttle | 1 | 0 | 0 | 0 |
| baseline | 1 | 0 | 0 | 0 |
| combo_al_mtd | 1 | 0 | 0 | 0 |
| combo_al_mtd_prior | 1 | 0 | 0 | 0 |
| combo_lh_al | 1 | 0 | 0 | 0 |
| combo_lh_mtd | 1 | 1 | 0 | 0 |
| combo_lh_mtd_prior | 1 | 1 | 0 | 0 |
| combo_prior_mtd_lh | 1 | 1 | 0 | 0 |
| mtd_gain_clip_prior | 1 | 1 | 0 | 0 |
| mtd_loss_halt | 3 | 0 | 0 | 0 |

**Totals:** soft=4 hard=0 promote=0 (board n=12)

**2024 loss_halt pick:** {'tau': 0.02, 'after_halt': 0.0} (sc=-0.7406)
**2024 after_loss pick:** {'after_loss': 0.25}
**lh+mtd nested pick:** {'tau': 0.015, 'after': 0.0}
**al+mtd nested pick:** {'tau': 0.015, 'after': 0.5}
**prior_mtd+lh nested pick:** {'tau': 0.02, 'after_halt': 0.0}

## Soft / hard / promote

**No promote.**

**Best soft IS:**

### `lock_mtd_t0.015_a0.0+lh_t0.02_ah0.0` (combo_prior_mtd_lh) — note=holdout_fail

| Window | Mean mo | %pos | Top3 |
|--------|--------:|-----:|-----:|
| 2024 (IS) | 1.05% | 73% | 52% |
| 2025_IS | 1.37% | 75% | 59% |
| 2025 (cal) | 0.79% | 73% | 56% |
| 2026 | 1.89% | 88% | 81% |
| holdout | 0.89% | 67% | 74% |

soft=True hard=False promote=False min_is_mo=1.05% max_is_top3=59%

### Top-8 by min_is_mo (diagnostic)

| idea | family | min_is_mo | pos24 | pos25is | top3_max | note |
|------|--------|----------:|------:|--------:|---------:|------|
| `lock_mtd_t0.015_a0.0+lh_t0.02_ah0.0` | combo_prior_mtd_lh | 1.05% | 73% | 75% | 59% | holdout_fail |
| `lock_lh_t0.02_ah0.0+mtd_t0.015_a0.0_prior` | combo_lh_mtd_prior | 1.02% | 73% | 75% | 59% | holdout_fail |
| `lock_lh_t0.02_ah0.0+mtd_t0.015_a0.0` | combo_lh_mtd | 1.02% | 73% | 75% | 59% | holdout_fail |
| `lock_mtd_t0.015_a0.0_prior_alone` | mtd_gain_clip_prior | 0.99% | 73% | 75% | 59% | year_clear_fail |
| `lock_al_0.25+mtd_t0.015_a0.0_prior` | combo_al_mtd_prior | 0.70% | 73% | 75% | 70% | soft_mean_fail |
| `lock_lh_t0.02_ah0.0` | mtd_loss_halt | 0.46% | 55% | 75% | 74% | is_fail |
| `lock_lh_t0.02_ah0.25_r2` | mtd_loss_halt | 0.45% | 55% | 75% | 74% | is_fail |
| `lock_lh_t0.03_ah0.0_r3` | mtd_loss_halt | 0.45% | 55% | 75% | 74% | is_fail |

## Verdict

- **Promote:** NO
- Locked tag remains `fx4plus_gbpcad_d1_voltarget_0025`
- Still blocked on FTMO CSVs: **YES** (`data/ftmo/` empty of CSVs)

Artifacts: `reports/quest_loss_halt_board.csv`, `reports/quest_loss_halt_consistency.md`, `configs/quest_loss_halt_selected.json`

