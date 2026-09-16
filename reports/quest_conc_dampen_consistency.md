# Trailing gain-concentration dampen consistency (locked sleeve)

**data_source:** `approximate_non_ftmo`  **RF:** 8%  **PORT_VOL_TARGET:** 0.0025  **signal_lag:** 1
**Locked tag (unchanged unless promote):** `fx4plus_gbpcad_d1_voltarget_0025`
**IS windows:** ['2024', '2025_IS']  **Confirm:** ['2025', '2026', 'holdout_365d']
**Selection:** nested on 2024 only; HO never for selection.
**Families:** baseline / conc_dampen / frozen_wrebal / wrebal+conc / wrebal+mtd / wrebal+mtd+conc / prior mtd. hi≤1. K=3.
**Soft mean floor:** 0.0095 on BOTH 2024 and 2025_IS.
**Frozen wrebal weights:** `[0.328198, 0.029051, 0.2413, 0.177605, 0.223846]`

## Board

| Family | N | Soft IS | Hard IS | Promote |
|--------|--:|--------:|--------:|--------:|
| baseline | 1 | 0 | 0 | 0 |
| conc_dampen | 1 | 0 | 0 | 0 |
| frozen_wrebal | 1 | 0 | 0 | 0 |
| frozen_wrebal_conc | 1 | 0 | 0 | 0 |
| frozen_wrebal_mtd | 2 | 2 | 0 | 0 |
| frozen_wrebal_mtd_conc | 1 | 1 | 0 | 0 |
| mtd_gain_clip_prior | 1 | 1 | 0 | 0 |

**Totals:** soft=4 hard=0 promote=0 (board n=8)

**2024 conc pick:** {'lookback_months': 8, 'thresh': 0.9, 'cool_scale': 0.65, 'k': 3} (sc=-0.7246)
**wrebal+conc nested:** {'lookback_months': 6, 'thresh': 0.8, 'cool_scale': 0.65, 'k': 3}
**wrebal+mtd+conc nested:** {'lookback_months': 6, 'thresh': 0.85, 'cool_scale': 0.65, 'k': 3}

## Soft / hard / promote

**No promote.**

**Best soft IS:**

### `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen` (frozen_wrebal_mtd) — note=holdout_fail

| Window | Mean mo | %pos | Top3 |
|--------|--------:|-----:|-----:|
| 2024 (IS) | 1.18% | 82% | 45% |
| 2025_IS | 1.74% | 75% | 64% |
| 2025 (cal) | 1.11% | 73% | 61% |
| 2026 | 2.41% | 75% | 80% |
| holdout | 1.69% | 67% | 71% |

soft=True hard=False promote=False min_is_mo=1.18% max_is_top3=64%

### Top-8 by min_is_mo (diagnostic)

| idea | family | min_is_mo | pos24 | pos25is | top3_max | note |
|------|--------|----------:|------:|--------:|---------:|------|
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5_frozen` | frozen_wrebal_mtd | 1.18% | 82% | 75% | 64% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5+conc_lb6_th0.85_cs0.65_k3` | frozen_wrebal_mtd_conc | 1.06% | 82% | 75% | 61% | holdout_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.65_frozen` | frozen_wrebal_mtd | 1.03% | 82% | 75% | 66% | holdout_fail |
| `lock_mtd_t0.015_a0.0_prior_alone` | mtd_gain_clip_prior | 0.99% | 73% | 75% | 59% | year_clear_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224_alone` | frozen_wrebal | 0.68% | 82% | 75% | 70% | soft_mean_fail |
| `wrebal_0.328_0.029_0.241_0.178_0.224+conc_lb6_th0.8_cs0.65_k3` | frozen_wrebal_conc | 0.58% | 82% | 75% | 69% | soft_mean_fail |
| `baseline_locked` | baseline | 0.42% | 55% | 75% | 74% | is_fail |
| `lock_conc_lb8_th0.9_cs0.65_k3` | conc_dampen | 0.25% | 55% | 75% | 74% | is_fail |

## Verdict

- **Promote:** NO
- Locked tag remains `fx4plus_gbpcad_d1_voltarget_0025`
- Still blocked on FTMO CSVs: **YES** (`data/ftmo/` empty of CSVs)

Artifacts: `reports/quest_conc_dampen_board.csv`, `reports/quest_conc_dampen_consistency.md`, `configs/quest_conc_dampen_selected.json`

