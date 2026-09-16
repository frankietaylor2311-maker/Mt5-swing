# Consistency overlay refine (locked sleeve only)

**data_source:** `approximate_non_ftmo`  **RF:** 8%  **PORT_VOL_TARGET:** 0.0025  **signal_lag:** 1
**Locked tag (unchanged unless promote):** `fx4plus_gbpcad_d1_voltarget_0025`
**IS windows:** ['2024', '2025_IS']  **Confirm:** ['2025', '2026', 'holdout_365d']
**Selection:** nested on 2024 (or max min(2024,2025_IS) for weights); HO never for selection.
**Overlays:** mtd / month_aware / runup / equity_curve_target / combos<=2 / IS weight rebalance. hi<=1.

## Board

| Family | N | Soft IS | Hard IS | Promote |
|--------|--:|--------:|--------:|--------:|
| baseline | 1 | 0 | 0 | 0 |
| combo_two | 5 | 2 | 0 | 0 |
| equity_curve_target | 1 | 0 | 0 | 0 |
| month_aware | 1 | 0 | 0 | 0 |
| mtd_gain_clip | 4 | 4 | 0 | 0 |
| runup_throttle | 1 | 0 | 0 | 0 |
| weight_plus_month | 1 | 0 | 0 | 0 |
| weight_plus_mtd | 1 | 1 | 0 | 0 |
| weight_rebalance | 2 | 1 | 0 | 0 |

**Totals:** soft=8 hard=0 promote=0 (board n=17)

**2024-only MTD pick:** {'tau': 0.015, 'after': 0.0} -> mean=0.99% pos=73% top3=52%
**2024-only month_aware pick:** {'strong_mo': 0.015, 'after_strong': 0.4, 'dd_trigger': 0.03, 'after_dd': 0.4}
**2024-only runup pick:** {'trail_bars': 21, 'runup_thresh': 0.03, 'cool_scale': 0.35}
**2024-only eqtarget pick:** {'target_mo_vol': 0.015, 'lookback_months': 4}
**IS weight rebalance:** [0.3282, 0.0291, 0.2413, 0.1776, 0.2238] score=0.0040

## Soft / hard / promote

**No promote.**

**Best soft IS:**

### `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5` (weight_plus_mtd) — note=holdout_fail

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
| `wrebal_0.328_0.029_0.241_0.178_0.224+mtd_t0.015_a0.5` | weight_plus_mtd | 1.20% | 82% | 75% | 64% | holdout_fail |
| `lock_mtd_t0.008_a0.25` | mtd_gain_clip | 1.06% | 73% | 75% | 65% | year_clear_fail |
| `lock_mtd_t0.015_a0.0+ru_tb21_th0.03_cs0.35` | combo_two | 0.99% | 73% | 75% | 59% | year_clear_fail |
| `lock_mtd_t0.015_a0.0` | mtd_gain_clip | 0.99% | 73% | 75% | 59% | year_clear_fail |
| `lock_mtd_t0.008_a0.35` | mtd_gain_clip | 0.97% | 73% | 75% | 66% | year_clear_fail |
| `lock_mtd_t0.015_a0.15` | mtd_gain_clip | 0.90% | 73% | 75% | 61% | year_clear_fail |
| `lock_mtd_t0.015_a0.0+eq_tv0.015_lb4` | combo_two | 0.78% | 73% | 75% | 60% | holdout_fail |
| `lock_mtd_t0.015_a0.0+mo_s0.015_as0.4_dd0.03_ad0.4` | combo_two | 0.77% | 64% | 75% | 68% | is_fail |

## Verdict

- **Promote:** NO
- Locked tag remains `fx4plus_gbpcad_d1_voltarget_0025`
- Still blocked on FTMO CSVs: **YES** (`data/ftmo/` empty of CSVs)

Artifacts: `reports/quest_consistency_overlay_board.csv`, `reports/quest_consistency_overlay_refine.md`, `configs/quest_consistency_overlay_selected.json`

