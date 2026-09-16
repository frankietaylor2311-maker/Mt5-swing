# D1 multi-year consistency repair + MTD gain-clip

**data_source:** `approximate_non_ftmo`  **RF:** 8%  **PORT_VOL_TARGET:** 0.0025  **signal_lag:** 1
**Locked tag (unchanged unless promote):** `fx4plus_gbpcad_d1_voltarget_0025`
**IS windows:** ['2024', '2025_IS']  **Confirm:** ['2025', '2026', 'holdout_365d']
**Selection:** maximize min(IS mean_mo) with %pos≥70%, top3 soft≤70% / hard≤55%. Holdout never for selection.

## Board

| Family | N | Soft IS | Hard IS | Promote |
|--------|--:|--------:|--------:|--------:|
| baseline | 1 | 0 | 0 | 0 |
| d1_alternate | 8 | 0 | 0 | 0 |
| lock_plus_clip | 2 | 0 | 0 | 0 |
| lock_plus_d1 | 28 | 0 | 0 | 0 |
| lock_plus_d1_pair | 20 | 0 | 0 | 0 |
| mtd_gain_clip | 1 | 1 | 0 | 0 |

**Totals:** soft=1 hard=0 promote=0 (board n=60)
**D1 pool screened:** 24  **solo IS-scored:** 7  **pair trials:** 10

**2024-only MTD clip pick:** tau=0.015 after=0.0 (2024 top3→52% mean→0.99% pos→73%)

## Soft / hard / promote

**No promote.**

**Best soft IS:**

### `lock_mtdclip_t0.015_a0.0` (mtd_gain_clip) — note=year_clear_fail

| Window | Mean mo | %pos | Top3 |
|--------|--------:|-----:|-----:|
| 2024 (IS) | 0.99% | 73% | 52% |
| 2025_IS | 1.45% | 75% | 59% |
| holdout | 1.37% | 75% | 66% |

soft=True hard=False promote=False min_is_mo=0.99% max_is_top3=59%


### Confirm windows for best soft (calendar)

| Window | Mean mo | %pos | Top3 | Gates |
|--------|--------:|-----:|-----:|:-----:|
| 2024 | 0.99% | 73% | 52% | PASS |
| 2025_IS | 1.45% | 75% | 59% | PASS |
| 2025 | **0.89%** | 73% | 56% | PASS |
| 2026 | 1.91% | 88% | 77% | PASS |
| holdout_365d | 1.37% | 75% | 66% | PASS |

Year-clear (≥1% mean & ≥70% pos on 2024/2025/2026): **FAIL** (2025 cal 0.89%; 2024 0.99%).
Hard top3≤55%: **FAIL** (2025_IS 59%). Soft top3≤70% on IS: **PASS**.

## Verdict

- **Promote:** NO
- Locked tag remains `fx4plus_gbpcad_d1_voltarget_0025`
- Still blocked on FTMO CSVs: **YES** (`data/ftmo/` empty of CSVs)

Artifacts: `reports/quest_d1_consistency_board.csv`, `quest_d1_consistency_pool.csv`, `quest_d1_consistency_solo.csv`, `quest_d1_consistency_repair.md`

