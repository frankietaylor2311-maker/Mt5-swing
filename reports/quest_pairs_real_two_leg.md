# Real two-leg pairs quest

**Data:** `approximate_non_ftmo`. **RF=0.08**. Locked `fx4plus_gbpcad_d1_voltarget_0025` unchanged unless promote.

## Diagnosis: Δz proxy vs real

- `proxy` / `z_vol` on EURAUD/AUDUSD: 2024 mean_mo=1.280%, pos=82%
- `real` / `z_vol` on EURAUD/AUDUSD: 2024 mean_mo=0.888%, pos=45%, amp≈181.5x
- `real` / `unit_residual` on EURAUD/AUDUSD: 2024 mean_mo=0.013%, pos=55%, amp≈181.5x

**Proxy illusory?** **YES** — unit_residual (honest notional) and/or costed z_vol cannot clear ≥1% on confirmation.

Board: 1080 candidates, soft=0, hard=0.
**Promote?** **NO**

## Best REAL candidate (confirm table)

Idea: `real_n3_z_vol` sizing=`z_vol`

| Window | Mean mo | %pos | Top3 | Gates |
|--------|--------:|-----:|-----:|:-----:|
| 2024 | 0.747% | 73% | 71% | True |
| 2025 | 0.575% | 73% | 58% | True |
| 2026 | 0.481% | 50% | 83% | True |
| holdout_365d | 0.296% | 50% | 59% | True |

confirm_min_mo=0.296%; promote=False; reason=['is_soft_fail', 'holdout_fail', 'years_fail']
