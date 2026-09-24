# Scholarly FX: capital-sleeve soft-stack wave (§70)

**Path:** Keep locked `fx4plus_gbpcad_d1_voltarget_0025` as **core** capital; allocate smaller fixed FTMO risk fractions to pre-registered soft-stack satellites `soft_ew_macro5` (§66/§69) and `soft_ew7_cip` (§69). **Separate capital sleeves** — NOT overlays/coolers on locked equity.
**Data:** `approximate_non_ftmo` + free FRED/OECD soft legs + Du–Schreger CIP. **RF=8%** **VT=0.0025**. Locked config sha256 `bafed8eff5aa…` untouched=YES.
**Primary (IS-only):** `core_100`. Mix grid fixed a priori (n=6).

## Full-sample mix summary

| Mix | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|-----|--------:|------:|-----:|-----:|-----:|-------:|
| core_100 | +1.717% | +2.88 | +2.16 | 72% | 38% | +1.48 |
| core85_soft15 | +1.462% | +2.90 | +2.17 | 72% | 38% | +1.48 |
| core70_soft30 | +1.207% | +2.93 | +2.18 | 72% | 38% | +1.48 |
| core60_soft40 | +1.036% | +2.95 | +2.19 | 72% | 38% | +1.48 |
| core80_soft10_cip10 | +1.377% | +2.91 | +2.17 | 72% | 38% | +1.48 |
| core70_soft20_cip10 | +1.207% | +2.93 | +2.18 | 72% | 38% | +1.48 |

## Consistency windows (primary)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| year_2024 | +1.31% | 64% | 71% | PASS | no |
| year_2025 | +1.42% | 73% | 68% | PASS | no |
| year_2026 | +2.62% | 75% | 85% | PASS | no |
| holdout_365d | +1.93% | 67% | 76% | PASS | no |

**Board:** n=6 soft_nw_pos=6 hard_nw_pos=6 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.
Primary sweep scale≈**1.341** bind=**daily**; scaled HO mean≈+2.59%/mo %pos 67% — clears: **NO**.
Soft-best = `core_100` (mean_mo=+1.717%).

Locked sleeve untouched PASS. No go-live under approximate_non_ftmo.

Artifacts: `reports/scholarly_fx_capital_sleeve_soft_*.csv`, `scholarly_fx_capital_sleeve_soft_meta.json`, `quest_locked_verify_capital_sleeve_soft.md`.
