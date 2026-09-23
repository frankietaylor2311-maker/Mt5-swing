# Scholarly FX: capital-sleeve mix wave (§53)

**Path:** Keep locked `fx4plus_gbpcad_d1_voltarget_0025` as **core** capital; allocate smaller fixed FTMO risk fractions to pre-registered scholarly satellites `bci_chg_xs` (§39) and `high_ppi_xs` (§51). **Separate capital sleeves** — NOT overlays/coolers on locked equity.
**Data:** `approximate_non_ftmo` + free FRED OECD BCI / PPI. **RF=8%** **VT=0.0025**. Locked config sha256 `bafed8eff5aa…` untouched=YES.
**Primary (IS-only):** `core_100`. Mix grid fixed a priori (n=6).

## Full-sample mix summary

| Mix | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|-----|--------:|------:|-----:|-----:|-----:|-------:|
| core_100 | +1.717% | +2.88 | +2.16 | 72% | 38% | +1.48 |
| core85_bci15 | +1.475% | +2.95 | +2.19 | 72% | 38% | +1.49 |
| core70_bci30 | +1.233% | +3.03 | +2.22 | 75% | 38% | +1.51 |
| core60_bci40 | +1.072% | +3.10 | +2.25 | 75% | 38% | +1.53 |
| core80_bci10_ppi10 | +1.389% | +2.95 | +2.19 | 72% | 38% | +1.49 |
| core70_bci20_ppi10 | +1.228% | +3.01 | +2.21 | 75% | 38% | +1.51 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| core_100 | year_2024 | +1.31% | 64% | PASS | no |
| core_100 | year_2025 | +1.42% | 73% | PASS | no |
| core_100 | year_2026 | +2.62% | 75% | PASS | no |
| core_100 | holdout_365d | +1.93% | 67% | PASS | no |
| core85_bci15 | year_2024 | +1.13% | 64% | PASS | no |
| core85_bci15 | year_2025 | +1.22% | 73% | PASS | no |
| core85_bci15 | year_2026 | +2.26% | 75% | PASS | no |
| core85_bci15 | holdout_365d | +1.66% | 67% | PASS | no |
| core70_bci30 | year_2024 | +0.96% | 73% | PASS | no |
| core70_bci30 | year_2025 | +1.01% | 73% | PASS | no |
| core70_bci30 | year_2026 | +1.90% | 75% | PASS | no |
| core70_bci30 | holdout_365d | +1.39% | 67% | PASS | no |
| core60_bci40 | year_2024 | +0.84% | 73% | PASS | no |
| core60_bci40 | year_2025 | +0.87% | 73% | PASS | no |
| core60_bci40 | year_2026 | +1.66% | 75% | PASS | no |
| core60_bci40 | holdout_365d | +1.22% | 67% | PASS | no |
| core80_bci10_ppi10 | year_2024 | +1.07% | 64% | PASS | no |
| core80_bci10_ppi10 | year_2025 | +1.15% | 73% | PASS | no |
| core80_bci10_ppi10 | year_2026 | +2.11% | 75% | PASS | no |
| core80_bci10_ppi10 | holdout_365d | +1.57% | 67% | PASS | no |
| core70_bci20_ppi10 | year_2024 | +0.95% | 73% | PASS | no |
| core70_bci20_ppi10 | year_2025 | +1.02% | 73% | PASS | no |
| core70_bci20_ppi10 | year_2026 | +1.87% | 75% | PASS | no |
| core70_bci20_ppi10 | holdout_365d | +1.39% | 67% | PASS | no |

**Board:** n=6 soft_nw_pos=6 hard_nw_pos=6 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_capital_sleeve_mix_*.csv`, `scholarly_fx_capital_sleeve_mix_meta.json`.
