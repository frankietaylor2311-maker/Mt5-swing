# Scholarly FX: terms-of-trade / commodity-currency wave

**Path:** Cashin–Céspedes–Sahay ToT — **distinct** from CRR commodity-*momentum* (§11). ToT change = export_mom − import_mom via frozen `COUNTRY_TOT_MAP`.
**Data:** `approximate_non_ftmo` + Yahoo commodity panel (oil/copper/gold/basket) already on disk. **PIT:** pub_lag=1d + signal_lag=1d; formation=63d skip=21d; calendar-date align.
**Primary:** `tot_country_ts`. Locked sleeve untouched. corr(tot_ts,crr_ts)=0.711; corr(tot_vs_g10,crr_xs)=0.310 — distinct enough to evaluate

### Frozen ToT map

| CCY | Export | Import | Traded? |
|-----|--------|--------|:-------:|
| AUD | copper | oil | yes |
| CAD | oil | copper | yes |
| NZD | basket | oil | yes |
| NOK | oil | copper | no (no pair) |
| ZAR | gold | oil | no (no pair) |

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| tot_country_ts | -0.181% | -0.92 | -1.06 | 47% | 12% | -0.29 |
| tot_xs | -0.005% | -0.07 | -0.06 | 48% | 10% | -0.02 |
| tot_vs_g10 | -0.052% | -0.85 | -1.07 | 35% | 15% | -0.27 |
| tot_ew | -0.080% | -0.85 | -0.94 | 48% | 13% | -0.26 |
| crr_country_ts_baseline | -0.037% | -0.23 | -0.30 | 38% | 11% | -0.06 |
| crr_xs_basket_baseline | +0.001% | +0.01 | +0.01 | 38% | 20% | -0.00 |

## Consistency windows (ToT legs)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| tot_country_ts | year_2024 | -1.02% | 18% | FAIL | no |
| tot_country_ts | year_2025 | +0.22% | 55% | PASS | no |
| tot_country_ts | year_2026 | -0.56% | 38% | PASS | no |
| tot_country_ts | holdout_365d | +0.05% | 42% | PASS | no |
| tot_xs | year_2024 | -0.32% | 27% | PASS | no |
| tot_xs | year_2025 | -0.32% | 36% | PASS | no |
| tot_xs | year_2026 | -0.00% | 50% | PASS | no |
| tot_xs | holdout_365d | +0.11% | 50% | PASS | no |
| tot_vs_g10 | year_2024 | -0.39% | 27% | PASS | no |
| tot_vs_g10 | year_2025 | +0.05% | 45% | PASS | no |
| tot_vs_g10 | year_2026 | -0.03% | 38% | PASS | no |
| tot_vs_g10 | holdout_365d | +0.18% | 50% | PASS | no |
| tot_ew | year_2024 | -0.58% | 27% | PASS | no |
| tot_ew | year_2025 | -0.02% | 64% | PASS | no |
| tot_ew | year_2026 | -0.20% | 50% | PASS | no |
| tot_ew | holdout_365d | +0.11% | 58% | PASS | no |

**Board:** n=4 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_tot_*.csv`, `scholarly_fx_tot_meta.json`.
