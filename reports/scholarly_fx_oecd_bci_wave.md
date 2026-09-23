# Scholarly FX: OECD BCI manufacturing-confidence differential wave (§39)

**Path:** OECD BCI via FRED `BSCICP02*M460S` (ISM/NAPM **404** fallback) — manufacturing / business-confidence XS — **distinct** from OECD CLI (§37), OECD CCI (§38), coincident macro-diff CPI/IP/UR, house-price (§35), money-growth (§33), IG OAS (§36), CA/TB, BIS REER/credit, reserves, equity-diff, GPR/EPU.
**Data:** `approximate_non_ftmo` + free FRED `BSCICP02*M460S`. EUR = `BSCICP02EZM460S` (EZ aggregate). AUD/CAD/NZD/JPY **unmapped** (BSCICP02 404; BSCICP03 amplitude stale); CHF **mapped** live. `n_long=n_short=1` a priori. **PIT:** pub_lag_months=2 (conservative monthly OECD BCI) + signal_lag_months=0 + 1d weight lag.
**Primary:** `high_bci_xs` (long high relative BCI YoY change / short low — business-confidence strength → appreciate). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_bci_xs | +0.081% | +1.02 | +0.98 | 57% | 13% | +0.24 |
| low_bci_xs | -0.105% | -1.35 | -1.29 | 42% | 17% | -0.29 |
| high_bci_z_xs | +0.126% | +1.71 | +1.90 | 56% | 15% | +0.37 |
| bci_chg_xs | +0.148% | +2.02 | +2.34 | 59% | 12% | +0.48 |
| us_bci_weak_fx | -0.008% | -0.28 | -0.32 | 9% | 43% | -0.06 |
| us_bci_haven_usd | +0.007% | +0.23 | +0.27 | 11% | 33% | +0.06 |
| bci_ew | +0.074% | +1.61 | +1.65 | 58% | 13% | +0.38 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| high_bci_xs | year_2024 | +0.25% | 73% | PASS | no |
| high_bci_xs | year_2025 | +0.44% | 82% | PASS | no |
| high_bci_xs | year_2026 | +0.48% | 88% | PASS | no |
| high_bci_xs | holdout_365d | +0.30% | 83% | PASS | no |
| low_bci_xs | year_2024 | -0.27% | 27% | PASS | no |
| low_bci_xs | year_2025 | -0.48% | 18% | PASS | no |
| low_bci_xs | year_2026 | -0.51% | 12% | PASS | no |
| low_bci_xs | holdout_365d | -0.33% | 17% | PASS | no |
| high_bci_z_xs | year_2024 | +0.24% | 73% | PASS | no |
| high_bci_z_xs | year_2025 | +0.37% | 73% | PASS | no |
| high_bci_z_xs | year_2026 | +0.28% | 75% | PASS | no |
| high_bci_z_xs | holdout_365d | +0.17% | 75% | PASS | no |
| bci_chg_xs | year_2024 | +0.21% | 73% | PASS | no |
| bci_chg_xs | year_2025 | +0.31% | 73% | PASS | no |
| bci_chg_xs | year_2026 | +0.48% | 88% | PASS | no |
| bci_chg_xs | holdout_365d | +0.24% | 75% | PASS | no |
| us_bci_weak_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_bci_weak_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_bci_weak_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_bci_weak_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_bci_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_bci_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_bci_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_bci_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| bci_ew | year_2024 | +0.16% | 73% | PASS | no |
| bci_ew | year_2025 | +0.25% | 73% | PASS | no |
| bci_ew | year_2026 | +0.32% | 88% | PASS | no |
| bci_ew | holdout_365d | +0.18% | 75% | PASS | no |

**Board:** n=7 soft_nw_pos=3 hard_nw_pos=1 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_oecd_bci_*.csv`, `scholarly_fx_oecd_bci_meta.json`.


## Risk sweep / board

- Primary `high_bci_xs` scale≈**0.593** bind=**daily**; scaled HO mean≈+0.18%/mo %pos 83% — clears 1% bar: **NO**.
- Soft-best: `bci_chg_xs` (~+14.8 bp/mo, NW t≈+2.34, hard).
- Board: n=7 soft_nw_pos=3 hard_nw_pos=1 promote=0.
- Locked `fx4plus_gbpcad_d1_voltarget_0025` untouched PASS (2024 0.42%/55%/74%; 2025 1.63%/73%/69%; 2026 2.30%/88%/87%; HO 1.52%/75%/81%).
- Data tag: `approximate_non_ftmo` (no FTMO MT5 D1 CSVs under `data/ftmo/`).

## Honest read

OECD BCI is the free PMI-style manufacturing/business-confidence structure (NAPM/ISM 404). Mildly positive primary (~+8 bp/mo) with soft/hard boards on acceleration/z legs, but year means still ≲0.5%/mo — not near 1%/mo. Thin EUR/GBP/CHF panel (`n_long=n_short=1`). No go-live.
