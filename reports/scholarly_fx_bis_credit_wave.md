# Scholarly FX: BIS private credit-to-GDP / credit-gap wave (§32)

**Path:** BIS private credit/GDP via FRED `Q*PAM770A` — Borio–Drehmann / Basel credit-cycle gap XS — **distinct** from government debt/GDP (§29 GGGDTA*), fiscal GGNLBA (§28), TB (§30), BIS REER (§31), CA, CB-BS, funding-liq, macro-diff, equity-diff, AI-GPR, ToT, dollar-beta, crash-skew.
**Data:** `approximate_non_ftmo` + free FRED `Q*PAM770A` (% GDP). Brief `CRDQ*APABIS` is absolute credit (documented unused). EUR = `QXMPAM770A` (euro-area; `QDEPAM770A` Germany alt). Full G10 mapped incl. NZD/CHF. Pre-computed credit-gap FRED IDs 404 — gap = level − trailing 180m mean (a priori). **PIT:** pub_lag_months=5 (conservative quarterly BIS) + signal_lag=1m + 1d weight lag.
**Primary:** `low_credit_gap_xs` (long negative credit gap / short positive — lean balance-sheet prior). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| low_credit_gap_xs | -0.013% | -0.22 | -0.25 | 46% | 10% | -0.02 |
| low_credit_xs | +0.000% | +0.00 | +0.00 | 48% | 11% | +0.05 |
| high_credit_xs | -0.002% | -0.04 | -0.05 | 51% | 13% | -0.05 |
| low_credit_z_xs | +0.037% | +0.48 | +0.53 | 51% | 11% | +0.10 |
| credit_chg_xs | +0.083% | +1.20 | +1.20 | 51% | 10% | +0.30 |
| us_credit_stress_fx | -0.003% | -0.14 | -0.19 | 6% | 55% | -0.04 |
| us_credit_haven_usd | +0.003% | +0.14 | +0.20 | 5% | 58% | +0.04 |
| credit_ew | +0.023% | +0.57 | +0.62 | 51% | 12% | +0.16 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| low_credit_gap_xs | year_2024 | +0.11% | 55% | PASS | no |
| low_credit_gap_xs | year_2025 | +0.00% | 55% | PASS | no |
| low_credit_gap_xs | year_2026 | -0.04% | 50% | PASS | no |
| low_credit_gap_xs | holdout_365d | +0.02% | 58% | PASS | no |
| low_credit_xs | year_2024 | +0.10% | 55% | PASS | no |
| low_credit_xs | year_2025 | +0.04% | 64% | PASS | no |
| low_credit_xs | year_2026 | +0.14% | 62% | PASS | no |
| low_credit_xs | holdout_365d | +0.04% | 50% | PASS | no |
| high_credit_xs | year_2024 | -0.10% | 45% | PASS | no |
| high_credit_xs | year_2025 | -0.04% | 36% | PASS | no |
| high_credit_xs | year_2026 | -0.14% | 38% | PASS | no |
| high_credit_xs | holdout_365d | -0.04% | 50% | PASS | no |
| low_credit_z_xs | year_2024 | +0.18% | 82% | PASS | no |
| low_credit_z_xs | year_2025 | -0.02% | 55% | PASS | no |
| low_credit_z_xs | year_2026 | -0.05% | 38% | PASS | no |
| low_credit_z_xs | holdout_365d | -0.25% | 25% | PASS | no |
| credit_chg_xs | year_2024 | +0.24% | 45% | PASS | no |
| credit_chg_xs | year_2025 | +0.14% | 55% | PASS | no |
| credit_chg_xs | year_2026 | -0.19% | 38% | PASS | no |
| credit_chg_xs | holdout_365d | -0.28% | 33% | PASS | no |
| us_credit_stress_fx | year_2024 | +0.00% | 0% | PASS | no |
| us_credit_stress_fx | year_2025 | +0.00% | 0% | PASS | no |
| us_credit_stress_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_credit_stress_fx | holdout_365d | +0.00% | 0% | PASS | no |
| us_credit_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| us_credit_haven_usd | year_2025 | +0.00% | 0% | PASS | no |
| us_credit_haven_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_credit_haven_usd | holdout_365d | +0.00% | 0% | PASS | no |
| credit_ew | year_2024 | +0.12% | 64% | PASS | no |
| credit_ew | year_2025 | +0.05% | 73% | PASS | no |
| credit_ew | year_2026 | -0.07% | 25% | PASS | no |
| credit_ew | holdout_365d | -0.09% | 33% | PASS | no |

**Board:** n=8 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_bis_credit_*.csv`, `scholarly_fx_bis_credit_meta.json`.
## Risk sweep / locked sleeve

Primary `low_credit_gap_xs` scale≈1.154 bind=daily; scaled HO mean≈0.02%/mo %pos 58% — clears **NO**.
Soft-best-ish: `credit_chg_xs` (+8.3 bp/mo, NW t≈1.20) — below soft |t|≥1.5.
Locked `fx4plus_gbpcad_d1_voltarget_0025` untouched; verify PASS (2024 0.42%/55%/74%; 2025 1.63%/73%/69%; 2026 2.30%/88%/87%; HO 1.52%/75%/81%).
Data tag: `approximate_non_ftmo`. EUR=`QXMPAM770A`; NZD/CHF mapped via `QNZPAM770A`/`QCHPAM770A`; brief `CRDQ*APABIS` = absolute (unused); precomputed credit-gap FRED IDs 404 (gap = trailing 180m mean deviation).
