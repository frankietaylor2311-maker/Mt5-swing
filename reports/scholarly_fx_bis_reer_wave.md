# Scholarly FX: BIS REER undervaluation wave (§31)

**Path:** Official BIS real broad effective exchange rates via FRED `RB*BIS` — multilateral REER undervaluation / mean-reversion XS — **distinct** from homemade bilateral PPP CPI real-FX, TB (§30), debt/GDP (§29), fiscal GGNLBA (§28), CA, CB-BS, macro-diff, equity-diff, AI-GPR, ToT, dollar-beta, crash-skew.
**Data:** `approximate_non_ftmo` + free FRED `RB*BIS`. EUR = `RBXMBIS` (euro-area; `RBDEBIS` Germany alt documented). Full G10 mapped incl. NZD/CHF. **PIT:** pub_lag_months=2 (conservative monthly BIS release) + signal_lag=1m + 1d weight lag. Primary z lookback=60m (alt 36m boarded).
**Primary:** `reer_cheap_xs` (long low REER z / short high — Rogoff/Taylor undervaluation mean-reversion). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| reer_cheap_xs | +0.034% | +0.51 | +0.60 | 52% | 13% | +0.15 |
| reer_cheap_xs_36 | +0.022% | +0.34 | +0.35 | 50% | 12% | +0.11 |
| reer_mom_xs | -0.044% | -0.65 | -0.76 | 48% | 13% | -0.18 |
| reer_chg_xs | +0.045% | +0.70 | +0.75 | 52% | 9% | +0.20 |
| us_reer_strong_usd | +0.022% | +0.42 | +0.46 | 27% | 17% | +0.11 |
| us_reer_meanrev_fx | -0.023% | -0.44 | -0.49 | 23% | 19% | -0.11 |
| reer_ew | +0.019% | +0.42 | +0.49 | 51% | 11% | +0.14 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| reer_cheap_xs | year_2024 | -0.11% | 36% | PASS | no |
| reer_cheap_xs | year_2025 | -0.27% | 27% | PASS | no |
| reer_cheap_xs | year_2026 | +0.16% | 62% | PASS | no |
| reer_cheap_xs | holdout_365d | +0.08% | 58% | PASS | no |
| reer_cheap_xs_36 | year_2024 | -0.19% | 27% | PASS | no |
| reer_cheap_xs_36 | year_2025 | -0.15% | 36% | PASS | no |
| reer_cheap_xs_36 | year_2026 | +0.05% | 50% | PASS | no |
| reer_cheap_xs_36 | holdout_365d | +0.11% | 58% | PASS | no |
| reer_mom_xs | year_2024 | +0.10% | 64% | PASS | no |
| reer_mom_xs | year_2025 | +0.25% | 73% | PASS | no |
| reer_mom_xs | year_2026 | -0.17% | 38% | PASS | no |
| reer_mom_xs | holdout_365d | -0.10% | 42% | PASS | no |
| reer_chg_xs | year_2024 | -0.02% | 45% | PASS | no |
| reer_chg_xs | year_2025 | +0.17% | 64% | PASS | no |
| reer_chg_xs | year_2026 | +0.06% | 38% | PASS | no |
| reer_chg_xs | holdout_365d | +0.17% | 50% | PASS | no |
| us_reer_strong_usd | year_2024 | +0.27% | 64% | PASS | no |
| us_reer_strong_usd | year_2025 | -0.40% | 9% | PASS | no |
| us_reer_strong_usd | year_2026 | +0.00% | 0% | PASS | no |
| us_reer_strong_usd | holdout_365d | +0.00% | 0% | PASS | no |
| us_reer_meanrev_fx | year_2024 | -0.27% | 27% | PASS | no |
| us_reer_meanrev_fx | year_2025 | +0.40% | 64% | PASS | no |
| us_reer_meanrev_fx | year_2026 | +0.00% | 0% | PASS | no |
| us_reer_meanrev_fx | holdout_365d | +0.00% | 0% | PASS | no |
| reer_ew | year_2024 | -0.13% | 36% | PASS | no |
| reer_ew | year_2025 | +0.10% | 45% | PASS | no |
| reer_ew | year_2026 | +0.07% | 50% | PASS | no |
| reer_ew | holdout_365d | +0.09% | 58% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_bis_reer_*.csv`, `scholarly_fx_bis_reer_meta.json`.

## Risk sweep / locked sleeve

Primary `reer_cheap_xs` scale≈2.924 bind=daily; scaled HO mean≈0.24%/mo %pos 58% — clears **NO**.
Soft-best-ish: `reer_chg_xs` (+4.5 bp/mo, NW t≈0.75) — below soft |t|≥1.5.
Locked `fx4plus_gbpcad_d1_voltarget_0025` untouched; verify PASS (2024 0.42%/55%/74%; 2025 1.63%/73%/69%; 2026 2.30%/88%/87%; HO 1.52%/75%/81%).
Data tag: `approximate_non_ftmo`. EUR=`RBXMBIS`; NZD/CHF mapped; `RBEZBIS`/`RBEMUBIS` 404.
