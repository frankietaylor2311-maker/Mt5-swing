# Scholarly FX: central-bank balance-sheet / QE differential wave

**Path:** Neely / Gagnon / Bauer–Neely portfolio-balance QE–FX channel — **distinct** from NFCI funding-liquidity and CA/GDP imbalances.
**Data:** `approximate_non_ftmo` + free FRED `WALCL` / `ECBASSETSW` / `JPNASSETS` (+ optional `GDP` for BS/GDP). **PIT:** weekly pub_lag=7d, monthly pub_lag=1m, GDP pub_lag=1Q + signal_lag=1d.
**Primary:** `walcl_pb_fx` (high Fed BS YoY → long FX / short USD). Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| walcl_pb_fx | +0.043% | +1.19 | +1.39 | 16% | 25% | +0.31 |
| walcl_haven_usd | -0.044% | -1.23 | -1.42 | 12% | 29% | -0.31 |
| walcl_chg_pb_fx | +0.023% | +0.65 | +0.76 | 18% | 25% | +0.17 |
| bs_diff_pb_fx | +0.013% | +0.33 | +0.38 | 16% | 25% | +0.08 |
| bs_peer_xs | -0.111% | -1.32 | -1.29 | 45% | 12% | -0.30 |
| bs_gdp_pb_fx | -0.031% | -1.02 | -1.02 | 13% | 38% | -0.27 |
| bs_ew | -0.018% | -0.48 | -0.53 | 47% | 13% | -0.12 |

## Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| walcl_pb_fx | year_2024 | +0.00% | 0% | PASS | no |
| walcl_pb_fx | year_2025 | +0.24% | 55% | PASS | no |
| walcl_pb_fx | year_2026 | -0.19% | 38% | PASS | no |
| walcl_pb_fx | holdout_365d | -0.01% | 42% | PASS | no |
| walcl_haven_usd | year_2024 | +0.00% | 0% | PASS | no |
| walcl_haven_usd | year_2025 | -0.25% | 45% | PASS | no |
| walcl_haven_usd | year_2026 | +0.19% | 62% | PASS | no |
| walcl_haven_usd | holdout_365d | +0.01% | 58% | PASS | no |
| walcl_chg_pb_fx | year_2024 | -0.04% | 18% | PASS | no |
| walcl_chg_pb_fx | year_2025 | +0.22% | 64% | PASS | no |
| walcl_chg_pb_fx | year_2026 | -0.28% | 25% | PASS | no |
| walcl_chg_pb_fx | holdout_365d | -0.11% | 42% | PASS | no |
| bs_diff_pb_fx | year_2024 | +0.00% | 0% | PASS | no |
| bs_diff_pb_fx | year_2025 | +0.25% | 55% | PASS | no |
| bs_diff_pb_fx | year_2026 | -0.19% | 38% | PASS | no |
| bs_diff_pb_fx | holdout_365d | -0.01% | 42% | PASS | no |
| bs_peer_xs | year_2024 | +0.13% | 64% | PASS | no |
| bs_peer_xs | year_2025 | +0.49% | 73% | PASS | no |
| bs_peer_xs | year_2026 | +0.12% | 38% | PASS | no |
| bs_peer_xs | holdout_365d | +0.18% | 50% | PASS | no |
| bs_gdp_pb_fx | year_2024 | +0.00% | 0% | PASS | no |
| bs_gdp_pb_fx | year_2025 | +0.00% | 0% | PASS | no |
| bs_gdp_pb_fx | year_2026 | +0.00% | 0% | PASS | no |
| bs_gdp_pb_fx | holdout_365d | +0.00% | 0% | PASS | no |
| bs_ew | year_2024 | +0.04% | 64% | PASS | no |
| bs_ew | year_2025 | +0.33% | 73% | PASS | no |
| bs_ew | year_2026 | -0.09% | 50% | PASS | no |
| bs_ew | holdout_365d | +0.05% | 58% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.

Artifacts: `reports/scholarly_fx_cb_bs_*.csv`, `scholarly_fx_cb_bs_meta.json`.

## Honest read

Neely / Gagnon portfolio-balance priors point the right *direction* and are **distinct** from NFCI funding (§20) and CA/GDP (§21): mild positive full-sample means on `walcl_pb_fx` / `walcl_chg_pb_fx` / `bs_diff_pb_fx` (~+1–4 bp/mo, NW t ≈ 0.4–1.4), with haven / peer-XS / BS-GDP legs flat-to-negative. Binary z≥1 episodes are infrequent → sparse %pos (~16%). Far from prop-firm 1%/mo + 70% hit-rate. Free FRED Fed+ECB+BoJ is sufficient; UKASSETS ends 2014 — excluded. Multi-CB coverage was **not** too thin — real-rate fallback deferred to next wave.

**Locked sleeve** `fx4plus_gbpcad_d1_voltarget_0025` untouched. **promote=0**.
