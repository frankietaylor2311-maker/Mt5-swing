# Scholarly FX: monthly trade-balance wave (§30)

**Path:** High-frequency external imbalance — OECD MEI merchandise exports/imports via FRED → TB/exports — **distinct** from quarterly CA (§21), debt/GDP (§29), fiscal GGNLBA (§28), CB-BS, macro-diff, equity-diff, AI-GPR, ToT, dollar-beta, crash-skew.

**Data:** `approximate_non_ftmo` + free FRED `XTEXVA01*M667S` / `XTIMVA01*M667S` + US `BOPGSTB` for tilts. EUR = Germany proxy (EZM ends ~2023-04). `USAB6BLTT02STSAM` / `BOPB12` **404**. **PIT:** pub_lag_months=2 + signal_lag=1m + 1d weight lag.

**Primary:** `tb_deficit_xs` (long deficit / short surplus — DCRS debtor premium). Full G10 mapped incl. NZD/CHF. Locked sleeve untouched.

## Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| tb_deficit_xs | +0.008% | +0.13 | +0.14 | 49% | 13% | +0.06 |
| tb_surplus_xs | −0.019% | −0.31 | −0.32 | 50% | 12% | −0.09 |
| tb_chg_xs | −0.077% | −1.24 | −1.14 | 44% | 12% | −0.27 |
| us_tb_gr_fx | −0.054% | −1.31 | −1.34 | 18% | 27% | −0.35 |
| us_tb_haven_usd | +0.052% | +1.27 | +1.30 | 22% | 21% | +0.33 |
| tb_ca_blend | +0.014% | +0.24 | +0.28 | 50% | 13% | +0.05 |
| tb_ew | −0.040% | −1.44 | −1.44 | 49% | 12% | −0.32 |

## Consistency windows (primary)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| tb_deficit_xs | year_2024 | +0.08% | 73% | PASS | no |
| tb_deficit_xs | year_2025 | −0.22% | 45% | PASS | no |
| tb_deficit_xs | year_2026 | −0.01% | 38% | PASS | no |
| tb_deficit_xs | holdout_365d | −0.06% | 42% | PASS | no |

**Board:** n=7 soft_nw_pos=0 hard_nw_pos=0 promote=0.
**Unscaled promote:** NO. **Scaled primary promote:** NO.
Primary risk sweep scale≈1.18 bind=daily; scaled HO clears: **NO**.

Locked `fx4plus_gbpcad_d1_voltarget_0025` **untouched**; re-verify PASS (see `quest_locked_verify_trade_balance.md`).

Artifacts: `reports/scholarly_fx_trade_balance_*.csv`, `scholarly_fx_trade_balance_meta.json`.
