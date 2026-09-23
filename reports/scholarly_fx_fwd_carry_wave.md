# Scholarly FX: swap- / forward-aware carry

**Prior (fixed):** Lustig–Verdelhan HML-FX sorts on *forward discounts*. Free proxy = OECD 3M money-market differentials + exact CIP-implied FD from IR3TIB01*; cash IRSTCI baseline for comparison. pub_lag=1m + signal_lag=1m; n_long=n_short=2; costs=1.5 bps/side + 5.0 bps TC haircut. No HO tuning.

**Data:** `approximate_non_ftmo` D1 FX + FRED OECD IR3M (`IR3TIB01*`) / immediate (`IRSTCI01*`). IR3M cols: AUD, CAD, CHF, EUR, GBP, JPY, NZD, USD. Spearman rank_corr(IRSTCI,IR3M)≈0.959; CIP-FD vs IR3M-diff≈1.000.

**Honesty:** Still **cash / money-market approximations**. No Bloomberg / broker FX swap points or outright forwards. Post-GFC CIP basis not modelled.

## Factor board (full sample, unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| carry_ir3m_xs | -0.034% | -0.43 | -0.51 | 52.8% | 10% | -0.13 |
| carry_irstci_xs | -0.011% | -0.14 | -0.16 | 51.7% | 10% | -0.07 |
| carry_cip_fd_xs | -0.034% | -0.43 | -0.51 | 52.8% | 10% | -0.13 |
| carry_ir3m_ew | -0.029% | -0.50 | -0.61 | 49.4% | 11% | -0.10 |
| carry_ir3m_tc5 | -0.035% | -0.45 | -0.52 | 52.8% | 10% | -0.13 |
| carry_blend_xs | -0.022% | -0.29 | -0.34 | 51.7% | 10% | -0.10 |

## Calendar / holdout windows (unscaled)

| Strategy | Window | mean_mo | %pos | top3 | t NW | Sharpe | gates | 1% bar | role |
|----------|--------|--------:|-----:|-----:|-----:|-------:|:-----:|:------:|------|
| carry_ir3m_xs | full_sample | -0.034% | 52.8% | 10% | -0.51 | -0.13 | FAIL | no | eval |
| carry_ir3m_xs | year_2024 | 0.063% | 72.7% | 64% | 0.24 | 0.39 | PASS | no | eval |
| carry_ir3m_xs | year_2025 | 0.087% | 72.7% | 60% | 0.48 | 0.12 | PASS | no | eval |
| carry_ir3m_xs | year_2026 | 0.190% | 75.0% | 69% | 0.92 | 0.96 | PASS | no | eval |
| carry_ir3m_xs | holdout_365d | 0.308% | 83.3% | 43% | 1.50 | 1.17 | PASS | no | confirm_only |
| carry_irstci_xs | full_sample | -0.011% | 51.7% | 10% | -0.16 | -0.07 | FAIL | no | eval |
| carry_irstci_xs | year_2024 | 0.128% | 63.6% | 63% | 0.56 | 0.68 | PASS | no | eval |
| carry_irstci_xs | year_2025 | 0.111% | 63.6% | 63% | 0.61 | 0.16 | PASS | no | eval |
| carry_irstci_xs | year_2026 | 0.156% | 62.5% | 82% | 0.79 | 1.15 | PASS | no | eval |
| carry_irstci_xs | holdout_365d | 0.328% | 75.0% | 55% | 1.46 | 1.38 | PASS | no | confirm_only |
| carry_cip_fd_xs | full_sample | -0.034% | 52.8% | 10% | -0.51 | -0.13 | FAIL | no | eval |
| carry_cip_fd_xs | year_2024 | 0.063% | 72.7% | 64% | 0.24 | 0.39 | PASS | no | eval |
| carry_cip_fd_xs | year_2025 | 0.087% | 72.7% | 60% | 0.48 | 0.12 | PASS | no | eval |
| carry_cip_fd_xs | year_2026 | 0.190% | 75.0% | 69% | 0.92 | 0.96 | PASS | no | eval |
| carry_cip_fd_xs | holdout_365d | 0.308% | 83.3% | 43% | 1.50 | 1.17 | PASS | no | confirm_only |
| carry_ir3m_ew | full_sample | -0.029% | 49.4% | 11% | -0.61 | -0.10 | PASS | no | eval |
| carry_ir3m_ew | year_2024 | -0.141% | 54.5% | 86% | -0.82 | -0.78 | PASS | no | eval |
| carry_ir3m_ew | year_2025 | 0.046% | 54.5% | 69% | 0.60 | 0.05 | PASS | no | eval |
| carry_ir3m_ew | year_2026 | 0.142% | 50.0% | 99% | 0.96 | 0.54 | PASS | no | eval |
| carry_ir3m_ew | holdout_365d | 0.091% | 50.0% | 83% | 0.84 | 0.50 | PASS | no | confirm_only |
| carry_ir3m_tc5 | full_sample | -0.035% | 52.8% | 10% | -0.52 | -0.13 | FAIL | no | eval |
| carry_ir3m_tc5 | year_2024 | 0.063% | 72.7% | 64% | 0.24 | 0.39 | PASS | no | eval |
| carry_ir3m_tc5 | year_2025 | 0.086% | 72.7% | 60% | 0.47 | 0.11 | PASS | no | eval |
| carry_ir3m_tc5 | year_2026 | 0.187% | 75.0% | 70% | 0.91 | 0.95 | PASS | no | eval |
| carry_ir3m_tc5 | holdout_365d | 0.307% | 83.3% | 43% | 1.50 | 1.16 | PASS | no | confirm_only |
| carry_blend_xs | full_sample | -0.022% | 51.7% | 10% | -0.34 | -0.10 | FAIL | no | eval |
| carry_blend_xs | year_2024 | 0.095% | 72.7% | 64% | 0.39 | 0.53 | PASS | no | eval |
| carry_blend_xs | year_2025 | 0.099% | 63.6% | 61% | 0.55 | 0.14 | PASS | no | eval |
| carry_blend_xs | year_2026 | 0.173% | 62.5% | 71% | 0.88 | 1.08 | PASS | no | eval |
| carry_blend_xs | holdout_365d | 0.318% | 75.0% | 46% | 1.50 | 1.30 | PASS | no | confirm_only |

## FTMO risk sweep

**Skipped** — no factor with positive full-sample IS mean (no unused DD budget to invent edge).

## Promote / 1%/mo bar

**Unscaled joint promote:** **NO**. **Scaled primary (`carry_ir3m_xs`) promote:** **NO**. Do not claim 1%/mo unless earned. Locked sleeve untouched; Yahoo/FRED ≠ FTMO MT5; no go-live.

### Honest gaps

- OECD IR3M differentials + CIP formula ≠ observed FX swap / outright forward points.
- Post-GFC cross-currency basis (CIP deviations) not modelled — no Bloomberg XCCY.
- Daily overnight ON rates exist only for USD/EUR/GBP on free FRED — not a G10 panel.
- Academic carry premia are mid–high single-digit ann. with crash risk — not prop 1%/mo.

Artifacts: `scholarly_fx_fwd_carry_*.csv`, `scholarly_fx_fwd_carry_meta.json`.
