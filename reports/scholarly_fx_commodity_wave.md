# Scholarly FX: commodity currencies (Chen–Rogoff–Rossi)

**Prior (fixed):** lagged commodity price momentum co-moves with / forecasts commodity-currency FX (AUD, CAD, NZD). pub_lag=1d + signal_lag=1d; formation=63d, skip=21d; costs=1.5 bps/side.

**Data:** `approximate_non_ftmo` Yahoo D1 FX + Yahoo commodity futures/ETF (`CL=F`, `HG=F`, `GC=F`, basket=`DBC`). Futures ≠ spot export baskets (CRR limitation).

**Map:** AUD→copper, CAD→oil, NZD→basket (no free dairy proxy).

## Factor board (full sample)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| commodity_country_ts | -0.037% | -0.23 | -0.30 | 37.8% | 11% | -0.06 |
| commodity_xs_basket | 0.001% | 0.01 | 0.01 | 37.8% | 20% | -0.00 |

## Calendar / holdout windows

| Strategy | Window | mean_mo | %pos | top3 | t NW | Sharpe | gates | 1% bar | role |
|----------|--------|--------:|-----:|-----:|-----:|-------:|:-----:|:------:|------|
| commodity_country_ts | full_sample | -0.037% | 37.8% | 11% | -0.30 | -0.06 | FAIL | no | eval |
| commodity_country_ts | year_2024 | -0.011% | 45.5% | 94% | -0.04 | -0.54 | PASS | no | eval |
| commodity_country_ts | year_2025 | 0.620% | 54.5% | 79% | 1.10 | 0.70 | PASS | no | eval |
| commodity_country_ts | year_2026 | -0.232% | 37.5% | 100% | -0.58 | 0.57 | PASS | no | eval |
| commodity_country_ts | holdout_365d | 0.406% | 50.0% | 68% | 0.72 | 0.41 | PASS | no | confirm_only |
| commodity_xs_basket | full_sample | 0.001% | 37.8% | 20% | 0.01 | -0.00 | PASS | no | eval |
| commodity_xs_basket | year_2024 | -0.207% | 36.4% | 97% | -1.00 | -1.31 | PASS | no | eval |
| commodity_xs_basket | year_2025 | -0.121% | 27.3% | 100% | -0.86 | -0.39 | PASS | no | eval |
| commodity_xs_basket | year_2026 | 0.118% | 62.5% | 72% | 1.10 | 1.27 | PASS | no | eval |
| commodity_xs_basket | holdout_365d | 0.196% | 66.7% | 50% | 1.42 | 0.90 | PASS | no | confirm_only |

## Local projections (diagnostic)

Prior: β > 0 (commodity ↑ → commodity FX appreciates vs USD).

| scope | ccy | regressor | h | β | t | n | sign_ok |
|-------|-----|-----------|--:|--:|--:|--:|:-------:|
| currency | AUD | commodity_mom | 1 | -0.00365 | -0.18 | 180 | no |
| currency | AUD | commodity_z | 1 | -0.00016 | -0.10 | 180 | no |
| currency | CAD | commodity_mom | 1 | -0.00829 | -1.72 | 180 | no |
| currency | CAD | commodity_z | 1 | -0.00020 | -0.16 | 180 | no |
| currency | NZD | commodity_mom | 1 | -0.03707 | -1.89 | 180 | no |
| currency | NZD | commodity_z | 1 | -0.00162 | -1.26 | 180 | no |
| pooled | POOLED | commodity_mom | 1 | -0.01119 | -1.51 | 540 | no |
| currency | AUD | commodity_mom | 3 | -0.03098 | -0.75 | 178 | no |
| currency | AUD | commodity_z | 3 | 0.00113 | 0.33 | 178 | yes |
| currency | CAD | commodity_mom | 3 | -0.00890 | -0.80 | 178 | no |
| currency | CAD | commodity_z | 3 | -0.00162 | -0.53 | 178 | no |
| currency | NZD | commodity_mom | 3 | -0.06731 | -1.19 | 178 | no |
| currency | NZD | commodity_z | 3 | -0.00516 | -1.53 | 178 | no |
| pooled | POOLED | commodity_mom | 3 | -0.01951 | -1.67 | 534 | no |

## Promote / 1%/mo bar

**Joint promote:** **NO**. Consistency clear requires mean_mo≥1%, %pos≥70%, top3≤55%, and FTMO gates — not earned unless tables above show YES.

### What this does NOT support

- No claim that commodity-FX alone delivers FTMO ~1%/mo consistency.
- No go-live; Yahoo futures ≠ FTMO MT5; no overlay on locked sleeve.
- CRR often study commodity *prices* forecasted by FX — we test the tradable commodity→FX direction with lags; results are sample-specific.

Artifacts: `scholarly_fx_commodity_*.csv`, `scholarly_fx_commodity_meta.json`.
