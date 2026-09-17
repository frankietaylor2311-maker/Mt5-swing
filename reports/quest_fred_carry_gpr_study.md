# Quest: FRED OECD carry + GPR/VIX regime (+ FX momentum)

**Generated:** 2026-09-17 07:33 BST  
**Data source:** `approximate_non_ftmo` (Yahoo D1) + FRED OECD immediate rates + Caldara–Iacoviello GPR + Yahoo VIX  
**Initial equity:** 10,000  
**Priors (fixed, no grid):** `n_long=n_short=2`, monthly rebalance, `signal_lag=1`, FRED `pub_lag_months=1`, GPR/VIX lag 1 day, momentum formation=63 / skip=21  
**Locked sleeve:** not modified (`fx4plus_gbpcad_d1_voltarget_0025` untouched)  
**Promote:** **NO** — carry_rank: NO — HO mean_mo=0.0020915681127683847 pos=0.6666666666666666 gates=True; carry_rank_gpr_vix: NO — HO mean_mo=0.001413413696472382 pos=0.5 gates=True; fx_momentum: NO — HO mean_mo=0.0038357461162056294 pos=0.75 gates=True; fred_carry_ew: NO — HO mean_mo=0.0005504490500872669 pos=0.5833333333333334 gates=True. Literature FX carry/momentum premia ≠ FTMO ~1%/month consistency; do not promote on approximate_non_ftmo.

## Literature

- Lustig, Roussanov & Verdelhan — cross-sectional FX carry / dollar–HML factors
- Menkhoff, Sarno, Schmeling & Schrimpf (2012a) *JF* — carry & global FX volatility
- Menkhoff et al. (2012b) *JFE* — currency momentum
- Caldara & Iacoviello (2022) *AER* — geopolitical risk (GPR)

## Point-in-time / ffill note

Rates are loaded with `pub_lag_months=1` then **forward-filled** so sparse
trailing NaNs on EUR / CHF / NZD (FRED series end earlier than USD/GBP/etc.)
do not empty the cross-section. Leading NaNs are not back-filled.
NaN counts before ffill (full panel): `{'USD': 0, 'EUR': 481, 'GBP': 282, 'AUD': 433, 'NZD': 386, 'JPY': 372, 'CAD': 246, 'CHF': 239}`.
Stale last-prints for CHF/NZD/EUR after their final FRED observation are a
documented limitation — not an invitation to invent rates.

## Windows

| Window key | Label |
|------------|-------|
| `2024` | calendar_2024 (research/WF) |
| `2025` | calendar_2025 (research/WF) |
| `2026` | calendar_2026_YTD (partial / mixed) |
| `holdout_365d` | pure holdout last ~365 calendar days (no tuning) |

## Results by sleeve × window

| Sleeve | Window | Mean mo | %pos mo | Total ret | Static DD | Daily DD | Gates | Sharpe |
|--------|--------|---------|---------|-----------|-----------|----------|-------|--------|
| carry_rank | 2024 | 0.06% | 72.73% | 1.58% | 0.57% | 0.93% | PASS | 0.39 |
| carry_rank | 2025 | -0.01% | 63.64% | -0.67% | 3.45% | 1.73% | PASS | -0.15 |
| carry_rank | 2026 | 0.09% | 50.00% | 1.25% | 0.07% | 0.49% | PASS | 0.57 |
| carry_rank | holdout_365d | 0.21% | 66.67% | 1.63% | 0.97% | 0.53% | PASS | 0.59 |
| carry_rank_gpr_vix | 2024 | 0.59% | 72.73% | 7.27% | 0.10% | 1.28% | PASS | 1.41 |
| carry_rank_gpr_vix | 2025 | -0.71% | 45.45% | -8.48% | 9.16% | 2.47% | PASS | -1.48 |
| carry_rank_gpr_vix | 2026 | 0.25% | 62.50% | 1.19% | 2.43% | 1.08% | PASS | 0.37 |
| carry_rank_gpr_vix | holdout_365d | 0.14% | 50.00% | 0.77% | 3.07% | 1.07% | PASS | 0.13 |
| fx_momentum | 2024 | -0.17% | 45.45% | -2.09% | 4.48% | 0.72% | PASS | -0.64 |
| fx_momentum | 2025 | 0.44% | 90.91% | 4.51% | 0.73% | 1.31% | PASS | 1.28 |
| fx_momentum | 2026 | 0.34% | 62.50% | 3.09% | 0.00% | 0.40% | PASS | 1.65 |
| fx_momentum | holdout_365d | 0.38% | 75.00% | 5.45% | 0.00% | 0.41% | PASS | 2.15 |
| fred_carry_ew | 2024 | 0.35% | 63.64% | 5.83% | 0.05% | 0.81% | PASS | 1.37 |
| fred_carry_ew | 2025 | -0.39% | 36.36% | -4.52% | 5.51% | 1.45% | PASS | -1.05 |
| fred_carry_ew | 2026 | 0.28% | 62.50% | 0.13% | 1.99% | 0.89% | PASS | 0.11 |
| fred_carry_ew | holdout_365d | 0.06% | 58.33% | 0.88% | 1.52% | 0.89% | PASS | 0.24 |

## Sleeve coverage

- `carry_rank`: 2011-09-15 → 2026-09-15 (3905 days)
- `carry_rank_gpr_vix`: 2011-09-15 → 2026-09-15 (3905 days)
- `fx_momentum`: 2011-09-15 → 2026-09-15 (3905 days)
- `fred_carry_ew`: 2011-09-15 → 2026-09-15 (3905 days)

## Interpretation / go-live

- This is a **research replication** of scholarly FX factors on approximate_non_ftmo data.
- Academic carry/momentum Sharpe is typically well below what a stable ≥1%/month
  FTMO hit-rate implies; crash risk in risk-off remains.
- GPR/VIX cooling may dampen stress periods; it is **not** insurance.
- **Do not go live** without FTMO MT5 history exports matching challenge symbols/feed.

CSV: `quest_fred_carry_gpr_study.csv`

