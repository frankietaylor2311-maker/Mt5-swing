# Scholarly FX research line (v1)

**Status:** Active (2026-09-17). Replaces the technical **overlay hunt** as the primary path toward FTMO-consistent ~1%/month.
**Data tag:** `approximate_non_ftmo` (Yahoo D1) + free FRED short rates + yfinance VIX + Caldara–Iacoviello GPR.
**Discipline:** `signal_lag≥1`, publication lags on macro, walk-forward / calendar windows, **no holdout tuning**.

---

## 1. What the literature actually supports

### 1.1 Carry

- **Claim (literature):** Currencies with high short-term interest rates tend to appreciate on average vs low-rate currencies (forward-premium puzzle / carry trade). Cross-sectional HML-FX style portfolios earn a positive premium with economically large crash risk in risk-off states.
- **Key refs:** Lustig, Roussanov & Verdelhan (2011), *JFE* / related; Menkhoff, Sarno, Schmeling & Schrimpf (2012a), “Carry Trades and Global Foreign Exchange Volatility,” *JF*; Burnside et al. on peso problems.
- **What we implement:** `strategies/carry_rank.py` — rank G10 by FRED immediate-rate differential vs USD (1-month publication lag), long top / short bottom, mapped to USD majors for FTMO.

### 1.2 Momentum

- **Claim:** Currencies with strong past excess returns continue to outperform over intermediate horizons (cross-sectional momentum); related to time-series momentum in FX.
- **Key refs:** Menkhoff, Sarno, Schmeling & Schrimpf (2012b), “Currency Momentum Strategies,” *JFE*; Asness, Moskowitz & Pedersen (2013) cross-asset momentum.
- **What we implement:** `strategies/fx_momentum.py` — formation ~63 trading days, skip ~21 days (3–1 style), monthly rebalance, n=2/2.

### 1.3 Dollar factor

- **Claim:** The average excess return of foreign currencies vs USD (“dollar”) is a priced factor in FX; many strategies load on it.
- **Key refs:** Lustig, Roussanov & Verdelhan (2011, 2014); related work surveyed in Nucera / Sarno reviews of FX risk premia.
- **What we implement:** `dollar_factor_returns` (equal-weight foreign vs USD) plus a simple lagged sign(“dollar TSMOM”) tradable proxy — **not** a claim of a new alpha.

### 1.4 FX volatility risk

- **Claim:** Innovations in global FX volatility are a state variable: high FX vol predicts carry underperformance; vol-sensitive risk premia matter for sorting currencies.
- **Key refs:** Menkhoff, Sarno, Schmeling & Schrimpf (2012a) *JF* (carry & global FX volatility).
- **Proxy here:** VIX (equity vol) as a freely available **uncertainty / risk-off** proxy — correlated with, but not identical to, global FX vol. Documented limitation.

### 1.5 Geopolitical risk (GPR)

- **Claim:** News-based geopolitical risk rises around wars / major tensions; elevated GPR is associated with risk-off asset moves, including safe-haven USD / CHF / JPY demand in many episodes.
- **Key refs:** Caldara & Iacoviello (2022), “Measuring Geopolitical Risk,” *American Economic Review*; data: https://www.matteoiacoviello.com/gpr.htm
- **What we implement:** `strategies/gpr_regime.py` — lag monthly GPR (+ VIX), z-score vs trailing year, cool gross exposure when stressed; optional USD tilt.

### 1.6 TPU / economic-policy uncertainty

- **Claim:** Trade-policy and economic-policy uncertainty (Baker–Bloom–Davis EPU / TPU) affect FX and risk premia around tariff / policy shocks.
- **Refs:** Baker, Bloom & Davis (2016) and policyuncertainty.com TPU series.
- **v1 status:** **Stub only** — place `data/macro/epu_tpu.csv` manually if needed (`load_epu_tpu_stub`). Not in the default factor board until a PIT CSV is cached.

---

## 2. What we can / cannot claim vs ~1%/month FTMO

| Statement | OK to claim? | Notes |
|-----------|:------------:|-------|
| Carry / momentum / dollar premia exist in long academic samples | **Yes** | With costs, capacity, and crash risk caveats |
| Our Yahoo+FRED replication will match paper Sharpe exactly | **No** | Different samples, costs, pair set, USD mapping |
| Literature factors alone deliver **stable ≥1% mean monthly** with ≥70% positive months under FTMO 10%/5% gates | **No — not supported** | Academic ann. returns are often mid–high single digits with Sharpe ≪ what 1%/mo + high hit-rate implies |
| Overlay-tweaking technical sleeves is the path | **Deprecated** | See `ONE_PCT_MONTH_QUEST.md` ethos update |
| GPR/VIX filter removes carry crash risk | **No** | May dampen; not insurance |
| Results on `approximate_non_ftmo` are golive-ready | **Never** | Need FTMO MT5 exports |

**Implied bar:** 1%/month ≈ 12%+ compounding with very high % positive months is a **prop-firm consistency** goal, not a restatement of the carry premium. Scholarly factors are building blocks / regime tools — success requires honest multi-year windows, not HO-fit overlays.

---

## 3. Point-in-time / publication delay

| Series | Source | Default lag |
|--------|--------|-------------|
| OECD immediate rates (G10) | FRED `IRSTCI01*M156N` | **1 month** |
| Overnight (USD/EUR/GBP) | FRED DFF / ECBDFR / IUDSOIA | **1 day** |
| VIX | Yahoo `^VIX` | **1 day** |
| GPR monthly / daily | Iacoviello Excel → `data/macro/gpr_*.csv` | **1 month** / **1 day** |
| FX prices | Yahoo D1 in `data/history/` | Strategy `signal_lag=1` |

If GPR HTTP is blocked: use `GprIndex.stub()` / drop XLS into `data/macro/` (instructions in `macro_uncertainty.download_gpr`).

---

## 4. FTMO mapping & netting limits

- Academic carry is often a multi-currency cash book. FTMO is a **CFD account**: we map currency weights → **USD majors** (`EURUSD`, `GBPUSD`, `AUDUSD`, `NZDUSD`, `USDJPY`, `USDCAD`, `USDCHF`) with sign conventions in `carry_rank.USD_PAIRS`.
- **Netting:** opposing signals on the same symbol net in `currency_weights_to_pair_weights`. Prefer net USD exposure over hedged cross pairs that inflate margin.
- **Hedging:** FTMO often allows hedges but margin on both legs — research default keeps gross \|w\| modest (top/bottom 2 equal-weight sides).
- **Leverage:** No RF hike in this line’s ethos; regime filter **scales down** only (`cool≤1`).

---

## 5. Code map

| Path | Role |
|------|------|
| `src/mt5_swing/data/fred_rates.py` | FRED download + PIT rate panel |
| `src/mt5_swing/data/macro_uncertainty.py` | VIX, GPR, EPU/TPU stub |
| `src/mt5_swing/strategies/carry_rank.py` | Cross-sectional carry |
| `src/mt5_swing/strategies/fx_momentum.py` | Currency momentum + dollar factor |
| `src/mt5_swing/strategies/gpr_regime.py` | GPR/VIX scale + USD tilt |
| `scripts/scholarly_fx_stats_report.py` | Means, t-stats, bootstrap, year windows, gates |
| `reports/scholarly_fx_stats_v1.md` | Empirical board (regenerated) |

---

## 6. Citations (short)

1. Caldara, D. & Iacoviello, M. (2022). Measuring Geopolitical Risk. *AER*. https://www.matteoiacoviello.com/gpr.htm
2. Menkhoff, L., Sarno, L., Schmeling, M. & Schrimpf, A. (2012). Carry Trades and Global Foreign Exchange Volatility. *Journal of Finance*.
3. Menkhoff, L., Sarno, L., Schmeling, M. & Schrimpf, A. (2012). Currency Momentum Strategies. *Journal of Financial Economics*.
4. Lustig, H., Roussanov, N. & Verdelhan, A. (2011). Common Risk Factors in Currency Markets. *Review of Financial Studies*.
5. Baker, S., Bloom, N. & Davis, S. (2016). Measuring Economic Policy Uncertainty. *QJE*. (EPU/TPU)
6. Surveys / related: work by Sarno; Nucera et al. on FX risk premia / dollar–carry structure (use for framing, not for claiming our sample matches their tables).

---

## 7. Next steps (research, not HO hunt)

1. Add simple transaction-cost / spread haircut on Yahoo D1.
2. Optional true FX-vol proxy (realized G10 vol) beside VIX.
3. Wire EPU/TPU CSV when downloaded.
4. Re-run stats on FTMO CSVs when available — **only then** discuss golive.
