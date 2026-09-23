# Scholarly FX research line (v1)

**Status:** Active (2026-09-23 BST, term-structure / yield-curve FX wave after Menkhoff FX-RV). Replaces the technical **overlay hunt** as the primary path toward FTMO-consistent ~1%/month.
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
- **Proxies here:**
  - **VIX** (equity vol) — free uncertainty / risk-off proxy (combo wave §8). Correlated with, but not identical to, global FX vol.
  - **True FX realized vol** (this wave §15): equal-weight mean of |currency-vs-USD daily returns| on available G10 USD majors; trailing 21d/63d RV → causal z; standalone USD-tilt / innov legs + carry cool / low-vol-only gate. `strategies/fx_realized_vol.py`. corr(FX-RV₂₁, VIX) ≈ 0.44 on this sample — **distinct** from VIX-only.

### 1.5 Geopolitical risk (GPR)

- **Claim:** News-based geopolitical risk rises around wars / major tensions; elevated GPR is associated with risk-off asset moves, including safe-haven USD / CHF / JPY demand in many episodes.
- **Country claim:** High *home-country* GPR (GPRC_*) is associated with subsequent depreciation of that currency vs USD (local-projection / sort evidence in this repo).
- **Key refs:** Caldara & Iacoviello (2022), “Measuring Geopolitical Risk,” *American Economic Review*; data: https://www.matteoiacoviello.com/gpr.htm (country indexes on same monthly export).
- **What we implement:**
  - `strategies/gpr_regime.py` — lag monthly/daily aggregate GPR (+ VIX), cool gross exposure; optional USD tilt.
  - `strategies/country_gpr_fx.py` + `scripts/scholarly_fx_country_gpr_wave.py` — country GPR → FX: lagged long-low/short-high sort + LP β at h=1,3,6 months (`pub_lag=1m`, `signal_lag=1`).


### 1.7 News / event intensity (GDELT / RSS / GPR proxy)

- **Claim:** Spikes in conflict/geopolitics news intensity coincide with risk-off FX (safe-haven USD) in many episodes; event studies around news shocks are a standard macro-finance design.
- **Free feeds tried:** GDELT DOC 2.0 `TimelineVol` (no key; ~3-month reliable window; frequent HTTP 429); Yahoo/Reuters **RSS** headline counts (ethical public feeds — recent items only, not a multi-year panel).
- **Fallback:** Caldara–Iacoviello **daily GPR spikes** as news-based intensity *proxy* (same newspaper-count foundation). Limitation: not signed entity/sentiment NLP.
- **What we implement:** `data/news_events.py` (feed interface + resolver) + `strategies/news_event_fx.py` (event vs control study; optional lagged long-USD rule if gate clears).
- **Paid NLP:** Needed for multi-year multilingual sentiment/entity panels (RavenPack / Refinitiv / Bloomberg / keyed GDELT Cloud). Free stack is insufficient for that claim.


### 1.8 PPP / real exchange-rate value

- **Claim:** Real exchange rates mean-revert toward PPP over long horizons (Rogoff PPP puzzle — half-lives often measured in *years*). Currencies that are expensive in real terms tend to depreciate subsequently.
- **Key refs:** Rogoff (1996), “The Purchasing Power Parity Puzzle,” *JEL*; Taylor & Taylor surveys; related real-FX value / mean-reversion work.
- **What we implement:** `strategies/ppp_real_fx.py` + `scripts/scholarly_fx_ppp_wave.py` — PIT real FX `q = S·(CPI_US/CPI_f)` from FRED CPI levels + Yahoo USD majors; trailing 60m/120m z; long undervalued / short overvalued (`pub_lag=1m`, `signal_lag=1m`). FTMO risk sweep on IS → OOS confirm via `backtest/ftmo_risk_sweep.py`.


### 1.9 Balassa–Samuelson / productivity-adjusted real FX

- **Claim:** Relative productivity in tradables raises relative prices (real appreciation) — Harrod–Balassa–Samuelson. Residuals of real FX vs relative productivity are a productivity-*adjusted* PPP / value signal.
- **Key refs:** Balassa (1964); Samuelson (1964); Chong, Jordà & Taylor (2012), “The Harrod–Balassa–Samuelson Hypothesis,” *IER*; Ricci, Milesi-Ferretti & Lee (real FX & fundamentals).
- **What we implement:** `strategies/balassa_samuelson_fx.py` + `scripts/scholarly_fx_bs_wave.py` — FRED IP *levels* as productivity proxy + CPI real FX; PIT `bs_gap` (z(log q)−z(p)), rolling-OLS `bs_resid`, `bs_prod` channel; pub_lag CPI=1m / IP=2m + `signal_lag=1m`. AUD/NZD/CHF IP missing on FRED.


### 1.10 Term structure / yield-curve FX

- **Claim:** Relative government yield-curve factors (level, slope) vs the USD help price currency risk premia; currencies with steeper curves (vs USD) tend to appreciate on average. Slope × carry interactions appear in Ang–Chen-style work; Lustig–Stathopoulos–Verdelhan study the *term structure* of carry risk premia. Related: Fama (1984) forward-premium / UIP puzzle on short-rate differentials.
- **Key refs:** Chen & Tsang (2013); Ang & Chen; Lustig, Stathopoulos & Verdelhan (2019); Fama (1984).
- **What we implement:** `strategies/yield_curve_fx.py` + `data/fred_yields.py` + `scripts/scholarly_fx_curve_wave.py` — OECD `IRLTLT01*` LT govt − `IRSTCI01*` immediate short = slope; differentials vs USD; PIT `pub_lag=1m` + `signal_lag=1m`. Legs: `curve_slope_xs`, `curve_lt_xs`, `curve_slope_z_xs`, `slope_x_carry`, `curve_ew`, secondary `uip_st_xs` / `uip_ir3m_xs`. Full G10 LT coverage on FRED (EUR EZ + DE bund gap-fill).

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
| Country GPR (GPRC_*) | Same monthly Excel → `gpr_country_monthly.csv` | **1 month** (+ strategy `signal_lag`) |
| OECD immediate rates (extended) | FRED `IRSTCI01*` G10 + SEK/NOK/DKK/MXN/… | **1 month** (see coverage CSV) |
| News intensity (GDELT/RSS/GPR proxy) | GDELT DOC / RSS / `gpr_daily.csv` | **1 day** (+ strategy `extra_lag`/`signal_lag`) |
| CPI index levels (PPP / real FX) | FRED CPIAUCSL / GBRCPIALLMINMEI / … | **1 month** (+ strategy `signal_lag`) |
| IP index levels (Balassa–Samuelson) | FRED INDPRO / *PROINDMISMEI / EA19… | **2 months** (+ strategy `signal_lag`) |
| Commodity futures (CRR) | Yahoo `CL=F`/`HG=F`/`GC=F` | **1 day** (+ strategy `signal_lag`) |
| OECD LT govt yields / curve slope | FRED `IRLTLT01*` − `IRSTCI01*` | **1 month** (+ strategy `signal_lag`) |
| OECD 3m interest rates (UIP secondary) | FRED `IR3TIB01*` | **1 month** (+ strategy `signal_lag`) |
| FX prices | Yahoo D1 in `data/history/` | Strategy `signal_lag=1` |
| Global FX realized vol (Menkhoff) | EW \|ccy ret\| from Yahoo USD majors | Trailing RV then `signal_lag=1` (no pub lag) |

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
| `src/mt5_swing/strategies/scholarly_combo.py` | EW carry+mom+dollar TSMOM; cool carry in high VIX/GPR; USD tilt on GPR |
| `scripts/scholarly_fx_stats_report.py` | Means, t-stats, bootstrap, year windows, gates |
| `scripts/scholarly_fx_combo_wave.py` | Combo board: OLS+Newey–West, top3, FTMO gates, GPR event study |
| `src/mt5_swing/strategies/country_gpr_fx.py` | Country GPR sort + local projections |
| `scripts/scholarly_fx_country_gpr_wave.py` | Country-GPR FX board + FRED coverage |
| `reports/scholarly_fx_stats_v1.md` | Empirical board (v1 factors) |
| `reports/scholarly_fx_combo_wave.md` | Combo wave board |
| `reports/scholarly_fx_country_gpr_wave.md` | Country-GPR wave board (this) |
| `reports/scholarly_fx_fred_rate_coverage.csv` | Extended OECD rate sparsity/missing |
| `src/mt5_swing/strategies/fx_realized_vol.py` | Menkhoff FX-RV level/z/innov + USD tilt + carry cool |
| `scripts/scholarly_fx_rv_wave.py` | FX-RV factor board + FTMO risk sweep |
| `reports/scholarly_fx_rv_wave.md` | FX-RV wave board (this) |

---

## 6. Citations (short)

1. Caldara, D. & Iacoviello, M. (2022). Measuring Geopolitical Risk. *AER*. https://www.matteoiacoviello.com/gpr.htm
2. Menkhoff, L., Sarno, L., Schmeling, M. & Schrimpf, A. (2012). Carry Trades and Global Foreign Exchange Volatility. *Journal of Finance*.
3. Menkhoff, L., Sarno, L., Schmeling, M. & Schrimpf, A. (2012). Currency Momentum Strategies. *Journal of Financial Economics*.
4. Lustig, H., Roussanov, N. & Verdelhan, A. (2011). Common Risk Factors in Currency Markets. *Review of Financial Studies*.
5. Baker, S., Bloom, N. & Davis, S. (2016). Measuring Economic Policy Uncertainty. *QJE*. (EPU/TPU)
6. Surveys / related: work by Sarno; Nucera et al. on FX risk premia / dollar–carry structure (use for framing, not for claiming our sample matches their tables).
7. Rogoff, K. (1996). The Purchasing Power Parity Puzzle. *Journal of Economic Literature*.
8. Chen, Y., Rogoff, K. & Rossi, B. (2010). Can Exchange Rates Forecast Commodity Prices? *QJE*.
9. Dahlquist, M. & Hasseltoft, H. — macro differentials and currency risk premia (framing).
10. Chen, Y. & Tsang, K. (2013). — relative yield-curve factors and exchange rates.
11. Lustig, H., Stathopoulos, A. & Verdelhan, A. (2019). The Term Structure of Currency Carry Trade Risk Premia. *Journal of Finance*.
12. Fama, E. (1984). Forward and Spot Exchange Rates. *Journal of Monetary Economics*. (UIP / forward premium)

---

## 7. Next steps (research, not HO hunt)

1. Add simple transaction-cost / spread haircut on Yahoo D1.
2. **Done:** true FX-vol proxy (realized G10 vol) beside VIX — see §15.
3. Wire EPU/TPU CSV when downloaded.
4. Macro-news NLP panel — **intensity layer wired** (GDELT interface + GPR proxy event study); signed NLP still needs paid API.
5. Re-run stats on FTMO CSVs when available — **only then** discuss golive.
6. Optional: AI-GPR bilateral / role decompositions (initiator vs spillover) if useful beyond GPRC_*.
7. **Done (2026-09-23):** PPP / real-FX value wave — promote=NO (see §13).
8. **Done (2026-09-23):** Commodity CRR + macro-diff waves — promote=NO (see §11–12).
9. **Done (2026-09-23):** Balassa–Samuelson / productivity wave — promote=NO (see §14).
10. **Done (2026-09-23):** True FX realized-vol risk factor (Menkhoff) — promote=NO (see §15).
11. **Done (2026-09-23):** Term-structure / yield-curve FX + UIP secondary — promote=NO (see §16).
12. Next scholarly candidates (not sleeve coolers): free order-flow / positioning proxies if available; FX option-implied vol (if free) vs RV; transaction-cost / swap-aware carry on better forwards; bilateral AI-GPR role decompositions.

---

## 8. Combo wave results (2026-09-17 BST) — carry × mom × dollar TSMOM × GPR/VIX

**Design (fixed priors, no HO tuning):** equal-weight 1/3 sleeves; Menkhoff-style **carry cool=0.35** when `max(z_VIX,z_GPR)≥1`; Caldara–Iacoviello **USD tilt≤0.15** on elevated GPR z; `signal_lag=1`; FRED rates +1m; VIX +1d; GPR daily +1d for regime/events.

### Full-sample (≈180 months)

| Factor | mean_mo | t OLS | t NW | %pos |
|--------|--------:|------:|-----:|-----:|
| carry_rank | −0.001% | −0.02 | −0.02 | 51% |
| fx_momentum | −0.064% | −0.90 | −0.92 | 46% |
| dollar_tsmom (weight map) | +0.012% | 0.09 | 0.11 | 50% |
| combo_ew_raw | −0.016% | −0.30 | −0.34 | 51% |
| **scholarly_combo** | **+0.028%** | 0.36 | 0.38 | 52% |

### Consistency windows (`scholarly_combo`)

| Window | mean_mo | %pos | top3 | gates | clears ≥1%/70%/top3≤55%? |
|--------|--------:|-----:|-----:|:-----:|:------------------------:|
| 2024 | 0.40% | 73% | 57% | PASS | **no** (mean & top3) |
| 2025 | 0.10% | 64% | 84% | PASS | **no** |
| 2026 | 0.36% | 62% | 84% | PASS | **no** |
| holdout_365d | 0.29% | 67% | 74% | PASS | **no** |

**Joint clear / promote:** **NO.** Closest window is 2024 (0.40%/73%) — still ~2.5× below the 1%/mo mean bar. Regime conditioning nudges full-sample mean slightly positive vs raw EW blend but does **not** approach prop-firm consistency.

### GPR top-decile event study (optional)

206 thinned events (top-decile lagged daily GPR, ≥5d apart). At h=+5, USD basket − risk FX ≈ **+0.15%**; at h=+10 ≈ **+0.06%** — mild safe-haven USD edge, economically small vs 1%/mo.

### Honest gap vs goal

Literature multi-factor + uncertainty gates are the right *prior*, but on Yahoo D1 they deliver **basis-point** monthly means, not percent. Missing for a fair FTMO test: broker CSVs, spreads/swap, country-level GPR / macro news NLP, true FX-vol (not only VIX).

Artifacts: `reports/scholarly_fx_combo_wave.md`, `scholarly_fx_combo_*.csv`, `scholarly_fx_gpr_event_study.csv`, `scholarly_fx_combo_meta.json`.

---

## 9. Country-GPR wave results (2026-09-17 BST)

**Design (fixed priors, no HO tuning):** map Caldara–Iacoviello `GPRC_*` → FX currencies (EUR = EW of DEU/FRA/ITA/ESP/NLD/BEL); 1m publication lag + 1m signal lag; trailing 60m z-score; long 2 lowest / short 2 highest home-GPR currencies vs USD. Local projections: cum. h-month FX return on lagged home GPR z.

### Lagged sort (full sample)

| mean_mo | t OLS | t NW | %pos | clears 1%/mo bar? |
|--------:|------:|-----:|-----:|:-----------------:|
| +0.034% | 0.55 | 0.64 | 48% | **NO** |

Year/holdout windows: FTMO 10%/5% gates PASS, but means are basis points and %pos ≪ 70%. Joint clear: **NO**.

### Local projections (pooled) — sign test

| h (months) | β (% per σ) | t | sign supports depreciation? |
|-----------:|------------:|--:|:---------------------------:|
| 1 | −0.067 | −1.12 | yes (weak) |
| 3 | −0.250 | −2.47 | **yes** |
| 6 | −0.579 | −4.28 | **yes** |

Per-currency: EUR/CAD/CHF/AUD mostly negative at h=6; **JPY** mixed/near-zero (safe-haven confounding). Economically small vs 1%/mo; statistically the pooled LP sign matches the prior at h≥3.

### FRED OECD carry expansion

- Extended panel: **26** `IRSTCI01*` series cached (G10 + SEK/NOK/DKK/MXN/KRW/PLN/CZK/HUF/ILS/ZAR/TRY/INR/BRL/CLP/ISK/CNY/RUB/IDR).
- **Sparse/stale:** SEK ends 2020-10 (discontinued on FRED); CHF last 2024-03; NZD last 2024-12; EUR/DKK/CNY/RUB lag several months.
- **Missing (404):** SGD, HKD, THB, PHP, MYR, TWD.
- Artifact: `reports/scholarly_fx_fred_rate_coverage.csv`.

### Honest gaps

| Gap | Status |
|-----|--------|
| News NLP | **Intensity layer wired** (GDELT/RSS interface + GPR proxy); signed NLP **not** wired |
| FTMO MT5 CSVs | **Not present** (`approximate_non_ftmo` Yahoo only) |
| `GPRC_NZL` | **Absent** in 44-country file |
| 1%/mo claim | **Not earned** |

Artifacts: `reports/scholarly_fx_country_gpr_wave.md`, `scholarly_fx_country_gpr_*.csv`, `scholarly_fx_country_gpr_meta.json`.

---

## 10. News/event wave results (2026-09-17 BST)

**Design (fixed priors, no HO tuning):** prefer GDELT DOC TimelineVol for conflict volume; Yahoo RSS counts tried ethically; **multi-year study uses GPR daily spike proxy** (GDELT n=82 days only; Reuters RSS 404). Top-decile lagged intensity events vs random controls; USD vs EUR/GBP/JPY/AUD; lagged long-USD hold only if event−control at h=+5 clears `diff>0` & `t≥1`. Costs 1.5 bps/side. **Do not claim 1%/mo.**

### Feeds

| source | usable for multi-year? | n |
|--------|:---------------------:|--:|
| GDELT DOC TimelineVol | no (short + flaky 429) | 82 |
| Yahoo EURUSD RSS counts | no (recent only) | 13 |
| Reuters world RSS | no (404) | 0 |
| **GPR daily spike proxy** | **yes (proxy)** | 15232 |

### Event vs control (206 events / 206 controls)

| series | h=+5 event | control | diff | t |
|--------|----------:|--------:|-----:|--:|
| EUR | +0.05% | +0.07% | −0.02% | −0.18 |
| GBP | +0.09% | +0.01% | +0.08% | 0.57 |
| JPY | +0.15% | −0.03% | +0.18% | 1.64 |
| AUD | −0.05% | +0.01% | −0.06% | −0.37 |
| **USD_BASKET** | **+0.07%** | **+0.01%** | **+0.06%** | **0.55** |

JPY shows the only mild USD-long edge vs controls; basket gate **fails** (t=0.55 < 1).

### Trading rule

Gate **NO** → gated strategy flat. Forced diagnostic (not for promotion): full-sample mean_mo ≈ +0.06%, %pos ≈ 30%; 2024 ≈ +0.37%/55%; holdout_365d ≈ −0.19%. FTMO 10%/5% PASS on flat/forced paths; **never** clears 1%/70%/top3. Joint clear: **NO**.

### Paid NLP?

**Yes.** Free GDELT/RSS cannot support a multi-year signed NLP panel. GPR is intensity proxy only. Paid options: RavenPack, Refinitiv News Analytics, Bloomberg, or keyed GDELT Cloud history.

Artifacts: `reports/scholarly_fx_news_event_wave.md`, `scholarly_fx_news_event_*.csv`, `scholarly_fx_news_event_meta.json`, cached `data/macro/gdelt_conflict_timeline.csv` + `yahoo_fx_rss_counts.csv`.


---

## 11. Commodity-currency wave results (2026-09-23 BST) — Chen–Rogoff–Rossi

**Design (fixed priors):** lagged commodity momentum → AUD (copper) / CAD (oil) / NZD (basket); pub_lag=1d + signal_lag=1d; formation 63d / skip 21d; costs 1.5 bps/side. Yahoo futures ≠ spot export baskets.

### Full-sample (unscaled)

| Factor | mean_mo | t NW | %pos | clears 1%/mo? |
|--------|--------:|-----:|-----:|:-------------:|
| commodity_country_ts | −0.037% | −0.30 | 38% | **NO** |
| commodity_xs_basket | +0.001% | +0.01 | 38% | **NO** |

Year/holdout: FTMO gates mostly PASS; means are basis points; joint promote: **NO**. Locked sleeve untouched.

Artifacts: `reports/scholarly_fx_commodity_wave.md`, `scholarly_fx_commodity_*.csv`.

---

## 12. Macro-differential wave results (2026-09-23 BST) — Dahlquist-style FRED

**Design (fixed priors):** lagged CPI/IP/UR differentials vs USD; pub_lags CPI/UR=1m, IP=2m + signal_lag=1m; n_long=n_short=2.

### Full-sample

| Factor | mean_mo | t NW | %pos | clears 1%/mo? |
|--------|--------:|-----:|-----:|:-------------:|
| macro_cpi | −0.022% | −0.39 | 46% | **NO** |
| macro_ip | −0.032% | −0.60 | 49% | **NO** |
| macro_ur | −0.008% | −0.15 | 46% | **NO** |
| macro_diff_ew | −0.020% | −0.59 | 46% | **NO** |

Joint promote: **NO**. Artifacts: `reports/scholarly_fx_macro_diff_wave.md`, `scholarly_fx_macro_diff_*.csv`.


---

## 13. PPP / real-FX value wave results (2026-09-23 BST) — Rogoff-style

**Design (fixed priors, no HO tuning):** real FX `q = S·(CPI_US/CPI_f)` from FRED CPI *levels* (pub_lag=1m) + Yahoo USD majors; trailing z at 60m/120m (`min_periods=36`); long 2 undervalued / short 2 overvalued; `signal_lag=1m` + 1d weight lag. Primary for sizing: `ppp_xs_60m`.

**Prop sizing:** IS risk sweep (`ftmo_risk_sweep`) pushes scale just under FTMO 10% static / 5% daily; OOS = last 365d confirm (no retune).

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| ppp_xs_60m | −0.010% | −0.15 | −0.15 | 42% | 12% | −0.03 |
| ppp_ts_60m | −0.014% | −0.25 | −0.30 | 39% | 13% | −0.05 |
| ppp_xs_120m | +0.007% | +0.11 | +0.11 | 43% | 12% | +0.03 |
| ppp_ts_120m | −0.016% | −0.28 | −0.32 | 41% | 13% | −0.06 |
| ppp_xs_ew | −0.001% | −0.02 | −0.02 | 42% | 12% | −0.00 |

### Consistency windows (`ppp_xs_60m`, unscaled)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| 2024 | −0.10% | 64% | 65% | PASS | no |
| 2025 | −0.35% | 27% | 100% | PASS | no |
| 2026 | +0.15% | 75% | 63% | PASS | no |
| holdout_365d | +0.10% | 67% | 52% | PASS | no |

### Risk sweep (IS → OOS)

| Strategy | scale | bind | IS mean_mo | IS static/daily | OOS mean_mo | OOS gates |
|----------|------:|:----:|-----------:|----------------:|------------:|:---------:|
| ppp_xs_60m | 1.15 | static | −0.017% | 10.00% / 4.27% | +0.11% | PASS |
| ppp_xs_120m | 1.15 | static | +0.003% | 10.00% / 4.27% | +0.11% | PASS |

IS already sits on the static DD budget at ~1.15× unit leverage — **no unused headroom**. Scaling does **not** create a 1%/mo edge: OOS mean ≈ +0.11%/mo ≪ 1%.

**Unscaled promote:** **NO**. **Scaled primary promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` unchanged. No technical overlay hunt.

### Honest read

Long-horizon PPP mean reversion is the right *prior*, but on this Yahoo+FRED sample the tradable monthly sort earns **basis points** (often negative) with %pos ≪ 70%. Slow Rogoff half-lives are incompatible with prop-firm monthly consistency without additional (non-overfit) edges. Free CPI levels are sufficient for the relative-z construction; absolute Big-Mac / ICP price levels are not required for this z-score design.

Artifacts: `reports/scholarly_fx_ppp_wave.md`, `scholarly_fx_ppp_*.csv`, `scholarly_fx_ppp_meta.json`, `scholarly_fx_ppp_risk_sweep.csv`.


---

## 14. Balassa–Samuelson / productivity wave results (2026-09-23 BST)

**Design (fixed priors, no HO tuning):** real FX `q = S·(CPI_US/CPI_f)` + relative productivity `p = log(IP_f)−log(IP_US)`; CPI pub_lag=1m, IP pub_lag=2m + `signal_lag=1m`; lookbacks 60m/120m.
- `bs_gap`: score = −(z(log q) − z(p)) — unit-coeff HBS residual after z-score
- `bs_resid`: score = −rolling OLS residual of log(q) on p
- `bs_prod`: score = z(p) — high relative productivity → long foreign
- `bs_ew`: EW of 60m legs

**Coverage:** IP on FRED for USD/EUR/GBP/JPY/CAD only (AUD/NZD/CHF missing). Thin G10 cross-section.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| bs_gap_60m | −0.045% | −0.71 | −0.76 | 41% | 14% | −0.15 |
| bs_resid_60m | +0.109% | +1.72 | +1.62 | 44% | 18% | +0.38 |
| bs_prod_60m | −0.011% | −0.17 | −0.19 | 47% | 12% | −0.07 |
| bs_gap_120m | −0.044% | −0.68 | −0.74 | 40% | 14% | −0.15 |
| bs_resid_120m | +0.117% | +1.81 | +1.73 | 45% | 17% | +0.39 |
| bs_prod_120m | −0.003% | −0.05 | −0.05 | 47% | 10% | −0.04 |
| bs_ew | +0.018% | +0.50 | +0.60 | 53% | 15% | +0.09 |

### Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| bs_resid_60m | holdout_365d | −0.02% | 50% | PASS | no |
| bs_resid_120m | holdout_365d | +0.02% | 50% | PASS | no |
| bs_resid_120m | 2024 | +0.27% | 64% | PASS | no |
| bs_ew | holdout_365d | +0.06% | 58% | PASS | no |
| bs_prod_60m | holdout_365d | +0.22% | 75% | PASS | no† |

† Holdout %pos hits 70% for `bs_prod_60m` but mean ≪ 1% and top3 > 55%; full-sample mean negative — not a promote.

### Risk sweep (IS → OOS)

Positive IS mean on resid/ew → sweep run. Best scaled IS mean (`bs_resid_120m` @ ~2.0×) ≈ **+0.24%/mo** still ≪ 1%; OOS means basis points. Scaled clears: **NO**.

**Unscaled promote:** **NO**. **Scaled primary promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` unchanged.

### Honest read

Rolling-OLS productivity-adjusted residual (`bs_resid`) is the only leg with a mild full-sample positive mean and NW t≈1.6–1.7 — economically **~0.1%/mo**, %pos ≪ 70%. Gap and raw productivity sorts are flat/negative. HBS is the right long-run prior; free IP levels + thin currency set do not deliver prop-firm monthly consistency. Distinct from plain PPP (§13) and macro-diff IP YoY (§12).

Artifacts: `reports/scholarly_fx_bs_wave.md`, `scholarly_fx_bs_*.csv`, `scholarly_fx_bs_meta.json`.


---

## 15. Menkhoff FX realized-vol wave results (2026-09-23 BST) — true G10 FX-RV

**Design (fixed priors, no HO tuning):** daily global FX vol = equal-weight mean of |currency-vs-USD returns| on EUR/GBP/AUD/NZD/JPY/CAD/CHF; trailing RV 21d/63d; z vs 252d (`min_periods=60`); `signal_lag=1`. Cutoffs frozen to match combo priors: `z_high=1`, `z_low=0`, `cool=0.35`. Costs 1.5 bps/side.

**Legs:**
- `fxrv_usd_tilt_*` — continuous long-USD intensity from FX-RV z
- `fxrv_innov_usd_*` — long USD on positive RV innovations (RV − trailing mean)
- `fxrv_highvol_usd_*` — binary long-USD when z ≥ 1 (else flat)
- `carry_fxrv_cool_*` — scholarly carry × FX-RV risk_scale ∈ [0.35, 1]
- `carry_fxrv_lowvol_only_*` — carry only when z ≤ 0 (else flat)
- `carry_raw` / `fxrv_ew_21d` baselines

**Distinctness:** corr(fx_rv_21d, lagged VIX) ≈ **0.44** — overlapping risk-off information but not a VIX clone.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| carry_raw | −0.002% | −0.03 | −0.03 | 51% | 10% | −0.04 |
| fxrv_usd_tilt_21d | +0.021% | +0.52 | +0.52 | 26% | 28% | +0.13 |
| fxrv_innov_usd_21d | +0.032% | +0.77 | +0.76 | 37% | 18% | +0.19 |
| fxrv_highvol_usd_21d | +0.024% | +0.65 | +0.65 | 13% | 32% | +0.17 |
| carry_fxrv_cool_21d | +0.018% | +0.34 | +0.34 | 49% | 12% | +0.02 |
| carry_fxrv_lowvol_only_21d | +0.012% | +0.25 | +0.25 | 40% | — | +0.05 |
| fxrv_usd_tilt_63d | +0.026% | +0.57 | +0.57 | 21% | — | +0.16 |
| fxrv_innov_usd_63d | +0.012% | +0.27 | +0.27 | 29% | — | +0.07 |
| fxrv_highvol_usd_63d | +0.033% | +0.73 | +0.73 | 12% | 31% | +0.20 |
| carry_fxrv_cool_63d | +0.023% | +0.41 | +0.41 | 51% | 12% | +0.04 |
| fxrv_ew_21d | +0.026% | +0.67 | +0.67 | 38% | 22% | +0.17 |

### Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| fxrv_innov_usd_21d | holdout_365d | +0.00% | 33% | PASS | no |
| fxrv_usd_tilt_21d | holdout_365d | −0.02% | 25% | PASS | no |
| carry_fxrv_cool_21d | holdout_365d | +0.29% | 83% | PASS | no† |
| carry_fxrv_cool_63d | holdout_365d | +0.28% | 75% | PASS | no† |
| fxrv_highvol_usd_63d | year_2024 | +0.37% | 27% | PASS | no |

† Holdout %pos can clear 70% on cooled carry, but mean ≪ 1%/mo and full-sample means are basis points — not a promote.

### Risk sweep (IS → OOS)

Positive IS means on standalone USD-tilt / innov legs → sweep run. Best scaled IS mean (`fxrv_highvol_usd_63d` @ ~3.2× daily-bound) ≈ **+0.12%/mo** still ≪ 1%; OOS means near zero / flat. Scaled clears: **NO**. Leverage does not invent a consistency edge.

**Unscaled promote:** **NO**. **Scaled primary (`carry_fxrv_cool_21d`) promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` unchanged.

### Honest read

True FX-RV is the right *prior* relative to VIX-only: mild positive full-sample means on long-USD-in-high-FX-vol rules (NW t ≈ 0.5–0.8) and a small lift of carry when cooled by FX-RV vs raw carry. Economically **~0.02–0.03%/mo** with %pos often ≪ 50% on standalone legs (sparse risk-off episodes). Far from prop-firm 1%/mo + 70% hit-rate. Yahoo D1 abs-return average ≠ OTC tick / option-implied FX vol.

Artifacts: `reports/scholarly_fx_rv_wave.md`, `scholarly_fx_rv_*.csv`, `scholarly_fx_rv_meta.json`.


---

## 16. Term-structure / yield-curve FX wave results (2026-09-23 BST) — Chen–Tsang / Ang–Chen

**Design (fixed priors, no HO tuning):** OECD LT govt (`IRLTLT01*`) − immediate short (`IRSTCI01*`) = slope; differentials vs USD; `pub_lag=1m` + `signal_lag=1m`; n_long=n_short=2; costs 1.5 bps/side. EUR EZ LT with DE bund gap-fill.

**Legs:**
- `curve_slope_xs` — long steep / short flat slope differentials (primary)
- `curve_lt_xs` — long high / short low LT yield differentials (level / long-carry)
- `curve_slope_z_xs` — same sort on trailing 60m z of slope_diff
- `slope_x_carry` — score = z(slope_diff)×z(st_diff); long high interaction
- `curve_ew` — EW of slope_xs + lt_xs
- `uip_st_xs` / `uip_ir3m_xs` — secondary UIP / forward-premium on IRSTCI and IR3TIB differentials

**Coverage:** Full G10 LT on FRED (USD/EUR/GBP/JPY/AUD/CAD/CHF/NZD). IR3M also full G10. Distinct from short-rate carry (§1.1) via the *slope* and LT-level channels.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| curve_slope_xs | +0.045% | +0.71 | +0.71 | 48% | 13% | +0.18 |
| curve_lt_xs | −0.006% | −0.07 | −0.08 | 50% | 11% | −0.05 |
| curve_slope_z_xs | +0.039% | +0.64 | +0.75 | 48% | 12% | +0.17 |
| slope_x_carry | −0.024% | −0.37 | −0.42 | 51% | 13% | −0.04 |
| uip_st_xs | −0.033% | −0.42 | −0.51 | 52% | 10% | −0.13 |
| uip_ir3m_xs | −0.033% | −0.42 | −0.49 | 53% | 10% | −0.13 |
| curve_ew | +0.020% | +0.34 | +0.42 | 50% | 14% | +0.06 |

### Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| curve_slope_xs | holdout_365d | −0.09% | 33% | PASS | no |
| curve_slope_xs | year_2024 | −0.11% | 45% | PASS | no |
| curve_lt_xs | holdout_365d | +0.32% | 75% | PASS | no† |
| uip_ir3m_xs | holdout_365d | +0.31% | 83% | PASS | no† |
| curve_ew | holdout_365d | +0.12% | 75% | PASS | no† |
| curve_slope_z_xs | holdout_365d | +0.08% | 50% | PASS | no |

† Holdout %pos can clear 70% on some legs, but mean ≪ 1%/mo and full-sample means are basis points — not a promote.

### Risk sweep (IS → OOS)

Positive IS means on slope / z / ew → sweep run. Best scaled IS mean (`curve_slope_z_xs` @ ~3.75× daily-bound) ≈ **+0.13%/mo** still ≪ 1%; primary `curve_slope_xs` scaled IS ≈ +0.11%/mo; OOS means mixed / near zero. Scaled clears: **NO**. Leverage does not invent consistency.

**Unscaled promote:** **NO**. **Scaled primary (`curve_slope_xs`) promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` unchanged.

### Honest read

Relative slope is the right *prior*: mild positive full-sample means on `curve_slope_xs` / `curve_slope_z_xs` (NW t ≈ 0.7) vs flat/negative LT-level and UIP/carry legs on this Yahoo+FRED sample. Economically **~0.04%/mo** with %pos ≪ 70%. Slope×carry interaction does not help here. Far from prop-firm 1%/mo + 70% hit-rate. Simple OECD LT−ST slope ≠ full Nelson–Siegel curve factors or swap-implied forwards.

Artifacts: `reports/scholarly_fx_curve_wave.md`, `scholarly_fx_curve_*.csv`, `scholarly_fx_curve_meta.json`.
