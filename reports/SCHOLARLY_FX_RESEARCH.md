# Scholarly FX research line (v1)

**Status:** Active (2026-09-17, news/event wave after country-GPR). Replaces the technical **overlay hunt** as the primary path toward FTMO-consistent ~1%/month.
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
| `src/mt5_swing/strategies/scholarly_combo.py` | EW carry+mom+dollar TSMOM; cool carry in high VIX/GPR; USD tilt on GPR |
| `scripts/scholarly_fx_stats_report.py` | Means, t-stats, bootstrap, year windows, gates |
| `scripts/scholarly_fx_combo_wave.py` | Combo board: OLS+Newey–West, top3, FTMO gates, GPR event study |
| `src/mt5_swing/strategies/country_gpr_fx.py` | Country GPR sort + local projections |
| `scripts/scholarly_fx_country_gpr_wave.py` | Country-GPR FX board + FRED coverage |
| `reports/scholarly_fx_stats_v1.md` | Empirical board (v1 factors) |
| `reports/scholarly_fx_combo_wave.md` | Combo wave board |
| `reports/scholarly_fx_country_gpr_wave.md` | Country-GPR wave board (this) |
| `reports/scholarly_fx_fred_rate_coverage.csv` | Extended OECD rate sparsity/missing |

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
4. Macro-news NLP panel — **intensity layer wired** (GDELT interface + GPR proxy event study); signed NLP still needs paid API.
5. Re-run stats on FTMO CSVs when available — **only then** discuss golive.
6. Optional: AI-GPR bilateral / role decompositions (initiator vs spillover) if useful beyond GPRC_*.

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
