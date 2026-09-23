# Scholarly FX research line (v1)

**Status:** Active (2026-09-23 BST, industrial-production PRINTO01 §42 after employment §41 / WUI §40 / OECD BCI §39 / CCI §38 / CLI §37; FX IV/RR + news-sentiment still blocked). Replaces the technical **overlay hunt** as the primary path toward FTMO-consistent ~1%/month.
**Data tag:** `approximate_non_ftmo` (Yahoo D1) + free FRED short rates / OECD IR3M money-market + Chicago NFCI/ANFCI + TED/CPFF/BAA + yfinance VIX + Caldara–Iacoviello GPR + free CFTC TFF/Legacy COT + Baker–Bloom–Davis EPU/TPU + IMF BOP CA/GDP (`{ISO3}B6BLTT02STSAQ`) + Fed/ECB/BoJ CB assets (`WALCL` / `ECBASSETSW` / `JPNASSETS`) + US TIPS/BE (`DFII10` / `T10YIE`) + Caldara–Iacoviello AI-GPR daily roles (`ai_gpr_daily.csv` threats/acts/oil-region) + IMF WEO fiscal balance (`GGNLBA*188N`) + US MTS (`MTSDS133FMS`) + IMF WEO gross debt (`GGGDTA*188N`; `GGXWDG*` 404) + US federal debt/GDP Q (`GFDEGDQ188S`) + BIS private credit/GDP (`Q*PAM770A`; `CRDQ*APABIS` absolute unused) + OECD broad-money growth (`MABMM301*M657S`; NZD/CHF 657S stale) + IMF IFS total reserves excl. gold (`TRESEG*M052N`; EUR=`TRESEGDEM052N`; NZD/CHF 404) + BIS real residential HPI (`Q*R628BIS`; EUR=`QDER628BIS`, `QEUR628BIS` 404) + ICE BofA OAS (`BAMLC0A0CM` IG / `BAMLH0A0HYM2` HY / `BAMLC0A4CBBB` BBB; public CSV ~3y ICE truncation). + OECD CLI amplitude-adjusted (`*LOLITOAASTSAM`; EUR=`DEULOLITOAASTSAM`; NZD/CHF stale unmapped). + OECD CCI balances (`CSCICP02*M460S`; USD=`USACSCICP02STSAM`; EUR=`CSCICP02EZM460S`; CAD/NZD/CHF unmapped; `CSCICP03*M665S` amplitude stale ~2024-01). + OECD BCI balances (`BSCICP02*M460S`; EUR=`BSCICP02EZM460S`; CHF live; AUD/CAD/NZD/JPY unmapped; `BSCICP03*M665S` amplitude stale ~2023-11..2024-01; NAPM/ISM 404). + Ahir–Bloom–Furceri country WUI (`WUIUSA`/`WUIDEU`/`WUIGBR`/`WUIJPN`/`WUICAN`/`WUIAUS`/`WUINZL`/`WUICHE`; EUR=Germany proxy; full G10 live ~2026-04; quarterly→monthly after pub_lag=4). + OECD MEI employment persons LFEMTTTT (`LFEMTTTTUSM647S`/`JPM647S`/`CAM647S`/`AUM647S` monthly; `LFEMTTTTDEQ647S` EUR Germany proxy — EZ Q647S stale 2022-10; `GBQ647S`/`NZQ647S`/`CHQ647S` quarterly; M657S/Q657S YoY stale/404 — not primary; YoY of levels; pub_lag=3). + OECD MEI industrial production YoY (`USAPRINTO01GYSAM`/`GBR`/`JPN`/`CAN` monthly; EUR=`FRAPRINTO01GYSAM` France proxy — EA19/DEU GYSAM stale; AUD/NZD/CHF=`*PRMNTO01GYSAQ` manufacturing quarterly — industry PRINTO01 404; PROINDMISMEI stale ~2024-03; pub_lag=2).
**Discipline:** `signal_lag≥1`, publication lags on macro, walk-forward / calendar windows, **no holdout tuning**.

---

## 1. What the literature actually supports

### 1.1 Carry

- **Claim (literature):** Currencies with high short-term interest rates tend to appreciate on average vs low-rate currencies (forward-premium puzzle / carry trade). Cross-sectional HML-FX style portfolios earn a positive premium with economically large crash risk in risk-off states.
- **Key refs:** Lustig, Roussanov & Verdelhan (2011), *JFE* / related; Menkhoff, Sarno, Schmeling & Schrimpf (2012a), “Carry Trades and Global Foreign Exchange Volatility,” *JF*; Burnside et al. on peso problems.
- **What we implement:** `strategies/carry_rank.py` — rank G10 by FRED immediate-rate differential vs USD (1-month publication lag), long top / short bottom, mapped to USD majors for FTMO. Forward-proxy upgrade: §1.12 / wave §19 (`IR3TIB01*` + CIP-implied FD).


### 1.12 Swap- / forward-aware carry (CIP / money-market proxy)

- **Claim:** Academic HML-FX sorts on *forward discounts* (not policy cash rates alone). Under CIP, forward discounts ≈ interest differentials at the forward tenor; Lustig–Roussanov–Verdelhan (2011) / Lustig–Verdelhan (2007) document a large carry premium with crash risk.
- **Key refs:** Lustig, Roussanov & Verdelhan (2011), *JFE*; Lustig & Verdelhan (2007); Menkhoff et al. (2012a); Fama (1984) forward-premium puzzle.
- **Free data reality:** True FX swap / outright forward points are vendor (Bloomberg, Refinitiv, broker). Free FRED OECD ``IR3TIB01*`` 3M money-market rates are the best G10 tenor match; ``IRSTCI01*`` immediate rates are the cash baseline; daily ON (DFF/ECBDFR/IUDSOIA) is too sparse for a G10 panel.
- **What we implement (wave §19):** `data/fred_forward_carry.py` + `strategies/forward_carry_fx.py` + `scripts/scholarly_fx_fwd_carry_wave.py` — IR3M XS, IRSTCI XS, CIP-implied FD XS, IR3M EW, blend, 5 bps TC haircut. PIT `pub_lag=1m` + `signal_lag=1m`. **Explicit:** still rate approximation — not observed forwards; post-GFC CIP basis unmodelled.


### 1.14 Funding liquidity / financial conditions (NFCI, TED, CP)

- **Claim:** Funding-liquidity spirals and tight financial conditions coincide with carry crashes and safe-haven USD demand; intermediaries withdraw when money-market / CP spreads and NFCI tighten.
- **Key refs:** Brunnermeier, Nagel & Pedersen (2008), "Carry Trades and Currency Crashes"; Menkhoff et al. (2012a) related FX-vol channel; Chicago Fed NFCI/ANFCI documentation.
- **What we implement (wave §20):** `data/fred_funding_liquidity.py` + `strategies/funding_liquidity_fx.py` + `scripts/scholarly_fx_funding_liq_wave.py` — NFCI/ANFCI USD tilts (z and level>0), ΔNFCI change tilts, TED/CPFF/BAA spreads, carry×NFCI cool / loose-only gate, carry×CPFF cool. PIT weekly `pub_lag=7d`, daily `pub_lag=1d`, `signal_lag=1`. **Explicit:** distinct from VIX/GPR/FX-RV/EPU; **not** overlaid on the locked sleeve.


### 1.15 Global imbalances / current-account FX

- **Claim:** External imbalances (current account / NFA) price currency risk premia and forecast dollar adjustment. Debtor (deficit) currencies earn a risk premium in Della Corte–Riddiough–Sarno; US external imbalance predicts FX adjustment in Gourinchas–Rey.
- **Key refs:** Della Corte, Riddiough & Sarno (2016), "Currency Premia and Global Imbalances," *RFS*; Gourinchas & Rey (2007), "International Financial Adjustment," *JPE*.
- **What we implement (wave §21):** `data/fred_current_account.py` + `strategies/current_account_fx.py` + `scripts/scholarly_fx_ca_wave.py` — IMF BOP CA/GDP % via FRED `{ISO3}B6BLTT02STSAQ`; legs `ca_debtor_xs` (primary), `ca_surplus_xs`, `ca_chg_xs`, `us_ca_gr_fx`, `us_ca_haven_usd`, `ca_ew`. PIT `pub_lag_quarters=2` (=6m) + `signal_lag=1m` + 1d weight lag. EUR: EA19 + DEU gap-fill. **Distinct** from PPP/BS/macro-diff (CPI/IP/UR) and Hau–Rey equity. **Not** overlaid on the locked sleeve.

### 1.16 Central-bank balance-sheet / QE differential FX

- **Claim:** Large-scale asset purchases / QE expand the CB balance sheet, compress term premia (portfolio-balance), and depreciate the expanding CB's currency vs peers (signalling + portfolio-balance channels). Relative BS growth differentials across CBs are a natural FX state variable.
- **Key refs:** Gagnon, Raskin, Remache & Sack (2011), "The Federal Reserve's Large-Scale Asset Purchases," *IJCB*; Neely (2015), "Unconventional Monetary Policy Effects on Exchange Rates," *JBF*; Bauer & Neely (2014), "International Channels of the Fed's Unconventional Monetary Policy," *JIMF*.
- **Free data:** FRED `WALCL` (Fed weekly), `ECBASSETSW` (ECB weekly), `JPNASSETS` (BoJ monthly). `UKASSETS` discontinued 2014-09 — excluded from primary panel. Optional GDP: `GDP` / `EUNNGDP` / `JPNNGDP` for BS/GDP ratios (within-currency z only — units heterogeneous).
- **What we implement (wave §22):** `data/fred_cb_balance_sheet.py` + `strategies/cb_balance_sheet_fx.py` + `scripts/scholarly_fx_cb_bs_wave.py` — legs `walcl_pb_fx` (primary Neely PB), `walcl_haven_usd`, `walcl_chg_pb_fx`, `bs_diff_pb_fx` (US−peer YoY), `bs_peer_xs` (EUR/JPY), `bs_gdp_pb_fx`, `bs_ew`. PIT weekly `pub_lag=7d`, monthly `pub_lag=1m`, GDP `pub_lag=1Q` + `signal_lag=1d`. YoY growth avoids FX conversion of level units. **Distinct** from NFCI funding (§20) and CA/GDP (§21). **Not** overlaid on the locked sleeve.



### 1.17 Real-rate / breakeven inflation differentials

- **Claim:** Real interest differentials (RID) and inflation expectations price FX: currencies with higher *real* rates tend to appreciate (Frankel); US TIPS real yields and breakevens are market state variables for the dollar; foreign LT−CPI proxies approximate real rates where linkers are unavailable.
- **Key refs:** Frankel (1979), "On the Mark," *AER*; Meese & Rogoff (1988), "Was It Real?," *JF*; Dahlquist & Hasseltoft (2013), "International Bond Risk Premia," *JIE*; Lustig, Stathopoulos & Verdelhan (2019), "Term Structure of Currency Carry Trade Risk Premia," *JF*; Hofmann, Shim & Shin (BIS) on bond risk premia / real rates and FX.
- **Free data:** FRED `DFII10` (US 10y TIPS real), `T10YIE` (10y breakeven), `DGS10` (nominal); foreign OECD LT (`IRLTLT01*`) − CPI YoY (honest proxy — **not** true linkers).
- **What we implement (wave §23):** `data/fred_real_rates.py` + `strategies/real_rate_fx.py` + `scripts/scholarly_fx_real_rate_wave.py` — legs `us_real_usd` (primary), `us_real_chg_usd`, `us_be_fx`, `us_be_usd`, `rr_xs`, `rr_z_xs`, `rr_chg_xs`, `rr_ew`. PIT daily `pub_lag=1d`, monthly `pub_lag=1m` + `signal_lag=1d/1m`. **Distinct** from nominal yield-curve (§16), PPP/CPI, and CB-BS (§22). **Not** overlaid on the locked sleeve.


### 1.18 Terms-of-trade / commodity currencies (Cashin–CCS refinement)

- **Claim:** Commodity-currency real exchange rates co-move with country *terms of trade* (export commodity prices relative to import prices), not only with a single export commodity. Positive ToT shocks appreciate commodity currencies vs USD.
- **Key refs:** Cashin, Céspedes & Sahay (2004), "Commodity Currencies and the Real Exchange Rate," *JDE*; Chen, Rogoff & Rossi (2010), *QJE* (export-commodity side); Amano & van Norden (oil–CAD).
- **What we implement (wave §24):** `COUNTRY_TOT_MAP` in `data/commodity_prices.py` + `strategies/tot_fx.py` + `scripts/scholarly_fx_tot_wave.py` — ToT change = export_mom − import_mom (AUD copper−oil, CAD oil−copper, NZD basket−oil; NOK/ZAR mapped, untraded). Legs `tot_country_ts`, `tot_xs`, `tot_vs_g10`, `tot_ew`. PIT pub_lag=1d + signal_lag=1d; calendar-date align. **Distinct** from CRR single-commodity mom (§11 / `commodity_fx.py`); corr≈0.71 / 0.31 on this sample. **Not** overlaid on the locked sleeve.



### 1.19 Bilateral AI-GPR role decompositions (threats / acts / oil-region)

- **Claim:** Geopolitical *threats* and *acts*, and oil-region GPR roles, associate with risk-off / safe-haven USD (and often CHF/JPY) and commodity-currency pressure — a finer decomposition than aggregate GPR or country-GPRC_* sorts.
- **Key refs:** Caldara & Iacoviello (2022), *AER*; AI-GPR / oil-region role indices (Iacoviello GPR page).
- **What we implement (wave §25):** `data/ai_gpr.py` + `strategies/ai_gpr_fx.py` + `scripts/scholarly_fx_ai_gpr_wave.py` — legs `ai_threats_usd` (primary), `ai_acts_usd`, `ai_gpr_usd`, `oil_gpr_usd`, `oil_threats_usd`, `oil_me_vs_non`, `carry_ai_threats_cool` (scholarly only), `ai_ew`. PIT daily `pub_lag=1d` + `signal_lag=1d`; trailing z=252d; binary z≥1; costs 1.5 bps/side. **Distinct** from aggregate GPR regime (§8), country-GPRC sorts (§9), news events (§10), CRR/ToT. **Not** overlaid on the locked sleeve.



### 1.20 Currency crash risk / return skewness (Brunnermeier–Nagel–Pedersen)

- **Claim:** Currencies with more *negative* return skewness (crash risk) earn a premium on average; carry trades load on this crash risk and unwind violently in risk-off states.
- **Key refs:** Brunnermeier, Nagel & Pedersen (2008), "Carry Trades and Currency Crashes," *RFS*. Related: Menkhoff et al. (2012a) FX-vol channel (level vol ≠ skew).
- **Free data:** Yahoo D1 OHLC only — trailing return skewness and left-tail shortfall (mean of returns ≤5th pct). No paid FX IV/RR / risk-reversal panel.
- **What we implement (wave §26):** `strategies/fx_crash_skew_fx.py` + `scripts/scholarly_fx_crash_skew_wave.py` — legs `crash_skew_xs` (primary 63d), `crash_skew_xs_126`, `left_tail_xs`, `mom_skew_regime`, scholarly `carry_crash_cool` (**not** on locked fx4plus), `crash_ew`. PIT `skip=1` + `signal_lag=1`; monthly XS rebalance; costs 1.5 bps/side. **Distinct** from Menkhoff FX-RV (§15), Lustig carry, AI-GPR (§25). **Not** overlaid on the locked sleeve.


### 1.21 Dollar-factor beta sorts (Lustig–Verdelhan)

- **Claim:** The average excess return of foreign currencies vs USD ("dollar" / RX) is a priced FX factor; currencies' rolling β on RX sorts into high-$β / low-$β portfolios with a premium that can flip with the average forward discount (AFD).
- **Key refs:** Lustig, Roussanov & Verdelhan (2011), *RFS*; Lustig, Roussanov & Verdelhan (2014); Verdelhan dollar-factor / beta-sort follow-ons.
- **What we implement (wave §27):** `strategies/dollar_beta_fx.py` + `scripts/scholarly_fx_dollar_beta_wave.py` — legs `dollar_beta_xs` (primary 60m monthly OLS β HML), `dollar_beta_xs_36m`, `dollar_beta_afd` (HML × sign(AFD) from FRED short rates), `dollar_rx_tsmom` (12m trailing RX sign), `dollar_ew`. PIT `skip=1m` + `signal_lag=1m`; monthly XS rebalance; costs 1.5 bps/side. **Distinct** from Lustig carry (rate sort), Menkhoff FX-RV, Hau–Rey equity-diff, BNP crash-skew (§26), AI-GPR. **Not** overlaid on the locked sleeve.






### 1.22 Fiscal-balance / government-budget differentials (twin deficits)

- **Claim:** Relative fiscal stance (general-government net lending/borrowing as % of GDP) prices FX via twin-deficits / external-adjustment and fiscal-sustainability channels: stronger relative fiscal balances tend to support subsequent appreciation; deep US deficits can foreshadow USD adjustment (or, alternately, safe-haven USD demand).
- **Key refs:** Abell (1990) twin deficits; Corsetti–Dedola–Leduc on fiscal shocks and exchange rates; Dai & Philippon on fiscal deficits / risk premia; related fiscal-sustainability FX surveys.
- **Free data:** IMF WEO general-government net lending/borrowing (% GDP) via FRED `GGNLBA*188N` (USD/EUR-DE/GBP/JPY/CAD/AUD). NZD/CHF **unmapped** on free FRED. US Monthly Treasury Statement `MTSDS133FMS` for higher-frequency US tilt.
- **What we implement (wave §28):** `data/fred_fiscal_balance.py` + `strategies/fiscal_balance_fx.py` + `scripts/scholarly_fx_fiscal_wave.py` — legs `fiscal_surplus_xs` (primary), `fiscal_deficit_xs`, `fiscal_chg_xs`, `us_fiscal_twin_fx`, `us_fiscal_haven_usd`, `us_mts_chg_usd`, scholarly `fiscal_ca_blend` (EW with CA surplus; **not** locked-sleeve overlay), `fiscal_ew`. PIT annual WEO `pub_lag_months=15` (~April Y+1) + MTS `pub_lag=1m` + `signal_lag=1m` + 1d weight lag. **Distinct** from CA/GDP (§21), CB-BS (§22), macro-diff. **Not** overlaid on the locked sleeve.

### 1.23 Government debt/GDP differentials (fiscal sustainability / debt overhang)

- **Claim:** Relative public debt/GDP (stock) prices FX via fiscal-sustainability / debt-overhang channels: low relative debt supports credibility and subsequent appreciation; high US debt can foreshadow USD adjustment (or, alternately, safe-haven USD demand). Distinct stock channel vs §28 fiscal-balance *flow*.
- **Key refs:** Reinhart–Rogoff debt overhang / fiscal sustainability lineage; twin-deficits / external-adjustment surveys (stock vs flow); Della Corte–Riddiough–Sarno debtor-premium honesty alternate.
- **Free data:** IMF WEO general-government gross debt (% GDP) via FRED `GGGDTA*188N` (USD/EUR-DE/GBP/JPY/CAD/AUD). Brief mnemonic `GGXWDG*188N` **404 on FRED** — documented. NZD/CHF **unmapped**. US quarterly `GFDEGDQ188S` for higher-frequency US tilts.
- **What we implement (wave §29):** `data/fred_debt_gdp.py` + `strategies/debt_gdp_fx.py` + `scripts/scholarly_fx_debt_gdp_wave.py` — legs `low_debt_xs` (primary), `high_debt_xs`, `debt_chg_xs`, `us_debt_twin_fx`, `us_debt_haven_usd`, scholarly `debt_fiscal_blend` (EW with fiscal surplus; **not** locked-sleeve overlay), `debt_ew`. PIT annual WEO `pub_lag_months=15` + US-Q `pub_lag=4m` + `signal_lag=1m` + 1d weight lag. **Distinct** from fiscal GGNLBA (§28), CA/GDP (§21), CB-BS, macro-diff, equity-diff, AI-GPR, ToT, dollar-beta, crash-skew. **Not** overlaid on the locked sleeve.





### 1.24 BIS real broad REER undervaluation / mean-reversion

- **Claim:** Official multilateral real effective exchange rates (BIS real broad EER) price FX via slow PPP / REER mean reversion: currencies with low trailing REER z (undervalued in real multilateral terms) subsequently appreciate vs those with high REER z (overvalued). Distinct from homemade bilateral CPI PPP DIY.
- **Key refs:** Rogoff (1996), "The Purchasing Power Parity Puzzle," *JEL*; Taylor / REER misalignment literature; BIS real broad effective exchange rates. Optional secondary: REER *momentum* (high z → long) as contrast prior.
- **Free data:** FRED `RB*BIS` monthly (USD/EUR-`RBXMBIS`/GBP/JPY/CAD/AUD/NZD/CHF). `RBEZBIS`/`RBEMUBIS` **404**; Germany `RBDEBIS` documented alt (unused — prefer euro-area `RBXMBIS`). Full G10 mapped.
- **What we implement (wave §31):** `data/fred_bis_reer.py` + `strategies/bis_reer_fx.py` + `scripts/scholarly_fx_bis_reer_wave.py` — legs `reer_cheap_xs` (primary, 60m z), `reer_cheap_xs_36`, `reer_mom_xs` (contrast), `reer_chg_xs` (Δ12), `us_reer_strong_usd`, `us_reer_meanrev_fx`, `reer_ew`. PIT `pub_lag_months=2` + `signal_lag=1m` + 1d weight lag. **Distinct** from `ppp_real_fx` bilateral CPI, TB (§30), debt (§29), fiscal (§28), CA, CB-BS, macro-diff. **Not** overlaid on the locked sleeve.

### 1.25 BIS private credit-to-GDP / credit-gap (Borio–Drehmann)

- **Claim:** Private credit/GDP and the credit gap (deviation from a long-run trend) predict financial-cycle stress and FX risk premia. Long low-gap / low-credit currencies (lean balance sheets) vs high-gap / high-credit (fragile / debtor premium honesty alternate).
- **Key refs:** Borio & Drehmann / BIS early-warning; Basel credit-to-GDP gap / financial-cycle literature.
- **Free data:** FRED BIS `Q*PAM770A` quarterly % GDP (USD/EUR-`QXMPAM770A`/GBP/JPY/CAD/AUD/NZD/CHF). Brief `CRDQ*APABIS` is **absolute** credit (documented unused). Pre-computed gap IDs (`BISCRDGAP*`, `CRDGAP*`, …) **404** — gap = level − trailing 180m mean (a priori HP-like one-sided).
- **What we implement (wave §32):** `data/fred_bis_credit.py` + `strategies/bis_credit_fx.py` + `scripts/scholarly_fx_bis_credit_wave.py` — legs `low_credit_gap_xs` (primary), `low_credit_xs`, `high_credit_xs`, `low_credit_z_xs` (5y z), `credit_chg_xs`, `us_credit_stress_fx`, `us_credit_haven_usd`, `credit_ew`. PIT `pub_lag_months=5` + `signal_lag=1m` + 1d weight lag. **Distinct** from debt (§29), fiscal (§28), TB (§30), REER (§31), CA, CB-BS, funding-liq. **Not** overlaid on the locked sleeve.



### 1.26 Monetary approach / money-growth differentials (Frenkel–Bilson)

- **Claim:** Sticky-price / monetary-approach FX: higher *relative* broad-money growth depreciates the expanding currency (Frenkel–Bilson; Dornbusch overshooting). Primary prior: long low relative money growth (tightness → appreciate).
- **Key refs:** Frenkel (1976); Bilson (1978); Dornbusch (1976) sticky-price monetary model.
- **Free data:** FRED OECD MEI `MABMM301*M657S` broad-money growth (USD/EUR-`EZ`/GBP/JPY/CAD/AUD). NZD/CHF `*657S` ends **2018-12** — unmapped on primary; NSA levels `*189N` archival only. US `M2SL` level documented alt.
- **What we implement (wave §33):** `data/fred_money_growth.py` + `strategies/money_growth_fx.py` + `scripts/scholarly_fx_money_growth_wave.py` — legs `low_money_growth_xs` (primary), `high_money_growth_xs`, `low_money_growth_z_xs`, `money_growth_chg_xs`, `us_money_stress_fx`, `us_money_haven_usd`, `money_ew`. PIT `pub_lag_months=2` + `signal_lag=1m` + 1d weight lag. **Distinct** from CB-BS/QE (§22), real-rate (§23), debt (§29), fiscal (§28), TB (§30), REER (§31), BIS credit (§32). **Not** overlaid on the locked sleeve.




### 1.27 IMF IFS reserves / external-buffer differentials (Aizenman–Jeanne–Rancière)

- **Claim:** External-buffer / reserve-adequacy channel: currencies with *high* relative international reserves (excl. gold) are more resilient → subsequent appreciation / lower crash risk. Honesty alternate: low-reserve / thin-buffer debtor premium.
- **Key refs:** Aizenman–Jeanne–Rancière reserve-adequacy literature; IMF reserve / Guidotti–Greenspan framing.
- **Free data:** FRED IMF IFS `TRESEG*M052N` (USD/EUR-Germany/`DEM`/GBP/JPY/CAD/AUD). `TRESEGEZM052N` ends **2018-04** — Germany proxy for EUR. NZD/CHF **404**. Reserves/GDP **not formed** (free GDP units heterogeneous / many 404). US `TOTRESNS`/`WRESBAL` are Fed bank reserves — documented alt, not IFS external buffer.
- **What we implement (wave §34):** `data/fred_reserves.py` + `strategies/reserves_fx.py` + `scripts/scholarly_fx_reserves_wave.py` — legs `high_reserves_xs` (primary), `low_reserves_xs`, `high_reserves_z_xs`, `reserves_chg_xs`, `us_reserves_stress_fx`, `us_reserves_haven_usd`, `reserves_ew`. PIT `pub_lag_months=3` + `signal_lag=1m` + 1d weight lag. **Distinct** from CA (§21), CB-BS (§22), fiscal (§28), debt (§29), TB (§30), REER (§31), credit (§32), money-growth (§33). **Not** overlaid on the locked sleeve.

### 1.28 Housing wealth / residential house-price differentials (Aoki–Proudman–Vlieghe)

- **Claim:** Housing wealth / collateral channel: currencies with *high* relative residential house-price *momentum* benefit from wealth and collateral effects → subsequent appreciation prior. Honesty alternate: low-HPI / housing-stress debtor premium.
- **Key refs:** Aoki, Proudman & Vlieghe (BoE / housing–credit literature); BIS residential property price statistics; housing–collateral channel surveys.
- **Free data:** FRED BIS `Q*R628BIS` real residential property price index (USD/EUR-Germany/`DER`/GBP/JPY/CAD/AUD/NZD/CHF). `QEUR628BIS` **404** — Germany proxy for EUR. Scores use YoY log-diff (not levels).
- **What we implement (wave §35):** `data/fred_house_prices.py` + `strategies/house_price_fx.py` + `scripts/scholarly_fx_house_price_wave.py` — legs `high_hpi_xs` (primary), `low_hpi_xs`, `high_hpi_z_xs`, `hpi_chg_xs`, `us_hpi_stress_fx`, `us_hpi_haven_usd`, `hpi_ew`. PIT `pub_lag_months=4` + `signal_lag=1m` + 1d weight lag. **Distinct** from BIS credit (§32), debt (§29), REER (§31), money-growth (§33), reserves (§34), CA/TB, equity-diff, funding-liq. **Not** overlaid on the locked sleeve.


### 1.29 ICE BofA IG OAS / corporate credit risk-appetite → FX

- **Claim:** Elevated corporate credit spreads (IG OAS) mark tight risk appetite / intermediary stress; USD haven demand and carry crashes coincide with widening OAS (BNP / Menkhoff risk-off channel). ICE BofA IG OAS is a *corporate* option-adjusted credit risk premium, distinct from Moody's Baa−Treasury (`BAA10Y`) already in funding-liq §20.
- **Key refs:** Brunnermeier, Nagel & Pedersen (2008); Menkhoff, Sarno, Schmeling & Schrimpf (2012a); ICE BofA US Corporate Index OAS (FRED `BAMLC0A0CM`).
- **Free data:** FRED `BAMLC0A0CM` (IG), `BAMLH0A0HYM2` (HY), optional `BAMLC0A4CBBB` (BBB). **Public CSV note:** free `fredgraph.csv` currently truncates ICE BofA OAS to ~3 calendar years without an API key.
- **What we implement (wave §36):** `data/fred_ig_oas.py` + `strategies/ig_oas_fx.py` + `scripts/scholarly_fx_ig_oas_wave.py` — legs `ig_oas_usd` (primary), `hy_oas_usd`, `ig_oas_chg_usd`, `ig_oas_lvl_usd`, `oas_stress_fx` (honesty wrong-signed), `carry_ig_oas_cool`, `carry_ig_oas_loose`, `oas_ew`. PIT daily `pub_lag=1d` + `signal_lag=1d`. **Distinct** from funding-liq §20 (NFCI/TED/CPFF/BAA10Y), VIX/GPR/FX-RV/EPU, crash-skew §26, house-price §35, credit-gap §32. **Not** overlaid on the locked sleeve.



### 1.30 OECD Composite Leading Indicator (CLI) differentials → FX

- **Claim:** Relative *leading* activity (OECD amplitude-adjusted CLI) forecasts business-cycle turns ahead of coincident IP/UR/CPI; currencies with high lagged CLI growth vs peers appreciate (leading-activity differential / Estrella–Mishkin channel framed for FX à la Dahlquist–Hasseltoft macro differentials).
- **Key refs:** OECD Composite Leading Indicators methodology; Estrella & Mishkin (1998) leading indicators / recession; Dahlquist & Hasseltoft (2013) macro–FX differential framing applied to leading (not coincident) activity.
- **Free data:** FRED `USALOLITOAASTSAM`, `DEULOLITOAASTSAM` (EUR Germany proxy; `EA19LOLITOAASTSAM` ends 2022-11), `GBRLOLITOAASTSAM`, `JPNLOLITOAASTSAM`, `CANLOLITOAASTSAM`, `AUSLOLITOAASTSAM`. NZD (`NZL…` ends 2019) / CHF (`CHE…` ends 2022) **unmapped**.
- **What we implement (wave §37):** `data/fred_oecd_cli.py` + `strategies/oecd_cli_fx.py` + `scripts/scholarly_fx_oecd_cli_wave.py` — legs `high_cli_xs` (primary), `low_cli_xs`, `high_cli_z_xs`, `cli_chg_xs`, `us_cli_weak_fx`, `us_cli_haven_usd`, `cli_ew`. PIT monthly `pub_lag=2m` + 1d weight lag (no extra month signal lag). **Distinct** from coincident macro-diff CPI/IP/UR, house-price §35, money-growth §33, IG OAS §36. **Not** overlaid on the locked sleeve.


### 1.31 OECD Consumer Confidence (CCI) differentials → FX

- **Claim:** Relative *consumer sentiment* / confidence differentials forecast risk appetite and FX; currencies with high lagged CCI (or CCI change) vs peers appreciate (Ludvigson-style sentiment channel framed for FX à la Dahlquist macro differentials — *sentiment*, not coincident IP/UR/CPI and not leading OECD CLI).
- **Key refs:** Ludvigson (2004) Consumer Confidence and Consumer Spending; OECD Consumer Confidence Indicators (MEI); Dahlquist & Hasseltoft macro–FX differential framing applied to sentiment.
- **Free data:** FRED `CSCICP02*M460S` balances (`CSCICP02GBM460S`, `CSCICP02EZM460S`, `CSCICP02JPM460S`, `CSCICP02AUM460S`); USD `USACSCICP02STSAM` (`CSCICP02USM460S` 404). Amplitude `CSCICP03*M665S` ends ~2024-01 — stale, not primary. CAD/NZD/CHF **unmapped**.
- **What we implement (wave §38):** `data/fred_oecd_cci.py` + `strategies/oecd_cci_fx.py` + `scripts/scholarly_fx_oecd_cci_wave.py` — legs `high_cci_xs` (primary), `low_cci_xs`, `high_cci_z_xs`, `cci_chg_xs`, `us_cci_weak_fx`, `us_cci_haven_usd`, `cci_ew`. PIT monthly `pub_lag=2m` + 1d weight lag. **Distinct** from OECD CLI §37, coincident macro-diff, house-price §35, money-growth §33, IG OAS §36. **Not** overlaid on the locked sleeve.



### 1.32 World Uncertainty Index (WUI) differentials → FX (Ahir–Bloom–Furceri)

- **Claim:** Currencies with *low* relative country WUI (calmer policy/macro uncertainty from EIU text) subsequently appreciate vs high-WUI peers; elevated US/global WUI → USD haven / risk-off (honesty alternate: stress / USD soft).
- **Key refs:** Ahir, Bloom & Furceri World Uncertainty Index (quarterly country series from Economist Intelligence Unit text); related uncertainty / risk-off FX literature (distinct from Baker–Bloom–Davis **EPU/TPU** newspaper counts and Caldara–Iacoviello **GPR**).
- **Free data:** FRED `WUIUSA`, `WUIDEU` (Germany EUR proxy; `WUIFRA`/`WUIITA`/`WUIESP` alts), `WUIGBR`, `WUIJPN`, `WUICAN`, `WUIAUS`, `WUINZL`, `WUICHE` — all quarterly, live through ~2026-04.
- **What we implement (wave §40):** `data/fred_wui.py` + `strategies/wui_fx.py` + `scripts/scholarly_fx_wui_wave.py` — legs `low_wui_xs` (primary), `high_wui_xs`, `low_wui_z_xs`, `wui_chg_xs`, `us_wui_stress_fx`, `us_wui_haven_usd`, `wui_ew`. Scores on **levels** (WUI is already an index — not YoY-first). PIT quarterly `pub_lag=4m` → monthly ffill + 1d weight lag. **Distinct** from EPU/TPU, GPR/AI-GPR, OECD CLI/CCI/BCI, IG OAS. **Not** overlaid on the locked sleeve.



### 1.33 Employment-growth differentials → FX (OECD MEI LFEMTTTT)

- **Claim:** Currencies with *high* relative employment growth (labour-market strength) subsequently appreciate vs low-growth peers (Dahlquist–Hasseltoft-style macro–FX differential). Honesty alternate: long low emp / labour-stress debtor premium.
- **Key refs:** Dahlquist & Hasseltoft macro–FX differentials applied to labour; OECD MEI employment; related labour-market / FX literature (distinct from *unemployment-rate* macro-diff UR).
- **Free data:** FRED `LFEMTTTT*M647S` / `*Q647S` employment **persons** levels (YoY in strategy). USD/JPY/CAD/AUD monthly M647S live; EUR=`LFEMTTTTDEQ647S` Germany proxy (`LFEMTTTTEZQ647S` EZ ends 2022-10); GBP/NZD/CHF quarterly Q647S. `*M657S`/`*Q657S` YoY mostly 404 or stale ~2023–24 — not primary. PAYEMS/CE16OV US alts; LREM64TT employment *rate* not primary. Retail-sales SARTMISMEI stale — skipped.
- **What we implement (wave §41):** `data/fred_employment.py` + `strategies/employment_fx.py` + `scripts/scholarly_fx_employment_wave.py` — legs `high_emp_xs` (primary), `low_emp_xs`, `high_emp_z_xs`, `emp_chg_xs`, `us_emp_stress_fx`, `us_emp_haven_usd`, `emp_ew`. Scores on **YoY % of levels**. PIT `pub_lag=3m` (labour a priori) + 1d weight lag. **Distinct** from macro-diff UR, OECD CLI/CCI/BCI, WUI. **Not** overlaid on the locked sleeve.



### 1.34 Industrial-production differentials → FX (OECD MEI PRINTO01)

- **Claim:** Currencies with *high* relative industrial-production YoY growth (real-activity / growth channel) subsequently appreciate vs low-growth peers (Molodtsova–Papell / Dahlquist–Hasseltoft-style macro–FX differential on *coincident* IP). Honesty alternate: long low IP / activity-stress debtor premium.
- **Key refs:** Molodtsova & Papell (and related Taylor-rule / growth-channel FX); Dahlquist & Hasseltoft macro–FX differentials applied to coincident industrial production; OECD MEI industry excl. construction.
- **Free data:** FRED `{ISO3}PRINTO01GYSAM` industry YoY monthly (USD/GBP/JPY/CAD live ~2026-06/07). EUR=`FRAPRINTO01GYSAM` France proxy (`EA19PRINTO01GYSAM` ends 2023-10; `DEUPRINTO01GYSAM` ends 2023-12). AUD/NZD/CHF industry PRINTO01 **404** → manufacturing `{ISO3}PRMNTO01GYSAQ` quarterly YoY. `*PROINDMISMEI` levels stale ~2024-03 — not primary. US `INDPRO`/`IPMAN` alts not mixed.
- **What we implement (wave §42):** `data/fred_industrial_production.py` + `strategies/industrial_production_fx.py` + `scripts/scholarly_fx_industrial_production_wave.py` — legs `high_ip_xs` (primary), `low_ip_xs`, `high_ip_z_xs`, `ip_chg_xs`, `us_ip_stress_fx`, `us_ip_haven_usd`, `ip_ew`. Scores on **YoY growth as reported**. PIT `pub_lag=2m` (IP a priori) + 1d weight lag. **Distinct** from macro-diff EW CPI/IP/UR blend, OECD CLI/CCI/BCI, employment §41. **Not** overlaid on the locked sleeve.

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

### 1.11 CFTC COT / speculative positioning

- **Claim:** Futures *speculative* net positions (Legacy Non-Commercial; TFF Leveraged Funds / Asset Managers) co-move with exchange rates; weekly *changes* in net speculative positions are associated with same-week FX moves (Klitgaard & Weir 2004). Classic traders also treat positioning *extremes* as mean-reversion setups. Predictive content beyond contemporaneous co-movement is mixed in the academic COT literature (Sanders / Irwin / Merrin and related).
- **Key refs:** Klitgaard & Weir (2004), “Exchange Rate Changes and Net Positions of Speculators in the Futures Market,” *NY Fed Economic Policy Review*; Sanders, Irwin & Merrin; Briese-style COT constructions; CFTC Traders in Financial Futures (TFF) documentation.
- **What we implement:** `data/cftc_cot.py` + `strategies/cot_positioning_fx.py` + `scripts/scholarly_fx_cot_wave.py` — free CFTC SODA TFF Futures-Only + Legacy Futures-Only for CME FX + ICE DX. PIT: Tuesday `report_date`, Friday release → `release_lag_days=3` known_date + `signal_lag=1` trading day. Legs: lev net/OI continuation, Δnet, z continuation, −z mean-reversion, Legacy NonComm, DX USD tilt, EW blend. Frozen priors — no HO tuning.

### 1.12 Equity–FX / Hau–Rey portfolio channel

- **Claim:** Relative local equity outperformance vs US tends to associate with local FX appreciation (portfolio / risk-appetite channel).
- **Key refs:** Hau & Rey (2006), “Exchange Rates, Equity Prices, and Capital Flows,” *RFS*.
- **What we implement:** `data/equity_indices.py` + `strategies/equity_diff_fx.py` + `scripts/scholarly_fx_equity_diff_wave.py` — Yahoo G10 equity indices; PIT `pub_lag=1d` + `signal_lag=1d`; formation=21d. Legs: `eq_diff_xs` (primary), `eq_diff_ts`, `eq_mom_xs`, `carry_eq_cool`. EW XS rank on (local−US) ≡ local-mom XS; distinctive = TS + carry cool. NZD omitted.


### 1.6 TPU / economic-policy uncertainty

- **Claim:** Trade-policy and economic-policy uncertainty (Baker–Bloom–Davis EPU / TPU) affect FX and risk premia around tariff / policy shocks; elevated *home-country* EPU is associated with subsequent FX depreciation vs USD (risk premium), and high US EPU/TPU coincides with risk-off / safe-haven USD.
- **Refs:** Baker, Bloom & Davis (2016), *QJE*; Davis (2016) GEPU; policyuncertainty.com categorical *Trade policy* = US TPU; All_Country monthly EPU workbook.
- **What we implement (this wave §18):** `data/macro_uncertainty.py` loaders (`load_epu`, `load_tpu`, `load_country_epu`, `load_epu_tpu_bundle`) + `strategies/epu_tpu_fx.py` + `scripts/scholarly_fx_epu_tpu_wave.py`. PIT `pub_lag=1m` + `signal_lag_months=1` + `signal_lag_days=1`. Legs: country-EPU XS, EPU−US relative, US-EPU/TPU/GEPU USD tilts, carry cooled by **EPU-only** / **TPU-only** (no VIX/GPR — distinct from §8 combo). CHF/NZD country EPU absent in the 22-country file.
- **FX IV / risk-reversal:** **No free PIT panel** (LSEG VolSurf, TFS-ICAP/CME DataMine, Bloomberg commercial only). Do **not** fabricate option IV from Yahoo OHLC. Documented blocker → EPU/TPU fallback this wave.

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
| CFTC COT (TFF / Legacy FX) | CFTC SODA `gpe5-46if` / `6dca-aqww` | **3 calendar days** (Tue→Fri) + strategy `signal_lag=1` trading day |
| US EPU (Baker–Bloom–Davis) | FRED `USEPUINDXM` / policyuncertainty.com | **1 month** (+ strategy `signal_lag`) |
| Global EPU (GEPU) | FRED `GEPUCURRENT` | **1 month** (+ strategy `signal_lag`) |
| US TPU (categorical Trade policy) | policyuncertainty.com Categorical EPU | **1 month** (+ strategy `signal_lag`) |
| Country EPU (22-country) | policyuncertainty.com All_Country_Data.xlsx | **1 month** (+ strategy `signal_lag`) |

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
| `src/mt5_swing/data/macro_uncertainty.py` | VIX, GPR, EPU/TPU/country-EPU loaders |
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
| `src/mt5_swing/data/cftc_cot.py` | Free CFTC TFF/Legacy SODA FX+DX panel + release lag |
| `src/mt5_swing/strategies/cot_positioning_fx.py` | Speculative pressure / extremes sorts + DX USD tilt |
| `scripts/scholarly_fx_cot_wave.py` | COT wave eval + FTMO risk sweep |
| `reports/scholarly_fx_cot_wave.md` | COT positioning wave board |
| `src/mt5_swing/data/fred_current_account.py` | IMF BOP CA/GDP % panel (FRED) + PIT 2Q lag |
| `src/mt5_swing/strategies/current_account_fx.py` | Debtor/surplus/ΔCA XS + US CA GR/haven tilts |
| `scripts/scholarly_fx_ca_wave.py` | CA imbalances wave eval + FTMO risk sweep |
| `reports/scholarly_fx_ca_wave.md` | Global imbalances / CA wave board |
| `src/mt5_swing/data/fred_cb_balance_sheet.py` | WALCL/ECBASSETSW/JPNASSETS + YoY + BS/GDP PIT |
| `src/mt5_swing/strategies/cb_balance_sheet_fx.py` | QE PB USD/FX tilts + peer XS + BS/GDP |
| `scripts/scholarly_fx_cb_bs_wave.py` | CB-BS / QE wave eval + FTMO risk sweep |
| `reports/scholarly_fx_cb_bs_wave.md` | CB balance-sheet / QE wave board |
| `src/mt5_swing/data/fred_oecd_cci.py` | OECD MEI CCI balances (CSCICP02) + US standardised CCI PIT |
| `src/mt5_swing/strategies/oecd_cci_fx.py` | High/low CCI XS + chg/z + US weak/haven tilts + EW |
| `scripts/scholarly_fx_oecd_cci_wave.py` | CCI sentiment wave eval + FTMO risk sweep |
| `reports/scholarly_fx_oecd_cci_wave.md` | OECD CCI consumer-confidence wave board |

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
9b. Ludvigson, S. (2004). Consumer Confidence and Consumer Spending. *JEP*.
10. Della Corte, P., Riddiough, S. & Sarno, L. (2016). Currency Premia and Global Imbalances. *RFS*.
11. Gourinchas, P.-O. & Rey, H. (2007). International Financial Adjustment. *JPE*.
12. Gagnon, J., Raskin, M., Remache, J. & Sack, B. (2011). The Federal Reserve's Large-Scale Asset Purchases. *IJCB*.
13. Neely, C. (2015). Unconventional Monetary Policy Effects on Exchange Rates. *JBF*.
14. Bauer, M. & Neely, C. (2014). International Channels of the Fed's Unconventional Monetary Policy. *JIMF*.
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
6. **Done (2026-09-23):** Bilateral AI-GPR role decompositions — promote=NO (see §25).
7. **Done (2026-09-23):** PPP / real-FX value wave — promote=NO (see §13).
8. **Done (2026-09-23):** Commodity CRR + macro-diff waves — promote=NO (see §11–12).
8b. **Done (2026-09-23):** Terms-of-trade / commodity-currency refinement — promote=NO (see §24).
9. **Done (2026-09-23):** Balassa–Samuelson / productivity wave — promote=NO (see §14).
10. **Done (2026-09-23):** True FX realized-vol risk factor (Menkhoff) — promote=NO (see §15).
11. **Done (2026-09-23):** Term-structure / yield-curve FX + UIP secondary — promote=NO (see §16).
12. **Done (2026-09-23):** CFTC COT positioning / speculative-pressure wave — promote=NO (see §17).
13. **Done (2026-09-23):** EPU/TPU, forward-carry, Hau–Rey equity, funding-liq, CA imbalances — promote=NO (§18–21).
14. **Done (2026-09-23):** CB balance-sheet / QE differential — promote=NO (see §22).
15. **Done (2026-09-23):** Real-rate / breakeven differentials — promote=NO (see §23).
16. **Done (2026-09-23):** Bilateral AI-GPR role decompositions — promote=NO (see §25).
17. **Done (2026-09-23):** Brunnermeier–Nagel–Pedersen crash-skew / left-tail wave — promote=NO (see §26).
18. **Done (2026-09-23):** Lustig–Verdelhan dollar-factor beta sorts — promote=NO (see §27).
19. **Done (2026-09-23):** FRED fiscal-balance / government-budget differentials — promote=NO (see §28).
19b. **Done (2026-09-23):** FRED government debt/GDP differentials — promote=NO (see §29).
19c. **Done (2026-09-23):** Monthly trade-balance FX wave — promote=NO (see §30).
19d. **Done (2026-09-23):** BIS/FRED REER undervaluation wave — promote=NO (see §31).
19e. **Done (2026-09-23):** BIS private credit-to-GDP / credit-gap wave — promote=NO (see §32).
19f. **Done (2026-09-23):** IMF IFS reserves / external-buffer FX wave — promote=NO (see §34).
19g. **Done (2026-09-23):** OECD/BIS residential house-price FX wave — promote=NO (see §35).
19h. **Done (2026-09-23):** ICE BofA IG OAS / credit risk-appetite FX wave — promote=NO (see §36).
19i. **Done (2026-09-23):** OECD CLI leading-indicator FX wave — promote=NO (see §37).
19j. **Done (2026-09-23):** OECD CCI / consumer-confidence FX wave — promote=NO (see §38).
19k. **Done (2026-09-23):** OECD MEI employment-growth LFEMTTTT FX wave — promote=NO (see §41).
20. Next scholarly candidates (not sleeve coolers): **FTMO MT5 CSV re-score** when exports arrive (all boards still `approximate_non_ftmo`); FX IV/RR **blocked**; news-based currency sentiment still blocked without free multi-year panel. (Industrial-production §42 / Employment §41 / WUI §40 / BCI §39 / CCI §38 / CLI §37 done; retail-sales SARTMISMEI confirmed stale.)

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

---

## 17. CFTC COT positioning wave results (2026-09-23 BST) — speculative pressure / extremes

**Design (fixed priors, no HO tuning):** free CFTC TFF Futures-Only + Legacy Futures-Only for EUR/GBP/JPY/CAD/CHF/AUD/NZD + ICE DX. Score = net speculative / open interest.
- PIT: Tuesday snapshot, Friday release → `release_lag_days=3` (known Friday) + `signal_lag=1` trading day; n_long=n_short=2; z_window=52 weeks; costs 1.5 bps/side.
- `cot_lev_net_xs` — long high / short low TFF leveraged-money net/OI (**continuation**, primary)
- `cot_lev_chg_xs` — sort on weekly Δ(lev_net/OI)
- `cot_lev_z_xs` — trailing z continuation
- `cot_lev_z_mr_xs` — −z (**mean-reversion** at extremes)
- `cot_noncomm_net_xs` — Legacy NonComm net/OI continuation
- `cot_dx_usd` — DX lev net/OI → long USD when DX speculative long
- `cot_ew` — EW of lev_net + lev_chg

**Coverage:** Full G10 FX + DX on free SODA (TFF from 2006-06; Legacy deeper). No paid NLP. Option-implied vol vs RV fallback **not needed** (CFTC download succeeded).

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| cot_lev_net_xs | −0.139% | −2.34 | −2.26 | 46% | 10% | −0.55 |
| cot_lev_chg_xs | −0.213% | −3.11 | −2.71 | 41% | 16% | −0.72 |
| cot_lev_z_xs | −0.129% | −1.86 | −1.87 | 44% | 15% | −0.42 |
| cot_lev_z_mr_xs | +0.078% | +1.12 | +1.16 | 54% | 13% | +0.25 |
| cot_noncomm_net_xs | −0.187% | −2.80 | −2.68 | 41% | 14% | −0.70 |
| cot_dx_usd | +0.148% | +0.94 | +0.86 | 53% | 14% | +0.23 |
| cot_ew | −0.176% | −3.70 | −3.18 | 41% | 12% | −0.87 |

### Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| cot_lev_net_xs | holdout_365d | +0.27% | 83% | PASS | no† |
| cot_lev_net_xs | year_2024 | +0.09% | 55% | PASS | no |
| cot_lev_z_mr_xs | holdout_365d | −0.03% | 58% | PASS | no |
| cot_dx_usd | holdout_365d | +0.44% | 50% | PASS | no |
| cot_ew | holdout_365d | +0.01% | 42% | PASS | no |

† Holdout %pos can clear 70% on primary continuation, but full-sample mean is **significantly negative** (NW t≈−2.3) — classic HO luck, not a promote.

### Risk sweep (IS → OOS)

Positive IS means only on MR / DX → sweep run for all legs. Best scaled IS mean (`cot_lev_z_mr_xs` @ ~2.07× daily-bound) ≈ **+0.18%/mo** still ≪ 1%; primary continuation scaled IS negative. OOS means mixed / basis points. Scaled clears: **NO**. Leverage does not invent consistency; flipping continuation after seeing negative full-sample t would be holdout tuning — we keep the pre-specified MR leg as the honest positive channel.

**Unscaled promote:** **NO**. **Scaled primary (`cot_lev_net_xs`) promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` unchanged.

### Honest read

On this Yahoo+CFTC sample, **continuation** speculative-pressure sorts (level, change, z, Legacy NonComm) earn **negative** full-sample means with NW |t| ≈ 1.9–3.2 — economically a few bp/week against the crowded side. The pre-specified **mean-reversion** z-leg and DX USD tilt are mildly positive (NW t ≈ 0.9–1.2) at **~0.08–0.15%/mo**, %pos ≪ 70%. Far from prop-firm 1%/mo + 70% hit-rate. Free CFTC SODA is sufficient; no need for paid positioning vendors this wave. CME FX futures positioning ≠ OTC spot order flow.

Artifacts: `reports/scholarly_fx_cot_wave.md`, `scholarly_fx_cot_*.csv`, `scholarly_fx_cot_meta.json`.


---

## 18. EPU / TPU wave results (2026-09-23 BST) — Baker–Bloom–Davis (IV/RR fallback)

**Path decision:** Primary goal was FX option-implied vol vs realized vol and/or risk-reversal/skew. **No free, downloadable, PIT-safe FX IV/RR panel exists** (commercial: LSEG VolSurf, TFS-ICAP/CME DataMine, Bloomberg). Fabricating IV from Yahoo OHLC is disallowed. **Fallback delivered:** deepen EPU/TPU with honest PIT release timing — distinct from prior VIX/GPR combo (§8) and Menkhoff FX-RV (§15).

**Design (fixed priors, no HO tuning):**
- Sources: FRED `USEPUINDXM` / `GEPUCURRENT`; policyuncertainty.com `All_Country_Data.xlsx` (country EPU); Categorical EPU *Trade policy* = US TPU (updated through present; Trade_Uncertainty_Data.xlsx freezes ~2019, corr=1.0 historically).
- PIT: `pub_lag_months=1` + `signal_lag_months=1` + `signal_lag_days=1`; n_long=n_short=2; costs 1.5 bps/side.
- Legs: `country_epu_xs` (long low / short high home EPU z — primary), `epu_diff_xs` (home−US), `epu_us_usd` / `tpu_us_usd` / `gepu_usd` (binary USD tilt when z≥1), `carry_epu_cool` / `carry_tpu_cool` (EPU-only / TPU-only cool — **no VIX/GPR**), `epu_ew`.
- Coverage: AUD/CAD/JPY/GBP/EUR/USD country EPU; **CHF/NZD missing** in 22-country file.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| country_epu_xs | +0.031% | +0.54 | +0.63 | 53% | 12% | +0.17 |
| epu_diff_xs | +0.031% | +0.54 | +0.63 | 53% | 12% | +0.17 |
| epu_us_usd | +0.011% | +1.06 | +0.87 | 17% | 29% | +0.28 |
| tpu_us_usd | +0.011% | +1.38 | +1.13 | 19% | 27% | +0.29 |
| gepu_usd | +0.001% | +0.14 | +0.11 | 19% | 33% | +0.03 |
| carry_epu_cool | +0.035% | +0.53 | +0.60 | 52% | 11% | +0.08 |
| carry_tpu_cool | −0.005% | −0.07 | −0.08 | 52% | 12% | −0.05 |
| epu_ew | +0.021% | +0.70 | +0.81 | 53% | 12% | +0.21 |

### Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| country_epu_xs | holdout_365d | +0.30% | 75% | PASS | no† |
| country_epu_xs | year_2024 | −0.07% | 36% | PASS | no |
| carry_epu_cool | holdout_365d | +0.31% | 75% | PASS | no† |
| tpu_us_usd | holdout_365d | +0.00% | 0% | PASS | no |
| epu_ew | holdout_365d | +0.15% | 75% | PASS | no† |

† Holdout %pos can clear 70% on XS / cooled carry, but mean ≪ 1%/mo and full-sample means are basis points — not a promote.

### Risk sweep (IS → OOS)

Positive IS means → sweep run. Best scaled IS mean (`tpu_us_usd` @ ~11.2× daily-bound) ≈ **+0.13%/mo** still ≪ 1%; primary `country_epu_xs` scaled IS ≈ +0.03%/mo; scaled HO mean ≈ +0.93% still below bar with clears=NO. Scaled clears: **NO**. Leverage does not invent consistency.

**Unscaled promote:** **NO**. **Scaled primary (`country_epu_xs`) promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` unchanged.

### Honest read

Country-EPU depreciation sorts and EPU-only carry cool are the right *priors* and are **distinct** from VIX/GPR: mild positive full-sample means (NW t ≈ 0.6–1.1) at **~0.01–0.04%/mo**, %pos ≪ 70% on sparse USD-tilt legs. Far from prop-firm 1%/mo + 70% hit-rate. Free policyuncertainty.com + FRED is sufficient; no paid NLP. FX IV/RR remains blocked without a vendor panel.

Artifacts: `reports/scholarly_fx_epu_tpu_wave.md`, `scholarly_fx_epu_tpu_*.csv`, `scholarly_fx_epu_tpu_meta.json`.


---

## 19. Swap- / forward-aware carry wave results (2026-09-23 BST) — Lustig–Verdelhan CIP proxy

**Design (fixed priors, no HO tuning):** Best free short-rate / money-market proxies for classic *forward-discount* carry.
- Sources: FRED OECD ``IR3TIB01*`` (3M money-market — primary) + ``IRSTCI01*`` (immediate cash baseline); CIP-implied FD from exact discrete ratio at 3M tenor.
- PIT: `pub_lag_months=1` + `signal_lag_months=1` + 1 trading-day weight lag; n_long=n_short=2.
- Costs: 1.5 bps/side baseline; **`carry_ir3m_tc5`** = same IR3M XS with **5 bps/side** TC haircut.
- Legs: `carry_ir3m_xs` (primary), `carry_irstci_xs`, `carry_cip_fd_xs`, `carry_ir3m_ew`, `carry_blend_xs`, `carry_ir3m_tc5`.
- Coverage: Full G10 on FRED IR3M + IRSTCI. Spearman rank_corr(IRSTCI,IR3M)≈**0.96**; CIP-FD vs IR3M-diff≈**1.00** (rank-identical on this panel).
- **Honesty:** Still cash / MM approximation — **not** Bloomberg FX swap or outright forward points; CIP basis not modelled.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| carry_ir3m_xs | −0.034% | −0.43 | −0.51 | 53% | 10% | −0.13 |
| carry_irstci_xs | −0.011% | −0.14 | −0.16 | 52% | 10% | −0.07 |
| carry_cip_fd_xs | −0.034% | −0.43 | −0.51 | 53% | 10% | −0.13 |
| carry_ir3m_ew | −0.029% | −0.50 | −0.61 | 49% | 11% | −0.10 |
| carry_ir3m_tc5 | −0.035% | −0.45 | −0.52 | 53% | 10% | −0.13 |
| carry_blend_xs | −0.022% | −0.29 | −0.34 | 52% | 10% | −0.10 |

### Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| carry_ir3m_xs | holdout_365d | +0.31% | 83% | PASS | no† |
| carry_irstci_xs | holdout_365d | +0.33% | 75% | PASS | no† |
| carry_ir3m_ew | holdout_365d | +0.09% | 50% | PASS | no |
| carry_blend_xs | holdout_365d | +0.32% | 75% | PASS | no† |

† Holdout %pos can clear 70% with mild positive mean, but **full-sample means are negative** (NW t ≈ −0.2 to −0.6) — classic HO luck, not a promote.

### Risk sweep (IS → OOS)

**Skipped** — no factor with positive full-sample IS mean (no unused DD budget to invent edge). Leverage cannot invent consistency from a negative-mean carry sleeve on this Yahoo+FRED sample.

**Unscaled promote:** **NO**. **Scaled primary (`carry_ir3m_xs`) promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` unchanged.

### Honest read

Money-market (IR3M) and CIP-implied FD sorts are the right *priors* relative to cash IRSTCI alone, and they are essentially rank-identical here (corr≈1). On this approximate_non_ftmo panel the forward-proxy carry earns **negative** full-sample means (~−0.01 to −0.03%/mo) with NW |t| ≪ 2 — IR3M does **not** improve on prior cash-rate carry toward the 1%/mo bar. TC haircut (5 bps) slightly worsens an already flat/negative sleeve. Far from prop-firm 1%/mo + 70% hit-rate. Free FRED OECD is sufficient for this claim; inventing Bloomberg forwards would be dishonest.

Artifacts: `reports/scholarly_fx_fwd_carry_wave.md`, `scholarly_fx_fwd_carry_*.csv`, `scholarly_fx_fwd_carry_meta.json`.

---

## 19. Hau–Rey equity-differential wave results (2026-09-23 BST)

**Design (fixed priors, no HO tuning):**
- Yahoo equity: `^GSPC` US, `^GDAXI` EUR, `^FTSE` GBP, `^N225` JPY, `^GSPTSE` CAD, `^AXJO` AUD, `^SSMI` CHF. Calendar-date normalize-before-join (session-collision fix).
- PIT: `pub_lag_days=1` + `signal_lag_days=1`; formation=21d; n_long=n_short=2; costs 1.5 bps/side.
- Legs: `eq_diff_xs` (primary), `eq_diff_ts`, `eq_mom_xs` (control), `carry_eq_cool` (cool carry when lagged US equity mom < 0).
- **XS identity:** EW rank on (local−US) ≡ local-mom XS — US term common across currencies.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| eq_diff_xs | −0.066% | −1.09 | −1.18 | 48% | 12% | −0.20 |
| eq_diff_ts | −0.019% | −0.26 | −0.30 | 51% | 14% | −0.07 |
| eq_mom_xs | −0.066% | −1.09 | −1.18 | 48% | 12% | −0.20 |
| carry_eq_cool | −0.024% | −0.39 | −0.48 | 50% | 12% | −0.13 |

### Consistency windows (primary `eq_diff_xs`)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| year_2024 | +0.04% | 55% | 88% | PASS | no |
| year_2025 | −0.08% | 45% | 91% | PASS | no |
| year_2026 | −0.18% | 50% | 87% | PASS | no |
| holdout_365d | −0.14% | 50% | 72% | PASS | no |

### Risk sweep

No positive IS mean on any factor → sweep skipped. Scaled promote: **NO**.

**Board:** n=4 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` unchanged.

### Honest read

Tradable Hau–Rey legs on Yahoo equity indices earn **~0 to −7 bp/mo** full-sample (NW |t| ≲ 1.2) — far from prop-firm 1%/mo + 70% hit-rate. Indices ≠ Hau–Rey *portfolio flow* data; free Yahoo is sufficient to reject this as a standalone FTMO sleeve.

Artifacts: `reports/scholarly_fx_equity_wave.md`, `scholarly_fx_equity_*.csv`, `scholarly_fx_equity_meta.json`.


---

## 20. Funding-liquidity / financial-conditions wave results (2026-09-23 BST) — BNP / NFCI–TED

**Design (fixed priors, no HO tuning):** Distinct carry-crash / FX risk channel vs prior VIX (§8), GPR, Menkhoff FX-RV (§15), and EPU/TPU (§18).
- Sources: FRED `NFCI` / `ANFCI` (+ risk/credit/leverage subindices), `TEDRATE` (ends 2022-01), `CPFF` (post-LIBOR CP−Tbill), `BAA10Y`.
- PIT: weekly NFCI family `pub_lag_days=7` (Chicago Fed mid-week release for Friday-ending week); daily spreads `pub_lag_days=1`; `signal_lag=1` trading day. Costs 1.5 bps/side.
- Frozen cutoffs: trailing z≥1 (match VIX/GPR/EPU); NFCI level>0 (Chicago Fed design — tighter than average); cool=0.35; usd_tilt=0.5.
- Legs: `nfci_usd` (primary), `anfci_usd`, `nfci_lvl_usd`, `nfci_chg_usd`, `cpff_usd`, `ted_usd`, `baa_usd`, `spread_usd` (TED⊕CPFF), `carry_nfci_cool`, `carry_nfci_loose` (carry only when NFCI≤0), `carry_cpff_cool`, `funding_ew`.
- **Explicit:** evaluating funding as a scholarly sleeve — **no cooler overlay** on locked `fx4plus_gbpcad_d1_voltarget_0025`.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| nfci_usd | +0.062% | +1.80 | +1.77 | 13% | 30% | +0.42 |
| nfci_lvl_usd | -0.014% | -1.11 | -1.00 | 1% | 100% | -0.33 |
| anfci_usd | +0.065% | +1.72 | +1.82 | 18% | 28% | +0.41 |
| nfci_chg_usd | +0.073% | +2.02 | +2.03 | 20% | 27% | +0.53 |
| cpff_usd | +0.015% | +0.41 | +0.50 | 23% | 28% | +0.11 |
| ted_usd | +0.048% | +1.70 | +2.05 | 13% | 30% | +0.40 |
| baa_usd | -0.022% | -0.72 | -0.81 | 11% | 36% | -0.15 |
| spread_usd | +0.055% | +1.40 | +1.78 | 20% | 25% | +0.39 |
| carry_nfci_cool | +0.016% | +0.25 | +0.29 | 52% | 12% | +0.01 |
| carry_nfci_loose | -0.012% | -0.16 | -0.18 | 51% | 10% | -0.08 |
| carry_cpff_cool | -0.021% | -0.31 | -0.34 | 54% | 12% | -0.12 |
| funding_ew | +0.050% | +1.85 | +1.88 | 30% | 25% | +0.44 |

### Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| nfci_usd | holdout_365d | +0.00% | 0% | PASS | no |
| nfci_chg_usd | holdout_365d | +0.14% | 25% | PASS | no |
| carry_nfci_cool | holdout_365d | +0.27% | 83% | PASS | no† |
| carry_nfci_loose | holdout_365d | +0.31% | 75% | PASS | no† |
| funding_ew | holdout_365d | +0.02% | 42% | PASS | no |

† Holdout %pos can clear 70% on funding-conditioned carry, but mean ≪ 1%/mo and full-sample carry cools are flat — classic HO luck, not a promote.

### Risk sweep (IS → OOS)

Positive IS means on several USD-tilt legs → sweep run. Best scaled IS mean (`spread_usd` @ ~3.56× daily-bound) ≈ **+0.23%/mo** still ≪ 1%; primary `nfci_usd` scaled IS ≈ +0.20%/mo; scaled HO flat/near zero. Scaled clears: **NO**. Leverage does not invent consistency.

**Unscaled promote:** **NO**. **Scaled primary (`nfci_usd`) promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` unchanged.

### Honest read

NFCI / ANFCI / ΔNFCI and TED USD tilts are the right *priors* and are **distinct** from VIX/GPR/FX-RV/EPU: mild positive full-sample means (NW t ≈ 1.8–2.0 on ΔNFCI and TED) at **~0.05–0.07%/mo**, but %pos is sparse (~13–20%) because binary stress episodes are infrequent. Absolute NFCI>0 gate and BAA credit tilt do not help. Funding-conditioned carry (cool / loose-only) does not lift the flat cash-rate carry sleeve toward the bar. Far from prop-firm 1%/mo + 70% hit-rate. Free FRED NFCI+CPFF is sufficient; TED ends 2022 (LIBOR) — CPFF fills.

**Next structure (if promote=0):** delivered as wave §21 (CA/GDP imbalances).

Artifacts: `reports/scholarly_fx_funding_liq_wave.md`, `scholarly_fx_funding_liq_*.csv`, `scholarly_fx_funding_liq_meta.json`.

---

## 21. Global imbalances / current-account wave results (2026-09-23 BST) — GR / DCRS

**Design (fixed priors, no HO tuning):** Distinct external-adjustment / NFA–CA channel vs prior PPP (§13), Balassa–Samuelson (§14), macro-diff CPI/IP/UR (§12), and Hau–Rey equity (§19).
- Sources: FRED IMF BOP `{ISO3}B6BLTT02STSAQ` current account / GDP (%), SA quarterly — full G10; EUR = EA19 with DEU gap-fill after ~2022-10.
- PIT: `pub_lag_quarters=2` (conservative BOP release + revision buffer = 6 months) + `signal_lag=1` month + 1 trading-day weight lag. Costs 1.5 bps/side. n_long=n_short=2.
- Legs: `ca_debtor_xs` (primary — Della Corte long deficit / short surplus), `ca_surplus_xs` (flow control), `ca_chg_xs` (ΔCA 4m), `us_ca_gr_fx` (Gourinchas–Rey: deep US CA z ≤ −1 → long foreign / short USD), `us_ca_haven_usd` (alternate: deep US deficit → long USD), `ca_ew` (debtor ⊕ chg ⊕ GR).
- **Explicit:** evaluating CA as a scholarly sleeve — **no cooler overlay** on locked `fx4plus_gbpcad_d1_voltarget_0025`. Free FRED only (OECD SDMX / IMF DataMapper blocked here).

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| ca_debtor_xs | +0.020% | +0.26 | +0.33 | 52% | 11% | +0.02 |
| ca_surplus_xs | −0.027% | −0.35 | −0.45 | 47% | 10% | −0.04 |
| ca_chg_xs | −0.021% | −0.36 | −0.35 | 48% | 14% | −0.05 |
| us_ca_gr_fx | −0.021% | −0.50 | −0.58 | 14% | 28% | −0.14 |
| us_ca_haven_usd | +0.021% | +0.50 | +0.58 | 16% | 24% | +0.13 |
| ca_ew | −0.007% | −0.20 | −0.24 | 55% | 12% | −0.06 |

### Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| ca_debtor_xs | holdout_365d | +0.30% | 75% | PASS | no† |
| ca_debtor_xs | year_2024 | −0.02% | 64% | PASS | no |
| ca_debtor_xs | year_2025 | +0.07% | 64% | PASS | no |
| ca_surplus_xs | holdout_365d | −0.30% | 25% | PASS | no |
| us_ca_haven_usd | holdout_365d | +0.01% | 58% | PASS | no |
| ca_ew | holdout_365d | +0.13% | 58% | PASS | no |

† Holdout %pos can clear 70% on the debtor XS with mild positive mean, but mean ≪ 1%/mo and full-sample NW |t| ≪ 2 — classic HO luck, not a promote.

### Risk sweep (IS → OOS)

Positive IS means on debtor / haven → sweep run. Best scaled IS mean (`us_ca_haven_usd` @ ~3.3× daily-bound) ≈ **+0.07%/mo** still ≪ 1%; primary `ca_debtor_xs` scaled IS ≈ +0.005%/mo; scaled HO mean ≈ +0.39% with %pos=75% but clears=NO. Scaled clears: **NO**. Leverage does not invent consistency.

**Board:** n=6 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. **Scaled primary (`ca_debtor_xs`) promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` unchanged.

### Honest read

Della Corte debtor sorts and Gourinchas–Rey US-CA tilts are the right *priors* and are **distinct** from PPP/BS/macro-diff: mild positive full-sample mean on `ca_debtor_xs` / `us_ca_haven_usd` (~+2 bp/mo, NW t ≈ 0.3–0.6), with the surplus / GR-adjustment / ΔCA legs flat-to-negative. Far from prop-firm 1%/mo + 70% hit-rate. Free FRED IMF BOP is sufficient; inventing NFA stock panels without free PIT data would be dishonest. EUR post-2022 uses DEU CA as EA19 gap-fill — documented limitation.

**Next structure (if promote=0):** delivered as wave §22 (CB balance-sheet / QE).

Artifacts: `reports/scholarly_fx_ca_wave.md`, `scholarly_fx_ca_*.csv`, `scholarly_fx_ca_meta.json`.

---

## 22. Central-bank balance-sheet / QE differential wave results (2026-09-23 BST) — Neely / Gagnon

**Design (fixed priors, no HO tuning):** Distinct QE / portfolio-balance channel vs funding-liquidity NFCI (§20) and CA/GDP imbalances (§21).
- Sources: FRED `WALCL` (Fed weekly total assets), `ECBASSETSW` (ECB weekly), `JPNASSETS` (BoJ monthly); optional US `GDP` for BS/GDP. `UKASSETS` discontinued 2014-09 — excluded.
- PIT: weekly `pub_lag_days=7`, monthly `pub_lag_months=1`, GDP `pub_lag_quarters=1` + `signal_lag=1` trading day. Costs 1.5 bps/side. YoY growth (52w / 12m) avoids FX conversion of heterogeneous level units.
- Legs: `walcl_pb_fx` (**primary** — Neely: high Fed BS YoY → long FX / short USD), `walcl_haven_usd` (crisis alternate), `walcl_chg_pb_fx` (13w Δ), `bs_diff_pb_fx` (US−EUR/JPY YoY differential), `bs_peer_xs` (long low / short high peer YoY), `bs_gdp_pb_fx` (US BS/GDP z), `bs_ew`.
- **Explicit:** evaluating CB-BS as a scholarly sleeve — **no cooler overlay** on locked `fx4plus_gbpcad_d1_voltarget_0025`. Free FRED only.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| walcl_pb_fx | +0.043% | +1.19 | +1.39 | 16% | 25% | +0.31 |
| walcl_haven_usd | −0.044% | −1.23 | −1.42 | 12% | 29% | −0.31 |
| walcl_chg_pb_fx | +0.023% | +0.65 | +0.76 | 18% | 25% | +0.17 |
| bs_diff_pb_fx | +0.013% | +0.33 | +0.38 | 16% | 25% | +0.08 |
| bs_peer_xs | −0.111% | −1.32 | −1.29 | 45% | 12% | −0.30 |
| bs_gdp_pb_fx | −0.031% | −1.02 | −1.02 | 13% | 38% | −0.27 |
| bs_ew | −0.018% | −0.48 | −0.53 | 47% | 13% | −0.12 |

### Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| walcl_pb_fx | holdout_365d | −0.01% | 42% | PASS | no |
| walcl_pb_fx | year_2024 | +0.00% | 0% | PASS | no |
| walcl_pb_fx | year_2025 | +0.24% | 55% | PASS | no |
| walcl_pb_fx | year_2026 | −0.19% | 38% | PASS | no |
| bs_peer_xs | holdout_365d | +0.18% | 50% | PASS | no |
| bs_peer_xs | year_2025 | +0.49% | 73% | PASS | no† |
| bs_ew | holdout_365d | +0.05% | 58% | PASS | no |
| walcl_haven_usd | holdout_365d | +0.01% | 58% | PASS | no |

† Year-2025 %pos can clear 70% on peer XS with mild mean, but full-sample mean is **negative** (NW t≈−1.3) and holdout mean ≪ 1% — classic window luck, not a promote.

### Risk sweep (IS → OOS)

Positive IS means on PB legs → sweep run. Best scaled IS mean (`walcl_pb_fx` @ ~4.7× daily-bound) ≈ **+0.23%/mo** still ≪ 1%; scaled HO flat/near zero. Scaled clears: **NO**. Leverage does not invent consistency.

**Board:** n=7 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. **Scaled primary (`walcl_pb_fx`) promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` unchanged.

### Honest read

Neely / Gagnon portfolio-balance priors are the right *direction* and are **distinct** from NFCI funding and CA/GDP: mild positive full-sample mean on `walcl_pb_fx` / `walcl_chg_pb_fx` / `bs_diff_pb_fx` (~+1–4 bp/mo, NW t ≈ 0.4–1.4), with haven and peer-XS / BS-GDP legs flat-to-negative. Binary z≥1 episodes are infrequent → sparse %pos (~16%). Far from prop-firm 1%/mo + 70% hit-rate. Free FRED multi-CB (Fed+ECB+BoJ) is sufficient; inventing BoE continuity after UKASSETS end-2014 would be dishonest. Multi-CB coverage was **not** too thin — real-rate fallback deferred.

**Next structure (if promote=0):** delivered as wave §23 (real-rate / breakeven).

Artifacts: `reports/scholarly_fx_cb_bs_wave.md`, `scholarly_fx_cb_bs_*.csv`, `scholarly_fx_cb_bs_meta.json`.


---

## 23. Real-rate / breakeven differential wave results (2026-09-23 BST) — Frankel / Meese–Rogoff

**Design (fixed priors, no HO tuning):** Distinct RID / inflation-expectations channel vs nominal yield-curve (§16), PPP/CPI real FX (§13), and CB-BS QE (§22).
- Sources: FRED `DFII10` (US 10y TIPS real), `T10YIE` (10y breakeven), `DGS10` (nominal); foreign OECD LT govt − CPI YoY as **honest proxy** (not true linkers where unavailable).
- PIT: daily `pub_lag_days=1`, monthly `pub_lag_months=1` + `signal_lag=1` trading day (daily tilts) / 1 month (XS) + 1d weight lag. Costs 1.5 bps/side.
- Legs: `us_real_usd` (**primary** — high US real → long USD), `us_real_chg_usd`, `us_be_fx` (high BE → long FX), `us_be_usd` (alternate), `rr_xs` / `rr_z_xs` / `rr_chg_xs` (foreign proxy − US TIPS sorts), `rr_ew`.
- **Explicit:** scholarly sleeve evaluation only — **no cooler overlay** on locked `fx4plus_gbpcad_d1_voltarget_0025`. Free FRED only.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| us_real_usd | +0.012% | +0.31 | +0.30 | 21% | 23% | +0.08 |
| us_real_chg_usd | +0.084% | +2.46 | +2.24 | 28% | 20% | +0.65 |
| us_be_fx | -0.062% | -1.81 | -1.97 | 16% | 35% | -0.51 |
| us_be_usd | +0.053% | +1.54 | +1.73 | 19% | 25% | +0.44 |
| rr_xs | -0.068% | -1.04 | -1.16 | 50% | 10% | -0.22 |
| rr_z_xs | -0.056% | -0.94 | -1.04 | 49% | 14% | -0.15 |
| rr_chg_xs | -0.034% | -0.51 | -0.48 | 50% | 12% | -0.10 |
| rr_ew | -0.027% | -0.75 | -0.70 | 48% | 12% | -0.16 |

### Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| us_real_usd | year_2024 | -0.09% | 9% | PASS | no |
| us_real_usd | year_2025 | -0.08% | 0% | PASS | no |
| us_real_usd | year_2026 | +0.04% | 25% | PASS | no |
| us_real_usd | holdout_365d | +0.03% | 17% | PASS | no |
| us_real_chg_usd | year_2024 | +0.13% | 36% | PASS | no |
| us_real_chg_usd | year_2025 | +0.01% | 18% | PASS | no |
| us_real_chg_usd | year_2026 | +0.02% | 38% | PASS | no |
| us_real_chg_usd | holdout_365d | +0.03% | 42% | PASS | no |
| us_be_usd | year_2024 | +0.10% | 18% | PASS | no |
| us_be_usd | year_2025 | -0.07% | 18% | PASS | no |
| us_be_usd | year_2026 | -0.05% | 12% | PASS | no |
| us_be_usd | holdout_365d | -0.03% | 8% | PASS | no |
| rr_xs | year_2024 | -0.10% | 73% | PASS | no |
| rr_xs | year_2025 | -0.19% | 45% | PASS | no |
| rr_xs | year_2026 | +0.14% | 62% | PASS | no |
| rr_xs | holdout_365d | +0.04% | 50% | PASS | no |
| rr_ew | year_2024 | -0.10% | 55% | PASS | no |
| rr_ew | year_2025 | -0.13% | 36% | PASS | no |
| rr_ew | year_2026 | +0.09% | 38% | PASS | no |
| rr_ew | holdout_365d | +0.03% | 33% | PASS | no |

### Risk sweep (IS → OOS)

Sweep run (positive IS on some legs). Primary `us_real_usd` scale≈3.32 bind=daily; scaled IS mean_mo≈+0.037% still ≪ 1%; scaled HO clears: **NO**. Leverage does not invent consistency.

**Board:** n=8 soft=2 hard=1 promote=0. **Unscaled promote:** **NO**. **Scaled primary (`us_real_usd`) promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` unchanged.

### Honest read

Frankel RID / Meese–Rogoff priors are the right *economic* direction and are **distinct** from nominal curve, PPP, and CB-BS: mild positive full-sample mean on `us_real_usd` / `us_real_chg_usd` / `us_be_usd` (~+1–8 bp/mo; `us_real_chg_usd` NW t≈+2.2 is the only soft/hard hit), but %pos is sparse (~21–28%) because binary z≥1 tilts fire infrequently. Cross-sectional LT−CPI proxies (`rr_xs` family) are flat-to-negative — consistent with proxy noise vs true linkers. Far from prop-firm 1%/mo + 70% hit-rate. Free FRED TIPS/BE panel is sufficient for the USD state claim; inventing foreign linker series would be dishonest.

**Next structure (if promote=0):** Done as §24 (ToT). Next: **bilateral AI-GPR role decompositions** (Caldara–Iacoviello AI-GPR threats/acts / oil-region roles — not plain GPRC_* sorts) — *not* another locked-sleeve cooler. FX IV/RR still blocked without a free panel.

Artifacts: `reports/scholarly_fx_real_rate_wave.md`, `scholarly_fx_real_rate_*.csv`, `scholarly_fx_real_rate_meta.json`.

---

## 24. Terms-of-trade / commodity-currency wave results (2026-09-23 BST) — Cashin–Céspedes–Sahay

**Design (fixed priors, no HO tuning):** ToT change = export_mom − import_mom from frozen `COUNTRY_TOT_MAP` (AUD copper−oil, CAD oil−copper, NZD basket−oil; NOK/ZAR documented, not traded — no USDNOK/USDZAR on Yahoo history). Formation 63d / skip 21d; pub_lag=1d + signal_lag=1d; calendar-date align; costs 1.5 bps/side. Reuses Yahoo commodity panel on disk.

**What is new vs §11 CRR:** Prior wave used *single* mapped commodity momentum (AUD→copper, CAD→oil, NZD→basket). This wave is an explicit **export vs import** differential. Distinctness: corr(`tot_country_ts`, `crr_country_ts`) ≈ **0.71**; corr(`tot_vs_g10`, `crr_xs_basket`) ≈ **0.31** — not thin; ToT path stands alone (no AI-GPR fallback this wave).

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| tot_country_ts | -0.181% | -0.92 | -1.06 | 47% | 12% | -0.29 |
| tot_xs | -0.005% | -0.07 | -0.06 | 48% | 10% | -0.02 |
| tot_vs_g10 | -0.052% | -0.85 | -1.07 | 35% | 15% | -0.27 |
| tot_ew | -0.080% | -0.85 | -0.94 | 48% | 13% | -0.26 |
| crr_country_ts_baseline | -0.037% | -0.23 | -0.30 | 38% | 11% | -0.06 |
| crr_xs_basket_baseline | +0.001% | +0.01 | +0.01 | 38% | 20% | -0.00 |

### Consistency windows (selected)

| Strategy | Window | mean_mo | %pos | gates | 1% bar |
|----------|--------|--------:|-----:|:-----:|:------:|
| tot_country_ts | year_2024 | -1.02% | 18% | FAIL | no |
| tot_country_ts | year_2025 | +0.22% | 55% | PASS | no |
| tot_country_ts | year_2026 | -0.56% | 38% | PASS | no |
| tot_country_ts | holdout_365d | +0.05% | 42% | PASS | no |
| tot_xs | year_2024 | -0.32% | 27% | PASS | no |
| tot_xs | year_2025 | -0.32% | 36% | PASS | no |
| tot_xs | year_2026 | -0.00% | 50% | PASS | no |
| tot_xs | holdout_365d | +0.11% | 50% | PASS | no |
| tot_vs_g10 | year_2024 | -0.39% | 27% | PASS | no |
| tot_vs_g10 | year_2025 | +0.05% | 45% | PASS | no |
| tot_vs_g10 | year_2026 | -0.03% | 38% | PASS | no |
| tot_vs_g10 | holdout_365d | +0.18% | 50% | PASS | no |
| tot_ew | year_2024 | -0.58% | 27% | PASS | no |
| tot_ew | year_2025 | -0.02% | 64% | PASS | no |
| tot_ew | year_2026 | -0.20% | 50% | PASS | no |
| tot_ew | holdout_365d | +0.11% | 58% | PASS | no |

### Risk sweep

Skipped — no ToT leg with positive IS mean monthly. Leverage would not invent consistency.

**Board:** n=4 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. **Scaled primary (`tot_country_ts`) promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` unchanged.

### Honest read

Cashin–CCS ToT priors are the right *economic* refinement of CRR and are **empirically distinct** on this free panel (corr≪1), but the export−import Yahoo futures proxy does not clear prop-firm 1%/mo + 70% hit-rate. Means are zero-to-negative (~−18 bp/mo on primary). Free futures ≠ true country export/import unit-value indices — a limitation shared with §11. Expanding to NOK/ZAR would need FX history we do not have under `approximate_non_ftmo`.

**Next structure (if promote=0):** Done as §25 (AI-GPR roles). FX IV/RR still blocked without a free panel; next free scholarly candidate if promote=0 again: news-based currency-specific sentiment (free multi-year panel) or FTMO MT5 CSV re-run.

Artifacts: `reports/scholarly_fx_tot_wave.md`, `scholarly_fx_tot_*.csv`, `scholarly_fx_tot_meta.json`.

---

## 25. Bilateral AI-GPR role decompositions wave results (2026-09-23 BST) — Caldara–Iacoviello

**Design (fixed priors, no HO tuning):** Daily AI-GPR roles from free `ai_gpr_daily.csv`. Primary `ai_threats_usd`: binary long-USD when lagged trailing-z(THREATS_GPR_AI) ≥ 1. Companion legs: `ai_acts_usd`, `ai_gpr_usd`, `oil_gpr_usd`, `oil_threats_usd`, `oil_me_vs_non` (ME oil − non-oil differential), scholarly `carry_ai_threats_cool` (cool≤1; **not** applied to locked fx4plus), `ai_ew`. PIT `pub_lag=1d` + `signal_lag=1d`; z_window=252; usd_tilt=0.5; costs 1.5 bps/side. Mapped to USD majors via `USD_LONG_PAIRS`.

**What is new vs §8–§9:** Aggregate GPR regime and monthly GPRC_* country sorts use classic / country indices. This wave uses **AI newspaper GPR role decompositions** (threats vs acts; oil vs non-oil; ME oil differential) — bilateral role structure, not another country sort or sleeve cooler.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| ai_threats_usd | -0.012% | -0.40 | -0.39 | 43% | 21% | -0.12 |
| ai_acts_usd | +0.001% | +0.05 | +0.06 | 37% | 24% | +0.01 |
| ai_gpr_usd | -0.045% | -1.52 | -1.55 | 33% | 28% | -0.43 |
| oil_gpr_usd | +0.020% | +0.76 | +0.71 | 41% | 17% | +0.20 |
| oil_threats_usd | -0.043% | -1.59 | -1.48 | 39% | 19% | -0.41 |
| oil_me_vs_non | -0.027% | -1.05 | -1.17 | 36% | 21% | -0.30 |
| carry_ai_threats_cool | -0.062% | -0.95 | -1.03 | 49% | 12% | -0.25 |
| ai_ew | -0.009% | -0.42 | -0.43 | 44% | 24% | -0.12 |

### Consistency windows (primary `ai_threats_usd`)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| year_2024 | -0.03% | 36% | 90% | PASS | no |
| year_2025 | -0.30% | 27% | 100% | PASS | no |
| year_2026 | +0.04% | 62% | 83% | PASS | no |
| holdout_365d | +0.07% | 58% | 74% | PASS | no |

### Risk sweep (IS → OOS)

Sweep run (positive IS mean on some legs, e.g. `oil_gpr_usd` / `ai_acts_usd`). Primary `ai_threats_usd` scale≈2.92 bind=static; scaled IS mean still negative / ≪ 1%; scaled HO clears: **NO**. Leverage does not invent consistency.

**Board:** n=8 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. **Scaled primary (`ai_threats_usd`) promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` verified unchanged PASS (2024 0.42%/55%/74%; 2025 1.63%/73%/69%; 2026 2.30%/88%/87%; HO 1.52%/75%/81%).

### Honest read

Caldara–Iacoviello AI-GPR *role* priors are the right economic refinement beyond aggregate GPR / GPRC_* sorts and are implementable on free daily data, but binary USD tilts on elevated threats/acts/oil roles do not clear prop-firm 1%/mo + 70% hit-rate on Yahoo D1. Full-sample means are ~0 to −6 bp/mo (primary −1.2 bp; best oil leg ~+2 bp with NW t≪1.5). Soft/hard NW boards empty. Distinctness from §8–§11/§24 is by construction (role columns, not country sorts or commodity prices). No go-live claim under `approximate_non_ftmo`.

**Next structure (if promote=0):** Done as §26 (BNP crash-skew). FX IV/RR + news sentiment still blocked; next free scholarly candidates if promote=0 again: Lustig–Verdelhan dollar-factor beta sorts or FRED fiscal-balance differentials — *not* another locked-sleeve cooler.

Artifacts: `reports/scholarly_fx_ai_gpr_wave.md`, `scholarly_fx_ai_gpr_*.csv`, `scholarly_fx_ai_gpr_meta.json`.


## 26. Brunnermeier–Nagel–Pedersen crash-risk / return-skewness wave results (2026-09-23 BST)

**Design (fixed priors, no HO tuning):** Free OHLC-only crash-risk proxies. Primary `crash_skew_xs`: monthly XS long high-crash-risk (more *negative* trailing return skewness) / short low, formation 63d, skip=1, signal_lag=1. Companion legs: `crash_skew_xs_126` (126d), `left_tail_xs` (mean of returns ≤5th pct over 63d), `mom_skew_regime` (Menkhoff-style mom within high-crash-score half), scholarly `carry_crash_cool` (cool≤1 when agg crash-skew z elevated; **not** applied to locked fx4plus), `crash_ew`. Costs 1.5 bps/side. Mapped to USD majors via `USD_PAIRS`.

**What is new vs §15 / carry / §25:** Menkhoff FX-RV is *level* absolute-return vol; Lustig carry is rate differentials; AI-GPR is newspaper role indices. This wave is **return skewness / left-tail shortfall** as the BNP crash-risk characteristic — buildable without paid IV/RR.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| crash_skew_xs | -0.151% | -2.31 | -2.02 | 43% | 14% | -0.54 |
| crash_skew_xs_126 | -0.066% | -0.99 | -1.04 | 47% | 19% | -0.22 |
| left_tail_xs | +0.070% | +0.94 | +1.07 | 51% | 15% | +0.23 |
| mom_skew_regime | -0.086% | -1.30 | -1.17 | 43% | 19% | -0.25 |
| carry_crash_cool | +0.002% | +0.03 | +0.04 | 51% | 12% | -0.03 |
| crash_ew | -0.057% | -1.37 | -1.39 | 45% | 20% | -0.32 |

### Consistency windows (primary `crash_skew_xs`)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| year_2024 | -0.23% | 45% | 89% | PASS | no |
| year_2025 | -0.06% | 64% | 80% | PASS | no |
| year_2026 | +0.03% | 75% | 79% | PASS | no |
| holdout_365d | +0.03% | 75% | 60% | PASS | no |

### Risk sweep (IS → OOS)

Sweep **run** (positive IS mean on `left_tail_xs`). Primary `crash_skew_xs` scale≈0.35 bind=static; scaled IS mean still negative / ≪1%; scaled HO clears: **NO**. Leverage does not invent consistency on a negative-mean primary.

**Board:** n=6 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. **Scaled primary (`crash_skew_xs`) promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` config **untouched**; re-verify gates PASS on all key windows (Yahoo D1 drift vs prior keepalive: 2024 ~0.36%/55%/78%; 2025 1.84%/73%/68%; 2026 2.43%/75%/88%; HO 1.54%/67%/84% — see `quest_locked_verify_crash_skew.md`).

### Honest read

BNP crash-risk / skewness is the right free-data structure given blocked FX IV/RR, and is cleanly distinct from Menkhoff level-vol and AI-GPR roles. On Yahoo D1 G10 the primary 63d skew HML is *wrong-signed* on this sample (full-sample ≈ −15 bp/mo, NW t ≈ −2.0) — currencies that *looked* crash-prone underperformed rather than earning a premium. Left-tail shortfall is the only soft-positive leg (~+7 bp/mo, NW t≈1.1) and still nowhere near 1%/mo + 70% hit-rate. Soft/hard NW boards empty for positive means. No go-live claim under `approximate_non_ftmo`.

**Next structure (if promote=0):** Done as §27 (dollar-factor beta). FX IV/RR + news-sentiment still **blocked**. Next free scholarly candidate: **FRED fiscal-balance / government budget differentials** — *not* another locked-sleeve cooler. Re-score all boards when FTMO MT5 CSVs arrive.

Artifacts: `reports/scholarly_fx_crash_skew_wave.md`, `scholarly_fx_crash_skew_*.csv`, `scholarly_fx_crash_skew_meta.json`.


## 27. Lustig–Verdelhan dollar-factor beta wave results (2026-09-23 BST)

**Design (fixed priors, no HO tuning):** Free FX panel only. Primary `dollar_beta_xs`: monthly OLS β of each currency's return on RX (EW foreign excess return vs USD) over **60 months**, skip=1m + signal_lag=1m; monthly XS long high-$β / short low-$β (n=2/2). Companion: `dollar_beta_xs_36m`; `dollar_beta_afd` (same HML × sign(AFD) from FRED short-rate differentials); `dollar_rx_tsmom` (sign of trailing 12m RX → long/short all FX vs USD); `dollar_ew`. Costs 1.5 bps/side. β estimated on **monthly** returns (Verdelhan 60m prior), weights expanded daily.

**What is new vs carry / FX-RV / equity-diff / crash-skew / AI-GPR:** Lustig carry sorts on *rates*; Menkhoff on *level* FX-RV; Hau–Rey on equity differentials; BNP on return *skew*; AI-GPR on newspaper roles. This wave sorts on **rolling dollar-factor β**.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| dollar_beta_xs | +0.014% | +0.18 | +0.20 | 39% | 15% | +0.05 |
| dollar_beta_xs_36m | −0.033% | −0.46 | −0.51 | 37% | 14% | −0.10 |
| dollar_beta_afd | +0.020% | +0.25 | +0.27 | 43% | 16% | +0.06 |
| dollar_rx_tsmom | −0.232% | −1.68 | −1.67 | 38% | 12% | −0.43 |
| dollar_ew | −0.057% | −1.33 | −1.20 | 38% | 17% | −0.34 |

### Consistency windows (primary `dollar_beta_xs`)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| year_2024 | −0.04% | 64% | 62% | PASS | no |
| year_2025 | −0.18% | 27% | 100% | PASS | no |
| year_2026 | +0.10% | 62% | 79% | PASS | no |
| holdout_365d | +0.18% | 58% | 67% | PASS | no |

### Risk sweep (IS → OOS)

Sweep **run** (positive IS mean on `dollar_beta_xs` / `dollar_beta_afd`). Primary scale≈0.89 bind=static; scaled IS mean still ≪1%; scaled HO clears: **NO**. AFD-conditioned leg is the soft-best (+2 bp/mo) but NW t≈0.27 — not a soft board hit.

**Board:** n=5 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. **Scaled primary (`dollar_beta_xs`) promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` config **untouched**; re-verify gates PASS on all key windows (Yahoo D1: 2024 0.36%/55%/78%; 2025 1.84%/73%/68%; 2026 2.43%/75%/88%; HO 1.54%/67%/84% — see `quest_locked_verify_dollar_beta.md`).

### Honest read

Dollar-factor β sorts are the right free-data Verdelhan structure after BNP crash-skew, and are cleanly distinct from carry / FX-RV / equity-diff / skew / AI-GPR. On Yahoo D1 G10 the primary 60m β HML is essentially **flat** (full-sample ≈ +1.4 bp/mo, NW t ≈ 0.2, %pos 39%) — nowhere near 1%/mo + 70% hit-rate. AFD conditioning helps only marginally (+2 bp/mo). Dollar RX TSMOM is wrong-signed on this sample (~−23 bp/mo). Soft/hard NW boards empty for positive means. No go-live claim under `approximate_non_ftmo`.

**Next structure (if promote=0):** Done as §28 (fiscal-balance). FX IV/RR + news-sentiment still **blocked**. Next free scholarly candidates: sovereign **debt/GDP** differentials or monthly **trade-balance** — *not* another locked-sleeve cooler. Re-score all boards when FTMO MT5 CSVs arrive.

Artifacts: `reports/scholarly_fx_dollar_beta_wave.md`, `scholarly_fx_dollar_beta_*.csv`, `scholarly_fx_dollar_beta_meta.json`.


## 28. Twin-deficits / fiscal-balance wave results (2026-09-23 BST)

**Design (fixed priors, no HO tuning):** IMF WEO general-government net lending/borrowing (% GDP) via FRED `GGNLBA*188N`. Primary `fiscal_surplus_xs`: monthly XS long high fiscal balance / short deficit (n=2/2) on mapped G10 (USD, EUR-DE proxy, GBP, JPY, CAD, AUD; NZD/CHF unmapped). Companion: `fiscal_deficit_xs` (risk-premium opposite), `fiscal_chg_xs` (YoY Δ), `us_fiscal_twin_fx` / `us_fiscal_haven_usd` (US deficit z tilts), `us_mts_chg_usd` (MTS surplus z → long USD), scholarly `fiscal_ca_blend` (EW with CA surplus; **not** on locked fx4plus), `fiscal_ew`. PIT `pub_lag_months=15` + `signal_lag=1m` + 1d weight lag; costs 1.5 bps/side.

**What is new vs CA / CB-BS / macro-diff / dollar-beta:** CA (§21) is external imbalance; CB-BS is QE assets; macro-diff is CPI/IP/UR; dollar-beta sorts on RX β. This wave is **fiscal balance / GDP** (budget stance) — twin-deficits stock-flow cousin of CA, not the same series.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| fiscal_surplus_xs | −0.037% | −0.75 | −0.75 | 48% | 12% | −0.18 |
| fiscal_deficit_xs | +0.035% | +0.70 | +0.70 | 52% | 12% | +0.17 |
| fiscal_chg_xs | +0.021% | +0.36 | +0.36 | 48% | 14% | +0.07 |
| us_fiscal_twin_fx | −0.029% | −0.79 | −0.79 | 13% | — | −0.20 |
| us_fiscal_haven_usd | +0.030% | +0.79 | +0.79 | 13% | — | +0.20 |
| us_mts_chg_usd | −0.003% | −0.11 | −0.11 | 16% | — | −0.03 |
| fiscal_ca_blend | −0.032% | −0.88 | −0.88 | 47% | 13% | −0.22 |
| fiscal_ew | −0.015% | −0.58 | −0.58 | 48% | 14% | −0.15 |

### Consistency windows (primary `fiscal_surplus_xs`)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| year_2024 | +0.05% | 64% | 73% | PASS | no |
| year_2025 | −0.18% | 36% | 96% | PASS | no |
| year_2026 | +0.18% | 62% | 92% | PASS | no |
| holdout_365d | +0.06% | 50% | 86% | PASS | no |

### Risk sweep (IS → OOS)

Sweep **run** (positive IS mean on `fiscal_deficit_xs` / `fiscal_chg_xs` / `us_fiscal_haven_usd`). Primary `fiscal_surplus_xs` scale≈0.76 bind=static; scaled IS mean still negative / ≪1%; scaled HO clears: **NO**. Deficit-premium opposite is the soft-best (+3.5 bp/mo, NW t≈0.7) — still nowhere near a soft board hit.

**Board:** n=8 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. **Scaled primary (`fiscal_surplus_xs`) promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` config **untouched**; re-verify gates PASS on all key windows (Yahoo D1: 2024 0.42%/55%/74%; 2025 1.63%/73%/69%; 2026 2.30%/88%/87%; HO 1.52%/75%/81% — see `quest_locked_verify_fiscal.md`).

### Honest read

Fiscal-balance / twin-deficits is the right free-data structure after dollar-factor β, and is cleanly distinct from CA/GDP, CB-BS, and macro-diff. On Yahoo D1 G10 the primary surplus HML is essentially **flat-to-negative** (full-sample ≈ −3.7 bp/mo, NW t ≈ −0.75, %pos 48%) — nowhere near 1%/mo + 70% hit-rate. The deficit-premium opposite and YoY-change legs are only marginally positive. US tilt legs rarely fire in 2024–26 under honest 15m WEO lag (annual fiscal state moves slowly). Soft/hard NW boards empty for positive means. NZD/CHF missing from free GGNLBA panel. No go-live claim under `approximate_non_ftmo`.

**Next structure (if promote=0):** Done as §29 (debt/GDP). FX IV/RR + news-sentiment still **blocked**. Next free scholarly candidates: monthly **trade-balance** (higher-freq vs quarterly CA) or BIS/FRED **REER** misalignment — *not* another locked-sleeve cooler. Re-score all boards when FTMO MT5 CSVs arrive.

Artifacts: `reports/scholarly_fx_fiscal_wave.md`, `scholarly_fx_fiscal_*.csv`, `scholarly_fx_fiscal_meta.json`.


## 29. Government debt/GDP wave results (2026-09-23 BST)

**Design (fixed priors, no HO tuning):** IMF WEO general-government gross debt (% GDP) via FRED `GGGDTA*188N` (brief `GGXWDG*188N` 404 — honest alias failure). Primary `low_debt_xs`: monthly XS long low debt/GDP / short high (n=2/2) on mapped G10 (USD, EUR-DE proxy, GBP, JPY, CAD, AUD; NZD/CHF unmapped). Companion: `high_debt_xs` (debtor-premium opposite), `debt_chg_xs` (YoY −Δ debt), `us_debt_twin_fx` / `us_debt_haven_usd` (US debt z tilts from quarterly `GFDEGDQ188S`), scholarly `debt_fiscal_blend` (EW with fiscal surplus; **not** on locked fx4plus), `debt_ew`. PIT `pub_lag_months=15` + US-Q `pub_lag=4m` + `signal_lag=1m` + 1d weight lag; costs 1.5 bps/side.

**What is new vs fiscal / CA / CB-BS / macro-diff:** Fiscal (§28) is *flow* net lending/borrowing; CA is external imbalance; this wave is the **debt stock / GDP** sustainability channel.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| low_debt_xs | +0.019% | +0.27 | +0.27 | 53% | 12% | +0.03 |
| high_debt_xs | −0.020% | −0.27 | −0.27 | 47% | — | — |
| debt_chg_xs | +0.014% | +0.27 | +0.27 | 52% | — | — |
| us_debt_twin_fx | −0.024% | −0.58 | −0.58 | 27% | — | — |
| us_debt_haven_usd | +0.023% | +0.56 | +0.56 | 28% | — | — |
| debt_fiscal_blend | −0.008% | −0.15 | −0.15 | 54% | — | — |
| debt_ew | +0.004% | +0.10 | +0.10 | 50% | — | — |

### Consistency windows (primary `low_debt_xs`)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| year_2024 | −0.04% | 64% | 68% | PASS | no |
| year_2025 | +0.32% | 73% | 60% | PASS | no |
| year_2026 | +0.06% | 50% | 81% | PASS | no |
| holdout_365d | +0.23% | 67% | 54% | PASS | no |

### Risk sweep (IS → OOS)

Sweep **run** (positive IS mean on several legs). Primary `low_debt_xs` scale≈0.67 bind=static; scaled IS mean ≈0 bp/mo ≪1%; scaled HO mean≈0.16%/mo %pos 67% — clears: **NO**. Best soft-ish leg is `us_debt_haven_usd` (+2.3 bp/mo, NW t≈0.56) — still nowhere near a soft board hit.

**Board:** n=7 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. **Scaled primary (`low_debt_xs`) promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` config **untouched**; re-verify gates PASS on all key windows (Yahoo D1: 2024 0.42%/55%/74%; 2025 1.63%/73%/69%; 2026 2.30%/88%/87%; HO 1.52%/75%/81% — see `quest_locked_verify_debt_gdp.md`).

### Honest read

Debt/GDP stock is the right free-data sustainability structure after fiscal-balance flow (§28), and is cleanly distinct from GGNLBA / CA / CB-BS. On Yahoo D1 G10 the primary low-debt HML is essentially **flat** (full-sample ≈ +1.9 bp/mo, NW t ≈ 0.27, %pos 53%) — nowhere near 1%/mo + 70% hit-rate. Change and haven legs are only marginally positive. Soft/hard NW boards empty for positive means. NZD/CHF missing; `GGXWDG*` mnemonic 404 (used live `GGGDTA*`). No go-live claim under `approximate_non_ftmo`.

**Next structure (if promote=0):** Done as §30 (monthly trade-balance). FX IV/RR + news-sentiment still **blocked**. Next free scholarly candidate: BIS/FRED **REER** misalignment — *not* another locked-sleeve cooler. Re-score all boards when FTMO MT5 CSVs arrive.

Artifacts: `reports/scholarly_fx_debt_gdp_wave.md`, `scholarly_fx_debt_gdp_*.csv`, `scholarly_fx_debt_gdp_meta.json`.


## 30. Monthly trade-balance FX wave results (2026-09-23 BST)

**Design (fixed priors, no HO tuning):** OECD MEI merchandise exports/imports via FRED `XTEXVA01*M667S` / `XTIMVA01*M667S`; normalised to **TB/exports = (EXP−IMP)/EXP**. Primary `tb_deficit_xs`: monthly XS long low TB (deficit) / short high surplus (n=2/2) on **full G10** (USD, EUR-DE proxy, GBP, JPY, CAD, AUD, NZD, CHF — NZD/CHF mapped unlike debt/fiscal). Companion: `tb_surplus_xs` (flow opposite), `tb_chg_xs` (YoY Δ), `us_tb_gr_fx` / `us_tb_haven_usd` (US BOPGSTB z tilts), scholarly `tb_ca_blend` (EW with CA debtor; **not** on locked fx4plus), `tb_ew`. PIT `pub_lag_months=2` + `signal_lag=1m` + 1d weight lag; costs 1.5 bps/side.

**What is new vs CA / debt / fiscal:** Quarterly IMF BOP CA/GDP (§21) is slow; debt/GDP (§29) is stock sustainability; fiscal (§28) is budget flow. This wave is the **monthly goods trade-balance** high-frequency external-balance channel.

**Data notes:** `XTEXVA01EZM667S`/`XTIMVA01EZM667S` live but end ~2023-04 — Germany DEM used as EUR proxy (continuity). `USAB6BLTT02STSAM` (monthly CA) and `BOPB12` **404** on FRED. US tilts use `BOPGSTB` (goods+services).

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| tb_deficit_xs | +0.008% | +0.13 | +0.14 | 49% | 13% | +0.06 |
| tb_surplus_xs | −0.019% | −0.31 | −0.32 | 50% | 12% | −0.09 |
| tb_chg_xs | −0.077% | −1.24 | −1.14 | 44% | 12% | −0.27 |
| us_tb_gr_fx | −0.054% | −1.31 | −1.34 | 18% | 27% | −0.35 |
| us_tb_haven_usd | +0.052% | +1.27 | +1.30 | 22% | 21% | +0.33 |
| tb_ca_blend | +0.014% | +0.24 | +0.28 | 50% | 13% | +0.05 |
| tb_ew | −0.040% | −1.44 | −1.44 | 49% | 12% | −0.32 |

### Consistency windows (primary `tb_deficit_xs`)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| year_2024 | +0.08% | 73% | 78% | PASS | no |
| year_2025 | −0.22% | 45% | 76% | PASS | no |
| year_2026 | −0.01% | 38% | 100% | PASS | no |
| holdout_365d | −0.06% | 42% | 80% | PASS | no |

### Risk sweep (IS → OOS)

Sweep **run** (positive IS mean on `tb_deficit_xs` / `us_tb_haven_usd` / `tb_ca_blend`). Primary `tb_deficit_xs` scale≈1.18 bind=daily; scaled IS mean ≈2 bp/mo ≪1%; scaled HO mean≈−0.08%/mo %pos 42% — clears: **NO**. Soft-best leg is `us_tb_haven_usd` (+5.2 bp/mo, NW t≈1.30) — still below soft board (|t|≥1.5).

**Board:** n=7 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. **Scaled primary (`tb_deficit_xs`) promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` config **untouched**; re-verify gates PASS on all key windows (Yahoo D1: 2024 0.42%/55%/74%; 2025 1.63%/73%/69%; 2026 2.30%/88%/87%; HO 1.52%/75%/81% — see `quest_locked_verify_trade_balance.md`).

### Honest read

Monthly trade-balance is the right free-data high-frequency external-balance structure after debt/GDP (§29), and is cleanly distinct from quarterly CA/GDP. On Yahoo D1 G10 the primary deficit HML is essentially **flat** (full-sample ≈ +0.8 bp/mo, NW t ≈ 0.14, %pos 49%) — nowhere near 1%/mo + 70% hit-rate. Change and GR legs are wrong-signed; the US haven tilt is the soft-best but still |NW t|<1.5. Soft/hard NW boards empty for positive means. Full G10 mapped (NZD/CHF present). No go-live claim under `approximate_non_ftmo`.

**Next structure (if promote=0):** Done as §31 (BIS REER). FX IV/RR + news-sentiment still **blocked**. Next free scholarly candidates: IMF/WEO **reserves** / external-buffer differentials or BIS **credit-to-GDP** — *not* another locked-sleeve cooler. Re-score all boards when FTMO MT5 CSVs arrive.

Artifacts: `reports/scholarly_fx_trade_balance_wave.md`, `scholarly_fx_trade_balance_*.csv`, `scholarly_fx_trade_balance_meta.json`.


## 31. BIS REER undervaluation FX wave results (2026-09-23 BST)

**Design (fixed priors, no HO tuning):** FRED BIS real broad EER `RB*BIS` monthly index. Primary `reer_cheap_xs`: XS long low 60m trailing REER z (undervalued) / short high z (n=2/2) on **full G10** (USD, EUR=`RBXMBIS` euro-area, GBP, JPY, CAD, AUD, NZD, CHF). Companion: `reer_cheap_xs_36` (36m z board), `reer_mom_xs` (high-z momentum contrast), `reer_chg_xs` (−Δ12 REER), `us_reer_strong_usd` / `us_reer_meanrev_fx` (US RBUSBIS z tilts), `reer_ew`. PIT `pub_lag_months=2` + `signal_lag=1m` + 1d weight lag; costs 1.5 bps/side.

**What is new vs PPP / TB / debt / fiscal:** Homemade bilateral CPI PPP (`ppp_real_fx`) is DIY real bilateral; TB (§30) is monthly goods external balance; debt/fiscal are fiscal stock/flow. This wave is the **official multilateral BIS REER** undervaluation / mean-reversion channel (Rogoff PPP puzzle / Taylor misalignment).

**Data notes:** `RBXMBIS` (euro area) preferred for EUR continuity; `RBDEBIS` Germany alt documented unused. `RBEZBIS`/`RBEMUBIS` **404** on FRED. Full G10 incl. NZD/CHF mapped. Latest raw month ~2026-07 as of 2026-09-23 — consistent with fixed pub_lag=2.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| reer_cheap_xs | +0.034% | +0.51 | +0.60 | 52% | 13% | +0.15 |
| reer_cheap_xs_36 | +0.022% | +0.34 | +0.35 | 50% | 12% | +0.11 |
| reer_mom_xs | −0.044% | −0.65 | −0.76 | 48% | 13% | −0.18 |
| reer_chg_xs | +0.045% | +0.70 | +0.75 | 52% | 9% | +0.20 |
| us_reer_strong_usd | +0.022% | +0.42 | +0.46 | 27% | 17% | +0.11 |
| us_reer_meanrev_fx | −0.023% | −0.44 | −0.49 | 23% | 19% | −0.11 |
| reer_ew | +0.019% | +0.42 | +0.49 | 51% | 11% | +0.14 |

### Consistency windows (primary `reer_cheap_xs`)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| year_2024 | −0.11% | 36% | 93% | PASS | no |
| year_2025 | −0.27% | 27% | 100% | PASS | no |
| year_2026 | +0.16% | 62% | 81% | PASS | no |
| holdout_365d | +0.08% | 58% | 66% | PASS | no |

### Risk sweep (IS → OOS)

Sweep **run** (positive IS mean on several legs). Primary `reer_cheap_xs` scale≈2.92 bind=daily; scaled IS mean ≈9 bp/mo ≪1%; scaled HO mean≈0.24%/mo %pos 58% — clears: **NO**. Soft-best-ish leg is `reer_chg_xs` (+4.5 bp/mo, NW t≈0.75) — still far below soft board (|t|≥1.5).

**Board:** n=7 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. **Scaled primary (`reer_cheap_xs`) promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` config **untouched**; re-verify gates PASS on all key windows (Yahoo D1: 2024 0.42%/55%/74%; 2025 1.63%/73%/69%; 2026 2.30%/88%/87%; HO 1.52%/75%/81% — see `quest_locked_verify_bis_reer.md`).

### Honest read

Official BIS multilateral REER is the right free-data undervaluation structure after TB (§30), and is cleanly distinct from homemade bilateral PPP. On Yahoo D1 G10 the primary cheap-REER HML is essentially **flat** (full-sample ≈ +3.4 bp/mo, NW t ≈ 0.60, %pos 52%) — nowhere near 1%/mo + 70% hit-rate. Momentum contrast is wrong-signed; change leg is the soft-best but |NW t|<1. Soft/hard NW boards empty for positive means. Full G10 mapped (NZD/CHF/`RBXMBIS` EUR present). No go-live claim under `approximate_non_ftmo`.

**Next structure (if promote=0):** Done as §33 (money-growth). FX IV/RR + news-sentiment still **blocked**. Next free scholarly candidate: IMF/WEO **reserves** / TRESEG* / total reserves % GDP, or OECD **house-price** differentials — *not* another locked-sleeve cooler. Re-score all boards when FTMO MT5 CSVs arrive.

Artifacts: `reports/scholarly_fx_bis_reer_wave.md`, `scholarly_fx_bis_reer_*.csv`, `scholarly_fx_bis_reer_meta.json`.


## 32. BIS private credit-to-GDP / credit-gap FX wave results (2026-09-23 BST)

**Design (fixed priors, no HO tuning):** FRED BIS private credit/GDP `Q*PAM770A` quarterly → monthly. Primary `low_credit_gap_xs`: XS long negative trailing-trend credit gap / short positive (n=2/2) on **full G10** (USD, EUR=`QXMPAM770A` euro-area, GBP, JPY, CAD, AUD, NZD, CHF). Gap = credit/GDP − trailing 180m mean (≈60q HP-like one-sided; precomputed FRED gap IDs 404). Companion: `low_credit_xs`, `high_credit_xs` (debtor/fragile alternate), `low_credit_z_xs` (5y z), `credit_chg_xs` (−Δ12), `us_credit_stress_fx` / `us_credit_haven_usd`, `credit_ew`. PIT `pub_lag_months=5` + `signal_lag=1m` + 1d weight lag; costs 1.5 bps/side.

**What is new vs debt / fiscal / TB / REER:** Government debt/GDP (§29 GGGDTA*) is *public* stock; fiscal GGNLBA (§28) is budget *flow*; TB (§30) is goods external balance; BIS REER (§31) is multilateral real FX. This wave is the **private credit / financial-cycle** Borio–Drehmann channel.

**Data notes:** Prefer `Q*PAM770A` (% GDP). Brief `CRDQ*APABIS` live but absolute bn — documented unused. `CRDQNZAPABIS` / precomputed gap IDs **404**. EUR=`QXMPAM770A` (Germany `QDEPAM770A` alt). Full G10 incl. NZD/CHF mapped. Latest raw quarter 2025-10 as of 2026-09-23 — consistent with fixed pub_lag=5.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| low_credit_gap_xs | −0.013% | −0.25 | −0.25 | 46% | — | — |
| low_credit_xs | +0.000% | +0.00 | +0.00 | 48% | — | — |
| high_credit_xs | −0.002% | −0.05 | −0.05 | 51% | — | — |
| low_credit_z_xs | +0.037% | +0.53 | +0.53 | 51% | — | — |
| credit_chg_xs | +0.083% | +1.20 | +1.20 | 51% | — | — |
| us_credit_stress_fx | −0.003% | −0.19 | −0.19 | 6% | — | — |
| us_credit_haven_usd | +0.003% | +0.20 | +0.20 | 5% | — | — |
| credit_ew | +0.023% | +0.62 | +0.62 | 51% | — | — |

### Consistency windows (primary `low_credit_gap_xs`)

| Window | mean_mo | %pos | gates | 1% bar |
|--------|--------:|-----:|:-----:|:------:|
| year_2024 | +0.11% | 55% | PASS | no |
| year_2025 | +0.00% | 55% | PASS | no |
| year_2026 | −0.04% | 50% | PASS | no |
| holdout_365d | +0.02% | 58% | PASS | no |

### Risk sweep (IS → OOS)

Sweep **run** (positive IS mean on several legs). Primary `low_credit_gap_xs` scale≈1.15 bind=daily; scaled HO mean≈0.02%/mo %pos 58% — clears: **NO**. Soft-best-ish leg is `credit_chg_xs` (+8.3 bp/mo, NW t≈1.20) — still below soft board (|t|≥1.5).

**Board:** n=8 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. **Scaled primary (`low_credit_gap_xs`) promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` config **untouched**; re-verify gates PASS on all key windows (Yahoo D1: 2024 0.42%/55%/74%; 2025 1.63%/73%/69%; 2026 2.30%/88%/87%; HO 1.52%/75%/81% — see `quest_locked_verify_bis_credit.md`).

### Honest read

BIS private credit/GDP is the right free-data financial-cycle structure after REER (§31), and is cleanly distinct from public debt/GDP. On Yahoo D1 G10 the primary low-gap HML is essentially **flat / slightly negative** (full-sample ≈ −1.3 bp/mo, NW t ≈ −0.25, %pos 46%) — nowhere near 1%/mo + 70% hit-rate. Change and 5y-z legs are the soft-best but |NW t|<1.5. Soft/hard NW boards empty for positive means. Full G10 mapped (NZD/CHF/`QXMPAM770A` EUR present). No go-live claim under `approximate_non_ftmo`.

**Next structure (if promote=0):** FX IV/RR + news-sentiment still **blocked**. Next free scholarly candidate: IMF/WEO **reserves** / TRESEG* / total reserves % GDP / external-buffer FX — *not* another locked-sleeve cooler. Re-score all boards when FTMO MT5 CSVs arrive.

Artifacts: `reports/scholarly_fx_bis_credit_wave.md`, `scholarly_fx_bis_credit_*.csv`, `scholarly_fx_bis_credit_meta.json`.


## 33. Monetary / money-growth FX wave results (2026-09-23 BST)

**Design (fixed priors, no HO tuning):** FRED OECD MEI `MABMM301*M657S` broad-money growth. Primary `low_money_growth_xs`: XS long low relative money growth / short high (n=2/2) on USD+EUR+GBP+JPY+CAD+AUD. Companion: `high_money_growth_xs`, `low_money_growth_z_xs` (5y z), `money_growth_chg_xs` (−Δ12 of growth), `us_money_stress_fx` / `us_money_haven_usd`, `money_ew`. PIT `pub_lag_months=2` + `signal_lag=1m` + 1d weight lag; costs 1.5 bps/side.

**What is new vs CB-BS / real-rate / credit:** CB-BS (§22) is *central-bank assets* (QE); real-rate (§23) is TIPS/BE; BIS credit (§32) is private credit/GDP. This wave is the classic **monetary-approach money-growth differential**.

**Data notes:** Prefer `MABMM301*M657S`. NZD/CHF 657S ends 2018 — **unmapped** on primary (NSA `*189N` archival). US `M2SL` level alt documented. EUR=`MABMM301EZM657S`.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| low_money_growth_xs | +0.034% | +0.55 | +0.59 | 52% | 11% | +0.12 |
| high_money_growth_xs | −0.058% | −0.92 | −0.99 | 46% | 12% | −0.21 |
| low_money_growth_z_xs | −0.003% | −0.04 | −0.04 | 46% | 13% | −0.06 |
| money_growth_chg_xs | +0.041% | +0.63 | +0.65 | 48% | 12% | +0.12 |
| us_money_stress_fx | −0.021% | −0.96 | −0.98 | 6% | 65% | −0.24 |
| us_money_haven_usd | +0.020% | +0.88 | +0.91 | 8% | 51% | +0.22 |
| money_ew | +0.018% | +0.48 | +0.54 | 48% | 12% | +0.10 |

### Consistency windows (primary `low_money_growth_xs`)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| year_2024 | +0.17% | 64% | 72% | PASS | no |
| year_2025 | −0.33% | 27% | 100% | PASS | no |
| year_2026 | +0.05% | 50% | 98% | PASS | no |
| holdout_365d | −0.11% | 42% | 96% | PASS | no |

### Risk sweep (IS → OOS)

Sweep **run**. Primary `low_money_growth_xs` scale≈3.637 bind=daily; scaled HO mean≈−0.41%/mo %pos 42% — clears: **NO** (OOS gates FAIL). Soft-best-ish leg is `money_growth_chg_xs` (~+4.1 bp/mo, NW t≈0.65) — below soft board (|t|≥1.5).

**Board:** n=7 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. **Scaled primary promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` config **untouched**; re-verify gates PASS (Yahoo D1: 2024 0.42%/55%/74%; 2025 1.63%/73%/69%; 2026 2.30%/88%/87%; HO 1.52%/75%/81% — see `quest_locked_verify_money_growth.md`).

### Honest read

Frenkel–Bilson money-growth XS is the right free-data monetary-approach structure after BIS credit (§32). On Yahoo D1 G10 the primary low-growth HML is essentially **flat** (full-sample ≈ +3.4 bp/mo, NW t ≈ 0.59, %pos 52%) — nowhere near 1%/mo + 70% hit-rate. Soft/hard NW boards empty. NZD/CHF unmapped on primary. No go-live claim under `approximate_non_ftmo`.

**Next structure (if promote=0):** Done as §34 (reserves). FX IV/RR + news-sentiment still **blocked**. Next free scholarly candidate: OECD **house-price** differentials or IG OAS (`BAMLC0A0CM`) risk-appetite — *not* another locked-sleeve cooler. Re-score all boards when FTMO MT5 CSVs arrive.

Artifacts: `reports/scholarly_fx_money_growth_wave.md`, `scholarly_fx_money_growth_*.csv`, `scholarly_fx_money_growth_meta.json`.


## 34. IMF IFS reserves / external-buffer FX wave results (2026-09-23 BST)

**Design (fixed priors, no HO tuning):** FRED IMF IFS `TRESEG*M052N` total reserves excl. gold (log USD mn). Primary `high_reserves_xs`: XS long high relative external buffer / short low (n=2/2) on USD+EUR(DE)+GBP+JPY+CAD+AUD. Companion: `low_reserves_xs`, `high_reserves_z_xs` (5y z), `reserves_chg_xs` (Δ12 of log), `us_reserves_stress_fx` / `us_reserves_haven_usd`, `reserves_ew`. PIT `pub_lag_months=3` + `signal_lag=1m` + 1d weight lag; costs 1.5 bps/side.

**What is new vs CA / TB / debt / money:** CA (§21) and TB (§30) are *flow* balances; debt (§29) is public *stock*; money-growth (§33) is monetary aggregates. This wave is the classic **external-buffer / reserve-adequacy** stock channel.

**Data notes:** Prefer `TRESEG*M052N`. EUR=`TRESEGDEM052N` (Germany; `TRESEGEZM052N` ends 2018-04). NZD/CHF **404** — unmapped. Reserves/GDP **not formed** (GDP units heterogeneous / many 404). US `TOTRESNS`/`WRESBAL` = Fed bank reserves (not IFS) — documented alt only.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_reserves_xs | −0.038% | −0.59 | −0.58 | 48% | 12% | −0.09 |
| low_reserves_xs | +0.036% | +0.57 | +0.56 | 52% | 13% | +0.09 |
| high_reserves_z_xs | −0.027% | −0.43 | −0.47 | 52% | 10% | −0.11 |
| reserves_chg_xs | +0.048% | +0.74 | +0.81 | 52% | 11% | +0.18 |
| us_reserves_stress_fx | −0.002% | −0.08 | −0.09 | 9% | 49% | −0.02 |
| us_reserves_haven_usd | +0.002% | +0.06 | +0.07 | 8% | 36% | +0.02 |
| reserves_ew | +0.003% | +0.10 | +0.10 | 46% | 16% | +0.06 |

### Consistency windows (primary `high_reserves_xs`)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| year_2024 | +0.04% | 36% | 100% | PASS | no |
| year_2025 | −0.32% | 27% | 100% | PASS | no |
| year_2026 | −0.06% | 50% | 93% | PASS | no |
| holdout_365d | −0.24% | 33% | 93% | PASS | no |

### Risk sweep (IS → OOS)

Sweep **run**. Primary `high_reserves_xs` scale≈2.333 bind=static; scaled HO mean≈−0.55%/mo %pos 33% — clears: **NO**. Soft-best-ish leg is `reserves_chg_xs` (~+4.8 bp/mo, NW t≈0.81) — below soft board (|t|≥1.5). Honesty alternate `low_reserves_xs` ~+3.6 bp/mo (NW t≈0.56).

**Board:** n=7 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. **Scaled primary promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` config **untouched**; re-verify gates PASS (Yahoo D1: 2024 0.42%/55%/74%; 2025 1.63%/73%/69%; 2026 2.30%/88%/87%; HO 1.52%/75%/81% — see `quest_locked_verify_reserves.md`).

### Honest read

IMF IFS external-buffer / reserve-adequacy is the right free-data structure after money-growth (§33), and is cleanly distinct from CA/TB flows and public debt. On Yahoo D1 G10 the primary high-reserves HML is essentially **flat-to-negative** (full-sample ≈ −3.8 bp/mo, NW t ≈ −0.58, %pos 48%) — nowhere near 1%/mo + 70% hit-rate. The change and low-reserves honesty legs are only marginally positive. Soft/hard NW boards empty. NZD/CHF unmapped; reserves/GDP not formed on free data. No go-live claim under `approximate_non_ftmo`.

**Next structure (if promote=0):** Done as §35 (house-price). FX IV/RR + news-sentiment still **blocked**. Next free scholarly candidate after §35: IG OAS (`BAMLC0A0CM`) risk-appetite — *not* another locked-sleeve cooler. Re-score all boards when FTMO MT5 CSVs arrive.

Artifacts: `reports/scholarly_fx_reserves_wave.md`, `scholarly_fx_reserves_*.csv`, `scholarly_fx_reserves_meta.json`.


## 35. OECD/BIS residential house-price FX wave results (2026-09-23 BST)

**Design (fixed priors, no HO tuning):** FRED BIS `Q*R628BIS` real residential property price index. Primary `high_hpi_xs`: XS long high relative HPI *momentum* (YoY log-diff) / short low (n=2/2) on full G10 USD+EUR(DE)+GBP+JPY+CAD+AUD+NZD+CHF. Companion: `low_hpi_xs`, `high_hpi_z_xs` (5y z of YoY), `hpi_chg_xs` (Δ12 of YoY), `us_hpi_stress_fx` / `us_hpi_haven_usd`, `hpi_ew`. PIT `pub_lag_months=4` + `signal_lag=1m` + 1d weight lag; costs 1.5 bps/side.

**What is new vs credit / debt / REER / money / reserves:** BIS credit (§32) is private credit/GDP; debt (§29) is public debt/GDP; REER (§31) is real effective FX; money-growth (§33) is monetary aggregates; reserves (§34) is external buffer. This wave is the **housing wealth / collateral** channel (Aoki–Proudman–Vlieghe; BIS residential property prices).

**Data notes:** Prefer `Q*R628BIS`. EUR=`QDER628BIS` (Germany; `QEUR628BIS` **404**). NZD/CHF **mapped** (`QNZR628BIS` / `QCHR628BIS`). Scores = **YoY log-diff** of index (not levels — units differ by country). FR/IT/ES alts documented coverage-only.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_hpi_xs | +0.036% | +0.55 | +0.60 | 52% | 12% | +0.16 |
| low_hpi_xs | −0.040% | −0.61 | −0.67 | 47% | 13% | −0.18 |
| high_hpi_z_xs | +0.028% | +0.45 | +0.46 | 52% | 12% | +0.16 |
| hpi_chg_xs | +0.108% | +1.60 | +1.37 | 54% | 12% | +0.40 |
| us_hpi_stress_fx | −0.015% | −0.45 | −0.65 | 13% | 28% | −0.11 |
| us_hpi_haven_usd | +0.013% | +0.41 | +0.59 | 16% | 25% | +0.10 |
| hpi_ew | +0.043% | +1.10 | +1.05 | 56% | 11% | +0.30 |

### Consistency windows (primary `high_hpi_xs`)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| year_2024 | −0.16% | 36% | 94% | PASS | no |
| year_2025 | +0.29% | 82% | 71% | PASS | no |
| year_2026 | +0.09% | 50% | 84% | PASS | no |
| holdout_365d | +0.11% | 50% | 70% | PASS | no |

### Risk sweep (IS → OOS)

Sweep **run**. Primary `high_hpi_xs` scale≈1.700 bind=static; scaled HO mean≈+0.19%/mo %pos 50% — clears: **NO**. Soft-best-ish leg is `hpi_chg_xs` (~+10.8 bp/mo, NW t≈1.37) — below soft board (|t|≥1.5).

**Board:** n=7 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. **Scaled primary promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` config **untouched**; re-verify gates PASS (Yahoo D1: 2024 0.42%/55%/74%; 2025 1.63%/73%/69%; 2026 2.30%/88%/87%; HO 1.52%/75%/81% — see `quest_locked_verify_house_price.md`).

### Honest read

BIS residential HPI / housing wealth-collateral is the right free-data structure after reserves (§34), and is cleanly distinct from private credit/GDP and public debt. On Yahoo D1 G10 the primary high-HPI-momentum HML is essentially **flat** (full-sample ≈ +3.6 bp/mo, NW t ≈ 0.60, %pos 52%) — nowhere near 1%/mo + 70% hit-rate. Acceleration (`hpi_chg_xs`) is the soft-best but |NW t|<1.5. Soft/hard NW boards empty. Full G10 mapped (NZD/CHF live). No go-live claim under `approximate_non_ftmo`.

**Next structure (if promote=0):** Done as §36 (IG OAS). FX IV/RR + news-sentiment still **blocked**. Next free scholarly candidate after §36: OECD **CLI** / PMI–ISM differentials — *not* another locked-sleeve cooler. Re-score all boards when FTMO MT5 CSVs arrive.

Artifacts: `reports/scholarly_fx_house_price_wave.md`, `scholarly_fx_house_price_*.csv`, `scholarly_fx_house_price_meta.json`.


## 36. ICE BofA IG OAS / credit risk-appetite FX wave results (2026-09-23 BST)

**Design (fixed priors, no HO tuning):** FRED ICE BofA `BAMLC0A0CM` (IG OAS) + companions `BAMLH0A0HYM2` (HY) / `BAMLC0A4CBBB` (BBB coverage). Primary `ig_oas_usd`: long USD when lagged z(IG OAS) ≥ 1.0 (binary usd_tilt=0.5). Companions: `hy_oas_usd`, `ig_oas_chg_usd`, `ig_oas_lvl_usd` (continuous intensity), `oas_stress_fx` (honesty wrong-signed risk-FX tilt), `carry_ig_oas_cool`, `carry_ig_oas_loose`, `oas_ew`. PIT daily `pub_lag=1d` + `signal_lag=1d`; z_window=252, min_periods=60, cool=0.35; costs 1.5 bps/side.

**What is new vs funding-liq / VIX / crash-skew / credit-gap / house-price:** Funding-liq §20 uses NFCI/TED/CPFF/Moody's `BAA10Y`; this wave uses ICE BofA *corporate* OAS (option-adjusted). Distinct from VIX/GPR/FX-RV/EPU, crash-skew §26, BIS credit-gap §32, house-price §35. USD-tilt / carry-conditioned (not country XS).

**Data notes:** Prefer FTMO MT5 D1 if present — absent → `approximate_non_ftmo` (Yahoo D1). Free FRED `fredgraph.csv` **truncates** ICE BofA OAS to ~3 calendar years (2023-09→2026-09) without an API key — documented limitation; z-scores use available window only.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| ig_oas_usd | −0.009% | −0.82 | −0.94 | 1% | 100% | −0.19 |
| ig_oas_lvl_usd | −0.010% | −0.84 | −0.99 | 3% | 93% | −0.21 |
| oas_stress_fx | +0.008% | +0.79 | +0.93 | 1% | 100% | +0.18 |
| hy_oas_usd | −0.012% | −0.92 | −1.06 | 1% | 100% | −0.26 |
| ig_oas_chg_usd | −0.017% | −1.05 | −0.96 | 6% | 62% | −0.42 |
| carry_ig_oas_cool | −0.003% | −0.04 | −0.05 | 52% | 10% | −0.04 |
| carry_ig_oas_loose | +0.024% | +1.18 | +0.91 | 11% | 26% | +0.27 |
| oas_ew | −0.013% | −0.98 | −1.00 | 7% | 58% | −0.33 |

### Consistency windows (primary `ig_oas_usd`)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| year_2024 | 0.00% | 0% | — | PASS | no |
| year_2025 | −0.14% | 9% | 100% | PASS | no |
| year_2026 | 0.00% | 0% | — | PASS | no |
| holdout_365d | 0.00% | 0% | — | PASS | no |

### Risk sweep (IS → OOS)

Sweep **run** (some non-primary legs had positive IS mean; primary IS mean **negative**). Primary `ig_oas_usd` scale≈4.038 bind=static; scaled HO mean≈0%/mo %pos 0% — clears: **NO**. Soft-best-ish leg is `carry_ig_oas_loose` (~+2.4 bp/mo, NW t≈0.91) — below soft board (|t|≥1.5). Honesty alternate `oas_stress_fx` ~+0.8 bp/mo (NW t≈0.93).

**Board:** n=8 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. **Scaled primary promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` config **untouched**; re-verify gates PASS (Yahoo D1: 2024 0.42%/55%/74%; 2025 1.63%/73%/69%; 2026 2.30%/88%/87%; HO 1.52%/75%/81% — see `quest_locked_verify_ig_oas.md`).

### Honest read

ICE BofA IG OAS is the right free-data *corporate* credit risk-appetite structure after house-price (§35), and is cleanly distinct from Moody's BAA10Y / NFCI funding-liq (§20). On Yahoo D1 G10 with the public ~3y OAS window the primary USD-haven tilt is essentially **flat-to-slightly-negative** (full-sample ≈ −0.9 bp/mo, NW t ≈ −0.94, sparse activity %pos ~1%) — nowhere near 1%/mo + 70% hit-rate. Carry×OAS-loose and the wrong-signed honesty leg are only marginally positive. Soft/hard NW boards empty. Public CSV ICE truncation is a material sample-length blocker for longer-history inference. No go-live claim under `approximate_non_ftmo`.

**Next structure (if promote=0):** Done as §37 (OECD CLI). FX IV/RR + news-sentiment still **blocked**. Next free scholarly candidate after §37: manufacturing **PMI**/ISM differentials or consumer-confidence differentials — *not* another locked-sleeve cooler. Re-score all boards when FTMO MT5 CSVs arrive (and optionally re-score OAS with a FRED API key for longer ICE history).

Artifacts: `reports/scholarly_fx_ig_oas_wave.md`, `scholarly_fx_ig_oas_*.csv`, `scholarly_fx_ig_oas_meta.json`.


## 37. OECD CLI leading-indicator FX wave results (2026-09-23 BST)

**Design (fixed priors, no HO tuning):** FRED OECD MEI amplitude-adjusted CLI `*LOLITOAASTSAM`. Primary `high_cli_xs`: long high relative CLI YoY growth (`CLI_t − CLI_{t−12}`) / short low. Companions: `low_cli_xs` (honesty), `high_cli_z_xs`, `cli_chg_xs` (Δ12 of YoY), `us_cli_weak_fx` / `us_cli_haven_usd` (US CLI YoY z ≤ −1.0), `cli_ew`. PIT monthly `pub_lag=2m` + 1d weight lag (no extra month signal lag); z_window=60, min_periods=24, usd_tilt=0.5; costs 1.5 bps/side.

**What is new vs macro-diff / house-price / money-growth / IG OAS:** Macro-diff uses coincident CPI/IP/UR; this wave uses *leading* OECD CLI. Distinct from housing HPI §35, broad-money §33, corporate OAS §36. Country XS on YoY CLI growth (not USD-tilt-only).

**Data notes:** Prefer FTMO MT5 D1 if present — absent → `approximate_non_ftmo` (Yahoo D1). EUR = Germany `DEULOLITOAASTSAM` (`EA19…` ends 2022-11; `EZ19…` 404). NZD/CHF **unmapped** (NZL ends 2019-11; CHE ends 2022-11).

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_cli_xs | +0.045% | +0.71 | +0.72 | 54% | 11% | +0.19 |
| low_cli_xs | −0.051% | −0.81 | −0.83 | 44% | — | — |
| high_cli_z_xs | −0.007% | −0.11 | −0.11 | 51% | — | — |
| cli_chg_xs | −0.033% | −0.50 | −0.49 | 47% | — | — |
| us_cli_weak_fx | +0.005% | +0.23 | +0.23 | 9% | — | — |
| us_cli_haven_usd | −0.005% | −0.26 | −0.26 | 9% | — | — |
| cli_ew | +0.005% | +0.15 | +0.15 | 47% | — | — |

### Consistency windows (primary `high_cli_xs`)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| year_2024 | +0.20% | 55% | 86% | PASS | no |
| year_2025 | +0.08% | 55% | 80% | PASS | no |
| year_2026 | +0.05% | 50% | 96% | PASS | no |
| holdout_365d | +0.16% | 58% | 78% | PASS | no |

### Risk sweep (IS → OOS)

Sweep **run**. Primary `high_cli_xs` scale≈1.505 bind=static; scaled HO mean≈+0.24%/mo %pos 58% — clears: **NO**. Soft-best-ish leg is primary itself (~+4.5 bp/mo, NW t≈0.72) — below soft board (|t|≥1.5).

**Board:** n=7 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. **Scaled primary promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` config **untouched**; re-verify gates PASS (Yahoo D1: 2024 0.42%/55%/74%; 2025 1.63%/73%/69%; 2026 2.30%/88%/87%; HO 1.52%/75%/81% — see `quest_locked_verify_oecd_cli.md`).

### Honest read

OECD CLI is the right free-data *leading*-activity structure after IG OAS (§36), and is cleanly distinct from coincident macro-diff CPI/IP/UR. On Yahoo D1 G10 the primary high-CLI-YoY HML is essentially **flat-to-mildly-positive** (full-sample ≈ +4.5 bp/mo, NW t ≈ 0.72, %pos 54%) — nowhere near 1%/mo + 70% hit-rate. Soft/hard NW boards empty. NZD/CHF coverage gaps documented. No go-live claim under `approximate_non_ftmo`.

**Next structure (if promote=0):** Done as §38 (OECD CCI). FX IV/RR + news-sentiment still **blocked**. Next free scholarly candidate: manufacturing **PMI**/ISM differentials (or OECD BCI) — *not* another locked-sleeve cooler. Re-score all boards when FTMO MT5 CSVs arrive.

Artifacts: `reports/scholarly_fx_oecd_cli_wave.md`, `scholarly_fx_oecd_cli_*.csv`, `scholarly_fx_oecd_cli_meta.json`.

## 38. OECD CCI / consumer-confidence FX wave results (2026-09-23 BST)

**Design (fixed priors, no HO tuning):** FRED OECD MEI consumer-confidence balances `CSCICP02*M460S` (+ US `USACSCICP02STSAM`; `CSCICP02USM460S` 404). Amplitude-adjusted `CSCICP03*M665S` probed but **stale** (~ends 2024-01) — not primary. Primary `high_cci_xs`: long high relative CCI YoY change (`CCI_t − CCI_{t−12}`) / short low. Companions: `low_cci_xs` (honesty), `high_cci_z_xs`, `cci_chg_xs` (Δ12 of YoY), `us_cci_weak_fx` / `us_cci_haven_usd` (US CCI YoY z ≤ −1.0), `cci_ew`. PIT monthly `pub_lag=2m` + 1d weight lag (no extra month signal lag); z_window=60, min_periods=24, usd_tilt=0.5; costs 1.5 bps/side.

**What is new vs CLI / macro-diff / IG OAS:** OECD CLI §37 is *leading* activity; this wave is *consumer sentiment* (Ludvigson-style). Distinct from coincident macro-diff CPI/IP/UR, house-price §35, money-growth §33, corporate OAS §36. Country XS on YoY CCI change among foreign CSCICP02 balances (USD tilt uses USACSCICP02STSAM).

**Data notes:** Prefer FTMO MT5 D1 if present — absent → `approximate_non_ftmo` (Yahoo D1). EUR = EZ `CSCICP02EZM460S` (live). CAD/NZD/CHF **unmapped** (`CSCICP02` 404; `CSCICP03` amplitude stale). Mapped: USD, EUR, GBP, JPY, AUD.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_cci_xs | +0.016% | +0.24 | +0.26 | 53% | 11% | +0.03 |
| low_cci_xs | −0.029% | −0.44 | −0.47 | 47% | 12% | −0.07 |
| high_cci_z_xs | −0.034% | −0.53 | −0.54 | 46% | 13% | −0.15 |
| cci_chg_xs | −0.027% | −0.43 | −0.50 | 52% | 11% | −0.13 |
| us_cci_weak_fx | +0.002% | +0.06 | +0.06 | 14% | 33% | +0.02 |
| us_cci_haven_usd | −0.004% | −0.12 | −0.12 | 11% | 34% | −0.03 |
| cci_ew | −0.003% | −0.07 | −0.08 | 52% | 12% | −0.05 |

### Consistency windows (primary `high_cci_xs`)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| year_2024 | +0.55% | 64% | 63% | PASS | no |
| year_2025 | +0.21% | 64% | 68% | PASS | no |
| year_2026 | +0.09% | 50% | 95% | PASS | no |
| holdout_365d | +0.22% | 58% | 54% | PASS | no |

### Risk sweep (IS → OOS)

Sweep **run**. Primary `high_cci_xs` scale≈0.864 bind=static; scaled HO mean≈+0.19%/mo %pos 58% — clears: **NO**. Soft-best-ish leg is primary itself (~+1.6 bp/mo, NW t≈0.26) — below soft board (|t|≥1.5).

**Board:** n=7 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. **Scaled primary promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` config **untouched**; re-verify gates PASS (Yahoo D1: 2024 0.42%/55%/74%; 2025 1.63%/73%/69%; 2026 2.30%/88%/87%; HO 1.52%/75%/81% — see `quest_locked_verify_oecd_cci.md`).

### Honest read

OECD CCI is the right free-data *consumer-sentiment* structure after CLI (§37), and is cleanly distinct from leading CLI and coincident macro-diff. Live path uses `CSCICP02*M460S` balances (amplitude `CSCICP03*M665S` too stale for 2025/2026). On Yahoo D1 G10 the primary high-CCI-YoY HML is essentially **flat** (full-sample ≈ +1.6 bp/mo, NW t ≈ 0.26, %pos 53%) — nowhere near 1%/mo + 70% hit-rate. Soft/hard NW boards empty. CAD/NZD/CHF coverage gaps documented. No go-live claim under `approximate_non_ftmo`.

**Next structure (if promote=0):** Done as §39 (OECD BCI / manufacturing-confidence). FX IV/RR + news-sentiment still **blocked**. Re-score all boards when FTMO MT5 CSVs arrive.

Artifacts: `reports/scholarly_fx_oecd_cci_wave.md`, `scholarly_fx_oecd_cci_*.csv`, `scholarly_fx_oecd_cci_meta.json`.

## 39. OECD BCI / manufacturing-confidence FX wave results (2026-09-23 BST)

**Design (fixed priors, no HO tuning):** FRED OECD MEI business-confidence balances `BSCICP02*M460S` (ISM/NAPM manufacturing PMI **404** on FRED — BCI is the free fallback). Amplitude-adjusted `BSCICP03*M665S` probed but **stale** (~ends 2023-11..2024-01) — not primary. Primary `high_bci_xs`: long high relative BCI YoY change (`BCI_t − BCI_{t−12}`) / short low. Companions: `low_bci_xs` (honesty), `high_bci_z_xs`, `bci_chg_xs` (Δ12 of YoY), `us_bci_weak_fx` / `us_bci_haven_usd` (US BCI YoY z ≤ −1.0), `bci_ew`. PIT monthly `pub_lag=2m` + 1d weight lag (no extra month signal lag); z_window=60, min_periods=24, usd_tilt=0.5; costs 1.5 bps/side. **`n_long=n_short=1` a priori** (foreign panel only EUR/GBP/CHF).

**What is new vs CLI / CCI / macro-diff / IG OAS:** OECD CLI §37 is *leading* activity; CCI §38 is *consumer* sentiment; this wave is *business / manufacturing* confidence (PMI-style). Distinct from coincident macro-diff CPI/IP/UR, house-price §35, money-growth §33, corporate OAS §36.

**Data notes:** Prefer FTMO MT5 D1 if present — absent → `approximate_non_ftmo` (Yahoo D1). EUR = EZ `BSCICP02EZM460S` (live through ~2026-01). CHF = `BSCICP02CHM460S` **live** (unlike CCI). AUD/CAD/NZD/JPY **unmapped** (`BSCICP02` 404; `BSCICP03` amplitude stale). Mapped: USD, EUR, GBP, CHF.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_bci_xs | +0.081% | +1.02 | +0.98 | 57% | 13% | +0.24 |
| low_bci_xs | −0.105% | −1.35 | −1.29 | 42% | 17% | −0.29 |
| high_bci_z_xs | +0.126% | +1.71 | +1.90 | 56% | 15% | +0.37 |
| bci_chg_xs | +0.148% | +2.02 | +2.34 | 59% | 12% | +0.48 |
| us_bci_weak_fx | −0.008% | −0.28 | −0.32 | 9% | 43% | −0.06 |
| us_bci_haven_usd | +0.007% | +0.23 | +0.27 | 11% | 33% | +0.06 |
| bci_ew | +0.074% | +1.61 | +1.65 | 58% | 13% | +0.38 |

### Consistency windows (primary `high_bci_xs`)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| year_2024 | +0.25% | 73% | 71% | PASS | no |
| year_2025 | +0.44% | 82% | 61% | PASS | no |
| year_2026 | +0.48% | 88% | 64% | PASS | no |
| holdout_365d | +0.30% | 83% | 56% | PASS | no |

### Risk sweep (IS → OOS)

Sweep **run**. Primary `high_bci_xs` scale≈0.593 bind=daily; scaled HO mean≈+0.18%/mo %pos 83% — clears: **NO**. Soft-best leg `bci_chg_xs` (~+14.8 bp/mo, NW t≈+2.34, hard board) — still far below 1%/mo consistency bar on year/HO means (~0.2–0.5%/mo).

**Board:** n=7 soft=3 hard=1 promote=0. **Unscaled promote:** **NO**. **Scaled primary promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` config **untouched**; re-verify gates PASS (Yahoo D1: 2024 0.42%/55%/74%; 2025 1.63%/73%/69%; 2026 2.30%/88%/87%; HO 1.52%/75%/81% — see `quest_locked_verify_oecd_bci.md`).

### Honest read

OECD BCI is the right free-data *manufacturing / business-confidence* structure after CLI (§37) and CCI (§38), and is cleanly distinct from both. Live path uses `BSCICP02*M460S` balances (amplitude `BSCICP03*M665S` too stale for 2025/2026; NAPM/ISM 404). On Yahoo D1 G10 the primary high-BCI-YoY HML is mildly positive (full-sample ≈ +8.1 bp/mo, NW t ≈ 0.98, %pos 57%) with strong recent-year %pos (73–88%) but means still ≲0.5%/mo — nowhere near 1%/mo. Soft board has 3 legs (`bci_chg_xs` hard NW t≈2.34; `high_bci_z_xs` / `bci_ew` soft); none clear promote. Thin foreign panel (EUR/GBP/CHF only) documented via `n_long=n_short=1`. No go-live claim under `approximate_non_ftmo`.

**Next structure (if promote=0):** Done as §40 (World Uncertainty Index / Ahir–Bloom–Furceri). FX IV/RR + news-sentiment still **blocked**. Re-score all boards when FTMO MT5 CSVs arrive.

Artifacts: `reports/scholarly_fx_oecd_bci_wave.md`, `scholarly_fx_oecd_bci_*.csv`, `scholarly_fx_oecd_bci_meta.json`.

## 40. World Uncertainty Index (WUI) FX wave results (2026-09-23 BST)

**Design (fixed priors, no HO tuning):** FRED Ahir–Bloom–Furceri country WUI (quarterly → monthly after `pub_lag_months=4`). Primary `low_wui_xs`: long low relative WUI **level** / short high (`n_long=n_short=2` full G10 foreign). Companions: `high_wui_xs` (honesty / fragile-uncertainty premium), `low_wui_z_xs` (60m z of levels), `wui_chg_xs` (−Δ12m ≈ −Δ4q of level), `us_wui_stress_fx` / `us_wui_haven_usd` (US WUI level z ≥ +1.0), `wui_ew` (EW of primary + chg + stress). PIT quarterly pub_lag=4m + 1d weight lag (no extra month signal lag); z_window=60, min_periods=24, usd_tilt=0.5; costs 1.5 bps/side. **Score basis = levels** (WUI is already an index — not YoY-first).

**What is new vs EPU/TPU / GPR / OECD CLI·CCI·BCI:** EPU/TPU are Baker–Bloom–Davis newspaper counts; GPR/AI-GPR are Caldara–Iacoviello geopolitics; OECD CLI/CCI/BCI are activity/sentiment balances. This wave is *EIU-text country uncertainty* (WUI) cross-section + US WUI haven/stress tilts.

**Data notes:** Prefer FTMO MT5 D1 if present — absent → `approximate_non_ftmo` (Yahoo D1). EUR = `WUIDEU` Germany proxy (`WUIFRA`/`WUIITA`/`WUIESP` alts; no clean EA aggregate). Full G10 mapped and live through PIT ~2026-08 (raw ~2026-04 + 4m lag). Retail-sales OECD SARTMISMEI probed separately and **stale** (~ends 2023–early 2024) — unsuitable; WUI chosen instead.

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| low_wui_xs | −0.071% | −1.11 | −1.07 | 49% | 12% | −0.27 |
| high_wui_xs | +0.061% | +0.96 | +0.93 | 52% | 13% | +0.24 |
| low_wui_z_xs | +0.073% | +1.17 | +1.23 | 57% | 9% | +0.25 |
| wui_chg_xs | +0.023% | +0.37 | +0.39 | 56% | 8% | +0.05 |
| us_wui_stress_fx | +0.018% | +0.54 | +0.67 | 13% | 27% | +0.14 |
| us_wui_haven_usd | −0.019% | −0.57 | −0.71 | 14% | 29% | −0.14 |
| wui_ew | −0.010% | −0.25 | −0.25 | 51% | 11% | −0.09 |

### Consistency windows (primary `low_wui_xs`)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| year_2024 | +0.00% | 64% | 71% | PASS | no |
| year_2025 | −0.24% | 18% | 100% | PASS | no |
| year_2026 | +0.21% | 62% | 68% | PASS | no |
| holdout_365d | +0.25% | 58% | 60% | PASS | no |

### Risk sweep (IS → OOS)

Sweep **run** (positive IS mean on some companions). Primary `low_wui_xs` scale≈0.608 bind=static; IS mean_mo negative (≈ −5.5 bp); scaled HO mean≈+0.15%/mo %pos 58% — clears: **NO**. Soft-best-ish leg is `low_wui_z_xs` (~+7.3 bp/mo, NW t≈+1.23) — below soft board (|t|≥1.5). Honesty `high_wui_xs` (~+6.1 bp/mo) also below soft.

**Board:** n=7 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. **Scaled primary promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` config **untouched**; re-verify gates PASS (Yahoo D1: 2024 0.42%/55%/74%; 2025 1.63%/73%/69%; 2026 2.30%/88%/87%; HO 1.52%/75%/81% — see `quest_locked_verify_wui.md`).

### Honest read

WUI is the right free-data *country uncertainty* structure after OECD BCI (§39), and is cleanly distinct from EPU/TPU and GPR. Full G10 live through ~2026-04 (PIT ~2026-08). On Yahoo D1 G10 the primary low-WUI HML is mildly **negative** (full-sample ≈ −7.1 bp/mo, NW t ≈ −1.07, %pos 49%) — nowhere near 1%/mo + 70% hit-rate. Soft/hard NW boards empty for positive means. Level-based scoring (a priori — WUI is an index) does not rescue the cross-section. No go-live claim under `approximate_non_ftmo`.

**Next structure (if promote=0):** Done as §41 (OECD MEI employment-growth LFEMTTTT). FX IV/RR + news-sentiment still **blocked**. Re-score all boards when FTMO MT5 CSVs arrive.

Artifacts: `reports/scholarly_fx_wui_wave.md`, `scholarly_fx_wui_*.csv`, `scholarly_fx_wui_meta.json`.

## 41. Employment-growth (LFEMTTTT) FX wave results (2026-09-23 BST)

**Design (fixed priors, no HO tuning):** FRED OECD MEI `LFEMTTTT*` employment persons (monthly `*M647S` + quarterly `*Q647S` → monthly after `pub_lag_months=3`). Primary `high_emp_xs`: long high relative emp **YoY %** / short low (`n_long=n_short=2` full G10 foreign). Companions: `low_emp_xs` (honesty / labour-stress debtor), `high_emp_z_xs` (60m z of YoY), `emp_chg_xs` (Δ12 of YoY), `us_emp_stress_fx` / `us_emp_haven_usd` (US emp YoY z ≤ −1.0), `emp_ew` (EW of primary + chg + stress). PIT pub_lag=3m (labour a priori — slower than BCI/money's 2) + 1d weight lag (no extra month signal lag); z_window=60, min_periods=24, usd_tilt=0.5; costs 1.5 bps/side. **Score basis = YoY % of persons levels** (`*M657S`/`*Q657S` YoY stale/404 — not primary).

**What is new vs macro-diff UR / OECD CLI·CCI·BCI / WUI:** Macro-diff uses *unemployment rates*; CLI/CCI/BCI are activity/sentiment balances; WUI is EIU-text uncertainty. This wave is *employment persons growth* (LFEMTTTT) cross-section + US labour stress/haven tilts. Retail-sales SARTMISMEI confirmed stale — not re-run.

**Data notes:** Prefer FTMO MT5 D1 if present — absent → `approximate_non_ftmo` (Yahoo D1). EUR = `LFEMTTTTDEQ647S` Germany proxy (`LFEMTTTTEZQ647S` EZ ends 2022-10; FR/IT/ES Q647S alts). Full G10 mapped (USD/JPY/CAD/AUD monthly; EUR/GBP/NZD/CHF quarterly).

### Full-sample (unscaled)

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_emp_xs | −0.072% | −1.14 | −1.24 | 48% | 13% | −0.29 |
| low_emp_xs | +0.062% | +0.98 | +1.06 | 52% | 10% | +0.26 |
| high_emp_z_xs | −0.020% | −0.32 | −0.35 | 49% | 12% | −0.08 |
| emp_chg_xs | −0.034% | −0.52 | −0.55 | 46% | 11% | −0.13 |
| us_emp_stress_fx | +0.027% | +1.21 | +1.36 | 7% | 28% | +0.28 |
| us_emp_haven_usd | −0.027% | −1.22 | −1.37 | 6% | 30% | −0.28 |
| emp_ew | −0.026% | −0.75 | −0.79 | 43% | 12% | −0.20 |

### Consistency windows (primary `high_emp_xs`)

| Window | mean_mo | %pos | top3 | gates | 1% bar |
|--------|--------:|-----:|-----:|:-----:|:------:|
| year_2024 | +0.01% | 55% | 80% | PASS | no |
| year_2025 | −0.10% | 36% | 89% | PASS | no |
| year_2026 | +0.14% | 75% | 70% | PASS | no |
| holdout_365d | +0.13% | 67% | 58% | PASS | no |

### Risk sweep (IS → OOS)

Sweep **run** (positive IS mean on honesty `low_emp_xs` / US stress). Primary `high_emp_xs` scale≈0.521 bind=static; IS mean_mo negative (≈ −4.6 bp); scaled HO mean≈+0.07%/mo %pos 67% — clears: **NO**. Soft-best-ish leg is honesty `low_emp_xs` (~+6.2 bp/mo, NW t≈+1.06) — below soft board (|t|≥1.5). `us_emp_stress_fx` NW t≈+1.36 also below soft.

**Board:** n=7 soft=0 hard=0 promote=0. **Unscaled promote:** **NO**. **Scaled primary promote:** **NO**. Locked `fx4plus_gbpcad_d1_voltarget_0025` config **untouched**; re-verify gates PASS (Yahoo D1: 2024 0.42%/55%/74%; 2025 1.63%/73%/69%; 2026 2.30%/88%/87%; HO 1.52%/75%/81% — see `quest_locked_verify_employment.md`).

### Honest read

Employment-growth LFEMTTTT is the right free-data *labour beyond UR* structure after WUI (§40), and is cleanly distinct from unemployment-rate macro-diff and OECD sentiment. Full G10 mapped via persons levels (YoY measure series stale/404). On Yahoo D1 G10 the primary high-emp-YoY HML is mildly **negative** (full-sample ≈ −7.2 bp/mo, NW t ≈ −1.24, %pos 48%) — nowhere near 1%/mo + 70% hit-rate. Soft/hard NW boards empty for positive means. Honesty low-emp alternate is weakly positive but below soft |t|≥1.5. No go-live claim under `approximate_non_ftmo`.

**Next structure (if promote=0):** FX IV/RR + news-sentiment still **blocked**. Employment §41 done. Next: re-score all boards when **FTMO MT5 CSVs** arrive — *not* another locked-sleeve cooler. (Other free scholarly structures largely exhausted on this free-data path.)

Artifacts: `reports/scholarly_fx_employment_wave.md`, `scholarly_fx_employment_*.csv`, `scholarly_fx_employment_meta.json`.


## 42. Industrial-production / OECD MEI PRINTO01 FX wave results (2026-09-23 BST)

**Design (fixed priors, no HO tuning):** FRED OECD MEI `{ISO3}PRINTO01GYSAM` industry YoY (monthly) + AUD/NZD/CHF `{ISO3}PRMNTO01GYSAQ` manufacturing YoY (quarterly → monthly after `pub_lag_months=2`). Primary `high_ip_xs`: long high relative IP YoY / short low (`n_long=n_short=2` full G10 foreign). Companions: `low_ip_xs` (honesty / activity-stress), `high_ip_z_xs` (60m z of YoY), `ip_chg_xs` (Δ12 of YoY), `us_ip_stress_fx` / `us_ip_haven_usd` (US IP YoY z ≤ −1.0), `ip_ew` (EW of primary + chg + stress). PIT pub_lag=2m (IP a priori) + 1d weight lag (no extra month signal lag); z_window=60, min_periods=24, usd_tilt=0.5; costs 1.5 bps/side. **Score basis = YoY growth as reported** (GYSAM/GYSAQ — not derived from levels). EUR = `FRAPRINTO01GYSAM` France proxy (EA19/DEU GYSAM stale).

**What is new vs macro-diff IP / OECD CLI·CCI·BCI / employment:** Macro-diff uses an EW CPI/IP/UR *blend*; CLI/CCI/BCI are leading/sentiment balances; employment §41 is LFEMTTTT labour persons growth. This wave is *coincident industrial-production YoY* cross-section (PRINTO01) + US IP stress/haven tilts.

**Data tag:** `approximate_non_ftmo` (no FTMO CSVs under `data/ftmo/`). Full G10 mapped.

### Full-sample factor summary

| Factor | mean_mo | t OLS | t NW | %pos | top3 | Sharpe |
|--------|--------:|------:|-----:|-----:|-----:|-------:|
| high_ip_xs | +0.101% | +1.81 | +1.88 | 54% | 9% | +0.37 |
| low_ip_xs | −0.122% | −2.16 | −2.23 | 44% | — | −0.44 |
| high_ip_z_xs | +0.078% | +1.33 | +1.35 | 54% | — | +0.29 |
| ip_chg_xs | +0.038% | +0.58 | +0.60 | 53% | — | +0.14 |
| us_ip_stress_fx | +0.005% | +0.18 | +0.18 | 11% | — | +0.04 |
| us_ip_haven_usd | −0.006% | −0.20 | −0.20 | 8% | — | −0.04 |
| ip_ew | +0.048% | +1.28 | +1.32 | 57% | — | +0.27 |

### Consistency windows (primary `high_ip_xs`)

| Window | mean_mo | %pos | gates | 1% bar |
|--------|--------:|-----:|:-----:|:------:|
| 2024 | +0.05% | 64% | PASS | no |
| 2025 | +0.16% | 64% | PASS | no |
| 2026 | +0.11% | 50% | PASS | no |
| holdout_365d | +0.16% | 50% | PASS | no |

**Board:** n=7 soft_nw_pos=**1** (`high_ip_xs` NW t≈+1.88) hard_nw_pos=**0** promote=**0**.

Sweep **run**. Primary `high_ip_xs` scale≈**1.975** bind=**static**; scaled HO mean≈+0.31%/mo %pos 50% — clears: **NO**. Soft-best = primary `high_ip_xs` (~+10.1 bp/mo, NW t≈+1.88).

Locked `fx4plus_gbpcad_d1_voltarget_0025` **untouched PASS**.

**Next structure (if promote=0):** Building permits / housing starts (distinct from house-price §35), NY Fed ACM term premium → USD risk-appetite FX, or FTMO CSV re-score when exports arrive. FX IV/RR + news-sentiment still **blocked**.

