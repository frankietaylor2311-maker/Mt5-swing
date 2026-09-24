# Macro / geopolitics / probability research line

**Steering (2026-09-17):** Stop overlay/indicator number-hunting on locked `fx4plus`.  
New line = math/statistics + economics/news/war factors grounded in scholar outcomes.

**Data label:** `approximate_non_ftmo` (Yahoo FX ≠ FTMO). Never golive without `data/ftmo/` MT5 exports.

---

## Idea vs tuning (this wave)

| Item | Class | Description |
|------|-------|-------------|
| FRED rate-diff carry (`fred_carry`) | **idea** | Replace price-trend `carry_proxy` with lagged policy-rate differentials |
| VIX risk gate | **idea** | Menkhoff-style global vol risk-off (fixed z≥1 → scale 0.35) |
| GPR risk gate | **idea** | Caldara–Iacoviello / Liu–Zhang geopolitical risk-off (fixed z≥1.5) |
| VIX+GPR stack | **idea** | Sequential literature gates — **not** a cool-knob grid |
| Overlay cool/throttle grids on locked equity | **rejected** | Prior keepalives; Frankie called out as number hunt |
| Mid-wave technical strats (mtf / range-expansion / session inventory) | **aborted** | Unregistered; not grid-tuned this wave |

---

## Scholar anchors (documented outcomes)

1. **Lustig, Roussanov & Verdelhan (2011), RFS** — Common risk factors in currency markets; carry sorted by interest differentials earns a premium.
2. **Lustig & Verdelhan (2007), AER** — Foreign currency risk premia and consumption growth risk.
3. **Menkhoff, Sarno, Schmeling & Schrimpf (2012), Journal of Finance** — Carry trades and *global FX volatility*: high-vol innovations → carry crashes / risk-off. We proxy global vol with **VIX** (public daily) when realized FX-vol panels are unavailable.
4. **Menkhoff et al. (2012), JFE** — Currency momentum distinct from carry (price momentum; not this wave’s primary test).
5. **Dahlquist & Hasseltoft (2020), JFE** — *Economic momentum*: long currencies with strong past macro trends (IP, retail sales, unemployment, CPI, PPI). Sharpe ~0.70 historically; subsumes carry alpha. **Needs multi-country macro panel** (see required datasets).
6. **Caldara & Iacoviello (2022), AER** — News-based **Geopolitical Risk (GPR)** index; spikes around wars/crises; foreshadows weaker investment / risk assets.
7. **Liu & Zhang (2024), JBF** — *Geopolitical risk and currency returns*: long high country-GPR / short low country-GPR ≈ **+5.72%** annual excess historically; not explained by standard FX factors. Aggregate GPR risk-off is a **weaker but testable** proxy until country GPR is wired.
8. **Du, Tepper & Verdelhan (2018), JF** — CIP deviations / intermediary constraints (funding stress).
8b. **Du, Keerati & Schreger (2025)** — *Decoupling Dollar and Treasury Privilege* — public gov-bond CIP + U.S. Treasury Premium panel (`cip_dataset_v4`); boarded as scholarly FX wave **§67**; CIP-conditioned Lustig–Verdelhan carry boarded **§68**; soft-stack CIP enrichment boarded **§69**.

---

## Required datasets

| Dataset | Role | Status in repo |
|---------|------|----------------|
| FRED short rates (US, EUR, GBP, JPY, AUD, CAD, CHF, NZD) | Carry sign (Lustig–Verdelhan) | **Present** `data/macro/fred_IRST*.csv`, `fred_DFF`, `fred_ECBDFR`, `fred_IUDSOIA` |
| VIX (`^VIX` / FRED VIXCLS) | Global vol risk (Menkhoff) | **Present** `data/macro/vix_yahoo.csv` |
| Caldara–Iacoviello GPR daily/monthly | Geopolitical risk | **Present** `gpr_daily.csv`, `gpr_monthly.csv` (+ xls) |
| AI-GPR daily (Iacoviello–Tong) | Alt news-based GPR | **Present** `ai_gpr_daily.csv` |
| Country-specific GPR | Liu–Zhang long/short GPR FX | **Partial** — monthly export has country columns; not yet wired to legs |
| Multi-country CPI / PPI / IP / retail / unemployment / wages / LFP / employment-rate / real GDP / construction / passenger-car regs / merchandise export-value / merchandise import-value / household credit / GFCF investment / private final consumption / soft-signal EW stack | Dahlquist economic momentum + Lustig–Verdelhan consumption-growth risk + Mian–Sufi / Borio household credit | **Mostly boarded** — IP §42 / retail §46 / employment §41 (persons) / PPI §51 / CPI §52 / **UR §54** / **wage/earnings §55** / **LFP/activity §56** / **employment-rate §57** / **real GDP growth §58** / **construction-production §59** / **passenger-car regs §60** / **merchandise export-value §61** / **merchandise import-value §62** / **household-credit §63** / **GFCF investment §64** / **private final consumption §65** / **soft-signal EW recombination §66** (SOFT_LEGS=bci_chg/high_ip/high_ppi/low_gdp/low_cars; primary soft_ew5; board soft=7 hard=7 promote=0) / **Du–Schreger CIP/basis §67** (cip_govt 5y; primary low_cip_xs; soft=2 hard=0 promote=0) / **CIP-conditioned carry §68** (carry×CIP stress; primary carry_low_cip_stress; soft=2 hard=0 promote=0) / **soft-stack CIP enrichment §69** (SOFT_LEGS_CIP=macro5+ust_premium_haven_usd+cip_ew; primary soft_ew7_cip; board soft=7 hard=7 promote=0). PPI STALE ~2022-12; CPI JPY STALE ~2021-06. Building-permits §43 distinct (leading flow). Capital-sleeve mix §53 boarded (promote=0). |
| News sentiment (FX-specific) | Event studies / war headlines beyond GPR | **Missing** — GDELT / RavenPack / Factiva (licensed) |
| Cross-currency basis / CIP | Intermediary stress / Treasury premium | **Present / boarded §67** — Du–Keerati–Schreger `cip_dataset_v4` gov CIP (`cip_govt` bps, tenor=5y a priori) + UST premium; primary `low_cip_xs`; soft=2 hard=0 promote=0. **CIP-conditioned carry boarded §68** — Lustig–Verdelhan carry × CIP-stress gate/cool; primary `carry_low_cip_stress`; soft=2 hard=0 promote=0. **Soft-stack CIP enrichment boarded §69** — macro soft5 + CIP soft2 EW; primary `soft_ew7_cip`; soft=7 hard=7 promote=0 |
| FTMO MT5 OHLC (H1/H4/D1) | Golive / true costs | **Missing** `data/ftmo/` empty |
| Yahoo FX OHLC | Approximate backtests | **Present** `data/history/` |

---

## Testable pieces implemented this wave

- `mt5_swing.features.macro_factors` — causal loaders (month/bar lags)
- `mt5_swing.strategies.fred_carry` — rate-diff carry strategy
- `mt5_swing.portfolio.macro_regimes` — VIX / GPR gates (hi≤1)
- `scripts/quest_macro_factor_literature_wf.py` — IS-only selection, HO confirm, FTMO gates

Promote only if ≥1% mean monthly **and** ≥70% positive months on each of 2024, 2025, 2026, holdout (no look-ahead / no OOS tune / no RF hike).
