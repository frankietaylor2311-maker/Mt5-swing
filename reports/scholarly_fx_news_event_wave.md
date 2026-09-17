# Scholarly FX news/event wave
**Date:** 2026-09-17 BST
**Intensity source:** `gpr_spike_proxy`
**Note:** Proxy: Caldara–Iacoviello daily GPR (newspaper article share). Not live NLP of Reuters/Yahoo headlines; multi-year coverage. Limitation: intensity ≠ signed sentiment / entity NLP.
**Data:** `approximate_non_ftmo` Yahoo D1 for ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']. Costs: 1.5 bps/side on turnover. `signal_lag=1`, hold=5d, top-decile thinned events.

## Feed attempts

| source | available | n_obs | note |
|---|---|---:|---|
| gdelt_doc | True | 82 | cached CSV gdelt_conflict_timeline.csv (DOC TimelineVol; ~recent window only) |
| gdelt_doc | True | 82 | cached CSV gdelt_conflict_timeline.csv (DOC TimelineVol; ~recent window only) — only n=82 days; too short for multi-year study → GPR proxy |
| yahoo_rss | True | 13 | RSS headline counts from https://finance.yahoo.com/rss/headline?s=EURUSD=X (recent feed items only) |
| reuters_rss | False | 0 | RSS fetch failed (HTTPError: HTTP Error 404: Not Found) |
| gpr_spike_proxy | True | 15232 | Proxy: Caldara–Iacoviello daily GPR (newspaper article share). Not live NLP of Reuters/Yahoo headlines; multi-year coverage. Limitation: intensity ≠ signed sent |

## Event vs control (USD strengthens = +)

Events: **206** (≥90%ile lagged intensity / top-decile, ≥5d apart). Controls: **206** random non-event days.

| series | h | event_cum | control_cum | diff | t |
|---|---:|---:|---:|---:|---:|
| EUR | 0 | -0.031% | 0.026% | -0.058% | -1.15 |
| EUR | 5 | 0.045% | 0.066% | -0.022% | -0.18 |
| EUR | 10 | -0.002% | 0.065% | -0.067% | -0.43 |
| GBP | 0 | 0.018% | 0.005% | 0.013% | 0.23 |
| GBP | 5 | 0.092% | 0.013% | 0.079% | 0.57 |
| GBP | 10 | 0.026% | 0.143% | -0.117% | -0.62 |
| JPY | 0 | 0.024% | -0.013% | 0.037% | 0.81 |
| JPY | 5 | 0.149% | -0.028% | 0.177% | 1.64 |
| JPY | 10 | 0.255% | -0.021% | 0.276% | 1.91 |
| AUD | 0 | -0.010% | 0.050% | -0.060% | -0.95 |
| AUD | 5 | -0.047% | 0.011% | -0.058% | -0.37 |
| AUD | 10 | -0.214% | 0.252% | -0.465% | -2.08 |
| USD_BASKET | 0 | 0.000% | 0.019% | -0.019% | -0.43 |
| USD_BASKET | 5 | 0.067% | 0.006% | 0.060% | 0.55 |
| USD_BASKET | 10 | 0.020% | 0.104% | -0.084% | -0.56 |

## Signal gate (fixed prior)

At h=+5: diff=0.060%, t=0.55 → **trade=NO** (no reliable USD edge vs control at fixed prior gate).

## Lagged USD rule (costs on)

Gate **failed** → gated strategy returns are flat (no trades). Forced diagnostic below is **not** for promotion.

| window | strategy | mean_mo | %pos | top3 | t_nw | gates | clears 1%/70%/top3? |
|---|---|---:|---:|---:|---:|:---:|:---:|
| full | news_usd_event_gated | 0.000% | 0.0% | nan% | 0.00 | PASS | **no** |
| full | news_usd_event_forced_diag | 0.061% | 30.0% | 19.0% | 0.76 | PASS | **no** |
| 2024 | news_usd_event_gated | 0.000% | 0.0% | nan% | 0.00 | PASS | **no** |
| 2024 | news_usd_event_forced_diag | 0.373% | 54.5% | 71.2% | 1.06 | PASS | **no** |
| 2025 | news_usd_event_gated | 0.000% | 0.0% | nan% | 0.00 | PASS | **no** |
| 2025 | news_usd_event_forced_diag | -0.408% | 18.2% | 100.0% | -1.65 | PASS | **no** |
| 2026 | news_usd_event_gated | 0.000% | 0.0% | nan% | 0.00 | PASS | **no** |
| 2026 | news_usd_event_forced_diag | 0.185% | 50.0% | 92.8% | 0.79 | PASS | **no** |
| holdout_365d | news_usd_event_gated | 0.000% | 0.0% | nan% | 0.00 | PASS | **no** |
| holdout_365d | news_usd_event_forced_diag | -0.188% | 41.7% | 81.2% | -0.58 | PASS | **no** |

**Joint clear / promote:** **NO.** Do **not** claim 1%/mo.

## Paid NLP?

**Yes for full news NLP.** Free GDELT DOC is rate-limited / short-history; RSS counts are recent-only; GPR spike is a newspaper-share *proxy*, not signed entity/sentiment NLP. A paid API (RavenPack, Refinitiv News Analytics, Bloomberg, or GDELT Cloud keyed history) would be needed for multi-year multilingual NLP panels.

## Artifacts

- `scholarly_fx_news_event_study.csv`, `scholarly_fx_news_event_snapshots.csv`, `scholarly_fx_news_event_windows.csv`, `scholarly_fx_news_event_monthly_returns.csv`, `scholarly_fx_news_event_meta.json`
