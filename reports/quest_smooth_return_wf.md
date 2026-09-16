# Quest wave: smooth-return portfolio + IS min(year mean_mo) WF

**data_source:** `approximate_non_ftmo`  **RF:** 8%  **warmup:** 250  **signal_lag=1**
**Selection:** maximize min(IS year mean_mo) s.t. %pos≥70%, top3≤55% (soft≤70%); holdout never for selection.
**Per-leg risk:** RF/n_legs (hard cap). **Port VT grid:** [0.002, 0.0025, 0.003, 0.0035, 0.004, 0.005] aiming ~0.8–1.2%/mo expected.
**Locked official:** `fx4plus_gbpcad_d1_voltarget_0025` (unchanged unless promote).

## Baseline (locked)

| Window | Mean mo | %pos | Top3 | Gates |
|--------|--------:|-----:|-----:|:-----:|
| 2024 | 0.42% | 55% | 74% | PASS |
| 2025 | 1.63% | 73% | 69% | PASS |
| 2026 | 2.30% | 88% | 87% | PASS |
| holdout_365d | 1.52% | 75% | 81% | PASS |
| roll12_m6 | 1.13% | 67% | 80% | PASS |

## IS selection board (top 15)

| Idea | min score | raw min mo | avg %pos | avg top3 | n | symbols |
|------|----------:|-----------:|---------:|---------:|--:|---------|
| lock_plus4|oos_sharpe|vt0.0035|hi3.0 | 0.55% | 0.81% | 74% | 62% | 9 | USDCHF|GBPUSD|CADJPY|AUDCAD|GBPCAD|EURUS |
| lock_plus4|oos_sharpe|vt0.004|hi3.0 | 0.55% | 0.81% | 74% | 62% | 9 | USDCHF|GBPUSD|CADJPY|AUDCAD|GBPCAD|EURUS |
| lock_plus4|oos_sharpe|vt0.005|hi3.0 | 0.55% | 0.81% | 74% | 62% | 9 | USDCHF|GBPUSD|CADJPY|AUDCAD|GBPCAD|EURUS |
| lock_plus4|oos_sharpe|vt0.003|hi3.0 | 0.53% | 0.80% | 74% | 62% | 9 | USDCHF|GBPUSD|CADJPY|AUDCAD|GBPCAD|EURUS |
| lock_plus3|oos_sharpe|vt0.005|hi3.0 | 0.53% | 0.81% | 74% | 62% | 8 | USDCHF|GBPUSD|CADJPY|AUDCAD|GBPCAD|EURUS |
| lock_plus3|oos_sharpe|vt0.004|hi3.0 | 0.53% | 0.81% | 74% | 62% | 8 | USDCHF|GBPUSD|CADJPY|AUDCAD|GBPCAD|EURUS |
| lock_plus3|oos_sharpe|vt0.0035|hi3.0 | 0.52% | 0.81% | 74% | 62% | 8 | USDCHF|GBPUSD|CADJPY|AUDCAD|GBPCAD|EURUS |
| lock_plus4|oos_sharpe|vt0.0025|hi3.0 | 0.51% | 0.79% | 74% | 62% | 9 | USDCHF|GBPUSD|CADJPY|AUDCAD|GBPCAD|EURUS |
| lock_plus3|oos_sharpe|vt0.003|hi3.0 | 0.51% | 0.80% | 74% | 62% | 8 | USDCHF|GBPUSD|CADJPY|AUDCAD|GBPCAD|EURUS |
| lock_plus3|oos_sharpe|vt0.002|hi3.0 | 0.50% | 0.77% | 74% | 62% | 8 | USDCHF|GBPUSD|CADJPY|AUDCAD|GBPCAD|EURUS |
| lock_plus3|oos_sharpe|vt0.0025|hi3.0 | 0.49% | 0.79% | 74% | 62% | 8 | USDCHF|GBPUSD|CADJPY|AUDCAD|GBPCAD|EURUS |
| lock_plus4|oos_sharpe|vt0.002|hi3.0 | 0.48% | 0.77% | 74% | 62% | 9 | USDCHF|GBPUSD|CADJPY|AUDCAD|GBPCAD|EURUS |
| lock_plus2|oos_sharpe|vt0.002|hi3.0 | 0.42% | 0.77% | 74% | 64% | 7 | USDCHF|GBPUSD|CADJPY|AUDCAD|GBPCAD|EURUS |
| lock_plus2|oos_sharpe|vt0.005|hi3.0 | 0.40% | 0.78% | 74% | 64% | 7 | USDCHF|GBPUSD|CADJPY|AUDCAD|GBPCAD|EURUS |
| lock_plus2|oos_sharpe|vt0.004|hi3.0 | 0.39% | 0.77% | 74% | 65% | 7 | USDCHF|GBPUSD|CADJPY|AUDCAD|GBPCAD|EURUS |

**Hard constraint passers:** 104 / 528
**Best IS:** `lock_plus4|oos_sharpe|vt0.0035|hi3.0` (score=0.55% raw_min=0.81%)

## Confirmation / promotion

| Idea | 2024 mo/%pos/top3 | 2025 mo/%pos | 2026 mo | HO mo/%pos/top3 | Promote | Why |
|------|------------------:|-------------:|--------:|----------------:|:-------:|-----|
| baseline_vt0025 | 0.42/55/74 | 1.63/73 | 2.30 | 1.52/75/81 | no | 2024 mean_mo=0.42% <1% |
| lock_plus4|oos_sharpe|vt0.0035|hi3.0 | 0.81/73/55 | 0.80/64 | 1.33 | 0.63/50/84 | no | 2024 mean_mo=0.81% <1% |
| lock_plus4|oos_sharpe|vt0.004|hi3.0 | 0.81/73/55 | 0.80/64 | 1.31 | 0.62/50/84 | no | 2024 mean_mo=0.81% <1% |
| lock_plus4|oos_sharpe|vt0.005|hi3.0 | 0.81/73/55 | 0.80/64 | 1.31 | 0.62/50/84 | no | 2024 mean_mo=0.81% <1% |
| lock_plus4|oos_sharpe|vt0.003|hi3.0 | 0.80/73/55 | 0.80/64 | 1.34 | 0.63/50/84 | no | 2024 mean_mo=0.80% <1% |
| lock_plus3|oos_sharpe|vt0.005|hi3.0 | 1.11/73/56 | 0.64/73 | 1.41 | 0.83/50/84 | no | 2025 mean_mo=0.64% <1% |

**PROMOTE:** none — official tag unchanged `fx4plus_gbpcad_d1_voltarget_0025`.

## Yahoo / FTMO limits (if stuck)

- All research uses `approximate_non_ftmo` (Yahoo via yfinance).
- Spreads/commissions/sessions are approximate; FTMO MT5 tick/spread/swap differ.
- Dual-year uncorrelated edge is scarce on Yahoo FX — many legs flip sign 2024↔2025.
- No equity index history in this repo (US30/NAS100/SPX) for diversifiers.
- H1 Yahoo depth ~2y; incomplete multi-year calendars.
- **FTMO `data/ftmo/` exports would unlock:** true FTMO spreads/commission, session filters,
  longer clean H1/M15, index CFDs if offered, and go-live candidacy (`ftmo_mt5_export`).

## Failures (honest)

- Smooth many-leg portfolios that hit %pos/top3 on IS often sit below 1%/mo on 2024.
- Raising VT toward 1%/mo expected tends to reintroduce burstiness (top3↑).
- Metals only eligible on dual-confirm; XAU dual scarce in WF summary.
- No RF hike (8% locked).

