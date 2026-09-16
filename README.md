# mt5_swing

Swing-trading **research + execution** toolkit for MetaTrader 5 (forex/CFDs).

This package provides point-in-time features, baseline strategies, a cost-aware bar
backtester, hard drawdown kill-switches, and **always out-of-sample** walk-forward
validation (rolling default + anchored). Live/paper routing uses the optional
`MetaTrader5` package on Windows; research runs fully offline from CSV or synthetic
sample data.

> **Honest default:** baseline strategies are research templates. They may **fail**
> the 10% / 5% risk gates on sample or interim public data. The framework reports IS/OOS metrics and
> pass/fail honestly — it does **not** claim OOS profitability or FTMO readiness
> on `approximate_non_ftmo` data.

## Hard risk rules

| Gate | Limit | Definition |
|------|-------|------------|
| FTMO 2-Step Max Loss | **&lt; 10%** | **Static** from initial capital: `(initial − equity) / initial` (not peak-to-trough) |
| FTMO 2-Step Max Daily Loss | **&lt; 5%** | Loss from balance at **00:00 Europe/Prague** / **initial**: `(E_prague_sod − E_t) / initial` |
| Peak-to-trough DD (info) | — | `(peak − equity) / peak` — reported but not the 2-Step gate |

On breach: **halt new entries**. If past the flatten threshold (defaults to the same
limit; configurable), **flatten** open exposure.

## Anti look-ahead / anti-overfitting

- Indicators use trailing windows only (`center=False`); `lag(n)` is explicit.
- Default `signal_lag=1`: features used for decisions are shifted so bar-*t* signals
  do not peek at unlagged same-bar values for execution at the next open.
- Walk-forward always evaluates **OOS** folds (rolling or anchored).
- `optimize` runs a **constrained** IS grid only; winners must be re-checked with
  walk-forward OOS (never select params on OOS inside the optimizer).

## Install

```bash
cd Mt5-swing
python -m pip install -e ".[dev]"
# or: uv pip install -e ".[dev]"
```

Optional MT5 (Windows + running terminal):

```bash
pip install -e ".[mt5]"
```

## FTMO Challenge research data

**Production path:** export OHLC from the **Windows FTMO MT5** terminal (History Center
or a script), then import:

```bash
python -m mt5_swing.cli import-ftmo-data \
  --file EURUSD_H4=/path/from/mt5/EURUSD_H4.csv \
  --file GBPUSD_H4=/path/from/mt5/GBPUSD_H4.csv
# writes data/ftmo/*.csv with data_source=ftmo_mt5_export
```

Config: [`src/mt5_swing/config/ftmo_2step.yaml`](src/mt5_swing/config/ftmo_2step.yaml) — **2-Step**
Challenge: **static** max loss **&lt; 10%** of initial capital; max daily loss **&lt; 5%** of
initial vs balance at **00:00 Europe/Prague** (not UTC). Peak-to-trough DD is informational.
Swing style: H4/D1. Basket: EURUSD, GBPUSD, USDJPY (+ optional XAUUSD/index when exported).

**Do not install MetaTrader 5 on Linux.** Live/paper against FTMO runs on the always-on
Windows PC later.

**Interim public data (Yahoo via yfinance):** optional only when FTMO exports are not
available yet. Every file/report must be labeled `approximate_non_ftmo` and must **not**
be treated as FTMO go-live pass criteria.

```bash
python -m mt5_swing.cli download-data --interim-public --all-ftmo-research
python -m mt5_swing.cli download-data --synthetic --symbol EURUSD  # offline tests
```

Batch walk-forward + constrained IS refine (holdout last ~1y never used for selection):

```bash
python scripts/run_ftmo_research.py
# → reports/walk_forward_summary.md
```

## Quick start


```bash
# Generate / ensure sample CSVs under data/sample/
python -m mt5_swing.cli download-data --synthetic --symbol EURUSD --timeframe H4

# Single backtest
python -m mt5_swing.cli backtest --symbol EURUSD --strategy trend_ma_adx

# Walk-forward (prints IS/OOS + PASS/FAIL vs 10%/5% gates)
python -m mt5_swing.cli walk-forward --symbol EURUSD --mode rolling
python -m mt5_swing.cli walk-forward --symbol EURUSD --mode anchored

# Constrained IS optimize (then validate with walk-forward)
python -m mt5_swing.cli optimize --strategy trend_ma_adx

# Paper (stub) / live (gated)
python -m mt5_swing.cli paper-trade --symbol EURUSD
python -m mt5_swing.cli live-trade --symbol EURUSD          # refuses without --live
python -m mt5_swing.cli live-trade --symbol EURUSD --live   # Windows + MT5 only
```

Console script alias: `mt5-swing` (same commands).

## Tests

```bash
pytest
```

Includes a **look-ahead sabotage** test (future OHLC corruption must not change past
features/signals) and walk-forward smoke tests.

## Package layout

```
src/mt5_swing/
  data/          OHLC loader, FTMO/MT5 CSV import, interim yfinance, sample generator
  features/      Point-in-time indicators + explicit lag
  strategies/    Strategy protocol + trend MA/ADX, mean-reversion+regime, breakout
  backtest/      Bar backtester (spread/commission/slippage, sizing hooks)
  risk/          ATR / fixed-fractional sizing, MaxDD + DailyDD, kill-switch
  validation/    Walk-forward (rolling + anchored), OOS report, overfit checks
  broker/mt5/    Client with stub mode; live gated by --live
  config/        default.yaml + ftmo.yaml (Challenge gates & symbol universe)
  cli.py         download-data, backtest, walk-forward, optimize, paper-trade, live-trade
```

## Methodology (short)

1. Load OHLC (CSV or sample). Build lagged features.
2. Strategy emits target direction ∈ {−1, 0, +1}.
3. Backtester fills at bar **open** when the (lagged) signal changes; applies spread,
   slippage (pips), and commission per lot; sizes via ATR risk fraction or fixed lots.
4. Kill-switch monitors peak-to-trough and daily DD; halts entries / flattens on breach.
5. Walk-forward splits IS/OOS chronologically; aggregate OOS DD is gated at 10%/5%.

## MT5 Windows notes

- The `MetaTrader5` Python package talks to a **local MT5 terminal** (Windows).
- Install MT5, enable Algo Trading, log into a demo/live account in the terminal.
- `pip install MetaTrader5` (or `pip install mt5_swing[mt5]`).
- This Linux/CI environment uses **stub mode** automatically when the package is absent.
- **No secrets in the repo.** Pass login/password/server only at runtime via your own
  environment or the already-logged-in terminal session — never commit `.env` credentials.

## Risk disclaimer

Trading forex/CFDs involves substantial risk of loss. This software is for research
and education. Past backtests (especially on synthetic sample data) do not imply
future results. You are solely responsible for any live trading decisions. The authors
provide the software **as-is** under the MIT license with no warranty.

## License

MIT — see [LICENSE](LICENSE).
