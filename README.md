# mt5_swing

Swing-trading **research + execution** toolkit for MetaTrader 5 (forex/CFDs).

This package provides point-in-time features, baseline strategies, a cost-aware bar
backtester, hard drawdown kill-switches, and **always out-of-sample** walk-forward
validation (rolling default + anchored). Live/paper routing uses the optional
`MetaTrader5` package on Windows; research runs fully offline from CSV or synthetic
sample data.

> **Honest default:** baseline strategies are research templates. They may **fail**
> the 10% / 5% risk gates on sample data. The framework reports IS/OOS metrics and
> pass/fail honestly — it does **not** claim OOS profitability.

## Hard risk rules

| Gate | Limit | Definition |
|------|-------|------------|
| Peak-to-trough max DD | **&lt; 10%** | `(peak_equity − equity) / peak_equity` |
| Daily DD | **&lt; 5%** | Equity vs **prior calendar-day close** (UTC date of the bar index): `(E_prior_day_close − E_t) / E_prior_day_close` |

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

## Quick start

```bash
# Generate / ensure sample CSVs under data/sample/
python -m mt5_swing.cli download-data --symbol EURUSD --timeframe H4

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
  data/          OHLC CSV loader, sample generator, symbol/pip metadata
  features/      Point-in-time indicators + explicit lag
  strategies/    Strategy protocol + trend MA/ADX, mean-reversion+regime, breakout
  backtest/      Bar backtester (spread/commission/slippage, sizing hooks)
  risk/          ATR / fixed-fractional sizing, MaxDD + DailyDD, kill-switch
  validation/    Walk-forward (rolling + anchored), OOS report, overfit checks
  broker/mt5/    Client with stub mode; live gated by --live
  config/        default.yaml (EURUSD, GBPUSD, USDJPY, H4/D1)
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
