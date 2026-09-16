"""CLI entry points for mt5_swing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from mt5_swing import __version__
from mt5_swing.backtest.engine import BacktestConfig, run_backtest
from mt5_swing.broker.mt5.client import MT5Client
from mt5_swing.config import load_config
from mt5_swing.data.loader import ensure_sample_data, generate_sample_ohlc, load_ohlc_csv, save_ohlc_csv
from mt5_swing.optimize import constrained_grid_search
from mt5_swing.strategies.registry import get_strategy, list_strategies
from mt5_swing.validation.walk_forward import WalkForwardConfig, run_walk_forward


def _repo_sample_dir() -> Path:
    # src/mt5_swing/cli.py -> parents[2] = repo root when editable install
    return Path(__file__).resolve().parents[2] / "data" / "sample"


def _load_data(symbol: str, timeframe: str, data_path: str | None) -> "pd.DataFrame":
    import pandas as pd

    if data_path:
        return load_ohlc_csv(data_path, symbol=symbol, timeframe=timeframe)
    sample_dir = _repo_sample_dir()
    paths = ensure_sample_data(sample_dir, symbols=[symbol], timeframes=[timeframe])
    key = f"{symbol}_{timeframe}"
    return load_ohlc_csv(paths[key], symbol=symbol, timeframe=timeframe)


@click.group()
@click.version_option(__version__, prog_name="mt5-swing")
def main() -> None:
    """Swing-trading research + execution toolkit for MetaTrader 5."""


@main.command("download-data")
@click.option("--symbol", default="EURUSD")
@click.option("--timeframe", default="H4")
@click.option("--bars", default=2000, show_default=True)
@click.option("--out", "out_path", type=click.Path(), default=None)
@click.option("--live", is_flag=True, default=False, help="Fetch from MT5 terminal (Windows).")
def download_data(symbol: str, timeframe: str, bars: int, out_path: str | None, live: bool) -> None:
    """Download OHLC via MT5 or write synthetic sample data."""
    out = Path(out_path) if out_path else _repo_sample_dir() / f"{symbol}_{timeframe}.csv"
    if live:
        client = MT5Client(live=True, stub=False)
        client.connect()
        try:
            df = client.copy_rates(symbol, timeframe, bars)
        finally:
            client.shutdown()
    else:
        click.echo("MT5 not requested (--live omitted); writing synthetic sample OHLC.")
        df = generate_sample_ohlc(symbol=symbol, timeframe=timeframe, n_bars=bars)
    save_ohlc_csv(df, out)
    click.echo(f"Wrote {len(df)} bars to {out}")


@main.command("backtest")
@click.option("--symbol", default="EURUSD")
@click.option("--timeframe", default="H4")
@click.option("--strategy", "strategy_name", default="trend_ma_adx")
@click.option("--data", "data_path", type=click.Path(exists=True), default=None)
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
def backtest_cmd(
    symbol: str, timeframe: str, strategy_name: str, data_path: str | None, config_path: str | None
) -> None:
    """Run a single-segment backtest on sample or CSV data."""
    cfg_file = load_config(config_path)
    risk = cfg_file.get("risk", {})
    bt = cfg_file.get("backtest", {})
    df = _load_data(symbol, timeframe, data_path)
    strat = get_strategy(strategy_name, **cfg_file.get("strategy", {}).get("params", {}))
    bt_cfg = BacktestConfig(
        symbol=symbol,
        initial_equity=float(bt.get("initial_equity", 10_000)),
        commission_per_lot=float(bt.get("commission_per_lot", 7.0)),
        slippage_pips=float(bt.get("slippage_pips", 0.5)),
        signal_lag=int(bt.get("signal_lag", 1)),
        max_dd=float(risk.get("max_peak_to_trough_dd", 0.10)),
        daily_dd=float(risk.get("max_daily_dd", 0.05)),
        risk_fraction=float(risk.get("risk_fraction", 0.01)),
        sizing=str(risk.get("sizing", "atr")),
    )
    result = run_backtest(df, strat, bt_cfg)
    m = result.metrics
    click.echo(f"Strategy: {strategy_name}  Symbol: {symbol}  Bars: {len(df)}")
    click.echo(f"total_return: {m.total_return:.2%}")
    click.echo(f"max_drawdown: {m.max_drawdown:.2%}  gate<10%: {'PASS' if m.passed_max_dd_gate else 'FAIL'}")
    click.echo(f"max_daily_dd: {m.max_daily_dd:.2%}  gate<5%:  {'PASS' if m.passed_daily_dd_gate else 'FAIL'}")
    click.echo(f"sharpe: {m.sharpe:.3f}  trades: {m.n_trades}  win_rate: {m.win_rate:.2%}")
    click.echo(f"gates_pass: {m.gates_pass}")
    click.echo("Note: baselines may fail gates; profitability is not claimed.")


@main.command("walk-forward")
@click.option("--symbol", default="EURUSD")
@click.option("--timeframe", default="H4")
@click.option("--strategy", "strategy_name", default="trend_ma_adx")
@click.option("--mode", type=click.Choice(["rolling", "anchored"]), default="rolling")
@click.option("--train-bars", default=500, show_default=True)
@click.option("--test-bars", default=100, show_default=True)
@click.option("--step-bars", default=100, show_default=True)
@click.option("--data", "data_path", type=click.Path(exists=True), default=None)
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
def walk_forward_cmd(
    symbol: str,
    timeframe: str,
    strategy_name: str,
    mode: str,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    data_path: str | None,
    config_path: str | None,
) -> None:
    """Always-OOS walk-forward (rolling default or anchored)."""
    cfg_file = load_config(config_path)
    risk = cfg_file.get("risk", {})
    df = _load_data(symbol, timeframe, data_path)
    params = cfg_file.get("strategy", {}).get("params", {})
    strat = get_strategy(strategy_name, **params)
    wf = WalkForwardConfig(
        mode=mode,  # type: ignore[arg-type]
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=step_bars,
        symbol=symbol,
        max_dd=float(risk.get("max_peak_to_trough_dd", 0.10)),
        daily_dd=float(risk.get("max_daily_dd", 0.05)),
        initial_equity=float(cfg_file.get("backtest", {}).get("initial_equity", 10_000)),
    )
    result = run_walk_forward(df, strat, wf)
    result.print_report()
    sys.exit(0 if result.gates_pass else 1)


@main.command("optimize")
@click.option("--symbol", default="EURUSD")
@click.option("--timeframe", default="H4")
@click.option("--strategy", "strategy_name", default="trend_ma_adx")
@click.option("--max-trials", default=20, show_default=True)
@click.option("--data", "data_path", type=click.Path(exists=True), default=None)
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
def optimize_cmd(
    symbol: str,
    timeframe: str,
    strategy_name: str,
    max_trials: int,
    data_path: str | None,
    config_path: str | None,
) -> None:
    """Constrained IS grid search; winners must still be walk-forward validated."""
    cfg_file = load_config(config_path)
    risk = cfg_file.get("risk", {})
    grid = cfg_file.get("optimize", {}).get("param_grid", {"adx_threshold": [15, 20, 25]})
    df = _load_data(symbol, timeframe, data_path)
    # Use first 60% as IS only
    cut = int(len(df) * 0.6)
    is_df = df.iloc[:cut]
    results = constrained_grid_search(
        is_df,
        strategy_name,
        grid,
        max_trials=max_trials,
        max_dd=float(risk.get("max_peak_to_trough_dd", 0.10)),
        daily_dd=float(risk.get("max_daily_dd", 0.05)),
        bt_config=BacktestConfig(
            symbol=symbol,
            max_dd=float(risk.get("max_peak_to_trough_dd", 0.10)),
            daily_dd=float(risk.get("max_daily_dd", 0.05)),
        ),
    )
    click.echo(f"Trials: {len(results)} (IS bars={len(is_df)})")
    for i, r in enumerate(results[:10]):
        m = r["metrics"]
        click.echo(
            f"#{i+1} params={r['params']} sharpe={r['sharpe']:.3f} "
            f"ret={m['total_return']:.2%} gates={'PASS' if r['gates_pass'] else 'FAIL'}"
        )
    click.echo("Reminder: select params on IS only; confirm with walk-forward OOS.")


@main.command("paper-trade")
@click.option("--symbol", default="EURUSD")
@click.option("--timeframe", default="H4")
@click.option("--strategy", "strategy_name", default="trend_ma_adx")
@click.option("--bars", default=300, show_default=True)
def paper_trade_cmd(symbol: str, timeframe: str, strategy_name: str, bars: int) -> None:
    """Stub/paper loop: generate signal on latest bar and log a stub order."""
    client = MT5Client(live=False, stub=True)
    client.connect()
    try:
        df = client.copy_rates(symbol, timeframe, bars)
        strat = get_strategy(strategy_name)
        from mt5_swing.features.indicators import apply_feature_pipeline

        feat = apply_feature_pipeline(df, signal_lag=1)
        sigs = strat.generate_signals(feat)
        last = int(sigs.iloc[-1])
        click.echo(f"[paper] {symbol} last_signal={last} equity={client.account_info()['equity']}")
        if last != 0:
            res = client.place_market_order(symbol, last, 0.01, comment="paper")
            click.echo(f"[paper] order ok={res.ok} ticket={res.ticket} stub={res.stub}")
        else:
            click.echo("[paper] flat — no order")
    finally:
        client.shutdown()


@main.command("live-trade")
@click.option("--symbol", default="EURUSD")
@click.option("--timeframe", default="H4")
@click.option("--strategy", "strategy_name", default="trend_ma_adx")
@click.option("--live", is_flag=True, default=False, help="REQUIRED flag to enable live routing.")
@click.option("--bars", default=300, show_default=True)
def live_trade_cmd(
    symbol: str, timeframe: str, strategy_name: str, live: bool, bars: int
) -> None:
    """
    Live trading gateway. Refuses to send orders unless ``--live`` is passed.

    Credentials (if needed) must come from the environment / MT5 terminal session,
    never from files in this repository.
    """
    if not live:
        click.echo(
            "Refusing live-trade without --live. "
            "Use paper-trade for stub mode, or re-run with --live on a Windows MT5 host.",
            err=True,
        )
        sys.exit(2)
    client = MT5Client(live=True, stub=False)
    try:
        client.connect()
    except Exception as exc:
        click.echo(f"MT5 connect failed: {exc}", err=True)
        sys.exit(1)
    try:
        df = client.copy_rates(symbol, timeframe, bars)
        strat = get_strategy(strategy_name)
        from mt5_swing.features.indicators import apply_feature_pipeline

        feat = apply_feature_pipeline(df, signal_lag=1)
        last = int(strat.generate_signals(feat).iloc[-1])
        info = client.account_info()
        click.echo(f"[live] equity={info.get('equity')} signal={last}")
        # Risk halt placeholder — operator must wire KillSwitch to live equity stream
        if last != 0:
            res = client.place_market_order(symbol, last, 0.01, comment="mt5_swing_live")
            click.echo(json.dumps({"ok": res.ok, "ticket": res.ticket, "message": res.message}))
        else:
            click.echo("[live] flat — no order")
    finally:
        client.shutdown()


@main.command("list-strategies")
def list_strategies_cmd() -> None:
    for name in list_strategies():
        click.echo(name)


if __name__ == "__main__":
    main()
