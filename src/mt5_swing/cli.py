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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sample_dir() -> Path:
    return _repo_root() / "data" / "sample"


def _ftmo_dir() -> Path:
    return _repo_root() / "data" / "ftmo"


def _history_dir() -> Path:
    return _repo_root() / "data" / "history"


def _data_source_for_path(path: Path) -> str:
    meta = path.with_suffix(".meta.json")
    if meta.exists():
        try:
            return json.loads(meta.read_text(encoding="utf-8")).get(
                "data_source", "unknown"
            )
        except Exception:
            pass
    # Heuristic by directory
    parts = {p.lower() for p in path.parts}
    if "ftmo" in parts:
        return "ftmo_mt5_export"
    if "history" in parts:
        return "approximate_non_ftmo"
    if "sample" in parts:
        return "synthetic_sample"
    return "unknown"


def _resolve_data_path(symbol: str, timeframe: str, data_path: str | None) -> tuple[Path, str]:
    if data_path:
        p = Path(data_path)
        return p, _data_source_for_path(p)
    key = f"{symbol.upper()}_{timeframe.upper()}.csv"
    for folder, src in (
        (_ftmo_dir(), "ftmo_mt5_export"),
        (_history_dir(), "approximate_non_ftmo"),
    ):
        cand = folder / key
        if cand.exists():
            return cand, _data_source_for_path(cand) if (folder / key).exists() else src
    # Fall back to sample
    paths = ensure_sample_data(_sample_dir(), symbols=[symbol], timeframes=[timeframe])
    p = paths[f"{symbol}_{timeframe}"]
    return p, "synthetic_sample"


def _load_data(symbol: str, timeframe: str, data_path: str | None):
    path, source = _resolve_data_path(symbol, timeframe, data_path)
    df = load_ohlc_csv(path, symbol=symbol, timeframe=timeframe)
    df.attrs["data_source"] = source
    df.attrs["data_path"] = str(path)
    return df, source, path


def _bt_from_cfg(cfg_file: dict, symbol: str) -> BacktestConfig:
    risk = cfg_file.get("risk", {})
    bt = cfg_file.get("backtest", {})
    ch = cfg_file.get("challenge", {})
    return BacktestConfig(
        symbol=symbol,
        initial_equity=float(bt.get("initial_equity", 10_000)),
        commission_per_lot=float(bt.get("commission_per_lot", 7.0)),
        slippage_pips=float(bt.get("slippage_pips", 0.5)),
        default_spread_pips=float(bt.get("default_spread_pips", 1.2)),
        signal_lag=int(bt.get("signal_lag", 1)),
        max_dd=float(risk.get("max_peak_to_trough_dd", ch.get("max_loss", 0.10))),
        daily_dd=float(risk.get("max_daily_dd", ch.get("max_daily_loss", 0.05))),
        risk_fraction=float(risk.get("risk_fraction", 0.01)),
        atr_stop_mult=float(risk.get("atr_stop_mult", 2.0)),
        sizing=str(risk.get("sizing", "atr")),
        vol_target=bool(risk.get("vol_target", False)),
        max_lot=float(risk.get("max_lot", 2.0)),
        flatten_on_breach=bool(risk.get("flatten_on_breach", True)),
        max_loss_mode=str(risk.get("max_loss_mode", ch.get("max_loss_mode", "static_initial"))),
        daily_loss_mode=str(risk.get("daily_loss_mode", ch.get("daily_loss_mode", "ftmo_initial"))),
        daily_tz=str(risk.get("daily_tz", ch.get("daily_tz", "Europe/Prague"))),
    )


@click.group()
@click.version_option(__version__, prog_name="mt5-swing")
def main() -> None:
    """Swing-trading research + execution toolkit for MetaTrader 5 / FTMO."""


@main.command("download-data")
@click.option("--symbol", default="EURUSD")
@click.option("--timeframe", default="H4")
@click.option("--bars", default=2000, show_default=True)
@click.option("--out", "out_path", type=click.Path(), default=None)
@click.option(
    "--synthetic",
    is_flag=True,
    default=False,
    help="Write synthetic sample OHLC (offline tests).",
)
@click.option(
    "--interim-public",
    is_flag=True,
    default=False,
    help="Fetch approximate public FX (yfinance). NOT FTMO-ready — labeled approximate_non_ftmo.",
)
@click.option(
    "--all-ftmo-research",
    is_flag=True,
    default=False,
    help="With --interim-public: download research_symbols × H4/D1 from ftmo.yaml.",
)
@click.option("--live", is_flag=True, default=False, help="Fetch from MT5 terminal (Windows).")
def download_data(
    symbol: str,
    timeframe: str,
    bars: int,
    out_path: str | None,
    synthetic: bool,
    interim_public: bool,
    all_ftmo_research: bool,
    live: bool,
) -> None:
    """Fetch OHLC: prefer FTMO MT5 exports via import-ftmo-data; interim public is labeled."""
    if live:
        out = Path(out_path) if out_path else _ftmo_dir() / f"{symbol}_{timeframe}.csv"
        client = MT5Client(live=True, stub=False)
        client.connect()
        try:
            df = client.copy_rates(symbol, timeframe, bars)
        finally:
            client.shutdown()
        df.attrs["data_source"] = "ftmo_mt5_export"
        save_ohlc_csv(df, out)
        click.echo(f"Wrote {len(df)} bars (MT5 live) to {out}")
        return

    if all_ftmo_research or interim_public:
        from mt5_swing.data.download import download_history, download_symbol_timeframe

        cfg = load_config(_repo_root() / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
        if all_ftmo_research:
            symbols = cfg.get("research_symbols", ["EURUSD", "GBPUSD", "USDJPY"])
            symbols = [
                s
                for s in symbols
                if not s.startswith(("XAU", "XAG", "US30", "US100", "US500", "GER", "UK"))
            ]
            tfs = cfg.get("timeframes", ["H4", "D1"])
            paths = download_history(symbols, tfs, out_dir=_history_dir())
            click.echo(
                f"Interim public download complete ({len(paths)} files). "
                "data_source=approximate_non_ftmo — NOT FTMO go-live criteria."
            )
            return
        out = Path(out_path) if out_path else _history_dir() / f"{symbol}_{timeframe}.csv"
        df = download_symbol_timeframe(symbol, timeframe)
        save_ohlc_csv(df, out)
        meta = out.with_suffix(".meta.json")
        meta.write_text(
            json.dumps(
                {
                    "data_source": "approximate_non_ftmo",
                    "symbol": symbol.upper(),
                    "timeframe": timeframe.upper(),
                    "bars": len(df),
                    "start": str(df.index.min()),
                    "end": str(df.index.max()),
                    "provider": "yfinance",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        click.echo(
            f"Wrote {len(df)} bars [approximate_non_ftmo] to {out}. "
            "Replace with FTMO MT5 exports before treating results as challenge-ready."
        )
        return

    # Default / --synthetic: offline sample (not market data)
    if not synthetic:
        click.echo(
            "Writing synthetic sample (pass --synthetic explicitly to silence this hint). "
            "FTMO research: export from Windows FTMO MT5 → import-ftmo-data. "
            "Interim public only: --interim-public (approximate_non_ftmo)."
        )
    out = Path(out_path) if out_path else _sample_dir() / f"{symbol}_{timeframe}.csv"
    df = generate_sample_ohlc(symbol=symbol, timeframe=timeframe, n_bars=bars)
    df.attrs["data_source"] = "synthetic_sample"
    save_ohlc_csv(df, out)
    click.echo(f"Wrote {len(df)} synthetic bars to {out}")



@main.command("import-ftmo-data")
@click.option(
    "--file",
    "files",
    multiple=True,
    required=True,
    help="Mapping SYMBOL_TF=path/to/mt5_export.csv (repeatable).",
)
@click.option(
    "--out-dir",
    type=click.Path(),
    default=None,
    help="Default: data/ftmo/",
)
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
def import_ftmo_data(files: tuple[str, ...], out_dir: str | None, config_path: str | None) -> None:
    """Import FTMO MT5 History CSV exports into data/ftmo/ (production research path)."""
    from mt5_swing.data.mt5_import import import_mt5_exports

    cfg_path = Path(config_path) if config_path else _repo_root() / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml"
    cfg = load_config(cfg_path) if cfg_path.exists() else {}
    spreads = cfg.get("default_spreads_pips", {})
    mapping: dict[str, str] = {}
    for item in files:
        if "=" not in item:
            raise click.ClickException(f"Expected SYMBOL_TF=path, got {item}")
        key, path = item.split("=", 1)
        mapping[key.strip()] = path.strip()
    dest = Path(out_dir) if out_dir else _ftmo_dir()
    paths = import_mt5_exports(mapping, dest, default_spreads=spreads)
    click.echo(f"Imported {len(paths)} FTMO/MT5 series → {dest} (data_source=ftmo_mt5_export)")


@main.command("backtest")
@click.option("--symbol", default="EURUSD")
@click.option("--timeframe", default="H4")
@click.option("--strategy", "strategy_name", default="trend_ma_adx")
@click.option("--data", "data_path", type=click.Path(exists=True), default=None)
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
def backtest_cmd(
    symbol: str, timeframe: str, strategy_name: str, data_path: str | None, config_path: str | None
) -> None:
    """Run a single-segment backtest on FTMO export, interim public, or sample data."""
    cfg_path = config_path or str(_repo_root() / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    cfg_file = load_config(cfg_path if Path(cfg_path).exists() else config_path)
    df, source, path = _load_data(symbol, timeframe, data_path)
    params = cfg_file.get("strategy", {}).get("params", {})
    strat = get_strategy(strategy_name, **params)
    bt_cfg = _bt_from_cfg(cfg_file, symbol)
    result = run_backtest(df, strat, bt_cfg)
    m = result.metrics
    click.echo(f"Strategy: {strategy_name}  Symbol: {symbol}  Bars: {len(df)}")
    click.echo(f"data_source: {source}  path: {path}")
    if source == "approximate_non_ftmo":
        click.echo("WARNING: approximate_non_ftmo — not FTMO go-live pass criteria.")
    click.echo(f"total_return: {m.total_return:.2%}")
    click.echo(f"max_drawdown: {m.max_drawdown:.2%}  gate<10%: {'PASS' if m.passed_max_dd_gate else 'FAIL'}")
    click.echo(f"max_daily_dd: {m.max_daily_dd:.2%}  gate<5%:  {'PASS' if m.passed_daily_dd_gate else 'FAIL'}")
    click.echo(f"sharpe: {m.sharpe:.3f}  trades: {m.n_trades}  win_rate: {m.win_rate:.2%}")
    click.echo(f"gates_pass: {m.gates_pass}")


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
    cfg_path = config_path or str(_repo_root() / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    cfg_file = load_config(cfg_path if Path(cfg_path).exists() else None)
    risk = cfg_file.get("risk", {})
    df, source, path = _load_data(symbol, timeframe, data_path)
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
        signal_lag=int(cfg_file.get("backtest", {}).get("signal_lag", 1)),
    )
    result = run_walk_forward(df, strat, wf)
    click.echo(f"data_source: {source}  path: {path}")
    if source == "approximate_non_ftmo":
        click.echo("WARNING: approximate_non_ftmo — not FTMO go-live pass criteria.")
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
    cfg_path = config_path or str(_repo_root() / "src" / "mt5_swing" / "config" / "ftmo_2step.yaml")
    cfg_file = load_config(cfg_path if Path(cfg_path).exists() else None)
    risk = cfg_file.get("risk", {})
    grid = cfg_file.get("optimize", {}).get("param_grid", {"adx_threshold": [15, 20, 25]})
    df, source, _path = _load_data(symbol, timeframe, data_path)
    click.echo(f"data_source: {source}")
    cut = int(len(df) * 0.6)
    is_df = df.iloc[:cut]
    results = constrained_grid_search(
        is_df,
        strategy_name,
        grid,
        max_trials=max_trials,
        max_dd=float(risk.get("max_peak_to_trough_dd", 0.10)),
        daily_dd=float(risk.get("max_daily_dd", 0.05)),
        bt_config=_bt_from_cfg(cfg_file, symbol),
    )
    click.echo(f"Trials: {len(results)} (IS bars={len(is_df)})")
    for i, r in enumerate(results[:10]):
        m = r["metrics"]
        click.echo(
            f"#{i+1} params={r['params']} sharpe={r['sharpe']:.3f} "
            f"ret={m['total_return']:.2%} gates={'PASS' if r['gates_pass'] else 'FAIL'}"
        )
    click.echo("Reminder: select params on IS only; confirm with walk-forward OOS. Never tune on holdout.")


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
    Live trading gateway (Windows FTMO MT5 terminal). Refuses without ``--live``.
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
