"""
MT5 client with stub mode when MetaTrader5 package is unavailable.

Live order routing is gated by ``live=True`` (CLI ``--live``). Paper/stub never
sends real orders. No credentials are stored in this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

try:
    import MetaTrader5 as mt5  # type: ignore

    HAS_MT5 = True
except ImportError:
    mt5 = None  # type: ignore
    HAS_MT5 = False


class MT5Unavailable(RuntimeError):
    """Raised when live MT5 is requested but the package/terminal is missing."""


@dataclass
class OrderResult:
    ok: bool
    ticket: int | None = None
    message: str = ""
    stub: bool = False


@dataclass
class MT5Client:
    """
    Thin wrapper around MetaTrader5.

    Parameters
    ----------
    live:
        If False (default), operate in stub/paper mode even if MT5 is installed.
    stub:
        Force stub mode (no terminal calls).
    """

    live: bool = False
    stub: bool = False
    connected: bool = False
    _stub_positions: list[dict] = field(default_factory=list)
    _stub_equity: float = 10_000.0

    def __post_init__(self) -> None:
        if self.stub or not HAS_MT5:
            self.stub = True

    def connect(self, *, login: int | None = None, password: str | None = None, server: str | None = None) -> bool:
        if self.stub or not self.live:
            self.connected = True
            return True
        if not HAS_MT5:
            raise MT5Unavailable(
                "MetaTrader5 package not installed. Install on Windows with "
                "`pip install mt5_swing[mt5]` and run the MT5 terminal."
            )
        if not mt5.initialize():
            raise MT5Unavailable(f"mt5.initialize failed: {mt5.last_error()}")
        if login is not None:
            # Credentials must be supplied by the operator at runtime — never from repo files.
            authorized = mt5.login(login, password=password or "", server=server or "")
            if not authorized:
                raise MT5Unavailable(f"mt5.login failed: {mt5.last_error()}")
        self.connected = True
        return True

    def shutdown(self) -> None:
        if self.stub or not HAS_MT5 or not self.live:
            self.connected = False
            return
        mt5.shutdown()
        self.connected = False

    def account_info(self) -> dict[str, Any]:
        if self.stub or not self.live:
            return {
                "equity": self._stub_equity,
                "balance": self._stub_equity,
                "mode": "stub",
                "live": False,
            }
        info = mt5.account_info()
        if info is None:
            return {}
        return info._asdict()

    def copy_rates(
        self,
        symbol: str,
        timeframe: str = "H4",
        count: int = 1000,
    ) -> pd.DataFrame:
        if self.stub or not self.live:
            from mt5_swing.data.loader import generate_sample_ohlc

            return generate_sample_ohlc(symbol=symbol, timeframe=timeframe, n_bars=count)

        tf_map = {
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_H4)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None:
            raise RuntimeError(f"copy_rates failed: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        return df.set_index("time")[["open", "high", "low", "close", "volume"]]

    def place_market_order(
        self,
        symbol: str,
        direction: int,
        lots: float,
        *,
        comment: str = "mt5_swing",
    ) -> OrderResult:
        if direction == 0 or lots <= 0:
            return OrderResult(ok=False, message="flat or zero lots")
        if self.stub or not self.live:
            ticket = len(self._stub_positions) + 1
            self._stub_positions.append(
                {
                    "ticket": ticket,
                    "symbol": symbol,
                    "direction": direction,
                    "lots": lots,
                    "time": datetime.now(timezone.utc).isoformat(),
                    "comment": comment,
                }
            )
            return OrderResult(ok=True, ticket=ticket, message="stub fill", stub=True)

        if not HAS_MT5:
            raise MT5Unavailable("Cannot place live order without MetaTrader5")

        order_type = mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(ok=False, message=f"no tick for {symbol}")
        price = tick.ask if direction > 0 else tick.bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 231100,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None:
            return OrderResult(ok=False, message=str(mt5.last_error()))
        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        return OrderResult(ok=ok, ticket=result.order, message=str(result.comment))

    def close_all(self, symbol: str | None = None) -> list[OrderResult]:
        if self.stub or not self.live:
            closed = []
            keep = []
            for p in self._stub_positions:
                if symbol is None or p["symbol"] == symbol:
                    closed.append(
                        OrderResult(ok=True, ticket=p["ticket"], message="stub close", stub=True)
                    )
                else:
                    keep.append(p)
            self._stub_positions = keep
            return closed
        # Live: close positions via opposite deals (simplified)
        results = []
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if not positions:
            return results
        for pos in positions:
            direction = -1 if pos.type == mt5.POSITION_TYPE_BUY else 1
            results.append(
                self.place_market_order(pos.symbol, direction, pos.volume, comment="close")
            )
        return results
