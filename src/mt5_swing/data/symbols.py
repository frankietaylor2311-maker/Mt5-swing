"""Symbol metadata and pip sizing helpers for forex majors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolMeta:
    """Static metadata for a traded symbol."""

    name: str
    digits: int
    contract_size: float = 100_000.0  # standard lot
    quote_currency: str = "USD"
    base_currency: str = "EUR"

    @property
    def pip_size(self) -> float:
        """Price increment of one pip (0.0001 for most FX; 0.01 for JPY pairs)."""
        return 10 ** -(self.digits - 1) if self.digits >= 3 else 10 ** -self.digits

    @property
    def point(self) -> float:
        return 10 ** -self.digits


_SYMBOLS: dict[str, SymbolMeta] = {
    "EURUSD": SymbolMeta("EURUSD", digits=5, base_currency="EUR", quote_currency="USD"),
    "GBPUSD": SymbolMeta("GBPUSD", digits=5, base_currency="GBP", quote_currency="USD"),
    "USDJPY": SymbolMeta("USDJPY", digits=3, base_currency="USD", quote_currency="JPY"),
    "USDCHF": SymbolMeta("USDCHF", digits=5, base_currency="USD", quote_currency="CHF"),
    "USDCAD": SymbolMeta("USDCAD", digits=5, base_currency="USD", quote_currency="CAD"),
    "AUDUSD": SymbolMeta("AUDUSD", digits=5, base_currency="AUD", quote_currency="USD"),
    "NZDUSD": SymbolMeta("NZDUSD", digits=5, base_currency="NZD", quote_currency="USD"),
    "EURGBP": SymbolMeta("EURGBP", digits=5, base_currency="EUR", quote_currency="GBP"),
    "EURJPY": SymbolMeta("EURJPY", digits=3, base_currency="EUR", quote_currency="JPY"),
    "GBPJPY": SymbolMeta("GBPJPY", digits=3, base_currency="GBP", quote_currency="JPY"),
    "AUDJPY": SymbolMeta("AUDJPY", digits=3, base_currency="AUD", quote_currency="JPY"),
    "EURCHF": SymbolMeta("EURCHF", digits=5, base_currency="EUR", quote_currency="CHF"),
    "CADJPY": SymbolMeta("CADJPY", digits=3, base_currency="CAD", quote_currency="JPY"),
    "NZDJPY": SymbolMeta("NZDJPY", digits=3, base_currency="NZD", quote_currency="JPY"),
    "EURCAD": SymbolMeta("EURCAD", digits=5, base_currency="EUR", quote_currency="CAD"),
    "AUDCAD": SymbolMeta("AUDCAD", digits=5, base_currency="AUD", quote_currency="CAD"),
    "GBPCAD": SymbolMeta("GBPCAD", digits=5, base_currency="GBP", quote_currency="CAD"),
    "EURAUD": SymbolMeta("EURAUD", digits=5, base_currency="EUR", quote_currency="AUD"),
    "NZDCAD": SymbolMeta("NZDCAD", digits=5, base_currency="NZD", quote_currency="CAD"),
    # Metals / indices — contract_size approximates FTMO CFD lots; verify in terminal
    "XAUUSD": SymbolMeta("XAUUSD", digits=2, contract_size=100.0, base_currency="XAU", quote_currency="USD"),
    "XAGUSD": SymbolMeta("XAGUSD", digits=3, contract_size=5000.0, base_currency="XAG", quote_currency="USD"),
    "US30": SymbolMeta("US30", digits=1, contract_size=1.0, base_currency="US30", quote_currency="USD"),
    "US100": SymbolMeta("US100", digits=1, contract_size=1.0, base_currency="US100", quote_currency="USD"),
    "US500": SymbolMeta("US500", digits=1, contract_size=1.0, base_currency="US500", quote_currency="USD"),
    "GER40": SymbolMeta("GER40", digits=1, contract_size=1.0, base_currency="GER40", quote_currency="EUR"),
    "UK100": SymbolMeta("UK100", digits=1, contract_size=1.0, base_currency="UK100", quote_currency="GBP"),
}


def get_symbol_meta(symbol: str) -> SymbolMeta:
    key = symbol.upper().replace("/", "")
    if key not in _SYMBOLS:
        # Sensible default for unknown FX-like symbols
        digits = 3 if key.endswith("JPY") else 5
        return SymbolMeta(name=key, digits=digits)
    return _SYMBOLS[key]


def pip_size(symbol: str) -> float:
    return get_symbol_meta(symbol).pip_size


def pip_value_per_lot(symbol: str, price: float | None = None) -> float:
    """
    Approximate USD pip value for 1.0 standard lot.

    For XXXUSD: pip_value ≈ contract_size * pip_size (in USD).
    For USDJPY: pip_value ≈ contract_size * pip_size / price (convert JPY→USD).
    """
    meta = get_symbol_meta(symbol)
    raw = meta.contract_size * meta.pip_size
    if meta.quote_currency == "USD":
        return raw
    if meta.base_currency == "USD" and price and price > 0:
        return raw / price
    # Fallback: treat as quote=USD
    return raw
