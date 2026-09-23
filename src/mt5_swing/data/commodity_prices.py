"""Commodity price panel for Chen–Rogoff–Rossi style commodity-currency research.

Downloads Yahoo futures/ETF closes via yfinance into ``data/macro/``:
- Oil (WTI): ``CL=F`` — primary for CAD (Amano & van Norden oil–CAD)
- Copper: ``HG=F`` — industrial metals / AUD
- Gold: ``GC=F`` — secondary / risk
- Broad basket: try ``DBC`` or ``GSG`` ETF; else equal-weight oil+copper+gold

Point-in-time: default ``pub_lag_days=1`` (futures close treated as known next session).

Limitations (document honestly):
- Futures continuous contracts ≠ spot country export baskets used in CRR (2010).
- No dairy futures proxy for NZD on free Yahoo reliably → use broad basket.
- Yahoo / approximate_non_ftmo — not FTMO MT5 ticks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Yahoo tickers
COMMODITY_TICKERS: dict[str, str] = {
    "oil": "CL=F",
    "copper": "HG=F",
    "gold": "GC=F",
}
BASKET_ETF_CANDIDATES: tuple[str, ...] = ("DBC", "GSG")

# Country → preferred commodity column (after panel construction)
COUNTRY_COMMODITY_MAP: dict[str, str] = {
    "AUD": "copper",  # industrial metals / China demand channel
    "CAD": "oil",  # Amano–van Norden oil–CAD
    "NZD": "basket",  # dairy proxy unavailable → broad basket
}


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"


def _ensure_macro() -> Path:
    d = macro_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def commodity_panel_path() -> Path:
    return macro_dir() / "commodity_yahoo_panel.csv"


def _download_one(ticker: str) -> pd.Series | None:
    import yfinance as yf

    try:
        raw = yf.Ticker(ticker).history(period="max", auto_adjust=True)
    except Exception as exc:  # noqa: BLE001 — network/yfinance flaky
        print(f"WARN commodity download failed for {ticker}: {exc}")
        return None
    if raw is None or raw.empty or "Close" not in raw.columns:
        print(f"WARN empty commodity download for {ticker}")
        return None
    s = pd.to_numeric(raw["Close"], errors="coerce").dropna()
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC")
    else:
        idx = idx.tz_localize("UTC")
    s.index = idx
    s.name = ticker
    return s


def download_commodity_panel(*, force: bool = False) -> Path:
    """Download/cache Yahoo commodity closes → ``data/macro/commodity_yahoo_panel.csv``.

    Columns: oil, copper, gold, basket (+ optional etf ticker name if used).
    Does not invent prices: if all downloads fail and no cache exists, raises.
    """
    path = commodity_panel_path()
    if path.exists() and not force and path.stat().st_size > 100:
        return path

    _ensure_macro()
    series: dict[str, pd.Series] = {}
    for name, ticker in COMMODITY_TICKERS.items():
        s = _download_one(ticker)
        if s is not None and len(s) > 50:
            series[name] = s
            # Also cache individual series for debugging
            one = _ensure_macro() / f"commodity_{name}_{ticker.replace('=', '_')}.csv"
            s.to_frame("close").to_csv(one)

    etf_used: str | None = None
    etf_s: pd.Series | None = None
    for etf in BASKET_ETF_CANDIDATES:
        etf_s = _download_one(etf)
        if etf_s is not None and len(etf_s) > 50:
            etf_used = etf
            break

    if not series and etf_s is None:
        if path.exists() and path.stat().st_size > 100:
            print(f"WARN download failed; keeping existing cache at {path}")
            return path
        raise RuntimeError(
            "Commodity Yahoo download failed and no cache at "
            f"{path}. Place CL=F / HG=F / GC=F closes manually or retry."
        )

    panel = pd.DataFrame(series).sort_index()
    panel.index = pd.DatetimeIndex(panel.index)
    if panel.index.tz is None:
        panel.index = panel.index.tz_localize("UTC")
    else:
        panel.index = panel.index.tz_convert("UTC")
    panel.index = panel.index.normalize()
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    if etf_s is not None and etf_used is not None:
        panel["etf"] = etf_s.reindex(panel.index)
        panel["basket"] = panel["etf"]
        panel.attrs["basket_source"] = etf_used
    else:
        # Equal-weight of available oil/copper/gold (normalized levels → mean of returns later)
        cols = [c for c in ("oil", "copper", "gold") if c in panel.columns]
        if not cols:
            raise RuntimeError("No commodity series available for EW basket")
        # Use average of price levels rebased to 100 at first common date
        sub = panel[cols].dropna(how="any")
        if sub.empty:
            sub = panel[cols].dropna(how="all")
        rebased = sub / sub.iloc[0] * 100.0
        panel["basket"] = rebased.mean(axis=1).reindex(panel.index)
        panel.attrs["basket_source"] = "ew_oil_copper_gold"

    panel.index.name = "date"
    # Persist attrs via sidecar meta would be nice; embed basket_source as column note in CSV comment
    out = panel.copy()
    out.to_csv(path)
    meta = path.with_suffix(".meta.json")
    import json

    meta.write_text(
        json.dumps(
            {
                "source": "yfinance",
                "data_tag": "approximate_non_ftmo",
                "tickers": {k: COMMODITY_TICKERS[k] for k in COMMODITY_TICKERS if k in series},
                "basket_source": panel.attrs.get("basket_source", "unknown"),
                "etf_used": etf_used,
                "columns": list(out.columns),
                "n_rows": int(len(out)),
                "start": str(out.index.min()),
                "end": str(out.index.max()),
            },
            indent=2,
        )
    )
    return path


def load_commodity_panel(
    *,
    download: bool = True,
    pub_lag_days: int = 1,
    force: bool = False,
) -> pd.DataFrame:
    """Load commodity close panel with optional publication lag (default 1 day).

    Index shifted forward by ``pub_lag_days`` so close of day t is first known on t+lag.
    """
    path = commodity_panel_path()
    if download or not path.exists():
        try:
            download_commodity_panel(force=force)
        except RuntimeError:
            if not path.exists():
                raise
    if not path.exists():
        raise FileNotFoundError(f"Commodity panel missing: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True).normalize()
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.duplicated(keep="last")]

    if pub_lag_days > 0:
        df.index = df.index + pd.Timedelta(days=int(pub_lag_days))
        df.index = pd.DatetimeIndex(df.index).normalize()
        df = df[~df.index.duplicated(keep="last")].sort_index()

    # Restore basket_source from meta if present
    meta = path.with_suffix(".meta.json")
    basket_src = "ew_oil_copper_gold"
    if meta.exists():
        import json

        try:
            basket_src = json.loads(meta.read_text()).get("basket_source", basket_src)
        except Exception:  # noqa: BLE001
            pass
    if "basket" not in df.columns:
        cols = [c for c in ("oil", "copper", "gold") if c in df.columns]
        if cols:
            sub = df[cols].dropna(how="all")
            rebased = sub / sub.iloc[0] * 100.0
            df["basket"] = rebased.mean(axis=1)

    df.attrs["pub_lag_days"] = int(pub_lag_days)
    df.attrs["source"] = "yfinance_commodity"
    df.attrs["data_tag"] = "approximate_non_ftmo"
    df.attrs["basket_source"] = basket_src
    df.attrs["country_map"] = dict(COUNTRY_COMMODITY_MAP)
    return df


def commodity_returns(
    panel: pd.DataFrame | None = None,
    *,
    columns: list[str] | None = None,
    pub_lag_days: int = 1,
    download: bool = True,
) -> pd.DataFrame:
    """Simple percent returns of commodity closes (after publication lag)."""
    if panel is None:
        panel = load_commodity_panel(download=download, pub_lag_days=pub_lag_days)
    cols = columns or [c for c in panel.columns if c in ("oil", "copper", "gold", "basket", "etf")]
    cols = [c for c in cols if c in panel.columns]
    if not cols:
        return pd.DataFrame(index=panel.index)
    ret = panel[cols].pct_change()
    ret.attrs.update(getattr(panel, "attrs", {}) or {})
    return ret


def commodity_for_currency(panel: pd.DataFrame, currency: str) -> pd.Series:
    """Mapped commodity level series for a commodity currency (AUD/CAD/NZD)."""
    key = COUNTRY_COMMODITY_MAP.get(currency.upper())
    if key is None:
        raise KeyError(f"No commodity map for {currency}")
    col = key if key in panel.columns else "basket"
    if col not in panel.columns:
        raise KeyError(f"Commodity column {col} missing from panel")
    s = panel[col].copy()
    s.name = f"{currency.upper()}_{col}"
    return s


# ---------------------------------------------------------------------------
# Terms-of-trade (ToT) country mapping — distinct from COUNTRY_COMMODITY_MAP
# ---------------------------------------------------------------------------
# Literature: Cashin, Céspedes & Sahay (2004) commodity currencies & ToT;
# Chen–Rogoff–Rossi (2010) commodity prices vs FX (export side);
# Amano & van Norden (oil–CAD).
#
# Frozen export vs *import* proxies from the free Yahoo panel (oil / copper /
# gold / basket). Prior CRR wave used only the export (or single) commodity;
# ToT = export momentum − import momentum is the refinement.
#
# NOK / ZAR documented for completeness; traded only when USDNOK / USDZAR
# (or XXXUSD) appear in the FX panel — current history has AUD/CAD/NZD only.

COUNTRY_TOT_MAP: dict[str, dict[str, str]] = {
    # Australia: industrial metals exporter, net crude oil importer
    "AUD": {"export": "copper", "import": "oil"},
    # Canada: energy exporter; industrial metals as import / relative proxy
    "CAD": {"export": "oil", "import": "copper"},
    # New Zealand: softs proxy via broad basket; oil importer
    "NZD": {"export": "basket", "import": "oil"},
    # Norway: oil exporter (trade when NOK pair available)
    "NOK": {"export": "oil", "import": "copper"},
    # South Africa: gold/metals exporter; oil importer
    "ZAR": {"export": "gold", "import": "oil"},
}

# Currencies we attempt to trade when USD pairs exist in the research panel
TOT_TRADEABLE_CCYS: tuple[str, ...] = ("AUD", "CAD", "NZD")
TOT_DOCUMENTED_CCYS: tuple[str, ...] = ("AUD", "CAD", "NZD", "NOK", "ZAR")


def tot_export_import_columns(currency: str) -> tuple[str, str]:
    """Return (export_col, import_col) for a currency from the frozen ToT map."""
    m = COUNTRY_TOT_MAP.get(currency.upper())
    if m is None:
        raise KeyError(f"No ToT map for {currency}")
    return m["export"], m["import"]
