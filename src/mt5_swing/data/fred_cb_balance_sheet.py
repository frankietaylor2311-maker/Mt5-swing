"""FRED central-bank balance-sheet / QE panels for FX (free, PIT).

Literature
----------
- Gagnon, Raskin, Remache & Sack (2011), "The Federal Reserve's Large-Scale
  Asset Purchases," *IJCB* — LSAP / QE compress term premia (portfolio-balance).
- Neely (2015), "Unconventional Monetary Policy Effects on Exchange Rates,"
  *JBF* — Fed LSAP announcements depreciate the USD via portfolio-balance /
  signalling channels.
- Bauer & Neely (2014), "International Channels of the Fed's Unconventional
  Monetary Policy," *JIMF* — spillover to foreign yields / FX.
- Related: Krishnamurthy & Vissing-Jorgensen (2011) QE channels; Joyce et al.
  on BoE QE; Ueda / others on BoJ QQE.

Free FRED series (no Bloomberg)
--------------------------------
| CB  | Series       | Freq    | Units              | Notes                    |
|-----|--------------|---------|--------------------|--------------------------|
| Fed | WALCL        | weekly  | mn USD             | H.4.1 total assets       |
| ECB | ECBASSETSW   | weekly  | mn EUR             | Eurosystem total assets  |
| BoJ | JPNASSETS    | monthly | 100 mn JPY         | BoJ total assets         |
| BoE | UKASSETS     | monthly | mn GBP             | **discontinued 2014-09** |

Optional GDP for BS/GDP (quarterly, nominal):
USD ``GDP``, EUR ``EUNNGDP``, JPY ``JPNNGDP``, GBP ``UKNGDP``.

PIT lags (conservative, frozen — not holdout-tuned)
---------------------------------------------------
- Weekly WALCL / ECBASSETSW: H.4.1 / ECB weekly FS typically release ~1–4 days
  after Wednesday observation → ``pub_lag_days=7``.
- Monthly JPNASSETS / UKASSETS: ``pub_lag_months=1``.
- Quarterly GDP: ``pub_lag_quarters=1`` (advance + buffer).

Growth rates (YoY) avoid FX-conversion of heterogeneous level units.
Do **not** overlay on the locked FTMO sleeve.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → (FRED id, frequency tag)
CB_ASSETS_SERIES: dict[str, tuple[str, str]] = {
    "USD": ("WALCL", "weekly"),
    "EUR": ("ECBASSETSW", "weekly"),
    "JPY": ("JPNASSETS", "monthly"),
    # Discontinued — loaded only when explicitly requested
    "GBP": ("UKASSETS", "monthly"),
}

GDP_SERIES: dict[str, str] = {
    "USD": "GDP",  # bn USD SAAR
    "EUR": "EUNNGDP",  # mn EUR
    "JPY": "JPNNGDP",  # bn JPY
    "GBP": "UKNGDP",  # mn GBP
}

DEFAULT_WEEKLY_PUB_LAG_DAYS = 7
DEFAULT_MONTHLY_PUB_LAG_MONTHS = 1
DEFAULT_GDP_PUB_LAG_QUARTERS = 1


def _ensure_utc(s: pd.Series) -> pd.Series:
    out = s.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out.index = idx
    return out[~out.index.duplicated(keep="last")].sort_index()


def _apply_day_lag(s: pd.Series, days: int) -> pd.Series:
    out = _ensure_utc(s)
    if days > 0:
        out.index = out.index + pd.Timedelta(days=int(days))
        out = out[~out.index.duplicated(keep="last")].sort_index()
    out.attrs["pub_lag_days"] = int(days)
    return out


def _apply_month_lag(s: pd.Series, months: int) -> pd.Series:
    out = _ensure_utc(s)
    if months > 0:
        out.index = out.index + pd.DateOffset(months=int(months))
        out = out[~out.index.duplicated(keep="last")].sort_index()
    out.attrs["pub_lag_months"] = int(months)
    return out


def _apply_quarter_lag(s: pd.Series, quarters: int) -> pd.Series:
    out = _ensure_utc(s)
    if quarters > 0:
        out.index = out.index + pd.DateOffset(months=3 * int(quarters))
        out = out[~out.index.duplicated(keep="last")].sort_index()
    out.attrs["pub_lag_quarters"] = int(quarters)
    return out


def load_cb_assets(
    currency: str,
    *,
    download: bool = True,
    weekly_pub_lag_days: int = DEFAULT_WEEKLY_PUB_LAG_DAYS,
    monthly_pub_lag_months: int = DEFAULT_MONTHLY_PUB_LAG_MONTHS,
    force: bool = False,
    allow_discontinued: bool = False,
) -> pd.Series:
    """Load one CB total-assets series with publication lag."""
    ccy = currency.upper()
    meta = CB_ASSETS_SERIES.get(ccy)
    if meta is None:
        raise KeyError(f"No CB assets mapping for {ccy}")
    sid, freq = meta
    if ccy == "GBP" and not allow_discontinued:
        raise ValueError("UKASSETS discontinued 2014-09; pass allow_discontinued=True")
    if download:
        download_fred_series(sid, force=force)
    s = load_fred_series(sid, download=False)
    if freq == "weekly":
        s = _apply_day_lag(s, weekly_pub_lag_days)
    else:
        s = _apply_month_lag(s, monthly_pub_lag_months)
    s.name = ccy
    s.attrs["source"] = f"fred_{sid}"
    s.attrs["frequency"] = freq
    s.attrs["fred_id"] = sid
    return s


def yoy_growth(s: pd.Series, *, periods: int | None = None) -> pd.Series:
    """Year-over-year percent change on native frequency."""
    freq = s.attrs.get("frequency", "")
    if periods is None:
        periods = 52 if freq == "weekly" else 12
    out = s.pct_change(periods=int(periods))
    out.name = f"{s.name}_yoy" if s.name else "yoy"
    out.attrs.update({k: v for k, v in s.attrs.items()})
    out.attrs["yoy_periods"] = int(periods)
    return out


def load_cb_assets_panel(
    currencies: Iterable[str] | None = None,
    *,
    download: bool = True,
    weekly_pub_lag_days: int = DEFAULT_WEEKLY_PUB_LAG_DAYS,
    monthly_pub_lag_months: int = DEFAULT_MONTHLY_PUB_LAG_MONTHS,
    force: bool = False,
    include_gbp: bool = False,
) -> dict[str, pd.Series]:
    """Load CB assets levels (PIT) for the active multi-CB set.

    Default currencies: USD, EUR, JPY. GBP only if ``include_gbp`` (thin / ended).
    """
    curs = [c.upper() for c in (currencies or ("USD", "EUR", "JPY"))]
    if include_gbp and "GBP" not in curs:
        curs.append("GBP")
    out: dict[str, pd.Series] = {}
    for c in curs:
        allow = include_gbp or c != "GBP"
        if c == "GBP" and not include_gbp:
            continue
        try:
            out[c] = load_cb_assets(
                c,
                download=download,
                weekly_pub_lag_days=weekly_pub_lag_days,
                monthly_pub_lag_months=monthly_pub_lag_months,
                force=force,
                allow_discontinued=allow and c == "GBP",
            )
        except Exception as exc:  # noqa: BLE001
            # Skip missing / discontinued without killing the panel
            out.setdefault("_errors", pd.Series(dtype=float))
            if "_notes" not in out:
                pass
            print(f"WARN CB assets {c}: {exc}")
    return {k: v for k, v in out.items() if not k.startswith("_")}


def load_cb_yoy_panel(
    currencies: Iterable[str] | None = None,
    *,
    download: bool = True,
    weekly_pub_lag_days: int = DEFAULT_WEEKLY_PUB_LAG_DAYS,
    monthly_pub_lag_months: int = DEFAULT_MONTHLY_PUB_LAG_MONTHS,
    force: bool = False,
    include_gbp: bool = False,
) -> pd.DataFrame:
    """YoY BS growth panel aligned to a weekly UTC calendar (ffill monthly).

    Columns = currency codes. Values = fraction YoY change (0.10 = +10%).
    Index = PIT *known* dates after publication lag.
    """
    levels = load_cb_assets_panel(
        currencies,
        download=download,
        weekly_pub_lag_days=weekly_pub_lag_days,
        monthly_pub_lag_months=monthly_pub_lag_months,
        force=force,
        include_gbp=include_gbp,
    )
    growth: dict[str, pd.Series] = {}
    for ccy, s in levels.items():
        g = yoy_growth(s)
        growth[ccy] = g

    if not growth:
        return pd.DataFrame()

    # Build weekly union index from weekly series; ffill monthly onto it
    weekly_idx = None
    for ccy, s in levels.items():
        if s.attrs.get("frequency") == "weekly":
            weekly_idx = s.index if weekly_idx is None else weekly_idx.union(s.index)
    if weekly_idx is None or len(weekly_idx) == 0:
        # Fallback: monthly union
        idx = None
        for s in growth.values():
            idx = s.index if idx is None else idx.union(s.index)
        panel = pd.DataFrame({c: growth[c] for c in growth}).sort_index()
        panel.attrs["calendar"] = "native"
        return panel

    weekly_idx = pd.DatetimeIndex(weekly_idx).sort_values().unique()
    cols = {}
    for ccy, g in growth.items():
        cols[ccy] = g.reindex(weekly_idx, method="ffill")
    panel = pd.DataFrame(cols).sort_index()
    panel.attrs["calendar"] = "weekly_ffill"
    panel.attrs["weekly_pub_lag_days"] = int(weekly_pub_lag_days)
    panel.attrs["monthly_pub_lag_months"] = int(monthly_pub_lag_months)
    return panel


def load_gdp_series(
    currency: str,
    *,
    download: bool = True,
    pub_lag_quarters: int = DEFAULT_GDP_PUB_LAG_QUARTERS,
    force: bool = False,
) -> pd.Series:
    """Nominal GDP with quarter publication lag."""
    ccy = currency.upper()
    sid = GDP_SERIES.get(ccy)
    if sid is None:
        raise KeyError(f"No GDP mapping for {ccy}")
    if download:
        download_fred_series(sid, force=force)
    s = load_fred_series(sid, download=False)
    s = _apply_quarter_lag(s, pub_lag_quarters)
    s.name = ccy
    s.attrs["source"] = f"fred_{sid}"
    s.attrs["fred_id"] = sid
    return s


def load_bs_gdp_ratio(
    currency: str = "USD",
    *,
    download: bool = True,
    weekly_pub_lag_days: int = DEFAULT_WEEKLY_PUB_LAG_DAYS,
    monthly_pub_lag_months: int = DEFAULT_MONTHLY_PUB_LAG_MONTHS,
    gdp_pub_lag_quarters: int = DEFAULT_GDP_PUB_LAG_QUARTERS,
    force: bool = False,
) -> pd.Series:
    """CB assets / GDP ratio on the assets calendar (ffill GDP).

    Units are heterogeneous across CBs — use **levels only for within-currency
    z-scores / growth**, never for raw cross-currency level sorts.
    """
    assets = load_cb_assets(
        currency,
        download=download,
        weekly_pub_lag_days=weekly_pub_lag_days,
        monthly_pub_lag_months=monthly_pub_lag_months,
        force=force,
        allow_discontinued=(currency.upper() == "GBP"),
    )
    gdp = load_gdp_series(
        currency, download=download, pub_lag_quarters=gdp_pub_lag_quarters, force=force
    )
    gdp_aligned = gdp.reindex(assets.index, method="ffill")
    # Scale heuristics so ratio is O(1)–O(10): WALCL mn / GDP bn → /1000
    ccy = currency.upper()
    if ccy == "USD":
        ratio = assets / (gdp_aligned * 1000.0)  # mn / bn
    elif ccy == "EUR":
        ratio = assets / gdp_aligned  # both mn EUR (EUNNGDP is mn)
    elif ccy == "JPY":
        # JPNASSETS: 100 mn yen; JPNNGDP: bn yen → assets*100 / (gdp*1000) = assets/(gdp*10)
        ratio = assets / (gdp_aligned * 10.0)
    else:
        ratio = assets / gdp_aligned.replace(0.0, pd.NA)
    ratio = ratio.replace([float("inf"), float("-inf")], pd.NA).dropna()
    ratio.name = f"{ccy}_bs_gdp"
    ratio.attrs["source"] = f"fred_{assets.attrs.get('fred_id')}/{gdp.attrs.get('fred_id')}"
    ratio.attrs["frequency"] = assets.attrs.get("frequency", "")
    return ratio


def us_vs_peer_yoy_diff(
    yoy_panel: pd.DataFrame,
    *,
    peers: Iterable[str] = ("EUR", "JPY"),
) -> pd.Series:
    """US YoY BS growth minus equal-weight peer YoY (portfolio-balance differential)."""
    if "USD" not in yoy_panel.columns:
        raise KeyError("yoy_panel missing USD")
    peer_cols = [c for c in peers if c in yoy_panel.columns]
    if not peer_cols:
        raise KeyError("no peer columns in yoy_panel")
    peer_mean = yoy_panel[peer_cols].mean(axis=1)
    diff = yoy_panel["USD"] - peer_mean
    diff.name = "us_peer_bs_yoy_diff"
    return diff


def cb_coverage(yoy_panel: pd.DataFrame) -> pd.DataFrame:
    """Small coverage table for reports."""
    rows = []
    for c in yoy_panel.columns:
        s = yoy_panel[c].dropna()
        meta = CB_ASSETS_SERIES.get(c, ("?", "?"))
        rows.append(
            {
                "currency": c,
                "fred_id": meta[0],
                "frequency": meta[1],
                "n_obs": int(len(s)),
                "start": str(s.index.min().date()) if len(s) else None,
                "end": str(s.index.max().date()) if len(s) else None,
            }
        )
    return pd.DataFrame(rows)
