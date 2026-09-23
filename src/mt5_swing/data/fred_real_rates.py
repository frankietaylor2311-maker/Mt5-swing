"""FRED real-rate / breakeven inflation panels for FX (free, PIT).

Literature
----------
- Frankel (1979), "On the Mark: A Theory of Floating Exchange Rates Based on
  Real Interest Differentials," *AER* — real interest differential model of FX.
- Meese & Rogoff (1988), "Was It Real? The Exchange Rate-Interest Differential
  Relation over the Modern Floating-Rate Period," *JF* — classic RID tests.
- Dahlquist & Hasseltoft (2013), "International Bond Risk Premia," *JIE* —
  real / local-currency bond premia and FX.
- Lustig, Stathopoulos & Verdelhan (2019), "The Term Structure of Currency
  Carry Trade Risk Premia," *JF* — real vs nominal term structure of carry.
- Hofmann, Shim & Shin (2020/22 BIS), bond risk premia / real rates and FX
  (EM focus; G10 channel still informative for USD state).

Free FRED series (no Bloomberg / paid linker panels)
----------------------------------------------------
| Series  | Freq  | Meaning                         |
|---------|-------|---------------------------------|
| DFII10  | daily | US 10y TIPS real yield (% p.a.)  |
| T10YIE  | daily | US 10y breakeven inflation (%)   |
| DGS10   | daily | US 10y nominal Treasury (%)      |

Foreign real-rate *proxy* (honest — not true inflation-linked yields where
unavailable): OECD LT govt yield (``IRLTLT01*``) minus CPI YoY from the
macro-diff CPI loader. US true linker (DFII10) is preferred for USD state
and for the USD leg of XS differentials.

PIT lags (conservative, frozen — not holdout-tuned)
---------------------------------------------------
- Daily TIPS / breakeven / DGS10: ``pub_lag_days=1`` (close known next session).
- Monthly LT + CPI YoY: ``pub_lag_months=1`` (via existing loaders).

Distinct from nominal yield-curve (§16), PPP/CPI real FX, and CB-BS QE.
Do **not** overlay on the locked FTMO sleeve.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_macro_diff import load_cpi_yoy_panel
from mt5_swing.data.fred_rates import (
    apply_publication_lag,
    download_fred_series,
    load_fred_series,
    macro_dir,
)
from mt5_swing.data.fred_yields import FRED_LT_GOVT_MONTHLY, load_lt_yield_panel

# Daily US real / breakeven / nominal
US_REAL_SERIES = "DFII10"
US_BREAKEVEN_SERIES = "T10YIE"
US_NOMINAL_10Y = "DGS10"

DEFAULT_DAILY_PUB_LAG_DAYS = 1
DEFAULT_MONTHLY_PUB_LAG_MONTHS = 1


def _ensure_utc(s: pd.Series) -> pd.Series:
    out = s.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out.index = idx
    return out[~out.index.duplicated(keep="last")].sort_index()


def _to_month_end_utc(s: pd.Series) -> pd.Series:
    """Resample daily → month-end last observation (UTC)."""
    out = _ensure_utc(s)
    m = out.resample("ME").last()
    m = m[~m.index.duplicated(keep="last")].sort_index()
    return m


def load_us_real_rate(
    *,
    download: bool = True,
    pub_lag_days: int = DEFAULT_DAILY_PUB_LAG_DAYS,
    force: bool = False,
) -> pd.Series:
    """US 10y TIPS real yield (DFII10), publication-lagged."""
    if download or force:
        download_fred_series(US_REAL_SERIES, force=force)
    s = apply_publication_lag(
        _ensure_utc(load_fred_series(US_REAL_SERIES, download=False)),
        lag_days=pub_lag_days,
    )
    s.name = "DFII10"
    s.attrs["source"] = "fred_tips_real_10y"
    s.attrs["pub_lag_days"] = int(pub_lag_days)
    s.attrs["units"] = "percent_per_annum"
    return s


def load_us_breakeven(
    *,
    download: bool = True,
    pub_lag_days: int = DEFAULT_DAILY_PUB_LAG_DAYS,
    force: bool = False,
) -> pd.Series:
    """US 10y breakeven inflation (T10YIE), publication-lagged."""
    if download or force:
        download_fred_series(US_BREAKEVEN_SERIES, force=force)
    s = apply_publication_lag(
        _ensure_utc(load_fred_series(US_BREAKEVEN_SERIES, download=False)),
        lag_days=pub_lag_days,
    )
    s.name = "T10YIE"
    s.attrs["source"] = "fred_breakeven_10y"
    s.attrs["pub_lag_days"] = int(pub_lag_days)
    s.attrs["units"] = "percent"
    return s


def load_us_nominal_10y(
    *,
    download: bool = True,
    pub_lag_days: int = DEFAULT_DAILY_PUB_LAG_DAYS,
    force: bool = False,
) -> pd.Series:
    """US 10y nominal Treasury (DGS10), publication-lagged."""
    if download or force:
        download_fred_series(US_NOMINAL_10Y, force=force)
    s = apply_publication_lag(
        _ensure_utc(load_fred_series(US_NOMINAL_10Y, download=False)),
        lag_days=pub_lag_days,
    )
    s.name = "DGS10"
    s.attrs["source"] = "fred_nominal_10y"
    s.attrs["pub_lag_days"] = int(pub_lag_days)
    return s


def load_foreign_real_proxy_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_MONTHLY_PUB_LAG_MONTHS,
    download: bool = True,
) -> pd.DataFrame:
    """Foreign real-rate *proxy* = OECD LT govt − CPI YoY (% p.a.).

    Honest limitation: these are **not** inflation-linked (linker) real yields
    where national TIPS markets are thin / absent on free FRED. Construction
    matches common research proxies when linkers are unavailable.
    """
    curs = [c.upper() for c in (currencies or FRED_LT_GOVT_MONTHLY.keys())]
    # Need foreign + USD for differentials; load all requested
    lt = load_lt_yield_panel(curs, pub_lag_months=pub_lag_months, download=download)
    cpi = load_cpi_yoy_panel(curs, pub_lag_months=pub_lag_months, download=download)
    common = sorted(set(lt.columns) & set(cpi.columns))
    if not common:
        raise RuntimeError("No overlap between LT yields and CPI YoY panels")
    # Align on month-start intersection
    idx = lt.index.intersection(cpi.index)
    lt_a = lt.reindex(idx)[common]
    cpi_a = cpi.reindex(idx)[common]
    # CPI YoY from loader is already in percent terms (e.g. 2.5 = 2.5%)
    real = lt_a - cpi_a
    real.attrs["definition"] = "lt_govt_minus_cpi_yoy_proxy_not_linker"
    real.attrs["pub_lag_months"] = int(pub_lag_months)
    real.attrs["source"] = "fred_oecd_lt_minus_cpi_yoy"
    real.attrs["honest_note"] = (
        "Foreign real rates are LT−CPI YoY proxies, not TIPS/linker yields"
    )
    return real


def load_real_rate_panels(
    currencies: Iterable[str] | None = None,
    *,
    daily_pub_lag_days: int = DEFAULT_DAILY_PUB_LAG_DAYS,
    monthly_pub_lag_months: int = DEFAULT_MONTHLY_PUB_LAG_MONTHS,
    download: bool = True,
) -> dict[str, object]:
    """Bundle US daily real/BE and monthly proxy real panel + differentials.

    Returns keys:
      us_real_d, us_be_d, us_nom_d,
      real_proxy_m (incl. USD LT−CPI),
      us_tips_m (DFII10 month-end),
      rr_diff_vs_usd (foreign proxy − US TIPS month-end),
      rr_diff_proxy_vs_usd (foreign proxy − US LT−CPI proxy).
    """
    curs = [c.upper() for c in (currencies or FRED_LT_GOVT_MONTHLY.keys())]
    if "USD" not in curs:
        curs = ["USD"] + list(curs)

    us_real_d = load_us_real_rate(download=download, pub_lag_days=daily_pub_lag_days)
    us_be_d = load_us_breakeven(download=download, pub_lag_days=daily_pub_lag_days)
    us_nom_d = load_us_nominal_10y(download=download, pub_lag_days=daily_pub_lag_days)

    real_proxy_m = load_foreign_real_proxy_panel(
        curs, pub_lag_months=monthly_pub_lag_months, download=download
    )
    us_tips_m = _to_month_end_utc(us_real_d)
    us_tips_m.name = "DFII10_ME"

    # Align tips month-end to month-start key used by OECD panels
    tips_ms = us_tips_m.copy()
    tips_ms.index = (
        pd.DatetimeIndex(tips_ms.index)
        .tz_convert(None)
        .to_period("M")
        .to_timestamp(how="start")
        .tz_localize("UTC")
    )
    tips_ms = tips_ms[~tips_ms.index.duplicated(keep="last")].sort_index()

    foreign = [c for c in real_proxy_m.columns if c != "USD"]
    # Primary XS: foreign proxy − US true TIPS
    rr_diff = real_proxy_m[foreign].sub(tips_ms.reindex(real_proxy_m.index), axis=0)
    rr_diff.attrs["definition"] = "foreign_lt_cpi_proxy_minus_us_dfii10"
    rr_diff.attrs["pub_lag_months"] = int(monthly_pub_lag_months)

    # Secondary: both sides LT−CPI proxy (apples-to-apples construction)
    if "USD" in real_proxy_m.columns:
        usd_proxy = real_proxy_m["USD"]
        rr_diff_proxy = real_proxy_m[foreign].sub(usd_proxy, axis=0)
        rr_diff_proxy.attrs["definition"] = "foreign_lt_cpi_minus_us_lt_cpi"
    else:
        rr_diff_proxy = rr_diff.copy()
        rr_diff_proxy.attrs["definition"] = "fallback_same_as_tips_diff"

    return {
        "us_real_d": us_real_d,
        "us_be_d": us_be_d,
        "us_nom_d": us_nom_d,
        "real_proxy_m": real_proxy_m,
        "us_tips_m": us_tips_m,
        "rr_diff_vs_usd": rr_diff,
        "rr_diff_proxy_vs_usd": rr_diff_proxy,
    }


def real_rate_coverage(
    panels: dict[str, object] | None = None,
    *,
    download: bool = False,
) -> pd.DataFrame:
    """Document coverage / staleness for real-rate / breakeven series."""
    if panels is None:
        panels = load_real_rate_panels(download=download)
    rows = []
    for key, label in (
        ("us_real_d", "DFII10"),
        ("us_be_d", "T10YIE"),
        ("us_nom_d", "DGS10"),
    ):
        s = panels[key]
        assert isinstance(s, pd.Series)
        rows.append(
            {
                "series": label,
                "kind": "us_daily",
                "n_obs": int(s.notna().sum()),
                "start": str(s.dropna().index.min().date()) if s.notna().any() else None,
                "end": str(s.dropna().index.max().date()) if s.notna().any() else None,
                "note": "true linker / market BE",
            }
        )
    proxy = panels["real_proxy_m"]
    assert isinstance(proxy, pd.DataFrame)
    for c in proxy.columns:
        s = proxy[c].dropna()
        rows.append(
            {
                "series": f"LT-CPI_{c}",
                "kind": "proxy_monthly",
                "n_obs": int(len(s)),
                "start": str(s.index.min().date()) if len(s) else None,
                "end": str(s.index.max().date()) if len(s) else None,
                "note": "proxy_not_linker" if c != "USD" else "US_proxy_LT_minus_CPI",
            }
        )
    return pd.DataFrame(rows)


def download_default_real_rate_panel(*, force: bool = False) -> dict[str, object]:
    """Ensure DFII10 / T10YIE / DGS10 CSVs exist under ``data/macro/``."""
    paths = {}
    for sid in (US_REAL_SERIES, US_BREAKEVEN_SERIES, US_NOMINAL_10Y):
        paths[sid] = download_fred_series(sid, force=force)
    _ = macro_dir()
    return paths
