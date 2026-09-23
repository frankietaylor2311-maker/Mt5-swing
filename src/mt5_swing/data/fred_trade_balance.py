"""FRED / OECD MEI monthly trade-balance panels for external-balance FX (free, PIT).

Literature
----------
- Della Corte, Riddiough & Sarno (2016), "Currency Premia and Global Imbalances,"
  *RFS* — debtor / external-imbalance currencies earn a risk premium.
- Gourinchas & Rey (2007), "International Financial Adjustment," *JPE* — US
  trade / external imbalance channel forecasts dollar adjustment.
- Higher-frequency cousin of quarterly CA/GDP (§21): monthly goods trade
  (exports − imports) reacts faster than IMF BOP CA.

Free data (no Bloomberg / OECD portal key)
-----------------------------------------
OECD MEI merchandise **exports** / **imports** (USD, SA) via FRED
``XTEXVA01{cc}M667S`` / ``XTIMVA01{cc}M667S``. Trade balance level =
exports − imports; **primary normalisation** = TB / exports =
``(EXP − IMP) / EXP`` so XS ranks are cross-sectionally comparable
(dimensionless surplus share of exports).

| Ccy | Exports              | Imports              | Notes                         |
|-----|----------------------|----------------------|-------------------------------|
| USD | XTEXVA01USM667S      | XTIMVA01USM667S      | + BOPGSTB overlay for US tilt |
| EUR | XTEXVA01DEM667S      | XTIMVA01DEM667S      | Germany proxy (EZM ends 2023) |
| GBP | XTEXVA01GBM667S      | XTIMVA01GBM667S      |                               |
| JPY | XTEXVA01JPM667S      | XTIMVA01JPM667S      |                               |
| CAD | XTEXVA01CAM667S      | XTIMVA01CAM667S      |                               |
| AUD | XTEXVA01AUM667S      | XTIMVA01AUM667S      |                               |
| NZD | XTEXVA01NZM667S      | XTIMVA01NZM667S      | mapped (unlike debt/fiscal)   |
| CHF | XTEXVA01CHM667S      | XTIMVA01CHM667S      | mapped (unlike debt/fiscal)   |

Tried and noted:
- ``XTEXVA01EZM667S`` / ``XTIMVA01EZM667S`` — live but ends ~2023-04; unused as
  primary (Germany proxy preferred for continuity, parallel to fiscal/debt).
- ``USAB6BLTT02STSAM`` (monthly CA) — **404** on FRED.
- ``BOPB12`` — **404**.

US goods+services overlay (US tilt legs only):
- ``BOPGSTB`` — Balance on Goods and Services (millions USD, monthly).

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Monthly OECD / BEA trade is typically released ~4–8 weeks after month-end.
Default ``pub_lag_months=2``: observation dated month-start ``t`` is first
known at ``t + 2 months``. Strategy modules add ``signal_lag`` months on top.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → (exports FRED id, imports FRED id)
TB_EXPORT_IMPORT: dict[str, tuple[str, str] | None] = {
    "USD": ("XTEXVA01USM667S", "XTIMVA01USM667S"),
    "EUR": ("XTEXVA01DEM667S", "XTIMVA01DEM667S"),  # Germany proxy
    "GBP": ("XTEXVA01GBM667S", "XTIMVA01GBM667S"),
    "JPY": ("XTEXVA01JPM667S", "XTIMVA01JPM667S"),
    "CAD": ("XTEXVA01CAM667S", "XTIMVA01CAM667S"),
    "AUD": ("XTEXVA01AUM667S", "XTIMVA01AUM667S"),
    "NZD": ("XTEXVA01NZM667S", "XTIMVA01NZM667S"),
    "CHF": ("XTEXVA01CHM667S", "XTIMVA01CHM667S"),
}

# Documented alternate / failed mnemonics
TB_FAILED_OR_ALT: dict[str, list[str]] = {
    "eur_ez_ends_2023": ["XTEXVA01EZM667S", "XTIMVA01EZM667S"],
    "monthly_ca_404": ["USAB6BLTT02STSAM"],
    "bopb12_404": ["BOPB12"],
}

US_GOODS_SERVICES_TB = "BOPGSTB"  # millions USD, monthly
EUR_EZ_EXPORTS = "XTEXVA01EZM667S"
EUR_EZ_IMPORTS = "XTIMVA01EZM667S"

DEFAULT_PUB_LAG_MONTHS = 2  # conservative monthly trade release lag


def _to_month_start(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = pd.DatetimeIndex(s.index)
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    ms = pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start")).tz_localize("UTC")
    s = s.copy()
    s.index = ms
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _load_monthly(series_id: str, *, download: bool, force: bool) -> pd.Series:
    if download:
        download_fred_series(series_id, force=force)
    return _to_month_start(load_fred_series(series_id, download=False))


def _apply_pub_lag(df: pd.DataFrame, *, pub_lag_months: int) -> pd.DataFrame:
    if df.empty or pub_lag_months <= 0:
        return df
    out = df.copy()
    out.index = out.index + pd.DateOffset(months=int(pub_lag_months))
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def _tb_ratio_from_exp_imp(exports: pd.Series, imports: pd.Series) -> pd.Series:
    """(EXP − IMP) / EXP — surplus share of exports (NaN if EXP≈0)."""
    common = exports.index.intersection(imports.index)
    ex = exports.loc[common].astype(float)
    im = imports.loc[common].astype(float)
    denom = ex.replace(0.0, pd.NA)
    ratio = (ex - im) / denom
    ratio = pd.to_numeric(ratio, errors="coerce")
    ratio.name = "tb_over_exports"
    return ratio.dropna()


def load_trade_balance_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Monthly TB/exports panel, publication-lagged.

    Columns = ISO currency codes. Values = (exports − imports) / exports.
    Index = UTC month-start of the PIT *known* date after ``pub_lag_months``.
    """
    curs = [c.upper() for c in (currencies or TB_EXPORT_IMPORT.keys())]
    cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}

    for c in curs:
        pair = TB_EXPORT_IMPORT.get(c)
        if pair is None:
            notes[c] = "unmapped_on_fred_xtexva"
            continue
        exp_id, imp_id = pair
        try:
            ex = _load_monthly(exp_id, download=download, force=force)
            im = _load_monthly(imp_id, download=download, force=force)
            ratio = _tb_ratio_from_exp_imp(ex, im)
            series_map[c] = f"{exp_id}-{imp_id}"
            if c == "EUR":
                notes[c] = (
                    f"Germany proxy via {exp_id}/{imp_id}; "
                    "EZM exports/imports end ~2023-04 (documented alt)"
                )
            cols[c] = ratio
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    df = pd.DataFrame(cols).sort_index()
    df = _apply_pub_lag(df, pub_lag_months=pub_lag_months)

    df.attrs["factor"] = "trade_balance_over_exports"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI XTEXVA01*/XTIMVA01* M667S; TB/exports = (EXP-IMP)/EXP"
    )
    df.attrs["unit"] = "tb_over_exports"
    df.attrs["unmapped"] = [c for c in curs if c not in cols]
    df.attrs["failed_or_alt"] = TB_FAILED_OR_ALT
    return df


def load_us_trade_balance(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US TB/exports from OECD MEI panel (same units as cross-section)."""
    panel = load_trade_balance_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US trade-balance series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_TB_OVER_EXPORTS"
    out.attrs["series_id"] = TB_EXPORT_IMPORT["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def load_us_bopgstb(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US Balance on Goods and Services (BOPGSTB, millions USD), PIT-lagged.

    Higher-quality US-only overlay (includes services) for GR / haven tilts.
    """
    s = _load_monthly(US_GOODS_SERVICES_TB, download=download, force=force)
    if pub_lag_months > 0:
        s = s.copy()
        s.index = s.index + pd.DateOffset(months=int(pub_lag_months))
        s = s[~s.index.duplicated(keep="last")].sort_index()
    out = s.dropna()
    out.name = "US_BOPGSTB"
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    out.attrs["series_id"] = US_GOODS_SERVICES_TB
    out.attrs["unit"] = "millions_usd"
    return out


def trade_balance_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_trade_balance_panel(download=False)
    rows = []
    notes = panel.attrs.get("notes", {})
    smap = panel.attrs.get("series_map", {})
    for c in panel.columns:
        s = panel[c].dropna()
        rows.append(
            {
                "currency": c,
                "fred_id": smap.get(c, ""),
                "n_obs": int(len(s)),
                "start": str(s.index.min().date()) if len(s) else None,
                "end": str(s.index.max().date()) if len(s) else None,
                "last": float(s.iloc[-1]) if len(s) else float("nan"),
                "note": notes.get(c, ""),
            }
        )
    for c in panel.attrs.get("unmapped", []):
        rows.append(
            {
                "currency": c,
                "fred_id": "",
                "n_obs": 0,
                "start": None,
                "end": None,
                "last": float("nan"),
                "note": notes.get(c, "unmapped_on_fred_xtexva"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
