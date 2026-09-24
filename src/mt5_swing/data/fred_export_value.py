"""FRED / OECD MEI merchandise export-value YoY (XTEXVA01) panels.

Literature
----------
- Dahlquist & Hasseltoft (2020, JFE) economic-momentum / growth-channel spirit
  applied to **external demand / export-growth differentials** (trade intensity).
- Currencies with *high* relative merchandise export-value YoY subsequently
  appreciate vs low-export-growth peers (external-demand prior).
- Distinct from: trade-balance §30 (TB/exports ratio), CA §21, ToT (UV 404),
  retail §46, IP §42, passenger-cars §60, construction §59, GDP §58.

Fall-through (§61 probe order, 2026-09-24)
-----------------------------------------
1. **CIP / cross-currency basis** — still **Missing** free G10 panel
   (Du–Tepper–Verdelhan); rates+spot CIP-implied FD already §19; do not fake
   basis → fall through.
2. **Hours** ``HOHWMN*`` — US-only / 404 abroad; **vacancies** ``LMJVTTUV*`` —
   partial + stale ~2023-24, JP/CA/NZ thin → document fall-through.
3. **Services production** ``PRSVTT*`` / **mfg turnover** ``ODMN*`` — 404.
4. **This wave:** OECD MEI merchandise export-value YoY ``XTEXVA01{ISO2}M657S``.

Free data (verified live on FRED probe 2026-09-24)
-------------------------------------------------
| Ccy | FRED id (primary)  | Freq | Unit / notes                                      |
|-----|--------------------|------|---------------------------------------------------|
| USD | XTEXVA01USM657S    | M    | YoY % as reported (~2026-06)                      |
| EUR | XTEXVA01DEM657S    | M    | **Germany proxy** (EZM657S 404; EZM667S stale)    |
| GBP | XTEXVA01GBM657S    | M    | YoY %                                             |
| JPY | XTEXVA01JPM657S    | M    | YoY %                                             |
| CAD | XTEXVA01CAM657S    | M    | YoY %                                             |
| AUD | XTEXVA01AUM657S    | M    | YoY %                                             |
| CHF | XTEXVA01CHM657S    | M    | YoY %                                             |
| NZD | XTEXVA01NZM657S    | M    | YoY %                                             |

Full G10 mapped. Companion levels ``*M667S`` exist (export **value** in currency
units — NOT volumes; do **not** mix levels into score panel; YoY 657S is primary).

Tried / documented (not primary):
- ``XTEXVA01EZM657S`` — **404** → Germany proxy a priori.
- ``XTEXVA01EZM667S`` — EA levels stale ~2023-04 (not mixed into YoY panel).
- ``XTEXVA01*M667S`` levels — value units, not volumes; not mixed into scores.
- CIP / cross-currency basis free G10 panel — Missing (do not fake).
- Hours ``HOHWMN*`` / vacancies ``LMJVTTUV*`` — thin/stale fall-through.
- Services production ``PRSVTT*`` / mfg turnover ``ODMN*`` — 404.

Unit choice (fixed a priori)
----------------------------
Panel stores **merchandise export-value YoY %** as reported by OECD M657S.
Strategy scores XS-rank on YoY. Acceleration = Δ12 of that YoY.
**Series unit = value (not volume).**

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Trade / export-value typically lands ~1–2 months after reference (same band as
IP §42 / retail §46 / cars §60). Default ``pub_lag_months=2``. Observation dated
month-start ``t`` is first known at ``t + 2 months``. Strategy modules add
``signal_lag=1`` month + a **1 trading-day** weight lag (mirror GDP §58 /
construction §59 / cars §60).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED OECD MEI merchandise export-value YoY (XTEXVA01 *M657S)
XVAL_SERIES: dict[str, str | None] = {
    "USD": "XTEXVA01USM657S",
    "EUR": "XTEXVA01DEM657S",  # Germany proxy; EZM657S 404; EZM667S stale ~2023-04
    "GBP": "XTEXVA01GBM657S",
    "JPY": "XTEXVA01JPM657S",
    "CAD": "XTEXVA01CAM657S",
    "AUD": "XTEXVA01AUM657S",
    "CHF": "XTEXVA01CHM657S",
    "NZD": "XTEXVA01NZM657S",
}

XVAL_FREQ: dict[str, str] = {
    "USD": "M",
    "EUR": "M",
    "GBP": "M",
    "JPY": "M",
    "CAD": "M",
    "AUD": "M",
    "CHF": "M",
    "NZD": "M",
}

XVAL_NATIVE_UNIT: dict[str, str] = {
    "USD": "yoy_pct_m657s",
    "EUR": "yoy_pct_m657s",
    "GBP": "yoy_pct_m657s",
    "JPY": "yoy_pct_m657s",
    "CAD": "yoy_pct_m657s",
    "AUD": "yoy_pct_m657s",
    "CHF": "yoy_pct_m657s",
    "NZD": "yoy_pct_m657s",
}

XVAL_SCORE_UNIT = "yoy_pct"

XVAL_FAILED_OR_ALT: dict[str, list[str]] = {
    "cip_cross_currency_basis_missing_free_g10_fallthrough": [
        "Du–Tepper–Verdelhan CIP/basis — no free multi-year G10 panel",
        "rates+spot CIP-implied FD already §19 — do not fake basis",
    ],
    "hours_hohwmn_us_only_or_404_fallthrough": [
        "HOHWMN02USM065S",
        "HOHWMN02USQ065S",
        "AWHMAN",
        "AWHAE",
    ],
    "vacancies_lmjvttuv_partial_stale_fallthrough": [
        "LMJVTTUVUSM647S",
        "LMJVTTUVGBM647S",
        "LMJVTTUVDEM647S",
        "LMJVTTUVCHM647S",
        "LMJVTTUVAUM647S",
        "JTSJOL",
        "JTSJOR",
    ],
    "services_production_prsvtt_404_fallthrough": [
        "USAPRSVTT01GYSAM",
        "CANPRSVTT01GYSAM",
        "DEUPRSVTT01GYSAM",
        "GBRPRSVTT01GYSAM",
    ],
    "mfg_turnover_odmn_404_fallthrough": [
        "USAODMNTO01GYSAM",
        "CANODMNTO01GYSAM",
        "DEUODMNTO01GYSAM",
        "GBRODMNTO01GYSAM",
    ],
    "eur_ez_m657s_404_germany_proxy": [
        "XTEXVA01EZM657S",  # 404
        "XTEXVA01DEM657S",  # Germany proxy (primary)
    ],
    "eur_ez_m667s_stale_2023_04": ["XTEXVA01EZM667S"],
    "levels_m667s_value_not_volume_not_mixed": [
        "XTEXVA01USM667S",
        "XTEXVA01DEM667S",
        "XTEXVA01GBM667S",
        "XTEXVA01JPM667S",
        "XTEXVA01CAM667S",
        "XTEXVA01AUM667S",
        "XTEXVA01CHM667S",
        "XTEXVA01NZM667S",
    ],
    "distinct_channels_not_this_wave": [
        "BOPGTB",  # trade-balance §30
        "AUSB6BLTT02STSAQ",  # CA §21
        "USAPRINTO01GYSAM",  # IP §42
        "USASLRTCR03GYSAM",  # passenger-cars §60
        "USAPRCNTO01GYSAM",  # construction §59
        "NAEXKP01USQ657S",  # GDP §58
        "SLRTTO01USA657S",  # retail §46
    ],
}

EUR_GERMANY_PROXY = "XTEXVA01DEM657S"
EUR_EZ_M657S_404 = "XTEXVA01EZM657S"
EUR_EZ_M667S_STALE = "XTEXVA01EZM667S"

DEFAULT_PUB_LAG_MONTHS = 2  # a priori: trade ~1–2m (same band as IP/retail/cars)


def _to_month_start(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = pd.DatetimeIndex(s.index)
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    ms = pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start")).tz_localize("UTC")
    s = s.copy()
    s.index = ms
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _load_series(series_id: str, *, download: bool, force: bool) -> pd.Series:
    if download:
        download_fred_series(series_id, force=force)
    return _to_month_start(load_fred_series(series_id, download=False))


def _apply_pub_lag_monthly(df: pd.DataFrame, *, pub_lag_months: int) -> pd.DataFrame:
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


def load_export_value_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """OECD MEI merchandise export-value YoY % → monthly PIT panel.

    Columns = ISO currency codes. Values = export-value **YoY %** (value, not
    volume). Index = UTC month-start of the PIT *known* date after
    ``pub_lag_months``. Full G10 mapped; EUR = Germany proxy.
    """
    curs = [c.upper() for c in (currencies or list(XVAL_SERIES.keys()))]
    monthly_cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}
    freq_map: dict[str, str] = {}
    native_map: dict[str, str] = {}

    for c in curs:
        sid = XVAL_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_xtexva01"
            continue
        try:
            s = _load_series(sid, download=download, force=force)
            series_map[c] = sid
            freq_map[c] = XVAL_FREQ.get(c, "M")
            native_map[c] = XVAL_NATIVE_UNIT.get(c, "yoy_pct")
            if c == "EUR":
                notes[c] = (
                    f"Germany proxy {sid} (M YoY % value); "
                    f"EZM657S 404; EZM667S stale ~2023-04"
                )
            else:
                notes[c] = f"OECD MEI merchandise export-value YoY {sid} (monthly; value not volume)"
            monthly_cols[c] = s
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    if not monthly_cols:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(monthly_cols).sort_index()
        df = _apply_pub_lag_monthly(df, pub_lag_months=pub_lag_months)
        df = df.dropna(how="all")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

    stale_note_parts = [
        "XTEXVA01EZM657S 404; XTEXVA01EZM667S stale ~2023-04",
        "EUR=XTEXVA01DEM657S Germany proxy a priori",
        "series unit = export VALUE YoY (not volume); *M667S levels not mixed",
        "CIP/cross-currency basis free G10 Missing → fall through",
        "hours HOHWMN* US-only/404 + vacancies LMJVTTUV* thin/stale → fall through",
        "services PRSVTT* / mfg turnover ODMN* 404 → fall through",
    ]

    df.attrs["factor"] = "oecd_mei_merchandise_export_value_xtexva01"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["freq_map"] = freq_map
    df.attrs["native_unit_map"] = native_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI XTEXVA01{ISO2}M657S merchandise export-value YoY "
        "(full G10; EUR=XTEXVA01DEM657S Germany proxy — EZM657S 404, "
        "EZM667S stale ~2023-04; unit=VALUE not volume; *M667S levels not mixed; "
        "CIP/basis Missing fallthrough; hours/vacancies thin fallthrough; "
        "PRSVTT*/ODMN* 404 fallthrough)"
    )
    df.attrs["unit"] = XVAL_SCORE_UNIT
    df.attrs["score_basis"] = "export_value_yoy_pct"
    df.attrs["unmapped"] = [c for c in curs if c not in series_map]
    df.attrs["failed_or_alt"] = XVAL_FAILED_OR_ALT
    df.attrs["stale"] = False  # full G10 live; EUR EZ documented separately
    df.attrs["stale_note"] = "; ".join(stale_note_parts)
    df.attrs["fallthrough_from"] = [
        "cip_cross_currency_basis_missing_free_g10",
        "hours_vacancies_thin_or_stale",
        "services_production_mfg_turnover_404",
    ]
    df.attrs["series_unit_caveat"] = "value_not_volume"
    return df


def load_us_export_value(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US merchandise export-value YoY % (XTEXVA01USM657S), PIT-lagged."""
    panel = load_export_value_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US merchandise export-value series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_XVAL_YOY"
    out.attrs["series_id"] = XVAL_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def export_value_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_export_value_panel(download=False)
    rows = []
    notes = panel.attrs.get("notes", {})
    smap = panel.attrs.get("series_map", {})
    fmap = panel.attrs.get("freq_map", {})
    for c in panel.columns:
        s = panel[c].dropna()
        rows.append(
            {
                "currency": c,
                "fred_id": smap.get(c, ""),
                "freq": fmap.get(c, ""),
                "n_obs": int(len(s)),
                "start": str(s.index.min().date()) if len(s) else None,
                "end": str(s.index.max().date()) if len(s) else None,
                "last": float(s.iloc[-1]) if len(s) else float("nan"),
                "note": notes.get(c, ""),
            }
        )
    for c in panel.attrs.get("unmapped", []):
        if c in panel.columns:
            continue
        rows.append(
            {
                "currency": c,
                "fred_id": "",
                "freq": "",
                "n_obs": 0,
                "start": None,
                "end": None,
                "last": float("nan"),
                "note": notes.get(c, "unmapped_on_fred_xtexva01"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
