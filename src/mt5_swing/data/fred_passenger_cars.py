"""FRED / OECD MEI passenger-car registrations (SLRTCR03) panels.

Literature
----------
- Dahlquist & Hasseltoft (2020, JFE) economic-momentum spirit on real-activity
  differentials → FX, applied to **durable consumer demand** (passenger cars).
- Growth-channel / Taylor-rule FX on auto registrations as a leading/coincident
  demand indicator (distinct from total retail volume).
- Distinct from: retail-sales §46 (SLRTTO01 / RSAFS — broad retail volume),
  building-permits §43 (ODCNPI03 — housing *flow*), construction §59
  (PRCNTO01 — construction *output*), IP §42, BCI §39, CLI §37, house-price §35.

Fall-through (§60 probe order, 2026-09-24)
-----------------------------------------
1. **Housing starts / completions** — OECD ``*ODCNPI02*`` / ``*ODCNPI04*`` all
   **404** on FRED; US ``HOUST`` / ``COMPUTSA`` live but foreign panel <~4 →
   document and fall through (distinct from permits §43 which already boarded
   ODCNPI03).
2. **Manufacturing new orders / inventories** — OECD ``*ODMNNO*`` / ``*ODMNTO*``
   / ``*ODMNIN*`` **404**; US ``AMTMNO`` / ``DGORDER`` / ``ISRATIO`` live only →
   thin panel → fall through.
3. **This wave:** OECD MEI passenger-car registrations YoY ``{ISO3}SLRTCR03GYSAM``.

Free data (verified live on FRED probe 2026-09-24)
-------------------------------------------------
| Ccy | FRED id (primary)    | Freq | Unit / notes                                      |
|-----|----------------------|------|---------------------------------------------------|
| USD | USASLRTCR03GYSAM     | M    | YoY % as reported (~2026-07)                      |
| EUR | ESPSLRTCR03GYSAM     | M    | YoY % Spain proxy (DEU/FRA GYSAM 404; EA19 stale) |
| GBP | GBRSLRTCR03GYSAM     | M    | YoY %                                             |
| JPY | JPNSLRTCR03GYSAM     | M    | YoY %                                             |
| CAD | CANSLRTCR03GYSAM     | M    | YoY % (~2026-06)                                  |
| AUD | AUSSLRTCR03GYSAM     | M    | YoY % (~2026-08)                                  |
| NZD | —                    | —    | **unmapped** (NZLSLRTCR03GYSAM stale ~2021-04)    |
| CHF | —                    | —    | **unmapped** (CHESLRTCR03GYSAM stale ~2018-12)    |

Tried / documented (not primary):
- ``DEUSLRTCR03GYSAM`` / ``FRASLRTCR03GYSAM`` — **404**.
- ``EA19SLRTCR03GYSAM`` — ends ~**2018-12** (stale) → Spain EUR proxy a priori.
- ``NZLSLRTCR03GYSAM`` stale ~2021-04; ``CHESLRTCR03GYSAM`` stale ~2018-12.
- ``SLRTCR03*Q657S`` QoQ family — stale ~2018 across G10 (not used).
- US ``TOTALSA`` / ``ALTSALES`` national unit sales — live alts, not mixed into
  OECD YoY panel (USA GYSAM live).
- Hours ``HOHWMN*`` / vacancies ``LMJVTTUV*`` — mostly 404/stale (labour alts
  documented under LFP/ER — not this wave).

Unit choice (fixed a priori)
----------------------------
Panel stores **passenger-car registrations YoY %** as reported by OECD GYSAM.
Strategy scores XS-rank on YoY. Acceleration = Δ12 of that YoY.

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Auto / retail registrations typically land ~1–2 months after reference (same
band as retail §46 / IP §42). Default ``pub_lag_months=2``. Observation dated
month-start ``t`` is first known at ``t + 2 months``. Strategy modules add
``signal_lag=1`` month + a **1 trading-day** weight lag (mirror GDP §58 /
construction §59 — *not* older IP §42 zero-month signal lag).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED OECD MEI passenger-car registrations YoY (SLRTCR03 GYSAM)
CARS_SERIES: dict[str, str | None] = {
    "USD": "USASLRTCR03GYSAM",
    "EUR": "ESPSLRTCR03GYSAM",  # Spain proxy; DEU/FRA 404; EA19 stale ~2018-12
    "GBP": "GBRSLRTCR03GYSAM",
    "JPY": "JPNSLRTCR03GYSAM",
    "CAD": "CANSLRTCR03GYSAM",
    "AUD": "AUSSLRTCR03GYSAM",
    "NZD": None,  # NZLSLRTCR03GYSAM stale ~2021-04
    "CHF": None,  # CHESLRTCR03GYSAM stale ~2018-12
}

CARS_FREQ: dict[str, str] = {
    "USD": "M",
    "EUR": "M",
    "GBP": "M",
    "JPY": "M",
    "CAD": "M",
    "AUD": "M",
}

CARS_NATIVE_UNIT: dict[str, str] = {
    "USD": "yoy_pct_gysam",
    "EUR": "yoy_pct_gysam",
    "GBP": "yoy_pct_gysam",
    "JPY": "yoy_pct_gysam",
    "CAD": "yoy_pct_gysam",
    "AUD": "yoy_pct_gysam",
}

CARS_SCORE_UNIT = "yoy_pct"

CARS_FAILED_OR_ALT: dict[str, list[str]] = {
    "housing_starts_odcnpi02_g10_404_fallthrough": [
        "USAODCNPI02GYSAM",
        "CANODCNPI02GYSAM",
        "AUSODCNPI02GYSAM",
        "DEUODCNPI02GYSAM",
        "FRAODCNPI02GYSAM",
        "GBRODCNPI02GYSAM",
        "JPNODCNPI02GYSAM",
        "CHEODCNPI02GYSAM",
        "NZLODCNPI02GYSAM",
        "EA19ODCNPI02GYSAM",
    ],
    "housing_completions_odcnpi04_404_fallthrough": [
        "USAODCNPI04GYSAM",
        "CANODCNPI04GYSAM",
        "AUSODCNPI04GYSAM",
        "DEUODCNPI04GYSAM",
    ],
    "us_housing_starts_live_thin_panel": ["HOUST", "HOUST1F", "COMPUTSA", "COMPU1USA"],
    "mfg_orders_odmn_g10_404_fallthrough": [
        "USAODMNNO01GYSAM",
        "CANODMNNO01GYSAM",
        "DEUODMNNO01GYSAM",
        "GBRODMNNO01GYSAM",
        "JPNODMNNO01GYSAM",
        "AUSODMNNO01GYSAM",
        "USAODMNTO01GYSAM",
        "CANODMNTO01GYSAM",
        "DEUODMNTO01GYSAM",
    ],
    "us_mfg_orders_live_thin_panel": ["AMTMNO", "DGORDER", "NEWORDER", "ISRATIO", "MNFCTRIRSA"],
    "eur_deu_fra_gysam_404": ["DEUSLRTCR03GYSAM", "FRASLRTCR03GYSAM"],
    "eur_ea19_gysam_stale_2018_12": ["EA19SLRTCR03GYSAM", "EA19SLRTCR03GYSAQ"],
    "eur_spain_proxy_primary": ["ESPSLRTCR03GYSAM"],
    "nzd_stale_2021_04": ["NZLSLRTCR03GYSAM"],
    "chf_stale_2018_12": ["CHESLRTCR03GYSAM"],
    "slrtcr03_q657s_stale_~2018": [
        "SLRTCR03USQ657S",
        "SLRTCR03CAQ657S",
        "SLRTCR03DEQ657S",
        "SLRTCR03GBQ657S",
        "SLRTCR03JPQ657S",
        "SLRTCR03AUQ657S",
        "SLRTCR03NZQ657S",
        "SLRTCR03CHQ657S",
    ],
    "us_national_unit_sales_alts_not_mixed": ["TOTALSA", "ALTSALES"],
    "hours_vacancies_404_or_stale_not_this_wave": [
        "HOHWMN02USM065S",
        "LMJVTTUVUSM647S",
        "JTSJOL",
    ],
    "distinct_channels_not_this_wave": [
        "SLRTTO01USA657S",  # retail §46
        "ODCNPI03USA661N",  # building-permits §43
        "USAPRCNTO01GYSAM",  # construction §59
        "USAPRINTO01GYSAM",  # IP §42
        "BSCICP02USM460S",  # BCI §39
        "USALOLITOAASTSAM",  # CLI §37
    ],
}

EUR_SPAIN_PROXY = "ESPSLRTCR03GYSAM"
EUR_EA19_STALE = "EA19SLRTCR03GYSAM"
NZD_STALE = "NZLSLRTCR03GYSAM"
CHF_STALE = "CHESLRTCR03GYSAM"

DEFAULT_PUB_LAG_MONTHS = 2  # a priori: auto/retail ~1–2m (same band as retail §46)


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


def load_passenger_cars_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """OECD MEI passenger-car registrations YoY % → monthly PIT panel.

    Columns = ISO currency codes. Values = car-registrations **YoY %**.
    Index = UTC month-start of the PIT *known* date after ``pub_lag_months``.
    Foreign live panel: EUR/GBP/JPY/CAD/AUD (≥ ~4–5 threshold); NZD/CHF
    unmapped (stale). EUR = Spain proxy.
    """
    curs = [c.upper() for c in (currencies or list(CARS_SERIES.keys()))]
    monthly_cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}
    freq_map: dict[str, str] = {}
    native_map: dict[str, str] = {}

    for c in curs:
        sid = CARS_SERIES.get(c)
        if sid is None:
            if c == "NZD":
                notes[c] = f"unmapped_stale:{NZD_STALE}~2021-04"
            elif c == "CHF":
                notes[c] = f"unmapped_stale:{CHF_STALE}~2018-12"
            else:
                notes[c] = "unmapped_on_fred_slrtcr03"
            continue
        try:
            s = _load_series(sid, download=download, force=force)
            series_map[c] = sid
            freq_map[c] = CARS_FREQ.get(c, "M")
            native_map[c] = CARS_NATIVE_UNIT.get(c, "yoy_pct")
            if c == "EUR":
                notes[c] = (
                    f"Spain proxy {sid} (M YoY %); "
                    f"DEU/FRA GYSAM 404; EA19 GYSAM stale ~2018-12"
                )
            else:
                notes[c] = f"OECD MEI passenger-car regs YoY {sid} (monthly)"
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
        "NZLSLRTCR03GYSAM stale ~2021-04 — NZD unmapped",
        "CHESLRTCR03GYSAM stale ~2018-12 — CHF unmapped",
        "DEUSLRTCR03GYSAM/FRASLRTCR03GYSAM 404; EA19SLRTCR03GYSAM stale ~2018-12",
        "EUR=ESPSLRTCR03GYSAM Spain proxy a priori",
        "housing starts ODCNPI02 G10 404 + US HOUST thin → fall through",
        "mfg orders ODMN* G10 404 + US AMTMNO thin → fall through",
    ]

    df.attrs["factor"] = "oecd_mei_passenger_car_regs_slrtcr03"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["freq_map"] = freq_map
    df.attrs["native_unit_map"] = native_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI {ISO3}SLRTCR03GYSAM passenger-car registrations YoY "
        "(USD/GBP/JPY/CAD/AUD live; EUR=ESPSLRTCR03GYSAM Spain proxy — "
        "DEU/FRA 404, EA19 stale ~2018-12; NZD/CHF unmapped stale; "
        "housing starts ODCNPI02 404 thin fallthrough; "
        "mfg orders ODMN* 404 thin fallthrough)"
    )
    df.attrs["unit"] = CARS_SCORE_UNIT
    df.attrs["score_basis"] = "passenger_cars_yoy_pct"
    df.attrs["unmapped"] = [c for c in curs if c not in series_map]
    df.attrs["failed_or_alt"] = CARS_FAILED_OR_ALT
    df.attrs["stale"] = True  # NZD/CHF + EUR EA19 documented
    df.attrs["stale_note"] = "; ".join(stale_note_parts)
    df.attrs["fallthrough_from"] = [
        "housing_starts_completions_thin_or_404",
        "manufacturing_new_orders_inventories_thin_or_404",
    ]
    return df


def load_us_passenger_cars(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US passenger-car registrations YoY % (USASLRTCR03GYSAM), PIT-lagged."""
    panel = load_passenger_cars_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US passenger-car registrations series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_CARS_YOY"
    out.attrs["series_id"] = CARS_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def passenger_cars_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_passenger_cars_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_fred_slrtcr03"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
