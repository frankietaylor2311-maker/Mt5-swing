"""FRED / BIS household credit (CRDQ*AHABIS) YoY panels for consumer-credit FX.

Literature
----------
- Household / consumer credit growth as a **retail demand / consumption-buffer**
  channel (Mian–Sufi household leverage; Borio–Drehmann financial-cycle spirit
  applied to the *household* sector rather than total PNFS).
- Currencies with *high* relative household-credit YoY subsequently appreciate
  vs low-growth peers (domestic-demand / consumer-credit prior; Dahlquist–
  Hasseltoft economic-momentum spirit on credit-financed absorption).
- Distinct from: BIS **private credit/GDP** §32 (``Q*PAM770A`` stock %GDP +
  credit-gap; absolute ``CRDQ*APABIS`` PNFS unused), money-growth §33, DSR §47,
  debt/GDP §29, house-price §35, retail §46, cars §60, import-value §62.

Fall-through (§63 probe order, 2026-09-24)
-----------------------------------------
1. **Merchandise trade VOLUME YoY** — ``XTINTVA*`` / ``XTEXVO*`` / ``XTIMVO*``
   **404** on FRED; OECD MEI ``XTEXVA01*M657S`` is VALUE (already §61/§62) —
   no clean G10 volume panel → fall through.
2. **Household saving rate** — ``NASAVP*`` / ``HHSBRG*`` **404**; only US
   ``PSAVERT`` live → cannot map ≥4 G10 → fall through.
3. **This wave:** BIS household credit absolute ``CRDQ{ISO2}AHABIS`` → YoY %
   (distinct retail/consumer credit growth; not PNFS credit/GDP §32).
4. BOP services / inventories / CLI order-books — thin or US-only (documented
   below); not primary once household-credit clears ≥4 G10.

Free data (verified live on FRED probe 2026-09-24)
-------------------------------------------------
| Ccy | FRED id (primary) | Freq | Unit / notes                                      |
|-----|-------------------|------|---------------------------------------------------|
| USD | CRDQUSAHABIS      | Q    | Household credit abs → YoY % (~2025-10)           |
| EUR | CRDQXMAHABIS      | Q    | **Euro-area** (CRDQEAAHABIS 404)                  |
| GBP | CRDQGBAHABIS      | Q    |                                                   |
| JPY | CRDQJPAHABIS      | Q    |                                                   |
| CAD | CRDQCAAHABIS      | Q    |                                                   |
| AUD | CRDQAUAHABIS      | Q    |                                                   |
| CHF | CRDQCHAHABIS      | Q    |                                                   |
| NZD | —                 | —    | ``CRDQNZAHABIS`` **404** unmapped                 |

Tried / documented (not primary):
- ``CRDQ*APABIS`` absolute **PNFS** credit — already documented unused in §32;
  not this wave (household AHABIS is the retail/consumer leg).
- ``Q*PAM770A`` private credit/GDP — §32 primary.
- ``CRDQNZAHABIS`` / ``CRDQEAAHABIS`` — **404**.
- Merchandise trade volume codes — **404**.
- Household saving ``NASAVP*`` / ``HHSBRG*`` — **404** (PSAVERT US-only).
- Inventories ``STMNTO*`` multi-country — **404** (US BUSINV only).
- Order books ``BSOITE02*`` — only US/DE/CH live (<4 G10); ``BSCICP03*``
  amplitude stale ~2023-11..2024-01 (already noted under BCI §39).

Unit choice (fixed a priori)
----------------------------
Panel stores **household-credit YoY %** computed from quarterly absolute
levels: ``100 * (L_t / L_{t-4} - 1)``. Strategy scores XS-rank on YoY.
Acceleration = Δ4 of that YoY (≈1y change of growth; quarterly cadence).

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
BIS quarterly credit releases typically land ~4–5 months after quarter-end
(same band as private credit/GDP §32 / DSR §47). Default ``pub_lag_months=5``.
Observation dated quarter-start ``t`` is first known at ``t + 5 months``.
Strategy modules add ``signal_lag=1`` month + a **1 trading-day** weight lag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED BIS household credit absolute (CRDQ*AHABIS)
HHCRED_SERIES: dict[str, str | None] = {
    "USD": "CRDQUSAHABIS",
    "EUR": "CRDQXMAHABIS",  # Euro-area; CRDQEAAHABIS 404
    "GBP": "CRDQGBAHABIS",
    "JPY": "CRDQJPAHABIS",
    "CAD": "CRDQCAAHABIS",
    "AUD": "CRDQAUAHABIS",
    "CHF": "CRDQCHAHABIS",
    "NZD": None,  # CRDQNZAHABIS 404
}

HHCRED_FREQ: dict[str, str] = {
    "USD": "Q",
    "EUR": "Q",
    "GBP": "Q",
    "JPY": "Q",
    "CAD": "Q",
    "AUD": "Q",
    "CHF": "Q",
    "NZD": "Q",
}

HHCRED_NATIVE_UNIT: dict[str, str] = {
    "USD": "abs_bn_to_yoy_pct",
    "EUR": "abs_bn_to_yoy_pct",
    "GBP": "abs_bn_to_yoy_pct",
    "JPY": "abs_bn_to_yoy_pct",
    "CAD": "abs_bn_to_yoy_pct",
    "AUD": "abs_bn_to_yoy_pct",
    "CHF": "abs_bn_to_yoy_pct",
}

HHCRED_SCORE_UNIT = "yoy_pct"

HHCRED_FAILED_OR_ALT: dict[str, list[str]] = {
    "trade_volume_xtintva_xtexvo_xtimvo_404_fallthrough": [
        "XTINTVA01USM657S",
        "XTEXVO01USM657S",
        "XTIMVO01USM657S",
        "XTEXVA01USM657S_is_value_not_volume_already_61",
        "XTIMVA01USM657S_is_value_not_volume_already_62",
    ],
    "household_saving_nasavp_hhsbrg_404_psavert_us_only_fallthrough": [
        "NASAVPUSA156N",
        "HHSBRG01USQ156N",
        "PSAVERT",
    ],
    "inventories_stmnnto_404_businv_us_only": [
        "STMNTO01USM657S",
        "USASTMNTO01GYSAM",
        "BUSINV",
    ],
    "order_books_bsoite02_thin_lt4_g10": [
        "BSOITE02USM460S",
        "BSOITE02DEM460S",
        "BSOITE02CHM460S",
        "BSCICP03USM665S_stale_bci39",
    ],
    "nzd_household_credit_404": ["CRDQNZAHABIS", "CRDQNZLAHABIS"],
    "eur_ea_404_use_xm": ["CRDQEAAHABIS", "CRDQEUAHABIS"],
    "pnfs_absolute_not_this_wave_see_32": [
        "CRDQUSAPABIS",
        "CRDQGBAPABIS",
        "QUSPAM770A",
    ],
    "distinct_channels_not_this_wave": [
        "QUSPAM770A",  # private credit/GDP §32
        "MABMM301USM657S",  # money growth §33
        "QUSPNFDSR",  # DSR §47 (SDMX)
        "GGGDTA188N",  # debt/GDP §29
        "QUSR628BIS",  # house price §35
        "SLRTTO01USA657S",  # retail §46
        "USASLRTCR03GYSAM",  # cars §60
        "XTIMVA01USM657S",  # import-value §62
    ],
}

EUR_EA_XM = "CRDQXMAHABIS"
EUR_EA_404 = "CRDQEAAHABIS"
NZD_404 = "CRDQNZAHABIS"

DEFAULT_PUB_LAG_MONTHS = 5  # a priori: same band as BIS credit §32 / DSR §47


def _to_quarter_start(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = pd.DatetimeIndex(s.index)
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    qs = pd.DatetimeIndex(naive.to_period("Q").to_timestamp(how="start")).tz_localize("UTC")
    s = s.copy()
    s.index = qs
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _load_quarterly(series_id: str, *, download: bool, force: bool) -> pd.Series:
    if download:
        download_fred_series(series_id, force=force)
    return _to_quarter_start(load_fred_series(series_id, download=False))


def _levels_to_yoy_pct(s: pd.Series) -> pd.Series:
    """Quarterly absolute levels → YoY % (lag-4)."""
    s = s.dropna().sort_index()
    yoy = 100.0 * (s / s.shift(4) - 1.0)
    yoy.name = s.name
    return yoy


def _expand_quarterly_to_monthly(
    df: pd.DataFrame,
    *,
    pub_lag_months: int,
) -> pd.DataFrame:
    """Shift quarterly index by pub_lag, then ffill to month-start."""
    if df.empty:
        return df
    out = df.copy()
    if pub_lag_months > 0:
        out.index = out.index + pd.DateOffset(months=int(pub_lag_months))
        out = out[~out.index.duplicated(keep="last")].sort_index()
    naive_min = out.index.min().tz_convert(None) if out.index.tz is not None else out.index.min()
    naive_max = out.index.max().tz_convert(None) if out.index.tz is not None else out.index.max()
    start = naive_min.to_period("M").to_timestamp(how="start")
    end = naive_max.to_period("M").to_timestamp(how="start")
    monthly_idx = pd.date_range(start, end, freq="MS", tz="UTC")
    out = out.reindex(monthly_idx.union(out.index)).sort_index()
    out = out.ffill()
    out = out.loc[monthly_idx]
    out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def load_household_credit_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """BIS household credit absolute → YoY % → monthly PIT panel.

    Columns = ISO currency codes. Values = household-credit **YoY %**.
    Index = UTC month-start of the PIT *known* date after ``pub_lag_months``.
    NZD unmapped (404). EUR = euro-area XM.
    """
    curs = [c.upper() for c in (currencies or list(HHCRED_SERIES.keys()))]
    quarterly_cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}
    freq_map: dict[str, str] = {}
    native_map: dict[str, str] = {}

    for c in curs:
        sid = HHCRED_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_crdq_ahabis (CRDQNZAHABIS 404)"
            continue
        try:
            levels = _load_quarterly(sid, download=download, force=force)
            yoy = _levels_to_yoy_pct(levels)
            series_map[c] = sid
            freq_map[c] = HHCRED_FREQ.get(c, "Q")
            native_map[c] = HHCRED_NATIVE_UNIT.get(c, "abs_bn_to_yoy_pct")
            if c == "EUR":
                notes[c] = (
                    f"Euro-area household credit YoY via {sid} "
                    f"(CRDQEAAHABIS 404); abs→YoY%"
                )
            else:
                notes[c] = f"BIS household credit abs {sid} → YoY % (quarterly)"
            quarterly_cols[c] = yoy
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    if not quarterly_cols:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(quarterly_cols).sort_index()
        df = _expand_quarterly_to_monthly(df, pub_lag_months=pub_lag_months)
        df = df.dropna(how="all")

    stale_note_parts = [
        "CRDQNZAHABIS 404 → NZD unmapped",
        "CRDQEAAHABIS 404 → EUR=CRDQXMAHABIS euro-area a priori",
        "score unit = household-credit YoY % from absolute levels (not %GDP)",
        "trade VOLUME XTINTVA/XTEXVO/XTIMVO 404 → fall through",
        "household saving NASAVP/HHSBRG 404; PSAVERT US-only → fall through",
        "inventories STMNTO* 404; order books BSOITE02 <4 G10 → not primary",
        "distinct from Q*PAM770A private credit/GDP §32",
    ]

    df.attrs["factor"] = "bis_household_credit_yoy_crdq_ahabis"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["freq_map"] = freq_map
    df.attrs["native_unit_map"] = native_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED BIS CRDQ{ISO2}AHABIS household credit absolute → YoY % "
        "(EUR=CRDQXMAHABIS euro-area — CRDQEAAHABIS 404; NZD CRDQNZAHABIS 404 "
        "unmapped; distinct from Q*PAM770A private credit/GDP §32; "
        "trade-volume/saving-rate/inventories/order-books fallthrough)"
    )
    df.attrs["unit"] = HHCRED_SCORE_UNIT
    df.attrs["score_basis"] = "household_credit_yoy_pct"
    df.attrs["unmapped"] = [c for c in curs if c not in series_map]
    df.attrs["failed_or_alt"] = HHCRED_FAILED_OR_ALT
    df.attrs["stale"] = False
    df.attrs["stale_note"] = "; ".join(stale_note_parts)
    df.attrs["fallthrough_from"] = [
        "merchandise_trade_volume_404",
        "household_saving_rate_404_us_only",
        # probe 3 is THIS wave (household credit) — succeeded
    ]
    df.attrs["series_unit_caveat"] = "abs_bn_to_yoy_not_pct_gdp"
    return df


def load_us_household_credit(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US household-credit YoY % (CRDQUSAHABIS→YoY), PIT-lagged."""
    panel = load_household_credit_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US household-credit series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_HHCRED_YOY"
    out.attrs["series_id"] = HHCRED_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def household_credit_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_household_credit_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_fred_crdq_ahabis"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
