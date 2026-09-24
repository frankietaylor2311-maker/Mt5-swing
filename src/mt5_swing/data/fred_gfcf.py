"""FRED / OECD MEI real GFCF / gross fixed capital formation (NAEXKP04) panels.

Literature
----------
- Dahlquist & Hasseltoft (2020, JFE) economic-momentum / growth differentials → FX,
  applied to **investment / GFCF** (not aggregate GDP).
- Classic macro-FX: currencies with *high* relative real GFCF / investment growth
  subsequently appreciate vs low-investment peers (capex / investment boom prior).
- Distinct from: GDP §58 (NAEXKP01 aggregate), construction §59, IP §42, BCI §39,
  building permits §43, HPI §35, CLI §37, retail §46, cars §60, export/import §61/§62,
  household credit §63, private credit/GDP §32.

Free data (verified live on FRED probe 2026-09-24)
-------------------------------------------------
OECD MEI Gross Domestic Product by Expenditure in Constant Prices:
Gross Fixed Capital Formation — FRED ``NAEXKP04{ISO2}Q657S`` = **Growth rate
previous period** (QoQ %, SA). Panel stores **YoY %** formed by compounding
four QoQ observations (``(∏(1+r_q/100)−1)×100``). Same unit family as GDP
``NAEXKP01`` (§58).

| Ccy | FRED id (primary)  | Freq | Notes                                         |
|-----|--------------------|------|-----------------------------------------------|
| USD | NAEXKP04USQ657S    | Q    | live ~2026-04                                   |
| EUR | NAEXKP04DEQ657S    | Q    | **Germany proxy** (EZQ657S stale ~2023-01)    |
| GBP | NAEXKP04GBQ657S    | Q    | live ~2026-04                                 |
| JPY | NAEXKP04JPQ657S    | Q    | live ~2026-04                                 |
| CAD | NAEXKP04CAQ657S    | Q    | live ~2026-04                                 |
| AUD | NAEXKP04AUQ657S    | Q    | live ~2026-04                                 |
| CHF | NAEXKP04CHQ657S    | Q    | live ~2026-04                                 |
| NZD | NAEXKP04NZQ657S    | Q    | live ~2026-01                                 |

Tried / documented (not primary) + fall-through probe:
- ``NAEXKP04EZQ657S`` — EA YoY-from-QoQ **stale ~2023-01** (Germany DEQ proxy used).
- ``NAEXKP04*Q659S`` same-period-previous-year — **live** and ≡ compounded Q657S
  YoY (US corr=1.0); kept Q657S→compound for consistency with GDP §58; not mixed.
- National GFCF alts not mixed with OECD MEI panel.
- **Fall-through skip:** PNFS absolute ``CRDQ*APABIS`` — too close to private
  credit-impulse companion of §32 (keepalive said skip).
- **Fall-through skip:** NFC ``ANFBIS`` — 404 on FRED.
- This wave = **GFCF investment** (NAEXKP04), not credit stock.

Unit choice (fixed a priori)
----------------------------
Panel stores **real GFCF YoY % growth** (compounded from Q657S QoQ). Strategy
scores XS-rank on YoY. Acceleration = Δ12 of that YoY (after Q→M ffill).

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Quarterly GFCF/NA is slow (same band as GDP §58). Default ``pub_lag_months=3``.
Observation dated quarter-start ``t`` is first known at ``t + 3 months``.
Quarterly series are pub-lagged then forward-filled to month-start (no
intra-quarter leak). Strategy modules add ``signal_lag=1`` month + a
**1 trading-day** weight lag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED OECD MEI real GFCF QoQ growth (NAEXKP04 *Q657S)
GFCF_SERIES: dict[str, str | None] = {
    "USD": "NAEXKP04USQ657S",
    "EUR": "NAEXKP04DEQ657S",  # Germany proxy; EZQ657S stale ~2023-01
    "GBP": "NAEXKP04GBQ657S",
    "JPY": "NAEXKP04JPQ657S",
    "CAD": "NAEXKP04CAQ657S",
    "AUD": "NAEXKP04AUQ657S",
    "CHF": "NAEXKP04CHQ657S",
    "NZD": "NAEXKP04NZQ657S",
}

GFCF_FREQ: dict[str, str] = {
    "USD": "Q",
    "EUR": "Q",
    "GBP": "Q",
    "JPY": "Q",
    "CAD": "Q",
    "AUD": "Q",
    "CHF": "Q",
    "NZD": "Q",
}

# Native series are QoQ %; loader compounds to YoY before return
GFCF_NATIVE_UNIT = "qoq_pct_q657s"
GFCF_SCORE_UNIT = "yoy_pct_compounded"

GFCF_FAILED_OR_ALT: dict[str, list[str]] = {
    "eur_ez_stale_germany_proxy": [
        "NAEXKP04EZQ657S",  # stale ~2023-01
        "NAEXKP04DEQ657S",  # Germany proxy (primary)
    ],
    "q659s_live_equiv_compounded_q657s_not_mixed": [
        "NAEXKP04USQ659S",  # live ≡ compound Q657S
        "NAEXKP04GBQ659S",
        "NAEXKP04DEQ659S",
    ],
    "fallthrough_pnfs_abs_crdq_apabis_skip_credit32_companion": [
        "CRDQUSAPABIS",
        "CRDQGBAPABIS",
        "CRDQDEAPABIS",
        "CRDQJPAPABIS",
    ],
    "fallthrough_nfc_anfbis_404": [
        "CRDQUSANFBIS",
        "CRDQGBANFBIS",
        "CRDQDEANFBIS",
    ],
    "national_gfcf_alts_not_mixed": [
        # national GFCF level/volume series — documented, not mixed into panel
    ],
    "distinct_channels_not_this_wave": [
        "NAEXKP01USQ657S",  # GDP §58
        "DEUPRCNTO01GYSAM",  # construction §59
        "USAPRINTO01GYSAM",  # IP §42
        "BSCICP02USM460S",  # BCI §39
        "CANODCNPI03GYSAM",  # building permits §43
        "QUSR628BIS",  # HPI §35
        "USALOLITOAASTSAM",  # CLI §37
        "CRDQUSAHABIS",  # household credit §63
        "QUSPAM770A",  # private credit/GDP §32
        "SLRTTO01USQ657S",  # retail §46
        "SLRTCR03USM657S",  # cars §60
        "XTEXVA01USM657S",  # export §61
        "XTIMVA01USM657S",  # import §62
    ],
}

EUR_EZ_ALT_STALE = "NAEXKP04EZQ657S"
US_GFCF_ALT = "NAEXKP04USQ657S"  # OECD MEI primary; national alts not mixed

DEFAULT_PUB_LAG_MONTHS = 3  # a priori: quarterly GFCF/NA is slow (same as GDP §58)


def _to_quarter_start(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = pd.DatetimeIndex(s.index)
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    qs = pd.DatetimeIndex(naive.to_period("Q").to_timestamp(how="start")).tz_localize("UTC")
    s = s.copy()
    s.index = qs
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _qoq_pct_to_yoy(qoq_pct: pd.Series) -> pd.Series:
    """Compound four QoQ % observations → YoY % (same as GDP §58 NAEXKP01)."""
    r = qoq_pct.astype(float) / 100.0
    yoy = (1.0 + r).rolling(4, min_periods=4).apply(
        lambda a: float(np.prod(a)) - 1.0, raw=True
    ) * 100.0
    yoy.name = qoq_pct.name
    return yoy


def _load_series_yoy(series_id: str, *, download: bool, force: bool) -> pd.Series:
    if download:
        download_fred_series(series_id, force=force)
    raw = load_fred_series(series_id, download=False)
    qoq = _to_quarter_start(raw)
    return _qoq_pct_to_yoy(qoq)


def _expand_quarterly_to_monthly(
    df: pd.DataFrame,
    *,
    pub_lag_months: int,
) -> pd.DataFrame:
    """Shift quarterly index by pub_lag, then ffill to month-start (no leak)."""
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


def load_gfcf_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """OECD MEI real GFCF YoY % (from Q657S QoQ) → monthly PIT panel.

    Columns = ISO currency codes. Values = real GFCF **YoY % growth**.
    Index = UTC month-start of the PIT *known* date after ``pub_lag_months``.
    Quarterly series are pub-lagged first, then forward-filled to month-start
    (no intra-quarter leak). Full G10 mapped (EUR = Germany proxy).
    """
    curs = [c.upper() for c in (currencies or list(GFCF_SERIES.keys()))]
    quarterly_cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}
    freq_map: dict[str, str] = {}

    for c in curs:
        sid = GFCF_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_naexkp04"
            continue
        freq = GFCF_FREQ.get(c, "Q")
        try:
            s = _load_series_yoy(sid, download=download, force=force)
            if c == "EUR":
                notes[c] = (
                    f"Germany proxy {sid} (Q→YoY %); "
                    f"EA {EUR_EZ_ALT_STALE} stale ~2023-01 — not primary"
                )
            else:
                notes[c] = f"OECD MEI real GFCF {sid} QoQ→YoY %"
            series_map[c] = sid
            freq_map[c] = freq
            quarterly_cols[c] = s
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    frames: list[pd.DataFrame] = []
    # Expand each quarterly series alone so a longer leg cannot ffill a shorter
    # one past its true last obs.
    for c, s in quarterly_cols.items():
        qdf = pd.DataFrame({c: s}).sort_index()
        qdf = _expand_quarterly_to_monthly(qdf, pub_lag_months=pub_lag_months)
        frames.append(qdf)

    if not frames:
        df = pd.DataFrame()
    else:
        df = pd.concat(frames, axis=1).sort_index()
        df = df.dropna(how="all")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

    stale_note_parts = []
    if "EUR" in series_map and series_map["EUR"] == "NAEXKP04DEQ657S":
        stale_note_parts.append(
            "EUR EA NAEXKP04EZQ657S stale ~2023-01 — Germany DEQ657S proxy used"
        )

    df.attrs["factor"] = "oecd_mei_gfcf_naexkp04"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["freq_map"] = freq_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI NAEXKP04*Q657S real GFCF / gross fixed capital formation "
        "growth rate previous period (QoQ %) → compounded to YoY % "
        "(USD/GBP/JPY/CAD/AUD/CHF/NZD *Q657S; "
        "EUR=NAEXKP04DEQ657S Germany proxy — EZQ657S stale ~2023-01; "
        "full G10 mapped; Q659S live≡compound Q657S not mixed; national GFCF alts not mixed; "
        "PNFS CRDQ*APABIS skipped (§32 companion); NFC ANFBIS 404)"
    )
    df.attrs["unit"] = GFCF_SCORE_UNIT
    df.attrs["native_unit"] = GFCF_NATIVE_UNIT
    df.attrs["score_basis"] = "gfcf_yoy_pct"
    df.attrs["unmapped"] = [c for c in curs if c not in series_map]
    df.attrs["failed_or_alt"] = GFCF_FAILED_OR_ALT
    df.attrs["stale"] = bool(
        "EUR" in series_map and series_map.get("EUR") == "NAEXKP04DEQ657S"
    )
    df.attrs["stale_note"] = (
        "; ".join(stale_note_parts)
        if stale_note_parts
        else "USD/GBP/JPY/CAD/AUD/CHF/NZD live into 2025–2026; EUR=DE proxy"
    )
    return df


def load_us_gfcf(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US real GFCF YoY % (from NAEXKP04USQ657S), PIT-lagged."""
    panel = load_gfcf_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US GFCF series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_GFCF_YOY"
    out.attrs["series_id"] = GFCF_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def gfcf_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_gfcf_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_fred_naexkp04"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
