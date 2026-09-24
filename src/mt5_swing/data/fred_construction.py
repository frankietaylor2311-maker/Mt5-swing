"""FRED / OECD MEI construction production (PRCNTO01) panels.

Literature
----------
- Growth-channel / Taylor-rule FX applied to construction *output* (coincident).
- Dahlquist & Hasseltoft (2020, JFE) economic-momentum spirit on real-activity
  differentials → FX.
- Distinct from: building-permits §43 (ODCNPI03 — leading housing *flow*),
  IP §42 (PRINTO01 — industry *excl.* construction), GDP §58 (NAEXKP01),
  CLI §37, BCI §39, retail §46, emp/UR/LFP/wages labour block.

Free data (verified live on FRED probe 2026-09-24)
-------------------------------------------------
OECD MEI Production of Total Construction — FRED ``PRCNTO01`` family.

| Ccy | FRED id (primary)    | Freq | Unit / notes                                      |
|-----|----------------------|------|---------------------------------------------------|
| USD | USAPRCNTO01GYSAM     | M    | YoY % as reported                                 |
| EUR | DEUPRCNTO01GYSAM     | M    | YoY % Germany proxy (live ~2026-06)               |
| GBP | GBRPRCNTO01GYSAM     | M    | YoY %                                             |
| CAD | CANPRCNTO01GYSAM     | M    | YoY %                                             |
| JPY | JPNPRCNTO01GYSAQ     | Q    | YoY % (monthly JPNPRCNTO01GYSAM stale ~2020-07)   |
| AUD | PRCNTO01AUQ657S      | Q    | QoQ % → compound to YoY (like GDP §58)            |
| NZD | PRCNTO01NZQ657S      | Q    | QoQ % → YoY                                       |
| CHF | PRCNTO01CHQ657S      | Q    | QoQ % → YoY                                       |

Tried / documented (not primary — do not mix):
- ``CHEPRCNTO01GYSAM`` — **404**.
- ``JPNPRCNTO01GYSAM`` — stale ~2020-07 → use quarterly GYSAQ.
- ``USAPRCNTO01IXOBSAM`` — index levels stale ~2024-03.
- Vacancies / hours alts — too stale / 404 (rejected elsewhere).

Unit choice (fixed a priori)
----------------------------
Panel stores **construction YoY %**. Monthly GYSAM / JPY GYSAQ are YoY as
reported. AUD/NZD/CHF Q657S are QoQ % compounded to YoY
(``(∏(1+r_q/100)−1)×100``). Strategy scores XS-rank on YoY. Acceleration =
Δ12 of that YoY (after Q→M ffill).

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Construction / IP-like. Default ``pub_lag_months=2``. Observation dated
month-/quarter-start ``t`` is first known at ``t + 2 months``. Quarterly
series are pub-lagged then forward-filled to month-start (no intra-quarter
leak). Strategy modules add ``signal_lag=1`` month + a **1 trading-day**
weight lag (mirror GDP §58 / employment-rate §57 — *not* older IP §42
zero-month signal lag).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED OECD MEI construction production (PRCNTO01)
CONS_SERIES: dict[str, str | None] = {
    "USD": "USAPRCNTO01GYSAM",
    "EUR": "DEUPRCNTO01GYSAM",  # Germany proxy; live ~2026-06
    "GBP": "GBRPRCNTO01GYSAM",
    "CAD": "CANPRCNTO01GYSAM",
    "JPY": "JPNPRCNTO01GYSAQ",  # quarterly YoY; monthly GYSAM stale ~2020-07
    "AUD": "PRCNTO01AUQ657S",  # QoQ % → YoY
    "NZD": "PRCNTO01NZQ657S",
    "CHF": "PRCNTO01CHQ657S",
}

CONS_FREQ: dict[str, str] = {
    "USD": "M",
    "EUR": "M",
    "GBP": "M",
    "CAD": "M",
    "JPY": "Q",
    "AUD": "Q",
    "NZD": "Q",
    "CHF": "Q",
}

# Native unit tag for each primary id
CONS_NATIVE_UNIT: dict[str, str] = {
    "USD": "yoy_pct_gysam",
    "EUR": "yoy_pct_gysam",
    "GBP": "yoy_pct_gysam",
    "CAD": "yoy_pct_gysam",
    "JPY": "yoy_pct_gysaq",
    "AUD": "qoq_pct_q657s",
    "NZD": "qoq_pct_q657s",
    "CHF": "qoq_pct_q657s",
}

CONS_SCORE_UNIT = "yoy_pct"

CONS_FAILED_OR_ALT: dict[str, list[str]] = {
    "che_gysam_404": ["CHEPRCNTO01GYSAM"],
    "jpn_monthly_gysam_stale_2020_07": ["JPNPRCNTO01GYSAM"],
    "jpn_quarterly_gysaq_primary": ["JPNPRCNTO01GYSAQ"],
    "usa_ixobsam_index_stale_2024_03": ["USAPRCNTO01IXOBSAM"],
    "eur_germany_proxy_primary": ["DEUPRCNTO01GYSAM"],
    "aud_nzd_chf_qoq_q657s_primary": [
        "PRCNTO01AUQ657S",
        "PRCNTO01NZQ657S",
        "PRCNTO01CHQ657S",
    ],
    "distinct_channels_not_this_wave": [
        "ODCNPI03USA661N",  # building-permits §43
        "USAPRINTO01GYSAM",  # IP §42 industry excl construction
        "NAEXKP01USQ657S",  # GDP §58
        "USALOLITOAASTSAM",  # CLI §37
        "BSCICP02USM460S",  # BCI §39
    ],
}

JPY_MONTHLY_STALE = "JPNPRCNTO01GYSAM"
CHE_GYSAM_404 = "CHEPRCNTO01GYSAM"
USA_IXOBSAM_STALE = "USAPRCNTO01IXOBSAM"

DEFAULT_PUB_LAG_MONTHS = 2  # a priori: construction/IP-like


def _to_month_start(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = pd.DatetimeIndex(s.index)
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    ms = pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start")).tz_localize("UTC")
    s = s.copy()
    s.index = ms
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


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
    """Compound four QoQ % observations → YoY %."""
    r = qoq_pct.astype(float) / 100.0
    yoy = (1.0 + r).rolling(4, min_periods=4).apply(
        lambda a: float(np.prod(a)) - 1.0, raw=True
    ) * 100.0
    yoy.name = qoq_pct.name
    return yoy


def _load_series_yoy(
    series_id: str,
    *,
    freq: str,
    native_unit: str,
    download: bool,
    force: bool,
) -> pd.Series:
    if download:
        download_fred_series(series_id, force=force)
    raw = load_fred_series(series_id, download=False)
    if freq.upper().startswith("Q"):
        q = _to_quarter_start(raw)
        if native_unit.startswith("qoq"):
            return _qoq_pct_to_yoy(q)
        return q  # already YoY (GYSAQ)
    return _to_month_start(raw)  # GYSAM YoY as reported


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


def load_construction_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """OECD MEI construction production YoY % → monthly PIT panel.

    Columns = ISO currency codes. Values = construction **YoY % growth**.
    Index = UTC month-start of the PIT *known* date after ``pub_lag_months``.
    Quarterly series are pub-lagged first, then forward-filled to month-start
    (no intra-quarter leak). Full G10 mapped (EUR = Germany proxy).
    """
    curs = [c.upper() for c in (currencies or list(CONS_SERIES.keys()))]
    monthly_cols: dict[str, pd.Series] = {}
    quarterly_cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}
    freq_map: dict[str, str] = {}
    native_map: dict[str, str] = {}

    for c in curs:
        sid = CONS_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_prcnto01"
            continue
        freq = CONS_FREQ.get(c, "M")
        native = CONS_NATIVE_UNIT.get(c, "yoy_pct")
        try:
            s = _load_series_yoy(
                sid, freq=freq, native_unit=native, download=download, force=force
            )
            series_map[c] = sid
            freq_map[c] = freq
            native_map[c] = native
            if c == "EUR":
                notes[c] = (
                    f"Germany proxy {sid} (M YoY %); live ~2026-06"
                )
            elif c == "JPY":
                notes[c] = (
                    f"OECD MEI construction YoY {sid} (quarterly; "
                    f"monthly {JPY_MONTHLY_STALE} stale ~2020-07 — not primary)"
                )
            elif native.startswith("qoq"):
                notes[c] = f"OECD MEI construction {sid} QoQ→YoY %"
            else:
                notes[c] = f"OECD MEI construction YoY {sid} (monthly)"
            if freq == "Q":
                quarterly_cols[c] = s
            else:
                monthly_cols[c] = s
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    frames: list[pd.DataFrame] = []
    if monthly_cols:
        mdf = pd.DataFrame(monthly_cols).sort_index()
        mdf = _apply_pub_lag_monthly(mdf, pub_lag_months=pub_lag_months)
        frames.append(mdf)
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

    stale_note_parts = [
        "JPY monthly JPNPRCNTO01GYSAM stale ~2020-07 — quarterly GYSAQ used",
        "CHEPRCNTO01GYSAM 404 — CHF via PRCNTO01CHQ657S QoQ→YoY",
        "USAPRCNTO01IXOBSAM index stale ~2024-03 — not primary",
    ]
    if "EUR" in series_map and series_map["EUR"] == "DEUPRCNTO01GYSAM":
        stale_note_parts.append("EUR=DEUPRCNTO01GYSAM Germany proxy")

    df.attrs["factor"] = "oecd_mei_construction_production_prcnto01"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["freq_map"] = freq_map
    df.attrs["native_unit_map"] = native_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI PRCNTO01 construction production "
        "(USD/EUR/GBP/CAD *GYSAM monthly YoY; "
        "JPY=JPNPRCNTO01GYSAQ quarterly YoY — monthly GYSAM stale ~2020-07; "
        "AUD/NZD/CHF=PRCNTO01*Q657S QoQ→YoY; "
        "EUR=DEUPRCNTO01GYSAM Germany proxy; "
        "CHEPRCNTO01GYSAM 404; USAPRCNTO01IXOBSAM stale ~2024-03 — not primary; "
        "full G10 mapped)"
    )
    df.attrs["unit"] = CONS_SCORE_UNIT
    df.attrs["score_basis"] = "construction_yoy_pct"
    df.attrs["unmapped"] = [c for c in curs if c not in series_map]
    df.attrs["failed_or_alt"] = CONS_FAILED_OR_ALT
    df.attrs["stale"] = True  # JPY monthly stale + CHF GYSAM 404 documented
    df.attrs["stale_note"] = "; ".join(stale_note_parts)
    return df


def load_us_construction(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US construction production YoY % (USAPRCNTO01GYSAM), PIT-lagged."""
    panel = load_construction_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US construction series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_CONS_YOY"
    out.attrs["series_id"] = CONS_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def construction_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_construction_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_fred_prcnto01"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
