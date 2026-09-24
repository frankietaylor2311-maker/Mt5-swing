"""FRED / OECD MEI real private final consumption (NAEXKP02) panels.

Note on naming
--------------
``PCE`` here = OECD MEI **private final consumption** (``NAEXKP02``), **not**
US BEA Personal Consumption Expenditures. Same QoQ→YoY family as GDP §58 /
PCE §64.

Literature
----------
- Lustig & Verdelhan (2007, AER) — foreign currency risk premia and
  **consumption-growth risk**.
- Dahlquist & Hasseltoft (2020, JFE) economic-momentum / growth differentials → FX,
  applied to **private consumption** (not aggregate GDP, not PCE, not retail volume).
- Classic macro-FX: currencies with *high* relative real private-consumption growth
  subsequently appreciate vs low-consumption peers (consumption-boom prior).
- Distinct from: GDP §58 (NAEXKP01), PCE §64 (NAEXKP04), retail §46, cars §60,
  household credit §63, construction §59, IP §42, CLI/CCI/BCI, emp/labour §41/54–57.

Free data (verified live on FRED probe 2026-09-24)
-------------------------------------------------
OECD MEI Gross Domestic Product by Expenditure in Constant Prices:
Private Final Consumption Expenditure — FRED ``NAEXKP02{ISO2}Q657S`` = **Growth
rate previous period** (QoQ %, SA). Panel stores **YoY %** formed by compounding
four QoQ observations (``(∏(1+r_q/100)−1)×100``). Same unit family as GDP
``NAEXKP01`` (§58) / PCE ``NAEXKP04`` (§64).

| Ccy | FRED id (primary)  | Freq | Notes                                         |
|-----|--------------------|------|-----------------------------------------------|
| USD | NAEXKP02USQ657S    | Q    | live ~2026-04                                   |
| EUR | NAEXKP02DEQ657S    | Q    | **Germany proxy** (EZQ657S stale ~2023-01)    |
| GBP | NAEXKP02GBQ657S    | Q    | live ~2026-04                                 |
| JPY | NAEXKP02JPQ657S    | Q    | live ~2026-04                                 |
| CAD | NAEXKP02CAQ657S    | Q    | live ~2026-04                                 |
| AUD | NAEXKP02AUQ657S    | Q    | live ~2026-04                                 |
| CHF | NAEXKP02CHQ657S    | Q    | live ~2026-04                                 |
| NZD | NAEXKP02NZQ657S    | Q    | live ~2026-01                                 |

Tried / documented (not primary):
- ``NAEXKP02EZQ657S`` — EA YoY-from-QoQ **stale ~2023-01** (Germany DEQ proxy used).
- ``NAEXKP02*Q659S`` same-period-previous-year — **live** and ≡ compounded Q657S
  YoY; kept Q657S→compound for consistency with GDP §58 / PCE §64; not mixed.
- National PCE / BEA PCE alts not mixed with OECD MEI panel.
- This wave = **private final consumption** (NAEXKP02), not BEA PCE, not retail.

Unit choice (fixed a priori)
----------------------------
Panel stores **real private-consumption YoY % growth** (compounded from Q657S QoQ).
Strategy scores XS-rank on YoY. Acceleration = Δ12 of that YoY (after Q→M ffill).

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Quarterly NA is slow (same band as GDP §58 / PCE §64). Default ``pub_lag_months=3``.
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

# Currency → FRED OECD MEI real private consumption QoQ growth (NAEXKP02 *Q657S)
PCE_SERIES: dict[str, str | None] = {
    "USD": "NAEXKP02USQ657S",
    "EUR": "NAEXKP02DEQ657S",  # Germany proxy; EZQ657S stale ~2023-01
    "GBP": "NAEXKP02GBQ657S",
    "JPY": "NAEXKP02JPQ657S",
    "CAD": "NAEXKP02CAQ657S",
    "AUD": "NAEXKP02AUQ657S",
    "CHF": "NAEXKP02CHQ657S",
    "NZD": "NAEXKP02NZQ657S",
}

PCE_FREQ: dict[str, str] = {
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
# PCE here = OECD private final consumption (NAEXKP02), not US BEA PCE
PCE_NATIVE_UNIT = "qoq_pct_q657s"
PCE_SCORE_UNIT = "yoy_pct_compounded"

PCE_FAILED_OR_ALT: dict[str, list[str]] = {
    "eur_ez_stale_germany_proxy": [
        "NAEXKP02EZQ657S",  # stale ~2023-01
        "NAEXKP02DEQ657S",  # Germany proxy (primary)
    ],
    "q659s_live_equiv_compounded_q657s_not_mixed": [
        "NAEXKP02USQ659S",  # live ≡ compound Q657S
        "NAEXKP02GBQ659S",
        "NAEXKP02DEQ659S",
    ],
    "national_bea_pce_alts_not_mixed": [
        # US BEA PCE / national private-consumption alts — documented, not mixed
    ],
    "distinct_channels_not_this_wave": [
        "NAEXKP01USQ657S",  # GDP §58
        "NAEXKP04USQ657S",  # GFCF §64
        "SLRTTO01USQ657S",  # retail §46
        "SLRTCR03USM657S",  # cars §60
        "CRDQUSAHABIS",  # household credit §63
        "DEUPRCNTO01GYSAM",  # construction §59
        "USAPRINTO01GYSAM",  # IP §42
        "BSCICP02USM460S",  # BCI §39
        "CSCICP02USM460S",  # CCI
        "USALOLITOAASTSAM",  # CLI §37
        "LFEMTTTTUSM647S",  # emp §41
        "LRHUTTTTUSM156S",  # UR §54
        "LCEAMN01USM657S",  # wages §55
        "LRAC64TTUSM156S",  # LFP §56
        "LREM64TTUSM156S",  # employment-rate §57
    ],
}

EUR_EZ_ALT_STALE = "NAEXKP02EZQ657S"
US_PCE_ALT = "NAEXKP02USQ657S"  # OECD MEI primary; BEA PCE alts not mixed

DEFAULT_PUB_LAG_MONTHS = 3  # a priori: quarterly NA is slow (same as GDP §58 / GFCF §64)


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


def load_pce_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """OECD MEI real private final consumption YoY % (from Q657S QoQ) → monthly PIT panel.

    Columns = ISO currency codes. Values = real private final consumption **YoY % growth**.
    Index = UTC month-start of the PIT *known* date after ``pub_lag_months``.
    Quarterly series are pub-lagged first, then forward-filled to month-start
    (no intra-quarter leak). Full G10 mapped (EUR = Germany proxy).
    """
    curs = [c.upper() for c in (currencies or list(PCE_SERIES.keys()))]
    quarterly_cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}
    freq_map: dict[str, str] = {}

    for c in curs:
        sid = PCE_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_naexkp02"
            continue
        freq = PCE_FREQ.get(c, "Q")
        try:
            s = _load_series_yoy(sid, download=download, force=force)
            if c == "EUR":
                notes[c] = (
                    f"Germany proxy {sid} (Q→YoY %); "
                    f"EA {EUR_EZ_ALT_STALE} stale ~2023-01 — not primary"
                )
            else:
                notes[c] = f"OECD MEI real private consumption {sid} QoQ→YoY %"
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
    if "EUR" in series_map and series_map["EUR"] == "NAEXKP02DEQ657S":
        stale_note_parts.append(
            "EUR EA NAEXKP02EZQ657S stale ~2023-01 — Germany DEQ657S proxy used"
        )

    df.attrs["factor"] = "oecd_mei_pce_naexkp02"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["freq_map"] = freq_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI NAEXKP02*Q657S real private final consumption "
        "growth rate previous period (QoQ %) → compounded to YoY % "
        "(USD/GBP/JPY/CAD/AUD/CHF/NZD *Q657S; "
        "EUR=NAEXKP02DEQ657S Germany proxy — EZQ657S stale ~2023-01; "
        "full G10 mapped; Q659S live≡compound Q657S not mixed; "
        "national BEA PCE alts not mixed; not US BEA PCE)"
    )
    df.attrs["unit"] = PCE_SCORE_UNIT
    df.attrs["native_unit"] = PCE_NATIVE_UNIT
    df.attrs["score_basis"] = "pce_yoy_pct"
    df.attrs["unmapped"] = [c for c in curs if c not in series_map]
    df.attrs["failed_or_alt"] = PCE_FAILED_OR_ALT
    df.attrs["stale"] = bool(
        "EUR" in series_map and series_map.get("EUR") == "NAEXKP02DEQ657S"
    )
    df.attrs["stale_note"] = (
        "; ".join(stale_note_parts)
        if stale_note_parts
        else "USD/GBP/JPY/CAD/AUD/CHF/NZD live into 2025–2026; EUR=DE proxy"
    )
    return df


def load_us_pce(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US real private final consumption YoY % (from NAEXKP02USQ657S), PIT-lagged."""
    panel = load_pce_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US PCE series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_PCE_YOY"
    out.attrs["series_id"] = PCE_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def pce_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_pce_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_fred_naexkp02"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
