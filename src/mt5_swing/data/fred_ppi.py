"""FRED / OECD MEI producer-price (PIEAMP01 / PPI) panels for PPI-YoY FX.

Literature
----------
- Dahlquist & Hasseltoft (2020, JFE) economic momentum: currencies with strong
  past macro trends (incl. PPI) subsequently appreciate.
- Primary prior: currencies with *high* relative PPI YoY appreciate vs
  low-PPI peers. Honesty alternate: long low PPI / cost-stress / soft-demand.
- Distinct from: IP §42 (volume), retail §46 (consumer sales), employment §41,
  ULC §48 / LP §49 / CU §50 (labour/capacity), building-permits §43, CLI/CCI/BCI,
  macro_diff EW CPI/IP/UR blend, CA/TB, fiscal/debt, BIS, reserves, house-price,
  money, equity, IG OAS, commodity, GPR family, ACM TP, WUI, EPU/TPU.

Free data (verified on FRED ~2026-09-23)
---------------------------------------
Prefer OECD MEI **manufacturing PPI YoY** ``{ISO3}PIEAMP01GYM`` where live.
**Panel is STALE** — GYM / M661N / Q661N ends ~**2022-12** (AUD Q ends
**2023-01**; NZD Q ends **2022-07**). ``{ISO3}PIEATI01GYSAM`` / ISO2
``PIEAMP01*M657S`` / ``*Q657S`` mostly **404**. Japan OECD ``PIEAMP01JP*`` /
``JPNPIEAMP01GYM`` **404** → ``JPNPPDMMINMEI`` domestic manufacturing PPI
index → YoY (same stale end). US national ``PPIACO`` / ``PPIFIS`` live
through ~2026-08 (alts — **not** mixed into OECD panel).

| Ccy | FRED id (primary)      | Freq | Unit   | Notes                                      |
|-----|------------------------|------|--------|--------------------------------------------|
| USD | USAPIEAMP01GYM         | M    | growth | ends ~2022-12 (stale)                      |
| EUR | DEUPIEAMP01GYM         | M    | growth | Germany proxy (EA19 GYM also ends 2022-12) |
| GBP | GBRPIEAMP01GYM         | M    | growth | ends ~2022-12 (stale)                      |
| JPY | JPNPPDMMINMEI → YoY    | M    | level  | OECD PIEAMP Japan **404**; ends ~2022-12   |
| CAD | CANPIEAMP01GYM         | M    | growth | ends ~2022-12 (stale)                      |
| AUD | PIEAMP01AUQ661N → YoY  | Q    | level  | GYM **404**; ends ~2023-01                 |
| NZD | PIEAMP01NZQ661N → YoY  | Q    | level  | GYM **404**; ends ~2022-07                 |
| CHF | CHEPIEAMP01GYM         | M    | growth | ends ~2022-12 (stale)                      |

Tried / documented:
- ``{ISO3}PIEATI01GYSAM`` / ``PIEAMP01GYSAM`` / ``IXOBSAM`` — **404**.
- ``PIEAMP01{ISO2}M657S`` / ``Q657S`` / ``A657S`` growth — **404**.
- ``PIEAMP01{ISO2}M661N`` index — ends **2022-12** (same stale; GYM preferred).
- ``JPNPIEAMP01GYM`` / ``PIEAMP01JPM661N`` — **404** → ``JPNPPDMMINMEI``.
- ``AUSPIEAMP01GYM`` / ``NZLPIEAMP01GYM`` / monthly AUM661N — **404** → Q661N.
- ``EA19PIEAMP01GYM`` — ends **2022-12** (same stale) → DEU EUR proxy.
- US ``PPIACO`` / ``PPIFIS`` / ``PPIFES`` — live national alts (not mixed).

Unit choice (fixed a priori)
----------------------------
Panel stores **PPI YoY %**. ``*GYM`` growth used as reported. Level/index
series (``JPNPPDMMINMEI``, ``PIEAMP01AUQ661N``, ``PIEAMP01NZQ661N``) convert
to YoY before pub lag (monthly ``pct_change(12)*100``; quarterly
``pct_change(4)*100``). Strategy scores XS-rank on these YoY values.

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Monthly PPI typically lands ~1–2 months after reference. Default
``pub_lag_months=2`` (picked once a priori — same band as IP §42 / retail §46).
Observation dated month-/quarter-start ``t`` is first known at ``t + 2 months``.
Quarterly series are pub-lagged then forward-filled to month-start (no
intra-quarter leak). Strategy modules add ``signal_lag=1`` month + a
**1 trading-day** weight lag.

**Staleness honesty:** free FRED OECD PPI panel ends late-2022 / early-2023;
post-2023 ranks are ffilled frozen. Full-sample still informative; 2025/2026
windows reflect that limitation. No go-live claim under ``approximate_non_ftmo``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED OECD MEI / national PPI (manufacturing preferred)
# NOTE: GYM / index ends ~2022-12 on free FRED (stale) — documented, still boarded.
PPI_SERIES: dict[str, str | None] = {
    "USD": "USAPIEAMP01GYM",
    "EUR": "DEUPIEAMP01GYM",  # Germany proxy; EA19 GYM also ends 2022-12
    "GBP": "GBRPIEAMP01GYM",
    "JPY": "JPNPPDMMINMEI",  # levels → YoY; OECD PIEAMP Japan 404
    "CAD": "CANPIEAMP01GYM",
    "AUD": "PIEAMP01AUQ661N",  # quarterly index → YoY; GYM 404
    "NZD": "PIEAMP01NZQ661N",  # quarterly index → YoY; GYM 404
    "CHF": "CHEPIEAMP01GYM",
}

# Native series unit: "growth" already % YoY; "level" needs pct_change → YoY
PPI_UNIT: dict[str, str] = {
    "USD": "growth",
    "EUR": "growth",
    "GBP": "growth",
    "JPY": "level",
    "CAD": "growth",
    "AUD": "level",
    "NZD": "level",
    "CHF": "growth",
}

PPI_FREQ: dict[str, str] = {
    "USD": "M",
    "EUR": "M",
    "GBP": "M",
    "JPY": "M",
    "CAD": "M",
    "AUD": "Q",
    "NZD": "Q",
    "CHF": "M",
}

PPI_FAILED_OR_ALT: dict[str, list[str]] = {
    "iso3_pieati_gysam_404": [
        "USAPIEATI01GYSAM",
        "DEUPIEATI01GYSAM",
        "FRAPIEATI01GYSAM",
        "GBRPIEATI01GYSAM",
        "JPNPIEATI01GYSAM",
        "CANPIEATI01GYSAM",
        "AUSPIEATI01GYSAM",
        "NZLPIEATI01GYSAM",
        "CHEPIEATI01GYSAM",
        "EA19PIEATI01GYSAM",
    ],
    "iso3_pieamp_gysam_404": [
        "USAPIEAMP01GYSAM",
        "DEUPIEAMP01GYSAM",
        "GBRPIEAMP01GYSAM",
        "JPNPIEAMP01GYSAM",
        "CANPIEAMP01GYSAM",
        "AUSPIEAMP01GYSAM",
        "NZLPIEAMP01GYSAM",
        "CHEPIEAMP01GYSAM",
        "EA19PIEAMP01GYSAM",
    ],
    "iso2_m657s_q657s_404": [
        "PIEAMP01USM657S",
        "PIEAMP01DEM657S",
        "PIEAMP01GBM657S",
        "PIEAMP01JPM657S",
        "PIEAMP01CAM657S",
        "PIEAMP01AUM657S",
        "PIEAMP01NZM657S",
        "PIEAMP01CHM657S",
        "PIEAMP01USQ657S",
        "PIEAMP01DEQ657S",
    ],
    "gym_m661n_stale_~2022_12": [
        "USAPIEAMP01GYM",
        "DEUPIEAMP01GYM",
        "GBRPIEAMP01GYM",
        "CANPIEAMP01GYM",
        "CHEPIEAMP01GYM",
        "EA19PIEAMP01GYM",
        "FRAPIEAMP01GYM",
        "PIEAMP01USM661N",
        "PIEAMP01DEM661N",
        "PIEAMP01GBM661N",
    ],
    "jpy_oecd_pieamp_404_jpnppdm_primary": [
        "JPNPIEAMP01GYM",
        "PIEAMP01JPM661N",
        "PIEAMP01JPQ661N",
        "JPNPPDMMINMEI",
    ],
    "aud_nzd_gym_404_q661n_primary": [
        "AUSPIEAMP01GYM",
        "NZLPIEAMP01GYM",
        "PIEAMP01AUM661N",
        "PIEAMP01NZM661N",
        "PIEAMP01AUQ661N",
        "PIEAMP01NZQ661N",
    ],
    "eur_ea19_gym_stale_2022_12": ["EA19PIEAMP01GYM", "EA19PIEATI01GYM"],
    "eur_germany_proxy_primary": ["DEUPIEAMP01GYM"],
    "eur_france_alt_stale": ["FRAPIEAMP01GYM", "FRAPIEATI01GYM"],
    "us_national_alts_not_mixed": ["PPIACO", "PPIFIS", "PPIFES"],
}

EUR_EA19_STALE = "EA19PIEAMP01GYM"
EUR_FRA_ALT = "FRAPIEAMP01GYM"
US_PPIACO_ALT = "PPIACO"
US_PPIFIS_ALT = "PPIFIS"

DEFAULT_PUB_LAG_MONTHS = 2  # a priori: PPI ~1–2m; use 2 (same band as IP/retail)


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


def _load_series(series_id: str, *, freq: str, download: bool, force: bool) -> pd.Series:
    if download:
        download_fred_series(series_id, force=force)
    raw = load_fred_series(series_id, download=False)
    if freq.upper().startswith("Q"):
        return _to_quarter_start(raw)
    return _to_month_start(raw)


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


def load_ppi_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """OECD MEI / national PPI YoY → monthly PIT panel.

    Columns = ISO currency codes. Values = PPI **YoY %**. Index = UTC
    month-start of the PIT *known* date after ``pub_lag_months``. Level/index
    series convert to YoY before the pub lag. Quarterly YoY series are
    pub-lagged first, then forward-filled to month-start. Strategy modules
    XS-rank on these values (foreign only).
    """
    curs = [c.upper() for c in (currencies or PPI_SERIES.keys())]
    monthly_cols: dict[str, pd.Series] = {}
    quarterly_cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}
    freq_map: dict[str, str] = {}
    unit_map: dict[str, str] = {}

    for c in curs:
        sid = PPI_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_ppi"
            continue
        freq = PPI_FREQ.get(c, "M")
        unit = PPI_UNIT.get(c, "growth")
        try:
            s = _load_series(sid, freq=freq, download=download, force=force)
            if unit == "level":
                if freq == "Q":
                    s = (s.pct_change(4) * 100.0).dropna()
                else:
                    s = (s.pct_change(12) * 100.0).dropna()
                if c == "JPY":
                    notes[c] = (
                        f"OECD MEI {sid} levels → YoY % "
                        f"(JPNPIEAMP01GYM / PIEAMP01JP* 404; ends ~2022-12 stale)"
                    )
                elif c in ("AUD", "NZD"):
                    notes[c] = (
                        f"OECD MEI {sid} quarterly index → YoY % "
                        f"(AUS/NZL PIEAMP01GYM 404; ends ~2022-07..2023-01 stale)"
                    )
                else:
                    notes[c] = f"{sid} levels/index → YoY %"
            elif c == "EUR":
                notes[c] = (
                    f"Germany proxy via {sid} (M growth as reported) "
                    f"(EA19PIEAMP01GYM ends 2022-12; FRA GYM alt; GYSAM 404)"
                )
            else:
                notes[c] = (
                    f"OECD MEI manufacturing PPI YoY {sid} (monthly as reported; "
                    f"ends ~2022-12 stale)"
                )
            series_map[c] = sid
            freq_map[c] = freq
            unit_map[c] = unit
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
    if quarterly_cols:
        qdf = pd.DataFrame(quarterly_cols).sort_index()
        qdf = _expand_quarterly_to_monthly(qdf, pub_lag_months=pub_lag_months)
        frames.append(qdf)

    if not frames:
        df = pd.DataFrame()
    else:
        df = pd.concat(frames, axis=1, sort=True).sort_index()
        df = df.dropna(how="all")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

    df.attrs["factor"] = "oecd_mei_ppi_pieamp01"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["freq_map"] = freq_map
    df.attrs["unit_map"] = unit_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI {ISO3}PIEAMP01GYM manufacturing PPI YoY monthly "
        "(USD/GBP/CAD/CHF; EUR=DEU Germany proxy — EA19 GYM stale same end; "
        "JPY=JPNPPDMMINMEI levels→YoY — OECD PIEAMP Japan 404; "
        "AUD/NZD=PIEAMP01*Q661N quarterly index→YoY — GYM 404; "
        "panel STALE ~2022-12; PPIACO/PPIFIS US alts not mixed)"
    )
    df.attrs["unit"] = "ppi_yoy_pct"
    df.attrs["score_basis"] = "yoy_growth_as_reported_or_derived"
    df.attrs["unmapped"] = [c for c in curs if c not in series_map]
    df.attrs["failed_or_alt"] = PPI_FAILED_OR_ALT
    df.attrs["stale"] = True
    df.attrs["stale_note"] = (
        "OECD MEI PPI on free FRED ends ~2022-12 (AUD Q ~2023-01; NZD Q ~2022-07); "
        "post-2023 ranks ffilled frozen"
    )
    return df


def load_us_ppi(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US OECD MEI manufacturing PPI YoY (USAPIEAMP01GYM), PIT-lagged."""
    panel = load_ppi_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US PPI series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_PPI"
    out.attrs["series_id"] = PPI_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def ppi_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_ppi_panel(download=False)
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
        rows.append(
            {
                "currency": c,
                "fred_id": "",
                "freq": "",
                "n_obs": 0,
                "start": None,
                "end": None,
                "last": float("nan"),
                "note": notes.get(c, "unmapped_on_fred_ppi"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
